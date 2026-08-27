//! PPO loss-gradient kernels.
//!
//! These produce the per-sample output gradients that feed the generic
//! GEMM/`elu_backward` backward backbone: the clipped-surrogate actor gradient
//! `g_mean` plus the state-independent `log_std` gradient contribution, and the
//! clipped value-loss gradient. Every per-sample tensor is row-major
//! `[rows x M]` (M = minibatch columns); one GPU thread handles one sample
//! column `m`, looping over the (small) action dimension internally.

use crate::utils::iterators::StepRng;
use crate::utils::limits::MAX_NUM_WORKGROUPS;
use khal_std::glamx::UVec3;
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

const WORKGROUP_SIZE: u32 = 256;
/// Largest dispatch these kernels accept: they stride by exactly this, so a host
/// dispatch must be clamped to it (fewer threads leave a gap in the stride, more
/// overruns the 65535-workgroup limit).
pub const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

// Guards `WORKGROUP_SIZE` against the `threads(...)` attribute it duplicates.
#[cfg(not(target_arch_is_gpu))]
static_assertions::const_assert_eq!(
    WORKGROUP_SIZE,
    <GpuPpoActorGradArgs<'static> as khal::shader::ShaderArgsType>::WORKGROUP_SIZE[0]
);

/// Scalar parameters for the actor PPO gradient (uniform buffer; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PpoActorParams {
    /// PPO clip epsilon.
    pub clip: f32,
    /// Entropy bonus coefficient (subtracted from the log_std gradient).
    pub entropy_coef: f32,
    /// Per-sample averaging factor `1 / minibatch_size`.
    pub scale: f32,
    /// `0.5·ln(2π)`: the Gaussian log-prob normalisation constant.
    pub log_sqrt_2pi: f32,
    /// Action dimensionality (rows).
    pub action_dim: u32,
    /// Number of sample columns `M`.
    pub num_cols: u32,
    pub pad0: u32,
    pub pad1: u32,
}

/// Scalar parameters for the clipped value-loss gradient (uniform; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PpoValueParams {
    /// PPO clip epsilon (value clipping range).
    pub clip: f32,
    /// Value-loss coefficient.
    pub value_coef: f32,
    /// Per-sample averaging factor `1 / minibatch_size`.
    pub scale: f32,
    /// Number of sample columns `M`.
    pub num_cols: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

/// Clipped-surrogate actor gradient + log_std gradient contribution, per sample.
///
/// For sample column `m` (one thread): compute the new diagonal-Gaussian
/// log-prob over the `action_dim` rows, the importance ratio
/// `exp(logp − logp_old)`, the PPO clip mask, then write `g_mean[k,m]` and
/// `g_logstd[k,m]` for every action dim `k`:
///   if !clipped: g_mean = −(adv·ratio·d/σ²)·scale,
///                g_logstd += −adv·ratio·(d²/σ² − 1)·scale,
///   always:      g_logstd += −entropy_coef·scale.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_ppo_actor_grad(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &PpoActorParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mean: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] action: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] log_std: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] adv: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] logp_old: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] g_mean: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] g_logstd: &mut [f32],
) {
    let a = params.action_dim as usize;
    let m_cols = params.num_cols as usize;
    let clip = params.clip;
    let scale = params.scale;
    let ent = params.entropy_coef;
    for m in StepRng::new(invocation_id.x..m_cols as u32, MAX_NUM_THREADS) {
        let m = m as usize;
        // New log-prob over the action dims.
        let mut logp = 0.0f32;
        for k in 0..a {
            let idx = k * m_cols + m;
            let ls = log_std.read(k);
            let std = ls.exp();
            let d = (action.read(idx) - mean.read(idx)) / std;
            logp += -0.5 * d * d - ls - params.log_sqrt_2pi;
        }
        let ratio = (logp - logp_old.read(m)).exp();
        let av = adv.read(m);
        let clipped = (av >= 0.0 && ratio > 1.0 + clip) || (av < 0.0 && ratio < 1.0 - clip);
        for k in 0..a {
            let idx = k * m_cols + m;
            let ls = log_std.read(k);
            let inv_var = (-2.0 * ls).exp(); // 1/σ²
            if clipped {
                *g_mean.at_mut(idx) = 0.0;
                *g_logstd.at_mut(idx) = -ent * scale;
            } else {
                let d = action.read(idx) - mean.read(idx);
                *g_mean.at_mut(idx) = -(av * ratio * d * inv_var) * scale;
                let dls = av * ratio * (d * d * inv_var - 1.0);
                *g_logstd.at_mut(idx) = (-dls - ent) * scale;
            }
        }
    }
}

/// Clipped value-loss gradient, per sample.
///
/// For sample column `m`: `v_clipped = value_old + clamp(v − value_old, ±clip)`,
/// and `dv = 2·(v_clipped − ret)` if the clipped squared error is larger else
/// `2·(v − ret)`; writes `g_v[m] = value_coef·dv·scale`.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_ppo_value_grad(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &PpoValueParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] v_pred: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] value_old: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] ret: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] g_v: &mut [f32],
) {
    let m_cols = params.num_cols as usize;
    let clip = params.clip;
    let scale = params.scale;
    for m in StepRng::new(invocation_id.x..m_cols as u32, MAX_NUM_THREADS) {
        let m = m as usize;
        let v = v_pred.read(m);
        let vo = value_old.read(m);
        let r = ret.read(m);
        let diff = v - vo;
        let clamped = if diff > clip {
            clip
        } else if diff < -clip {
            -clip
        } else {
            diff
        };
        let v_clipped = vo + clamped;
        let l_un = (v - r) * (v - r);
        let l_cl = (v_clipped - r) * (v_clipped - r);
        let dv = if l_cl > l_un {
            2.0 * (v_clipped - r)
        } else {
            2.0 * (v - r)
        };
        *g_v.at_mut(m) = params.value_coef * dv * scale;
    }
}

/// Scalar parameters for the PPO batch staging (uniform buffer; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct PpoStageParams {
    /// Observation dimensionality (rows).
    pub dim: u32,
    /// Environments per rollout step.
    pub n: u32,
    /// Rollout steps `T`; the raw buffer is step-blocked `[T][dim][n]`.
    pub steps: u32,
    /// Total batch columns of `out` (its row stride).
    pub total_cols: u32,
    /// First output column this dispatch writes (the mirrored or the original
    /// half of the batch).
    pub col_offset: u32,
    /// 0 = batch mode: the columns cover all `T·n` samples, env-major.
    /// Otherwise single-step mode, staging only rollout step `step_select - 1`,
    /// so the columns are the `n` envs (the per-step policy-input staging).
    pub step_select: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// Builds (one half of) the `[dim x total_cols]` row-major PPO batch straight
/// from the step-blocked raw rollout observations, applying the signed-perm
/// mirror, the normalizer affine and the ±5 clamp in one dispatch.
///
/// The mirror is an explicit signed permutation (`perm`/`sign`, identity tables
/// for the un-mirrored half). It has to be applied to raw observations, ahead of
/// the lossy clamp: a mirror of already-normalized values is wrong for every
/// saturated feature.
///
/// Batch columns are env-major (`col = e·T + t`) while the raw buffer is
/// step-blocked, hence the `raw[(t·dim + perm[d])·n + e]` gather. Dispatch
/// `[cols, dim, 1]` threads.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_ppo_stage_batch(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &PpoStageParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] raw: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mean: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] inv_std: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] perm: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sign: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] out: &mut [f32],
) {
    // `x` walks the batch columns (contiguous within an `out` row, so the writes
    // coalesce), `y` the observation dimensions. A flat 1-D dispatch of
    // `cols · dim` threads would overrun the 65535-workgroup limit.
    let cols = if params.step_select != 0 {
        params.n
    } else {
        params.steps * params.n
    };
    let x = invocation_id.x;
    let d = invocation_id.y;
    if x >= cols || d >= params.dim {
        return;
    }
    let (t, e) = if params.step_select != 0 {
        (params.step_select - 1, x)
    } else {
        (x % params.steps, x / params.steps)
    };
    let src_d = perm.read(d as usize);
    let v = raw.read(((t * params.dim + src_d) * params.n + e) as usize) * sign.read(d as usize);
    let v = ((v - mean.read(d as usize)) * inv_std.read(d as usize))
        .max(-5.0)
        .min(5.0);
    out.write((d * params.total_cols + params.col_offset + x) as usize, v);
}
