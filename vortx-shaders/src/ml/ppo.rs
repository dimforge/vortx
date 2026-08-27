//! PPO loss-gradient kernels (added for zealot's GPU policy update).
//!
//! These produce the per-sample OUTPUT gradients that feed the generic
//! GEMM/`elu_backward` backward backbone: the clipped-surrogate actor gradient
//! `g_mean` plus the state-independent `log_std` gradient contribution, and the
//! clipped value-loss gradient. An exact port of `zealot-rl`'s `minibatch_step`
//! (ppo.rs). Every per-sample tensor is row-major `[rows x M]` (M = minibatch
//! columns); one GPU thread handles one sample column `m`, looping over the
//! (small) action dimension internally.

use crate::utils::limits::MAX_NUM_WORKGROUPS;
use glamx::UVec3;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;

const WORKGROUP_SIZE: u32 = 256;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Scalar parameters for the actor PPO gradient (uniform buffer; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct PpoActorParams {
    /// PPO clip epsilon.
    pub clip: f32,
    /// Entropy bonus coefficient (subtracted from the log_std gradient).
    pub entropy_coef: f32,
    /// Per-sample averaging factor `1 / minibatch_size`.
    pub scale: f32,
    /// `0.5·ln(2π)` — the Gaussian log-prob normalisation constant.
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
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
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
/// `g_logstd[k,m]` for every action dim `k`. Matches `minibatch_step`:
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
    for m in (invocation_id.x as usize..m_cols).step_by(MAX_NUM_THREADS as usize) {
        // New log-prob over the action dims (matches ActorCritic::logp).
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
        let clipped =
            (av >= 0.0 && ratio > 1.0 + clip) || (av < 0.0 && ratio < 1.0 - clip);
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
/// `2·(v − ret)`; writes `g_v[m] = value_coef·dv·scale`. Matches `minibatch_step`.
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
    for m in (invocation_id.x as usize..m_cols).step_by(MAX_NUM_THREADS as usize) {
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
