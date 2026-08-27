//! PPO loss-gradient host dispatch.
//!
//! Wraps the two PPO output-gradient kernels (clipped-surrogate actor gradient
//! plus log_std contribution, and clipped value-loss gradient). They need no
//! `Shape` uniform or `TensorLayoutBuffers`: dimensions ride in the params
//! struct and indexing is row-major.

use crate::shaders::ml::{GpuPpoActorGrad, GpuPpoStageBatch, GpuPpoValueGrad};
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};

// Re-export the params structs from the shader crate.
pub use vortx_shaders::ml::ppo::{PpoActorParams, PpoStageParams, PpoValueParams};

/// PPO loss-gradient kernels.
#[derive(Shader)]
pub struct Ppo {
    /// Clipped-surrogate actor gradient + log_std contribution.
    pub actor_grad: GpuPpoActorGrad,
    /// Clipped value-loss gradient.
    pub value_grad: GpuPpoValueGrad,
    /// On-device staging of the PPO minibatch from raw rollout observations.
    pub stage_batch: GpuPpoStageBatch,
}

impl Ppo {
    /// Actor PPO gradient. All per-sample tensors are row-major `[action_dim x M]`
    /// except `log_std` (`[action_dim]`), `adv` / `logp_old` (`[M]`). Writes
    /// `g_mean` and `g_logstd` (`[action_dim x M]`). `params.num_cols` must equal `M`.
    #[allow(clippy::too_many_arguments)]
    pub fn actor_grad(
        &self,
        pass: &mut GpuPass,
        params: impl AsTensorRef<PpoActorParams>,
        mean: impl AsTensorRef<f32>,
        action: impl AsTensorRef<f32>,
        log_std: impl AsTensorRef<f32>,
        adv: impl AsTensorRef<f32>,
        logp_old: impl AsTensorRef<f32>,
        mut g_mean: impl AsTensorMut<f32>,
        mut g_logstd: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let params = params.as_tensor_ref();
        let mean = mean.as_tensor_ref();
        let action = action.as_tensor_ref();
        let log_std = log_std.as_tensor_ref();
        let adv = adv.as_tensor_ref();
        let logp_old = logp_old.as_tensor_ref();
        let mut g_mean = g_mean.as_tensor_mut();
        let mut g_logstd = g_logstd.as_tensor_mut();

        // One thread per sample column, clamped to the kernel's stride.
        let num_threads = (adv.len() as u32).min(vortx_shaders::ml::ppo::MAX_NUM_THREADS);
        let mut buf_g_mean = g_mean.buffer_mut();
        let mut buf_g_logstd = g_logstd.buffer_mut();

        self.actor_grad.call(
            pass,
            num_threads,
            &params.buffer(),
            &mean.buffer(),
            &action.buffer(),
            &log_std.buffer(),
            &adv.buffer(),
            &logp_old.buffer(),
            &mut buf_g_mean,
            &mut buf_g_logstd,
        )
    }

    /// Clipped value-loss gradient. `v_pred` / `value_old` / `ret` are `[M]`;
    /// writes `g_v` (`[M]`). `params.num_cols` must equal `M`.
    pub fn value_grad(
        &self,
        pass: &mut GpuPass,
        params: impl AsTensorRef<PpoValueParams>,
        v_pred: impl AsTensorRef<f32>,
        value_old: impl AsTensorRef<f32>,
        ret: impl AsTensorRef<f32>,
        mut g_v: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let params = params.as_tensor_ref();
        let v_pred = v_pred.as_tensor_ref();
        let value_old = value_old.as_tensor_ref();
        let ret = ret.as_tensor_ref();
        let mut g_v = g_v.as_tensor_mut();

        let num_threads = (v_pred.len() as u32).min(vortx_shaders::ml::ppo::MAX_NUM_THREADS);
        let mut buf_g_v = g_v.buffer_mut();

        self.value_grad.call(
            pass,
            num_threads,
            &params.buffer(),
            &v_pred.buffer(),
            &value_old.buffer(),
            &ret.buffer(),
            &mut buf_g_v,
        )
    }

    /// Stages (one half of) the PPO batch on device: reads the step-blocked raw
    /// rollout observations, applies the signed-perm mirror, the normalizer
    /// affine and the ±5 clamp, and writes row-major `[dim x total_cols]`
    /// columns starting at `params.col_offset`.
    ///
    /// `mean` / `inv_std` / `perm` / `sign` are all `[dim]`; pass identity
    /// tables in `perm` / `sign` for the un-mirrored half. See
    /// [`gpu_ppo_stage_batch`](vortx_shaders::ml::ppo::gpu_ppo_stage_batch) for
    /// the layout contract.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_batch(
        &self,
        pass: &mut GpuPass,
        params: impl AsTensorRef<PpoStageParams>,
        raw: impl AsTensorRef<f32>,
        mean: impl AsTensorRef<f32>,
        inv_std: impl AsTensorRef<f32>,
        perm: impl AsTensorRef<u32>,
        sign: impl AsTensorRef<f32>,
        mut out: impl AsTensorMut<f32>,
        cols: u32,
        dim: u32,
    ) -> Result<(), GpuBackendError> {
        let params = params.as_tensor_ref();
        let raw = raw.as_tensor_ref();
        let mean = mean.as_tensor_ref();
        let inv_std = inv_std.as_tensor_ref();
        let perm = perm.as_tensor_ref();
        let sign = sign.as_tensor_ref();
        let mut out = out.as_tensor_mut();

        let mut buf_out = out.buffer_mut();

        self.stage_batch.call(
            pass,
            [cols, dim, 1],
            &params.buffer(),
            &raw.buffer(),
            &mean.buffer(),
            &inv_std.buffer(),
            &perm.buffer(),
            &sign.buffer(),
            &mut buf_out,
        )
    }
}
