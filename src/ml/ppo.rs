//! PPO loss-gradient host dispatch. Added for zealot's GPU policy update.
//!
//! Wraps the two PPO output-gradient kernels (clipped-surrogate actor gradient +
//! log_std contribution, and clipped value-loss gradient). These have no `Shape`
//! uniform — dimensions ride in the params struct and indexing is row-major — so
//! no `TensorLayoutBuffers` is needed.

use crate::shaders::linalg::{GpuPpoActorGrad, GpuPpoValueGrad};
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};

// Re-export the params structs from the shader crate.
pub use vortx_shaders::linalg::ppo::{PpoActorParams, PpoValueParams};

/// PPO loss-gradient kernels.
#[derive(Shader)]
pub struct Ppo {
    /// Clipped-surrogate actor gradient + log_std contribution.
    pub actor_grad: GpuPpoActorGrad,
    /// Clipped value-loss gradient.
    pub value_grad: GpuPpoValueGrad,
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

        let num_threads = adv.len() as u32; // one thread per sample column
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

        let num_threads = v_pred.len() as u32;
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
}
