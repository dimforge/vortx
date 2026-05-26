//! Optimizer host dispatch (Adam). Added for zealot.

use crate::shaders::linalg::GpuAdam;
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};

// Re-export the params struct from the shader crate.
pub use vortx_shaders::linalg::optim::AdamParams;

/// The Adam optimizer kernel.
#[derive(Shader)]
pub struct Adam {
    /// One in-place Adam update step.
    pub adam: GpuAdam,
}

impl Adam {
    /// Performs one in-place Adam step: updates `theta`, `m`, `v` from `grad`.
    ///
    /// `params` is a scalar `Tensor<AdamParams>` (UNIFORM usage); `theta`, `grad`,
    /// `m`, `v` all share the same shape.
    pub fn step(
        &self,
        backend: &GpuBackend,
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        params: impl AsTensorRef<AdamParams>,
        mut theta: impl AsTensorMut<f32>,
        grad: impl AsTensorRef<f32>,
        mut m: impl AsTensorMut<f32>,
        mut v: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let params = params.as_tensor_ref();
        let mut theta = theta.as_tensor_mut();
        let grad = grad.as_tensor_ref();
        let mut m = m.as_tensor_mut();
        let mut v = v.as_tensor_mut();

        let shape = theta.layout().canonicalize();
        let num_threads = theta.len() as u32;

        shapes.insert(backend, shape)?;
        let shape_buf = shapes.get(shape).unwrap();
        let mut buf_theta = theta.buffer_mut();
        let mut buf_m = m.buffer_mut();
        let mut buf_v = v.buffer_mut();

        self.adam.call(
            pass,
            num_threads,
            &shape_buf.as_slice(),
            &params.buffer(),
            &mut buf_theta,
            &grad.buffer(),
            &mut buf_m,
            &mut buf_v,
        )
    }
}
