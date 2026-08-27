//! Backward passes of the element-wise activations (host dispatch).
//!
//! The forward directions are provided by [`crate::ml::Unary`]
//! (`UnaryOp::Tanh`, `UnaryOp::Elu`); only the gradients live here.

use crate::shaders::ml::GpuTanhBackward;
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};

/// Element-wise activation gradient kernels.
#[derive(Shader)]
pub struct ActivationBackward {
    /// In-place tanh backward (`g *= 1 - y^2`).
    pub tanh_backward: GpuTanhBackward,
}

impl ActivationBackward {
    /// In-place tanh backward: `g *= 1 - y^2`, where `y = tanh(x)` is the forward output.
    /// `g` and `y` must have the same shape.
    pub fn tanh_backward(
        &self,
        backend: &GpuBackend,
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut g: impl AsTensorMut<f32>,
        y: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut g = g.as_tensor_mut();
        let y = y.as_tensor_ref();
        let shape_g = g.layout().canonicalize();
        let shape_y = y.layout().canonicalize();
        let num_threads = g.len() as u32;

        shapes.insert(backend, shape_g)?;
        shapes.insert(backend, shape_y)?;
        let shape_g_buf = shapes.get(shape_g).unwrap();
        let shape_y_buf = shapes.get(shape_y).unwrap();
        let mut buf_g = g.buffer_mut();

        self.tanh_backward.call(
            pass,
            num_threads,
            &shape_g_buf.as_slice(),
            &shape_y_buf.as_slice(),
            &mut buf_g,
            &y.buffer(),
        )
    }
}
