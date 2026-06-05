//! Element-wise activation functions (host dispatch).
//!
//! Added for zealot's MLP policy — vortx upstream has no activations.

use crate::shaders::linalg::{GpuElu, GpuEluBackward, GpuEluVec4, GpuTanh, GpuTanhBackward};
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};

/// Element-wise activation kernels.
#[derive(Shader)]
pub struct Activation {
    /// In-place tanh.
    pub tanh: GpuTanh,
    /// In-place tanh backward (`g *= 1 - y^2`).
    pub tanh_backward: GpuTanhBackward,
    /// In-place ELU.
    pub elu: GpuElu,
    /// In-place ELU backward (`g *= 1 if y > 0 else y + 1`).
    pub elu_backward: GpuEluBackward,
    /// In-place ELU, vec4 (4 contiguous f32 per thread).
    pub elu_vec4: GpuEluVec4,
}

impl Activation {
    /// In-place tanh: `a = tanh(a)`.
    pub fn tanh(
        &self,
        backend: &GpuBackend,
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut a: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut a = a.as_tensor_mut();
        let shape_a = a.layout().canonicalize();
        let num_threads = a.len() as u32;

        shapes.insert(backend, shape_a)?;
        let shape_a_buf = shapes.get(shape_a).unwrap();
        let mut buf_a = a.buffer_mut();

        self.tanh
            .call(pass, num_threads, &shape_a_buf.as_slice(), &mut buf_a)
    }

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

    /// In-place ELU (alpha = 1): `a = a if a > 0 else exp(a) - 1`.
    pub fn elu(
        &self,
        backend: &GpuBackend,
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut a: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut a = a.as_tensor_mut();
        let shape_a = a.layout().canonicalize();
        let num_threads = a.len() as u32;

        shapes.insert(backend, shape_a)?;
        let shape_a_buf = shapes.get(shape_a).unwrap();
        let mut buf_a = a.buffer_mut();

        self.elu
            .call(pass, num_threads, &shape_a_buf.as_slice(), &mut buf_a)
    }

    /// In-place ELU, vec4: 4 contiguous f32 per thread (128-bit transactions).
    /// Buffer length must be a multiple of 4 and contiguous (dense activations).
    pub fn elu_vec4(
        &self,
        backend: &GpuBackend,
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut a: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut a = a.as_tensor_mut();
        let shape_a = a.layout().canonicalize();
        let num_threads = (a.len() / 4) as u32;

        shapes.insert(backend, shape_a)?;
        let shape_a_buf = shapes.get(shape_a).unwrap();
        let buf_a = a.buffer_mut();
        // Same bytes, viewed as vec4 (4 f32 -> 1 Vec4) for 128-bit transactions.
        let mut buf_v4 = buf_a.reinterpret::<glamx::Vec4>();

        self.elu_vec4
            .call(pass, num_threads, &shape_a_buf.as_slice(), &mut buf_v4)
    }

    /// In-place ELU backward: `g *= 1 if y > 0 else y + 1`, where `y = elu(x)` is
    /// the cached forward output. `g` and `y` must have the same shape.
    pub fn elu_backward(
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

        self.elu_backward.call(
            pass,
            num_threads,
            &shape_g_buf.as_slice(),
            &shape_y_buf.as_slice(),
            &mut buf_g,
            &y.buffer(),
        )
    }
}
