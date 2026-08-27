//! Gather operation: gathers elements from a tensor based on indices along an axis.

use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef, TensorBuilder};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};

/// Shader for the Gather operation.
#[derive(Shader)]
pub struct Gather {
    pub gather: vortx_shaders::ml::gather::Gather,
}

impl Gather {
    /// Launch the gather operation.
    ///
    /// Gathers elements from `src` along `axis` based on `indices`, writing to `dest`.
    pub fn launch(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut dest: impl AsTensorMut<f32>,
        src: impl AsTensorRef<f32>,
        indices: impl AsTensorRef<i32>,
        axis: u32,
    ) -> Result<(), GpuBackendError> {
        let mut dest = dest.as_tensor_mut();
        let src = src.as_tensor_ref();
        let indices = indices.as_tensor_ref();
        let len = dest.len() as u32;
        let max_threads = 65535u32;

        // Upload axis parameter
        let axis_buf = TensorBuilder::scalar(BufferUsages::STORAGE | BufferUsages::COPY_DST)
            .build_init(backend, &[axis])?;

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, dest.layout())?;
            shapes.insert(backend, src.layout())?;

            let shape_dest = shapes.get(dest.layout()).unwrap();
            let shape_src = shapes.get(src.layout()).unwrap();
            let mut buf_dest = dest.buffer_mut();

            self.gather.call(
                pass,
                [len.min(max_threads), 1, 1],
                &shape_dest.as_slice(),
                &shape_src.as_slice(),
                &mut buf_dest,
                &src.buffer(),
                &indices.buffer(),
                &axis_buf.buffer().as_slice(),
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = crate::shaders::linalg::Shapes2 {
                shape_a: dest.layout().into(),
                shape_b: src.layout().into(),
            };
            let mut buf_dest = dest.buffer_mut();

            self.gather.call(
                pass,
                [len.min(max_threads), 1, 1],
                &mut buf_dest,
                &src.buffer(),
                &indices.buffer(),
                &axis_buf.buffer().as_slice(),
                shapes_val,
            )
        }
    }
}
