//! Concat operation: concatenates tensors along a given axis.

use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef, TensorBuilder};

/// Shader for the Concat operation.
#[derive(Shader)]
pub struct Concat {
    pub concat_copy: vortx_shaders::ml::ConcatCopy,
}

impl Concat {
    /// Copy a source tensor into a slice of the destination tensor along a given axis.
    ///
    /// This is used to implement concat by calling it once per input tensor.
    /// `offset` indicates where this tensor's data should be placed along the axis.
    pub fn launch_copy(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut dest: impl AsTensorMut<f32>,
        src: impl AsTensorRef<f32>,
        axis: u32,
        offset: u32,
    ) -> Result<(), GpuBackendError> {
        let mut dest = dest.as_tensor_mut();
        let src = src.as_tensor_ref();
        let len = src.len() as u32;
        let max_threads = 65535u32;

        // Upload params [axis, offset]
        let params_buf = TensorBuilder::scalar(BufferUsages::STORAGE | BufferUsages::COPY_DST)
            .build_init(backend, &[axis, offset])?;

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, dest.layout())?;
            shapes.insert(backend, src.layout())?;

            let shape_dest = shapes.get(dest.layout()).unwrap();
            let shape_src = shapes.get(src.layout()).unwrap();
            let mut buf_dest = dest.buffer_mut();

            self.concat_copy.call(
                pass,
                [len.min(max_threads), 1, 1],
                &shape_dest.as_slice(),
                &shape_src.as_slice(),
                &mut buf_dest,
                &src.buffer(),
                &params_buf.buffer().as_slice(),
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = crate::shaders::linalg::Shapes2 {
                shape_a: dest.layout().into(),
                shape_b: src.layout().into(),
            };
            let mut buf_dest = dest.buffer_mut();

            self.concat_copy.call(
                pass,
                [len.min(max_threads), 1, 1],
                &mut buf_dest,
                &src.buffer(),
                &params_buf.buffer().as_slice(),
                shapes_val,
            )
        }
    }
}
