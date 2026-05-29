use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef, Tensor};
use bytemuck::Pod;
use khal::Shader;
use khal::backend::{DeviceValue, GpuBackend, GpuBackendError, GpuPass};

// Use spirv_bindgen-generated wrapper types from the shader crate.
use crate::shaders::linalg::Contiguous as GpuContiguous;
use crate::shaders::linalg::ContiguousWithOffset;

#[derive(Shader)]
/// Module for conversion from a non-contiguous tensor into a contiguous tensor.
pub struct Contiguous {
    /// Shader for copying a non-contiguous tensor into a row-major contiguous tensor.
    pub contiguous: GpuContiguous,
    pub contiguous_with_offset: ContiguousWithOffset,
}

impl Contiguous {
    /// Launch the kernel that copies the content of a `tensor` with non-contiguous layout into
    /// the contiguous tensor `out`.
    ///
    /// Panics if `T` doesn't have a size of 4 bytes exactly.
    pub fn launch<T: DeviceValue + Pod>(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut out: impl AsTensorMut<T>,
        tensor: impl AsTensorRef<T>,
        offset: Option<&Tensor<u32>>,
    ) -> Result<(), GpuBackendError> {
        assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<u32>(),
            "Contiguous only supports tensors with 4-bytes elements."
        );
        let mut out = out.as_tensor_mut();
        let tensor = tensor.as_tensor_ref();
        let mut tensor_shape = tensor.layout();
        let out_shape = out.layout();
        assert_eq!(tensor_shape.size, out_shape.size);
        assert!(
            out.is_contiguous(),
            "Output tensor must be contiguous: {:?}.",
            out.layout()
        );

        // println!("Tensor shape: {:?}", tensor_shape);
        tensor_shape = tensor_shape.canonicalize();
        // println!("Tensor shape (canon): {:?}", tensor_shape);

        let num_threads = tensor_shape.len() as u32;

        if let Some(offset) = offset {
            #[cfg(not(feature = "push_constants"))]
            {
                shapes.insert(backend, tensor_shape)?;
                let shape = shapes.get(tensor_shape).unwrap_or_else(|| unreachable!());
                let buf_dest = out.buffer_mut();

                self.contiguous_with_offset.call(
                    pass,
                    num_threads,
                    &shape.as_slice(),
                    &mut buf_dest.cast(),
                    &tensor.raw_buffer().as_slice().cast(),
                    &offset.buffer().as_slice(),
                )
            }

            #[cfg(feature = "push_constants")]
            {
                let mut buf_dest = out.buffer_mut();

                self.contiguous_with_offset.call(
                    pass,
                    num_threads,
                    &mut buf_dest.cast(),
                    &tensor.raw_buffer().as_slice().cast(),
                    &offset.buffer().as_slice(),
                    crate::shaders::linalg::Shapes1 {
                        shape: tensor_shape.into(),
                    },
                )
            }
        } else {
            #[cfg(not(feature = "push_constants"))]
            {
                shapes.insert(backend, tensor_shape)?;
                let shape = shapes.get(tensor_shape).unwrap_or_else(|| unreachable!());
                let buf_dest = out.buffer_mut();

                self.contiguous.call(
                    pass,
                    num_threads,
                    &shape.as_slice(),
                    &mut buf_dest.cast(),
                    &tensor.buffer().cast(),
                )
            }

            #[cfg(feature = "push_constants")]
            {
                let mut buf_dest = out.buffer_mut();

                self.contiguous.call(
                    pass,
                    num_threads,
                    &mut buf_dest.cast(),
                    &tensor.buffer().cast(),
                    crate::shaders::linalg::Shapes1 {
                        shape: tensor_shape.into(),
                    },
                )
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
    use khal::{BufferUsages, Shader};
    use nalgebra::DMatrix;
    use wgpu::{Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_contiguous_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        gpu_contiguous_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gpu_contiguous_cpu() {
        gpu_contiguous_generic(&GpuBackend::Cpu).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    async fn gpu_contiguous_cuda() {
        let cuda = GpuBackend::Cuda(khal::backend::cuda::Cuda::new(0).unwrap());
        gpu_contiguous_generic(&cuda).await;
    }

    #[cfg(feature = "metal")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_contiguous_metal() {
        let metal = GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap());
        gpu_contiguous_generic(&metal).await;
    }

    async fn gpu_contiguous_generic(backend: &GpuBackend) {
        let contiguous = super::Contiguous::from_backend(backend).unwrap();

        let mut shapes = TensorLayoutBuffers::new(backend);

        const NROWS: u32 = 256;
        const NCOLS: u32 = 128;

        let tensor = DMatrix::<f32>::new_random(NROWS as usize, NCOLS as usize);
        let output = DMatrix::<f32>::new_random(NCOLS as usize, NROWS as usize);

        let gpu_tensor = Tensor::matrix_from_na(backend, &tensor, BufferUsages::STORAGE).unwrap();
        let mut gpu_output = Tensor::matrix_from_na(
            backend,
            &output,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("contiguous", None);
        contiguous
            .launch(
                backend,
                &mut shapes,
                &mut pass,
                &mut gpu_output,
                gpu_tensor.as_view().transpose(0, 1),
                None,
            )
            .unwrap();
        drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();
        let mut computed = vec![0.0; output.len()];
        backend
            .slow_read_buffer(gpu_output.buffer(), &mut computed)
            .await
            .unwrap();
        let expected = crate::linalg::to_row_major(&tensor.transpose());

        // NOTE: we don't use assert_eq because, in case of failure, it prints
        //       the entire buffer which tends to break the IDE's test runner.
        assert!(computed == expected);
    }
}
