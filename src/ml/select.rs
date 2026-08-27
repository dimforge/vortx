use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};

#[derive(Shader)]
pub struct Select {
    pub select: vortx_shaders::ml::select::Select,
}

impl Select {
    pub fn launch(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut dest: impl AsTensorMut<f32>,
        src: impl AsTensorRef<f32>,
        idx: impl AsTensorRef<u32>,
    ) -> Result<(), GpuBackendError> {
        let mut dest = dest.as_tensor_mut();
        let src = src.as_tensor_ref();
        let idx = idx.as_tensor_ref();
        let len = dest.len() as u32;

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, dest.layout())?;
            shapes.insert(backend, src.layout())?;

            let shape_dest = shapes.get(dest.layout()).unwrap();
            let shape_src = shapes.get(src.layout()).unwrap();
            let mut buf_dest = dest.buffer_mut();

            self.select.call(
                pass,
                [len, 1, 1],
                &shape_dest.as_slice(),
                &shape_src.as_slice(),
                &mut buf_dest,
                &src.buffer(),
                &idx.buffer(),
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = crate::shaders::linalg::Shapes2 {
                shape_a: dest.layout().into(),
                shape_b: src.layout().into(),
            };
            let mut buf_dest = dest.buffer_mut();

            self.select.call(
                pass,
                [len, 1, 1],
                &mut buf_dest,
                &src.buffer(),
                &idx.buffer(),
                shapes_val,
            )
        }
    }
}

#[cfg(test)]
mod test {
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
    use khal::{BufferUsages, Shader};
    use wgpu::{Features, Limits};

    /// Select rows from a matrix by index: dest[i] = src[idx[i], :].
    async fn test_select_generic(backend: &GpuBackend) {
        let select = super::Select::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        // Source matrix 8x4
        let src_data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let src = Tensor::matrix(backend, 8, 4, &src_data, BufferUsages::STORAGE).unwrap();

        // Index vector: pick rows [2, 0, 5, 7]
        let idx_data: Vec<u32> = vec![2, 0, 5, 7];
        let idx = Tensor::vector(backend, &idx_data, BufferUsages::STORAGE).unwrap();

        // Destination matrix 4x4
        let mut dest = Tensor::<f32>::matrix_uninit(
            backend,
            4,
            4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        select
            .launch(backend, &mut shapes, &mut pass, &mut dest, &src, &idx)
            .unwrap();
        drop(pass);
        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        let mut result = vec![0.0f32; 16];
        backend
            .slow_read_buffer(dest.buffer(), &mut result)
            .await
            .unwrap();

        // Expected: rows 2, 0, 5, 7 of the source
        let expected: Vec<f32> = vec![
            8.0, 9.0, 10.0, 11.0, // row 2
            0.0, 1.0, 2.0, 3.0, // row 0
            20.0, 21.0, 22.0, 23.0, // row 5
            28.0, 29.0, 30.0, 31.0, // row 7
        ];
        assert_eq!(result, expected);
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn select_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_select_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn select_cpu() {
        let backend = GpuBackend::Cpu;
        test_select_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn select_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_select_generic(&backend).await;
    }
}
