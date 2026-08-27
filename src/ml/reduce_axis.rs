//! Axis-based reduction operations (ReduceSum, ReduceMean, etc.)

use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef, TensorBuilder};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};

/// Type of reduction operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// Shader for axis-based reduction operations.
#[derive(Shader)]
pub struct ReduceAxis {
    pub reduce_sum_axis: vortx_shaders::ml::reduce_axis::ReduceSumAxis,
    pub reduce_mean_axis: vortx_shaders::ml::reduce_axis::ReduceMeanAxis,
    pub reduce_max_axis: vortx_shaders::ml::reduce_axis::ReduceMaxAxis,
    pub reduce_min_axis: vortx_shaders::ml::reduce_axis::ReduceMinAxis,
}

impl ReduceAxis {
    /// Launch a reduction operation along a single axis.
    ///
    /// `axis` is the axis to reduce along (0-3).
    /// `reduce_size` is the size of the dimension being reduced.
    pub fn launch(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        op: ReduceOp,
        mut dest: impl AsTensorMut<f32>,
        src: impl AsTensorRef<f32>,
        axis: u32,
        reduce_size: u32,
    ) -> Result<(), GpuBackendError> {
        let mut dest = dest.as_tensor_mut();
        let src = src.as_tensor_ref();
        let len = dest.len() as u32;
        let max_threads = 65535u32;

        // Upload params [axis, reduce_size]
        let params_buf = TensorBuilder::scalar(BufferUsages::STORAGE | BufferUsages::COPY_DST)
            .build_init(backend, &[axis, reduce_size])?;

        macro_rules! dispatch_reduce {
            ($wrapper:expr) => {{
                #[cfg(not(feature = "push_constants"))]
                {
                    shapes.insert(backend, dest.layout())?;
                    shapes.insert(backend, src.layout())?;

                    let shape_dest = shapes.get(dest.layout()).unwrap();
                    let shape_src = shapes.get(src.layout()).unwrap();
                    let mut buf_dest = dest.buffer_mut();

                    $wrapper.call(
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

                    $wrapper.call(
                        pass,
                        [len.min(max_threads), 1, 1],
                        &mut buf_dest,
                        &src.buffer(),
                        &params_buf.buffer().as_slice(),
                        shapes_val,
                    )
                }
            }};
        }

        match op {
            ReduceOp::Sum => dispatch_reduce!(&self.reduce_sum_axis),
            ReduceOp::Mean => dispatch_reduce!(&self.reduce_mean_axis),
            ReduceOp::Max => dispatch_reduce!(&self.reduce_max_axis),
            ReduceOp::Min => dispatch_reduce!(&self.reduce_min_axis),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
    use khal::{BufferUsages, Shader};
    use wgpu::{Features, Limits};

    async fn test_reduce_sum_axis_generic(backend: &GpuBackend) {
        let reduce = ReduceAxis::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        // Input: 3x4 matrix
        let src_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // row 0: sum=10
            5.0, 6.0, 7.0, 8.0, // row 1: sum=26
            9.0, 10.0, 11.0, 12.0, // row 2: sum=42
        ];
        let src = Tensor::matrix(backend, 3, 4, &src_data, BufferUsages::STORAGE).unwrap();

        // Reduce along axis 1 (columns) → output shape [3, 1]
        let mut dest = Tensor::<f32>::vector_uninit(
            backend,
            3,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        reduce
            .launch(
                backend,
                &mut shapes,
                &mut pass,
                ReduceOp::Sum,
                &mut dest,
                &src,
                1,
                4,
            )
            .unwrap();
        drop(pass);
        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        let mut result = vec![0.0f32; 3];
        backend
            .slow_read_buffer(dest.buffer(), &mut result)
            .await
            .unwrap();

        let expected = [10.0, 26.0, 42.0];
        for (i, (a, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-4,
                "ReduceSum row {i}: gpu={a} expected={e}"
            );
        }
    }

    async fn test_reduce_max_axis_generic(backend: &GpuBackend) {
        let reduce = ReduceAxis::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        let src_data: Vec<f32> = vec![
            3.0, 1.0, 4.0, 1.0, // row 0: max=4
            5.0, 9.0, 2.0, 6.0, // row 1: max=9
        ];
        let src = Tensor::matrix(backend, 2, 4, &src_data, BufferUsages::STORAGE).unwrap();

        let mut dest = Tensor::<f32>::vector_uninit(
            backend,
            2,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        reduce
            .launch(
                backend,
                &mut shapes,
                &mut pass,
                ReduceOp::Max,
                &mut dest,
                &src,
                1,
                4,
            )
            .unwrap();
        drop(pass);
        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        let mut result = vec![0.0f32; 2];
        backend
            .slow_read_buffer(dest.buffer(), &mut result)
            .await
            .unwrap();

        assert!(
            (result[0] - 4.0).abs() < 1e-4,
            "ReduceMax row 0: {}",
            result[0]
        );
        assert!(
            (result[1] - 9.0).abs() < 1e-4,
            "ReduceMax row 1: {}",
            result[1]
        );
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn reduce_sum_axis_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_reduce_sum_axis_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn reduce_sum_axis_cpu() {
        let backend = GpuBackend::Cpu;
        test_reduce_sum_axis_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn reduce_sum_axis_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_reduce_sum_axis_generic(&backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn reduce_max_axis_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_reduce_max_axis_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn reduce_max_axis_cpu() {
        let backend = GpuBackend::Cpu;
        test_reduce_max_axis_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn reduce_max_axis_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_reduce_max_axis_generic(&backend).await;
    }
}
