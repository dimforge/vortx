use khal::backend::{DispatchGrid, GpuBackend, GpuBackendError, GpuPass};
use khal::Shader;
use nalgebra::DVector;
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef};

#[derive(Shader)]
/// Shader implementing the layer normalization kernel.
pub struct LayerNorm {
    pub layernorm_cols: vortx_shaders::ml::layernorm::LayernormCols,
    pub layernorm_rows: vortx_shaders::ml::layernorm::LayernormRows,
}

impl LayerNorm {
    pub fn launch_cols(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut output: impl AsTensorMut<f32>,
        input: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let input = input.as_tensor_ref().canonicalize();
        let mut output = output.as_tensor_mut().canonicalize();
        let shape_input = input.layout();
        let shape_output = output.layout();
        assert_eq!(
            shape_input.size, shape_output.size,
            "LayerNorm: dimension mismatch."
        );

        let grid = [
            shape_input.size[3],
            shape_input.size[1],
            shape_input.size[0],
        ];

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, shape_input)?;
            shapes.insert(backend, shape_output)?;
            let in_shape = shapes.get(shape_input).unwrap();
            let out_shape = shapes.get(shape_output).unwrap();
            let mut out_buf = output.buffer_mut();

            self.layernorm_cols.call(
                pass,
                DispatchGrid::Grid(grid),
                &in_shape.as_slice(),
                &out_shape.as_slice(),
                &input.buffer(),
                &mut out_buf,
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = shape_input.into();
            let mut out_buf = output.buffer_mut();

            self.layernorm_cols.call(
                pass,
                DispatchGrid::Grid(grid),
                &input.buffer(),
                &mut out_buf,
                shapes_val,
            )
        }
    }

    pub fn launch_rows(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut output: impl AsTensorMut<f32>,
        input: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let input = input.as_tensor_ref().canonicalize();
        let mut output = output.as_tensor_mut().canonicalize();
        let shape_input = input.layout();
        let shape_output = output.layout();

        assert_eq!(
            shape_input.size, shape_output.size,
            "LayerNorm: dimension mismatch."
        );

        let grid = [
            shape_input.size[2],
            shape_input.size[1],
            shape_input.size[0],
        ];

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, shape_input)?;
            shapes.insert(backend, shape_output)?;
            let in_shape = shapes.get(shape_input).unwrap();
            let out_shape = shapes.get(shape_output).unwrap();
            let mut out_buf = output.buffer_mut();

            self.layernorm_rows.call(
                pass,
                DispatchGrid::Grid(grid),
                &in_shape.as_slice(),
                &out_shape.as_slice(),
                &input.buffer(),
                &mut out_buf,
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = shape_input.into();
            let mut out_buf = output.buffer_mut();

            self.layernorm_rows.call(
                pass,
                DispatchGrid::Grid(grid),
                &input.buffer(),
                &mut out_buf,
                shapes_val,
            )
        }
    }

    /// The layernorm function.
    ///
    /// See <https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html> for details on the
    /// math.
    pub fn run_cpu(res: &mut DVector<f32>, v: &DVector<f32>) {
        const NUDGE_FACTOR: f32 = 1.0e-5;
        let mean = v.mean();
        res.zip_apply(v, |y, v| *y = v - mean);
        let variance = res.norm_squared() / (res.len() as f32);
        let scale = 1.0 / (variance + NUDGE_FACTOR).sqrt();
        *res *= scale;
    }
}

#[cfg(test)]
mod test {
    use crate::ml::LayerNorm;
    use khal::backend::WebGpu;
    use khal::backend::{Backend, Encoder, GpuBackend};
    use khal::{BufferUsages, Shader};
    use nalgebra::DVector;
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use wgpu::{Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_layernorm_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        gpu_layernorm_generic(&backend).await;
    }

    async fn gpu_layernorm_generic(backend: &GpuBackend) {
        let layernorm = super::LayerNorm::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        const LEN: u32 = 1757;

        let v0 = DVector::new_random(LEN as usize);
        let out = DVector::new_random(LEN as usize);
        let mut out_read = DVector::zeros(LEN as usize);
        let gpu_v0 =
            Tensor::vector(backend, &v0, BufferUsages::STORAGE | BufferUsages::COPY_SRC).unwrap();
        let mut gpu_out =
            Tensor::vector(backend, &v0, BufferUsages::STORAGE | BufferUsages::COPY_SRC).unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        layernorm
            .launch_rows(
                backend,
                &mut shapes,
                &mut pass,
                &mut gpu_out,
                gpu_v0.as_view(),
            )
            .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_out.buffer(), out_read.as_mut_slice())
            .await
            .unwrap();

        let mut cpu_result = out;
        LayerNorm::run_cpu(&mut cpu_result, &v0);

        approx::assert_relative_eq!(out_read, cpu_result, epsilon = 1.0e-3);
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gpu_layernorm_cpu() {
        let backend = GpuBackend::Cpu;
        gpu_layernorm_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_layernorm_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        gpu_layernorm_generic(&backend).await;
    }
}
