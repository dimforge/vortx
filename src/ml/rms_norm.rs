use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::Shader;
use nalgebra::{DVector, Dyn, Storage, Vector};
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef, Tensor};

#[derive(Shader)]
/// Shader implementing the RMS norm kernel.
pub struct RmsNorm {
    pub rms_norm: vortx_shaders::ml::rms_norm::RmsNorm,
}

pub use vortx_shaders::ml::rms_norm::RmsNormConfig;

impl RmsNorm {
    pub fn launch(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        config: &Tensor<RmsNormConfig>,
        mut result: impl AsTensorMut<f32>,
        value: impl AsTensorRef<f32>,
        weight: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let value = value.as_tensor_ref().canonicalize();
        let weight = weight.as_tensor_ref().canonicalize();
        let mut result = result.as_tensor_mut().canonicalize();

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, value.layout())?;
            shapes.insert(backend, weight.layout())?;
            shapes.insert(backend, result.layout())?;
            let shape_v = shapes.get(value.layout()).unwrap();
            let shape_w = shapes.get(weight.layout()).unwrap();
            let shape_out = shapes.get(result.layout()).unwrap();
            let mut out_buf = result.buffer_mut();

            self.rms_norm.call(
                pass,
                [1; 3],
                &shape_v.as_slice(),
                &shape_w.as_slice(),
                &shape_out.as_slice(),
                &value.buffer(),
                &weight.buffer(),
                &mut out_buf,
                &config.buffer().as_slice(),
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = value.layout().into();
            let mut out_buf = result.buffer_mut();

            self.rms_norm.call(
                pass,
                [1; 3],
                &value.buffer(),
                &weight.buffer(),
                &mut out_buf,
                &config.buffer().as_slice(),
                shapes_val,
            )
        }
    }

    pub fn run_cpu<SW: Storage<f32, Dyn>>(
        out: &mut DVector<f32>,
        a: &DVector<f32>,
        w: &Vector<f32, Dyn, SW>,
    ) {
        const NUDGE_FACTOR: f32 = 1.0e-5;
        let rms = 1.0 / (a.norm_squared() / (a.nrows() as f32) + NUDGE_FACTOR).sqrt();
        out.zip_zip_apply(a, w, |o, a, w| *o = (a * rms) * w);
    }
}

#[cfg(test)]
mod test {
    use crate::ml::{RmsNorm, RmsNormConfig};
    use khal::backend::WebGpu;
    use khal::backend::{Backend, Encoder, GpuBackend};
    use khal::{BufferUsages, Shader};
    use nalgebra::DVector;
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use wgpu::{Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_rms_norm_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        gpu_rms_norm_generic(&backend).await;
    }

    async fn gpu_rms_norm_generic(backend: &GpuBackend) {
        let rmsnorm = super::RmsNorm::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        const LEN: u32 = 1757;

        let result = DVector::new_random(LEN as usize);
        let value = DVector::new_random(LEN as usize);
        let weight = DVector::new_random(LEN as usize);
        let mut gpu_result_read = DVector::zeros(LEN as usize);

        let mut gpu_result = Tensor::vector(
            backend,
            &result,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let gpu_value = Tensor::vector(backend, &value, BufferUsages::STORAGE).unwrap();
        let gpu_weight = Tensor::vector(backend, &weight, BufferUsages::STORAGE).unwrap();
        let config = Tensor::scalar(
            backend,
            RmsNormConfig {
                nudge_factor: 1.0e-6,
            },
            BufferUsages::UNIFORM | BufferUsages::STORAGE,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        rmsnorm
            .launch(
                backend,
                &mut shapes,
                &mut pass,
                &config,
                &mut gpu_result,
                gpu_value.as_view(),
                gpu_weight.as_view(),
            )
            .unwrap();
        drop(pass);
        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_result.buffer(), gpu_result_read.as_mut_slice())
            .await
            .unwrap();

        let mut cpu_result = result;
        RmsNorm::run_cpu(&mut cpu_result, &value, &weight);

        approx::assert_relative_eq!(gpu_result_read, cpu_result, epsilon = 1.0e-4);
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gpu_rms_norm_cpu() {
        let backend = GpuBackend::Cpu;
        gpu_rms_norm_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_rms_norm_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        gpu_rms_norm_generic(&backend).await;
    }
}
