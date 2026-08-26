use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::Shader;
use nalgebra::DVector;
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef};

#[derive(Shader)]
/// Shader implementing the Silu activation function.
pub struct Silu {
    pub silu: vortx_shaders::ml::silu::Silu,
}

impl Silu {
    pub fn launch(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut in_out_h1: impl AsTensorMut<f32>,
        in_h2: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut h1 = in_out_h1.as_tensor_mut().canonicalize();
        let h2 = in_h2.as_tensor_ref().canonicalize();
        let len = h1.len() as u32;

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, h1.layout())?;
            shapes.insert(backend, h2.layout())?;
            let shape_a = shapes.get(h1.layout()).unwrap();
            let shape_b = shapes.get(h2.layout()).unwrap();
            let mut in_out_a = h1.buffer_mut();

            self.silu.call(
                pass,
                [len, 1, 1],
                &shape_a.as_slice(),
                &shape_b.as_slice(),
                &mut in_out_a,
                &h2.buffer(),
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = h1.layout().into();
            let mut in_out_a = h1.buffer_mut();

            self.silu
                .call(pass, [len, 1, 1], &mut in_out_a, &h2.buffer(), shapes_val)
        }
    }

    pub fn run_cpu(h1: &mut DVector<f32>, h2: &DVector<f32>) {
        // SwiGLU non-linearity.
        fn swish(x: f32, beta: f32) -> f32 {
            // This is the swish function from https://youtu.be/Mn_9W1nCFLo?si=LT6puSAfzgpP6ydz&t=3973
            x / (1.0 + (-beta * x).exp())
        }

        h1.zip_apply(h2, |h, h2| *h = h2 * swish(*h, 1.0));
    }
}

#[cfg(test)]
mod test {
    use khal::backend::WebGpu;
    use khal::backend::{Backend, Encoder, GpuBackend};
    use khal::{BufferUsages, Shader};
    use nalgebra::DVector;
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use wgpu::{Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_silu_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        gpu_silu_generic(&backend).await;
    }

    async fn gpu_silu_generic(backend: &GpuBackend) {
        let silu = super::Silu::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        const LEN: u32 = 1757;

        let h1 = DVector::new_random(LEN as usize);
        let h2 = DVector::new_random(LEN as usize);
        let mut h1_read = DVector::zeros(LEN as usize);

        let mut gpu_h1 =
            Tensor::vector(backend, &h1, BufferUsages::STORAGE | BufferUsages::COPY_SRC).unwrap();
        let gpu_h2 = Tensor::vector(backend, &h2, BufferUsages::STORAGE).unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("silu_test", None);
        silu.launch(
            backend,
            &mut shapes,
            &mut pass,
            &mut gpu_h1,
            gpu_h2.as_view(),
        )
        .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_h1.buffer(), h1_read.as_mut_slice())
            .await
            .unwrap();

        let mut cpu_result = h1;
        super::Silu::run_cpu(&mut cpu_result, &h2);

        approx::assert_relative_eq!(h1_read, cpu_result, epsilon = 1.0e-5);
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gpu_silu_cpu() {
        let backend = GpuBackend::Cpu;
        gpu_silu_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_silu_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        gpu_silu_generic(&backend).await;
    }
}
