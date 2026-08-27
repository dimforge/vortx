use crate::shapes::TensorLayoutBuffers;
use crate::tensor::AsTensorMut;
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use nalgebra::{Dyn, StorageMut, Vector};

/*
layout (push_constant) uniform parameter
{
    uint KX;
    uint KY;
    uint ne00;
    uint ne01;
    uint ne02;
    uint ne12;
    uint ne13;
    uint nb11;
    uint nb12;
    uint nb13;
    float scale;
    float max_bias;
    float m0;
    float m1;
    uint n_head_log2;
    uint nrows_x;
    uint has_sinks;
} p;

#include "types.glsl"

layout(constant_id = 0) const uint BLOCK_SIZE = 32;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) readonly buffer Y {B_TYPE data_b[];};
layout (binding = 2) readonly buffer Z {float data_c[];};
layout (binding = 3) buffer D {D_TYPE data_d[];};

struct SoftMaxGgmlUniforms {

}

struct SoftMaxGgmlArgs<'a> {

}
 */

#[derive(Shader)]
/// Shader implementing the softmax kernel.
pub struct SoftMax {
    pub softmax: vortx_shaders::ml::softmax::Softmax,
    pub log_softmax: vortx_shaders::ml::softmax::LogSoftmax,
}

impl SoftMax {
    pub fn launch(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut in_out_mat: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut in_out_mat = in_out_mat.as_tensor_mut().canonicalize();
        let size = in_out_mat.layout().size;

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, in_out_mat.layout())?;
            let shape_buf = shapes.get(in_out_mat.layout()).unwrap();
            let mut mat_buf = in_out_mat.buffer_mut();

            self.softmax.call(
                pass,
                [size[2], size[1], size[0]],
                &shape_buf.as_slice(),
                &mut mat_buf,
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = in_out_mat.layout().into();
            let mut mat_buf = in_out_mat.buffer_mut();

            self.softmax
                .call(pass, [size[2], size[1], size[0]], &mut mat_buf, shapes_val)
        }
    }

    pub fn launch_log(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut in_out_mat: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut in_out_mat = in_out_mat.as_tensor_mut().canonicalize();
        let size = in_out_mat.layout().size;

        #[cfg(not(feature = "push_constants"))]
        {
            shapes.insert(backend, in_out_mat.layout())?;
            let shape_buf = shapes.get(in_out_mat.layout()).unwrap();
            let mut mat_buf = in_out_mat.buffer_mut();

            self.log_softmax.call(
                pass,
                [size[2], size[1], size[0]],
                &shape_buf.as_slice(),
                &mut mat_buf,
            )
        }

        #[cfg(feature = "push_constants")]
        {
            let shapes_val = in_out_mat.layout().into();
            let mut mat_buf = in_out_mat.buffer_mut();

            self.log_softmax
                .call(pass, [size[2], size[1], size[0]], &mut mat_buf, shapes_val)
        }
    }

    /// The softmax function.
    ///
    /// Converts a set of real number into a probability distribution.
    /// See <https://fr.wikipedia.org/wiki/Fonction_softmax>
    pub fn run_cpu<S: StorageMut<f32, Dyn>>(vals: &mut Vector<f32, Dyn, S>) {
        // Note that llama2.c also introduces a bias based on the max value
        // to improve numerical stability. So it is effectively computing:
        // softmax(z) = (e^z - max) / (e^z - max).sum()
        let max_val = vals.max();
        let mut sum = 0.0;

        vals.apply(|x| {
            *x = (*x - max_val).exp();
            sum += *x;
        });

        *vals /= sum;
    }
}

#[cfg(test)]
mod test {
    use crate::ml::SoftMax;
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use khal::backend::WebGpu;
    use khal::backend::{Backend, Encoder, GpuBackend};
    use khal::{BufferUsages, Shader};
    use nalgebra::DVector;
    use wgpu::{Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_softmax_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        gpu_softmax_generic(&backend).await;
    }

    async fn gpu_softmax_generic(backend: &GpuBackend) {
        let softmax = super::SoftMax::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        const LEN: u32 = 1757;

        let v0 = DVector::from_fn(LEN as usize, |i, _| i as f32);
        let mut gpu_v0_read = DVector::zeros(LEN as usize);
        let mut gpu_v0 =
            Tensor::vector(backend, &v0, BufferUsages::STORAGE | BufferUsages::COPY_SRC).unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test", None);
        softmax
            .launch(backend, &mut shapes, &mut pass, &mut gpu_v0)
            .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_v0.buffer(), gpu_v0_read.as_mut_slice())
            .await
            .unwrap();

        let mut cpu_result = v0;
        SoftMax::run_cpu(&mut cpu_result);

        approx::assert_relative_eq!(gpu_v0_read, cpu_result, epsilon = 1.0e-7);
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gpu_softmax_cpu() {
        let backend = GpuBackend::Cpu;
        gpu_softmax_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_softmax_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        gpu_softmax_generic(&backend).await;
    }
}
