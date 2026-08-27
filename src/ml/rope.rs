use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, Tensor};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use nalgebra::{DVector, DVectorViewMut, Rotation2, vector};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RoPEVariant {
    // The original version of RoPE, where the rotated entries are adjacent.
    Original,
    // A variant of RoPE where the rotated entries are separated by `head_size / 2` elements.
    Neox,
}

#[derive(Shader)]
/// Shader implementing the Rotary Positional Encoding kernel.
pub struct RoPE {
    pub rope_neox: vortx_shaders::ml::rope::RopeNeox,
    pub rope: vortx_shaders::ml::rope::Rope,
}

pub use vortx_shaders::ml::rope::RoPEConfig;

impl RoPE {
    pub fn launch(
        &self,
        backend: &GpuBackend,
        #[cfg_attr(feature = "push_constants", allow(unused_variables))]
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        variant: RoPEVariant,
        config: &Tensor<RoPEConfig>,
        mut in_out_q: impl AsTensorMut<f32>,
        mut in_out_k: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut in_out_q = in_out_q.as_tensor_mut();
        let mut in_out_k = in_out_k.as_tensor_mut();

        assert_eq!(in_out_q.len() % 2, 0);
        assert_eq!(in_out_k.len() % 2, 0);
        assert!(
            in_out_q.len() >= in_out_k.len(),
            "The Query vector must be larger than, or as large as, the Key vector."
        );

        let num_threads = [in_out_q.len() as u32 / 2, 1, 1];

        macro_rules! dispatch_rope {
            ($wrapper:expr) => {{
                #[cfg(not(feature = "push_constants"))]
                {
                    shapes.insert(backend, in_out_q.layout()).unwrap();
                    shapes.insert(backend, in_out_k.layout()).unwrap();
                    let shape_q = shapes.get(in_out_q.layout()).unwrap();
                    let shape_k = shapes.get(in_out_k.layout()).unwrap();
                    let mut buf_q = in_out_q.buffer_mut();
                    let mut buf_k = in_out_k.buffer_mut();

                    $wrapper.call(
                        pass,
                        num_threads,
                        &shape_q.as_slice(),
                        &shape_k.as_slice(),
                        &config.buffer().as_slice(),
                        &mut buf_q,
                        &mut buf_k,
                    )
                }

                #[cfg(feature = "push_constants")]
                {
                    let shapes_val = crate::shaders::linalg::Shapes2 {
                        shape_a: in_out_q.layout().into(),
                        shape_b: in_out_k.layout().into(),
                    };
                    let mut buf_q = in_out_q.buffer_mut();
                    let mut buf_k = in_out_k.buffer_mut();

                    $wrapper.call(
                        pass,
                        num_threads,
                        &config.buffer().as_slice(),
                        &mut buf_q,
                        &mut buf_k,
                        shapes_val,
                    )
                }
            }};
        }

        match variant {
            RoPEVariant::Original => dispatch_rope!(&self.rope),
            RoPEVariant::Neox => dispatch_rope!(&self.rope_neox),
        }
    }

    // Rotary Positional Encoding (RoPE): complex-valued rotate q and k in each head.
    pub fn run_cpu(
        q: &mut DVector<f32>,
        k: &mut DVectorViewMut<f32>,
        head_size: usize,
        dim: usize,
        kv_dim: usize,
        pos: usize,
    ) {
        for i in (0..dim).step_by(2) {
            // For RoPE, we have one rotation matrix like https://youtu.be/Mn_9W1nCFLo?si=GLIXuFLGVG8q6v2u&t=1963
            // for each head. So we need to transform `i` into the corresponding index within
            // the head.
            let head_dim = (i % head_size) as f32;
            // Not that the formulae from the video linked above would be:
            //     10000.0.powf(-2.0 * ((i / 2) as f32 - 1.0) / dim as f32)
            // Although in the paper shown in the video, their index is 1-based which his why thy
            // have to subtract 1.0 whereas we don't need to.The `i / 2` and multiplication by 2.0
            // are both accounted for by stepping only on even values for `i`.
            // Therefore, the formulae below is equivalent to the RoPE paper's formulae.
            let theta = 10000.0_f32.powf(-head_dim / head_size as f32);
            let m_theta = pos as f32 * theta;
            let rot = Rotation2::new(m_theta);

            let qi = vector![q[i], q[i + 1]];
            let mut out_q = q.fixed_rows_mut::<2>(i);
            out_q.copy_from(&(rot * qi));

            // When i >= kv_dim, we are done rotating all the elements from the keys. That's
            // because there are less key heads than query heads, but each key head sub-vector has
            // the same dimension as the query head (they loose dimension when multiplied with the
            // key weight matrices).
            if i < kv_dim {
                let ki = vector![k[i], k[i + 1]];
                let mut out_k = k.fixed_rows_mut::<2>(i);
                out_k.copy_from(&(rot * ki));
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::RoPEConfig;
    use crate::ml::{RoPE, RoPEVariant};
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use khal::backend::WebGpu;
    use khal::backend::{Backend, Encoder, GpuBackend};
    use khal::{BufferUsages, Shader};
    use nalgebra::DVector;
    use wgpu::{Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_rope_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        gpu_rope_generic(&backend).await;
    }

    async fn gpu_rope_generic(backend: &GpuBackend) {
        let rope = super::RoPE::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        const HEAD_SIZE: u32 = 128;
        const LEN_Q: u32 = 13 * HEAD_SIZE;
        const LEN_K: u32 = 9 * HEAD_SIZE;

        let rope_indices = RoPEConfig {
            head_size: HEAD_SIZE,
            kv_dim: LEN_K,
            pos: 10,
            base_freq: 1.0e4,
        };

        let mut q = DVector::new_random(LEN_Q as usize);
        let mut k = DVector::new_random(LEN_K as usize);
        let mut result_q = DVector::zeros(LEN_Q as usize);
        let mut result_k = DVector::zeros(LEN_K as usize);

        let gpu_indices = Tensor::scalar(
            backend,
            rope_indices,
            BufferUsages::UNIFORM | BufferUsages::STORAGE,
        )
        .unwrap();
        let mut gpu_q =
            Tensor::vector(backend, &q, BufferUsages::STORAGE | BufferUsages::COPY_SRC).unwrap();
        let mut gpu_k =
            Tensor::vector(backend, &k, BufferUsages::STORAGE | BufferUsages::COPY_SRC).unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("rope_test", None);
        rope.launch(
            backend,
            &mut shapes,
            &mut pass,
            RoPEVariant::Original,
            &gpu_indices,
            &mut gpu_q,
            &mut gpu_k,
        )
        .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_q.buffer(), result_q.as_mut_slice())
            .await
            .unwrap();
        backend
            .slow_read_buffer(gpu_k.buffer(), result_k.as_mut_slice())
            .await
            .unwrap();

        RoPE::run_cpu(
            &mut q,
            &mut k.rows_mut(0, LEN_K as usize),
            rope_indices.head_size as usize,
            LEN_Q as usize,
            rope_indices.kv_dim as usize,
            rope_indices.pos as usize,
        );

        // TODO: why is the epsilon so high? Is it a difference in sin/cos implementations?
        approx::assert_relative_eq!(result_q, q, epsilon = 1.0e-5);
        approx::assert_relative_eq!(result_k, k, epsilon = 1.0e-5);
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gpu_rope_cpu() {
        let backend = GpuBackend::Cpu;
        gpu_rope_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_rope_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        gpu_rope_generic(&backend).await;
    }
}
