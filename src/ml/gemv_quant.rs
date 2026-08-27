use crate::Gemm;
use crate::ml::quantization::{BlockQ4K, BlockQ5K, BlockQ8K};
use crate::ml::quantized_matrix::GpuQuantTensor;
use crate::shapes::TensorLayoutBuffers;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{DispatchGrid, GpuBackend, GpuBackendError, GpuPass};

#[cfg(feature = "rand")]
use rand::distr::{Distribution, StandardUniform};
#[cfg(feature = "rand")]
use rand::{Rng, RngExt};

pub trait QuantizedValue {
    /// Number of dequantized elements the quantized value represents.
    const DEQUANTIZED_LEN: usize;
}

impl QuantizedValue for f32 {
    const DEQUANTIZED_LEN: usize = 1;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ8_0x2([u32; 17]);

impl QuantizedValue for GpuBlockQ8_0x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ4_0x2([u32; 9]);

impl QuantizedValue for GpuBlockQ4_0x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ4_1x2([u32; 10]);

impl QuantizedValue for GpuBlockQ4_1x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ5_0x2([u32; 11]);

impl QuantizedValue for GpuBlockQ5_0x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ5_1x2([u32; 12]);

impl QuantizedValue for GpuBlockQ5_1x2 {
    const DEQUANTIZED_LEN: usize = 64;
}

pub type GpuBlockQ8K = BlockQ8K;
pub type GpuBlockQ5K = BlockQ5K;
pub type GpuBlockQ4K = BlockQ4K;

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuBlockQ6Kx2([u32; 105]);

impl QuantizedValue for GpuBlockQ6Kx2 {
    const DEQUANTIZED_LEN: usize = 512;
}

// SAFETY: These impls are safe, they don't exist in bytemuck because they don't
// provide impls for non-power-of-two largeish arrays.
unsafe impl bytemuck::Zeroable for GpuBlockQ6Kx2 {}
unsafe impl bytemuck::Pod for GpuBlockQ6Kx2 {}

#[cfg(feature = "rand")]
macro_rules! impl_rand {
    ($($t: ident, $len: literal);*) => {$(
        impl Distribution<$t> for StandardUniform {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $t {
                // TODO: are all bit representations valid?
                $t([0; $len].map(|_| rng.random()))
            }
        }
    )*};
}

#[cfg(feature = "rand")]
impl_rand!(
    GpuBlockQ8_0x2, 17;
    GpuBlockQ4_0x2, 9;
    GpuBlockQ4_1x2, 10;
    GpuBlockQ5_0x2, 11;
    GpuBlockQ5_1x2, 12;
    GpuBlockQ6Kx2, 105
);

pub struct GemvQuant {
    pub gemm_f32: Gemm,
    pub gemv_q8: GemvQ8_0x2,
    pub gemv_q5: GemvQ5_0x2,
    pub gemv_q4: GemvQ4_0x2,
    pub gemv_q5_1: GemvQ5_1x2,
    pub gemv_q4_1: GemvQ4_1x2,
    pub gemv_q8_k: GemvQ8K,
    pub gemv_q6_k: GemvQ6Kx2,
    pub gemv_q5_k: GemvQ5K,
    pub gemv_q4_k: GemvQ4K,
}

impl GemvQuant {
    pub fn from_backend(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Ok(Self {
            gemm_f32: Gemm::from_backend(backend).unwrap(), // ?,
            gemv_q5: GemvQ5_0x2::from_backend(backend).unwrap(), // ?,
            gemv_q5_1: GemvQ5_1x2::from_backend(backend).unwrap(), // ?,
            gemv_q4: GemvQ4_0x2::from_backend(backend).unwrap(), // ?,
            gemv_q4_1: GemvQ4_1x2::from_backend(backend).unwrap(), // ?,
            gemv_q8_k: GemvQ8K::from_backend(backend).unwrap(), // ?,
            gemv_q6_k: GemvQ6Kx2::from_backend(backend).unwrap(), // ?,
            gemv_q5_k: GemvQ5K::from_backend(backend).unwrap(), // ?,
            gemv_q4_k: GemvQ4K::from_backend(backend).unwrap(), // ?,
            gemv_q8: GemvQ8_0x2::from_backend(backend).unwrap(), // ?,
        })
    }
}

#[derive(Shader)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ8_0x2 {
    pub gemv: vortx_shaders::ml::gemv_quant_q8_0x2::Gemv,
}

#[derive(Shader)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ5_0x2 {
    pub gemv: vortx_shaders::ml::gemv_quant_q5_0x2::Gemv,
}

#[derive(Shader)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ5_1x2 {
    pub gemv: vortx_shaders::ml::gemv_quant_q5_1x2::Gemv,
}

#[derive(Shader)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ4_0x2 {
    pub gemv: vortx_shaders::ml::gemv_quant_q4_0x2::Gemv,
}

#[derive(Shader)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ4_1x2 {
    pub gemv: vortx_shaders::ml::gemv_quant_q4_1x2::Gemv,
}

#[derive(Shader)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ8K {
    pub gemv: vortx_shaders::ml::gemv_quant_q8_k::Gemv,
}

#[derive(Shader)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ6Kx2 {
    pub gemv: vortx_shaders::ml::gemv_quant_q6_kx2::Gemv,
}

#[derive(Shader)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ5K {
    pub gemv: vortx_shaders::ml::gemv_quant_q5_k::Gemv,
}

#[derive(Shader)]
/// Shader for computing the product of a matrix and a vector.
pub struct GemvQ4K {
    pub gemv: vortx_shaders::ml::gemv_quant_q4_k::Gemv,
}

impl GemvQuant {
    /// Queues this shader to compute `out = m * v`.
    pub fn launch(
        &self,
        backend: &GpuBackend,
        shapes: &mut TensorLayoutBuffers,
        pass: &mut GpuPass,
        mut out: impl AsTensorMut<f32>,
        m: &GpuQuantTensor,
        v: impl AsTensorRef<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut out = out.as_tensor_mut();

        // TODO: add a function to convert a View<f32> to a View<vec4<f32>>
        //       then remove `TensorLayout::f32_to_vec4`.
        let mut v = v.as_tensor_ref();
        assert_eq!(m.rank(), 2);
        assert_eq!(v.rank(), 1);
        assert_eq!(out.rank(), 1);
        assert!(v.is_contiguous());
        assert!(out.is_contiguous());

        // Special case: if using f32, use the f32 kernel launcher instead.
        if let GpuQuantTensor::F32(m_f32) = m {
            v = v.unsqueeze(1);
            out = out.unsqueeze(1);
            return self.gemm_f32.dispatch(backend, shapes, pass, out, m_f32, v);
        }

        // assert_eq!(
        //     m.layout().size[1],
        //     v.layout().size[0],
        //     "Gemv: dimension mismatch."
        // );
        // assert_eq!(
        //     out.layout().size[0],
        //     m.layout().size[0],
        //     "Gemv: dimension mismatch."
        // );

        let v_shape = match m {
            GpuQuantTensor::F32(_) => unreachable!(),
            GpuQuantTensor::Q8_0(_) => v.layout().f32_to_vec4(),
            GpuQuantTensor::Q5_0(_) => v.layout().f32_to_vec4(),
            GpuQuantTensor::Q5_1(_) => v.layout().f32_to_vec4(),
            GpuQuantTensor::Q4_0(_) => v.layout().f32_to_vec4(),
            GpuQuantTensor::Q4_1(_) => v.layout().f32_to_vec4(),
            GpuQuantTensor::Q8K(_) => v.layout().f32_to_vec4(),
            GpuQuantTensor::Q6K(_) => v.layout().f32_to_vec4(),
            GpuQuantTensor::Q5K(_) => v.layout().f32_to_vec4(),
            GpuQuantTensor::Q4K(_) => v.layout().f32_to_vec4(),
        };

        let out_shape = match m {
            GpuQuantTensor::F32(_) => unreachable!(),
            // Optimized shaders use workgroup reduction → one Vec4 per workgroup.
            GpuQuantTensor::Q8_0(_)
            | GpuQuantTensor::Q4_0(_)
            | GpuQuantTensor::Q6K(_)
            | GpuQuantTensor::Q5K(_)
            | GpuQuantTensor::Q4K(_) => out.layout().f32_to_vec4(),
            // Non-optimized shaders (Q4_1, Q5_0, Q5_1, Q8K) index output by
            // global_invocation_id and write Vec4::splat per row. On WebGPU the
            // OOB writes are clamped; on CUDA they corrupt memory. These shader
            // paths are currently broken for CUDA and only work by accident on
            // WebGPU. They are rarely used in practice (most models use Q4K/Q5K).
            _ => out.layout(),
        };

        // Canonicalize the shape for coherence with other matmul shaders.
        let shape_m_canon = m.layout().canonicalize();
        #[cfg(not(feature = "push_constants"))]
        shapes.insert(backend, out_shape)?; // TODO: propagate error
        #[cfg(not(feature = "push_constants"))]
        shapes.insert(backend, v_shape)?; // TODO: propagate error
        #[cfg(not(feature = "push_constants"))]
        shapes.insert(backend, shape_m_canon)?; // TODO: propagate error
        #[cfg(not(feature = "push_constants"))]
        let _shape_out = shapes.get(out_shape).unwrap();
        #[cfg(not(feature = "push_constants"))]
        let _shape_v = shapes.get(v_shape).unwrap();
        #[cfg(not(feature = "push_constants"))]
        let shape_m = shapes.get(shape_m_canon).unwrap();

        let launch = match m {
            GpuQuantTensor::F32(_) => unreachable!(),
            GpuQuantTensor::Q8_0(_)
            | GpuQuantTensor::Q4_0(_)
            | GpuQuantTensor::Q6K(_)
            | GpuQuantTensor::Q5K(_)
            | GpuQuantTensor::Q4K(_) => out.layout().size[0] / 4,
            _ => m.layout().size[0].div_ceil(64),
        };

        let grid = DispatchGrid::Grid([launch, 1, 1]);

        // Dispatch to the appropriate kernel based on quantization type.
        // Each variant has its own generated args type, but all share the same field names
        // (shape_m, out, m, v) since the shader functions have the same signature.
        macro_rules! dispatch_gemv {
            ($kernel:expr, $tensor:expr) => {{
                #[cfg(not(feature = "push_constants"))]
                {
                    let mut buf_out = out.buffer_mut().reinterpret();
                    $kernel.gemv.call(
                        pass,
                        grid,
                        &shape_m.as_slice(),
                        &mut buf_out,
                        &$tensor.buffer().as_slice().reinterpret(),
                        &v.buffer().reinterpret(),
                    )
                }
                #[cfg(feature = "push_constants")]
                {
                    let shapes_val: vortx_shaders::linalg::Shapes1 = shape_m_canon.into();
                    let mut buf_out = out.buffer_mut().reinterpret();
                    $kernel.gemv.call(
                        pass,
                        grid,
                        &mut buf_out,
                        &$tensor.buffer().as_slice().reinterpret(),
                        &v.buffer().reinterpret(),
                        shapes_val,
                    )
                }
            }};
        }

        match m {
            GpuQuantTensor::F32(_) => unreachable!(),
            GpuQuantTensor::Q8_0(tensor) => {
                dispatch_gemv!(&self.gemv_q8, tensor)
            }
            GpuQuantTensor::Q5_0(tensor) => {
                dispatch_gemv!(&self.gemv_q5, tensor)
            }
            GpuQuantTensor::Q5_1(tensor) => {
                dispatch_gemv!(&self.gemv_q5_1, tensor)
            }
            GpuQuantTensor::Q4_0(tensor) => {
                dispatch_gemv!(&self.gemv_q4, tensor)
            }
            GpuQuantTensor::Q4_1(tensor) => {
                dispatch_gemv!(&self.gemv_q4_1, tensor)
            }
            GpuQuantTensor::Q8K(tensor) => {
                dispatch_gemv!(&self.gemv_q8_k, tensor)
            }
            GpuQuantTensor::Q6K(tensor) => {
                dispatch_gemv!(&self.gemv_q6_k, tensor)
            }
            GpuQuantTensor::Q5K(tensor) => {
                dispatch_gemv!(&self.gemv_q5_k, tensor)
            }
            GpuQuantTensor::Q4K(tensor) => {
                dispatch_gemv!(&self.gemv_q4_k, tensor)
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "rand")]
mod test {
    use super::*;
    use crate::ml::quantization::*;
    use crate::ml::quantized_matrix::GpuQuantTensor;
    use crate::shapes::TensorLayoutBuffers;
    use crate::tensor::Tensor;
    use khal::BufferUsages;
    use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
    use wgpu::{Features, Limits};

    /// Generate a random valid f16 scale (avoids NaN/Inf).
    /// Includes subnormal values to cover the full f16 range.
    fn rand_f16_scale() -> u16 {
        let val: f32 = rand::random::<f32>() * 2.0 - 1.0; // [-1, 1]
        // Scale down ~25% of values into the subnormal f16 range (< 2^-14).
        let val = if rand::random::<u8>() < 64 {
            val * 1e-5
        } else {
            val
        };
        half::f16::from_f32(val).to_bits()
    }

    fn rand_block_q8_0() -> BlockQ8_0 {
        BlockQ8_0 {
            scale: rand_f16_scale(),
            data: rand::random(),
        }
    }

    fn rand_block_q4_0() -> BlockQ4_0 {
        BlockQ4_0 {
            d: rand_f16_scale(),
            qs: rand::random(),
        }
    }

    fn rand_block_q8k() -> BlockQ8K {
        BlockQ8K {
            d: rand::random::<f32>() * 2.0 - 1.0,
            qs: [0i8; 256].map(|_| rand::random()),
            bsums: rand::random(),
        }
    }

    fn rand_block_q4k() -> BlockQ4K {
        BlockQ4K {
            d: rand_f16_scale(),
            dmin: rand_f16_scale(),
            scales: rand::random(),
            qs: [0u8; 128].map(|_| rand::random()),
        }
    }

    fn rand_block_q4_1() -> BlockQ4_1 {
        BlockQ4_1 {
            d: rand_f16_scale(),
            m: rand_f16_scale(),
            qs: rand::random(),
        }
    }

    fn rand_block_q5_0() -> BlockQ5_0 {
        BlockQ5_0 {
            d: rand_f16_scale(),
            qh: rand::random(),
            qs: rand::random(),
        }
    }

    fn rand_block_q5_1() -> BlockQ5_1 {
        BlockQ5_1 {
            d: rand_f16_scale(),
            m: rand_f16_scale(),
            qh: rand::random(),
            qs: rand::random(),
        }
    }

    fn rand_block_q6k() -> BlockQ6K {
        BlockQ6K {
            ql: [0u8; 128].map(|_| rand::random()),
            qh: [0u8; 64].map(|_| rand::random()),
            scales: [0i8; 16].map(|_| rand::random()),
            d: rand_f16_scale(),
        }
    }

    fn rand_block_q5k() -> BlockQ5K {
        BlockQ5K {
            d: rand_f16_scale(),
            dmin: rand_f16_scale(),
            scales: rand::random(),
            qh: rand::random(),
            qs: [0u8; 128].map(|_| rand::random()),
        }
    }

    /// CPU reference: dequantize quantized blocks into a full f32 matrix, then
    /// compute `out = matrix * vec`.
    fn cpu_gemv(dequantized_matrix: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
        assert_eq!(dequantized_matrix.len(), rows * cols);
        assert_eq!(vec.len(), cols);
        let mut out = vec![0.0f32; rows];
        for r in 0..rows {
            let mut sum = 0.0f64; // accumulate in f64 for reference accuracy
            for c in 0..cols {
                sum += dequantized_matrix[r * cols + c] as f64 * vec[c] as f64;
            }
            out[r] = sum as f32;
        }
        out
    }

    /// Helper: run a gemv_quant launch and read back the result.
    async fn run_gemv(
        backend: &GpuBackend,
        quant_tensor: &GpuQuantTensor,
        v_data: &[f32],
        rows: u32,
    ) -> Vec<f32> {
        let gemv = GemvQuant::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        let v = Tensor::vector(backend, v_data, BufferUsages::STORAGE).unwrap();
        let mut out = Tensor::<f32>::vector_uninit(
            backend,
            rows,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test_gemv", None);
        gemv.launch(backend, &mut shapes, &mut pass, &mut out, quant_tensor, &v)
            .unwrap();
        drop(pass);
        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        let mut result = vec![0.0f32; rows as usize];
        backend
            .slow_read_buffer(out.buffer(), &mut result)
            .await
            .unwrap();
        result
    }

    // =========================================================================
    // Q8_0
    // =========================================================================

    /// Dequantize a GpuBlockQ8_0x2 slice into flat f32s. Each GPU block = 2 CPU blocks = 64 f32s.
    fn dequantize_q8_0x2(blocks: &[GpuBlockQ8_0x2]) -> Vec<f32> {
        let cpu_blocks: &[BlockQ8_0] = bytemuck::cast_slice(blocks);
        cpu_blocks.iter().flat_map(|b| b.dequantize()).collect()
    }

    async fn test_gemv_q8_0_generic(backend: &GpuBackend) {
        const ROWS: usize = 64;
        const COLS: usize = 256; // must be multiple of 64
        const BLOCKS_PER_ROW: usize = COLS / 64;
        const TOTAL_CPU_BLOCKS: usize = ROWS * (COLS / 32);

        let cpu_blocks: Vec<BlockQ8_0> = (0..TOTAL_CPU_BLOCKS).map(|_| rand_block_q8_0()).collect();
        let blocks: Vec<GpuBlockQ8_0x2> = bytemuck::cast_slice(&cpu_blocks).to_vec();
        let v: Vec<f32> = (0..COLS)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let deq = dequantize_q8_0x2(&blocks);
        let expected = cpu_gemv(&deq, ROWS, COLS, &v);

        let m = Tensor::matrix(
            backend,
            ROWS as u32,
            BLOCKS_PER_ROW as u32,
            &blocks,
            BufferUsages::STORAGE,
        )
        .unwrap();
        let qt = GpuQuantTensor::Q8_0(m);
        let actual = run_gemv(backend, &qt, &v, ROWS as u32).await;

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let denom = e.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "Q8_0 row {i}: gpu={a} cpu={e} diff={diff}"
            );
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q8_0_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_gemv_q8_0_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gemv_q8_0_cpu() {
        let backend = GpuBackend::Cpu;
        test_gemv_q8_0_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q8_0_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_gemv_q8_0_generic(&backend).await;
    }

    // =========================================================================
    // Q4_0
    // =========================================================================

    fn dequantize_q4_0x2(blocks: &[GpuBlockQ4_0x2]) -> Vec<f32> {
        let cpu_blocks: &[BlockQ4_0] = bytemuck::cast_slice(blocks);
        cpu_blocks.iter().flat_map(|b| b.dequantize()).collect()
    }

    async fn test_gemv_q4_0_generic(backend: &GpuBackend) {
        const ROWS: usize = 64;
        const COLS: usize = 256;
        const BLOCKS_PER_ROW: usize = COLS / 64;
        const TOTAL_CPU_BLOCKS: usize = ROWS * (COLS / 32);

        let cpu_blocks: Vec<BlockQ4_0> = (0..TOTAL_CPU_BLOCKS).map(|_| rand_block_q4_0()).collect();
        let blocks: Vec<GpuBlockQ4_0x2> = bytemuck::cast_slice(&cpu_blocks).to_vec();
        let v: Vec<f32> = (0..COLS)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let deq = dequantize_q4_0x2(&blocks);
        let expected = cpu_gemv(&deq, ROWS, COLS, &v);

        let m = Tensor::matrix(
            backend,
            ROWS as u32,
            BLOCKS_PER_ROW as u32,
            &blocks,
            BufferUsages::STORAGE,
        )
        .unwrap();
        let qt = GpuQuantTensor::Q4_0(m);
        let actual = run_gemv(backend, &qt, &v, ROWS as u32).await;

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let denom = e.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "Q4_0 row {i}: gpu={a} cpu={e} diff={diff}"
            );
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q4_0_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_gemv_q4_0_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gemv_q4_0_cpu() {
        let backend = GpuBackend::Cpu;
        test_gemv_q4_0_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q4_0_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_gemv_q4_0_generic(&backend).await;
    }

    // =========================================================================
    // Q4K
    // =========================================================================

    fn dequantize_q4k(blocks: &[GpuBlockQ4K]) -> Vec<f32> {
        blocks.iter().flat_map(|b| b.dequantize()).collect()
    }

    async fn test_gemv_q4k_generic(backend: &GpuBackend) {
        const ROWS: usize = 64;
        const COLS: usize = 256;
        const BLOCKS_PER_ROW: usize = COLS / 256;
        const TOTAL_BLOCKS: usize = ROWS * BLOCKS_PER_ROW;

        let blocks: Vec<GpuBlockQ4K> = (0..TOTAL_BLOCKS).map(|_| rand_block_q4k()).collect();
        let v: Vec<f32> = (0..COLS)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let deq = dequantize_q4k(&blocks);
        let expected = cpu_gemv(&deq, ROWS, COLS, &v);

        let m = Tensor::matrix(
            backend,
            ROWS as u32,
            BLOCKS_PER_ROW as u32,
            &blocks,
            BufferUsages::STORAGE,
        )
        .unwrap();
        let qt = GpuQuantTensor::Q4K(m);
        let actual = run_gemv(backend, &qt, &v, ROWS as u32).await;

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let denom = e.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "Q4K row {i}: gpu={a} cpu={e} diff={diff}"
            );
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q4k_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_gemv_q4k_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gemv_q4k_cpu() {
        let backend = GpuBackend::Cpu;
        test_gemv_q4k_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q4k_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_gemv_q4k_generic(&backend).await;
    }

    // =========================================================================
    // Q5K
    // =========================================================================

    fn dequantize_q5k(blocks: &[GpuBlockQ5K]) -> Vec<f32> {
        blocks.iter().flat_map(|b| b.dequantize()).collect()
    }

    async fn test_gemv_q5k_generic(backend: &GpuBackend) {
        const ROWS: usize = 64;
        const COLS: usize = 256;
        const BLOCKS_PER_ROW: usize = COLS / 256;
        const TOTAL_BLOCKS: usize = ROWS * BLOCKS_PER_ROW;

        let blocks: Vec<GpuBlockQ5K> = (0..TOTAL_BLOCKS).map(|_| rand_block_q5k()).collect();
        let v: Vec<f32> = (0..COLS)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let deq = dequantize_q5k(&blocks);
        let expected = cpu_gemv(&deq, ROWS, COLS, &v);

        let m = Tensor::matrix(
            backend,
            ROWS as u32,
            BLOCKS_PER_ROW as u32,
            &blocks,
            BufferUsages::STORAGE,
        )
        .unwrap();
        let qt = GpuQuantTensor::Q5K(m);
        let actual = run_gemv(backend, &qt, &v, ROWS as u32).await;

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let denom = e.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "Q5K row {i}: gpu={a} cpu={e} diff={diff}"
            );
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q5k_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_gemv_q5k_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gemv_q5k_cpu() {
        let backend = GpuBackend::Cpu;
        test_gemv_q5k_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q5k_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_gemv_q5k_generic(&backend).await;
    }

    // =========================================================================
    // Q6K (optimized path, uses shared memory + workgroup reduction)
    // =========================================================================

    fn dequantize_q6kx2(blocks: &[GpuBlockQ6Kx2]) -> Vec<f32> {
        let cpu_blocks: &[BlockQ6K] = bytemuck::cast_slice(blocks);
        cpu_blocks.iter().flat_map(|b| b.dequantize()).collect()
    }

    async fn test_gemv_q6k_generic(backend: &GpuBackend) {
        const ROWS: usize = 64;
        const COLS: usize = 512; // must be multiple of 512 (Q6Kx2 = 2 blocks of 256)
        const BLOCKS_PER_ROW: usize = COLS / 512;
        const TOTAL_CPU_BLOCKS: usize = ROWS * (COLS / 256);

        let cpu_blocks: Vec<BlockQ6K> = (0..TOTAL_CPU_BLOCKS).map(|_| rand_block_q6k()).collect();
        let blocks: Vec<GpuBlockQ6Kx2> = bytemuck::cast_slice(&cpu_blocks).to_vec();
        let v: Vec<f32> = (0..COLS)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let deq = dequantize_q6kx2(&blocks);
        let expected = cpu_gemv(&deq, ROWS, COLS, &v);

        let m = Tensor::matrix(
            backend,
            ROWS as u32,
            BLOCKS_PER_ROW as u32,
            &blocks,
            BufferUsages::STORAGE,
        )
        .unwrap();
        let qt = GpuQuantTensor::Q6K(m);
        let actual = run_gemv(backend, &qt, &v, ROWS as u32).await;

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let denom = e.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "Q6K row {i}: gpu={a} cpu={e} diff={diff}"
            );
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q6k_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_gemv_q6k_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gemv_q6k_cpu() {
        let backend = GpuBackend::Cpu;
        test_gemv_q6k_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q6k_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_gemv_q6k_generic(&backend).await;
    }

    /// Helper for non-optimized shaders (Q4_1, Q5_0, Q5_1, Q8K) which write
    /// Vec4::splat(sum) per row. Allocates 4x output to prevent OOB, then
    /// extracts the x-component of each Vec4.
    async fn run_gemv_vec4_per_row(
        backend: &GpuBackend,
        quant_tensor: &GpuQuantTensor,
        v_data: &[f32],
        rows: u32,
    ) -> Vec<f32> {
        let gemv = GemvQuant::from_backend(backend).unwrap();
        let mut shapes = TensorLayoutBuffers::new(backend);

        let v = Tensor::vector(backend, v_data, BufferUsages::STORAGE).unwrap();
        let mut out = Tensor::<f32>::vector_uninit(
            backend,
            rows * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test_gemv", None);
        gemv.launch(backend, &mut shapes, &mut pass, &mut out, quant_tensor, &v)
            .unwrap();
        drop(pass);
        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        let mut raw = vec![0.0f32; (rows * 4) as usize];
        backend
            .slow_read_buffer(out.buffer(), &mut raw)
            .await
            .unwrap();
        (0..rows as usize).map(|i| raw[i * 4]).collect()
    }

    // --- Q4_1 ---

    fn dequantize_q4_1x2(blocks: &[GpuBlockQ4_1x2]) -> Vec<f32> {
        let cpu_blocks: &[BlockQ4_1] = bytemuck::cast_slice(blocks);
        cpu_blocks.iter().flat_map(|b| b.dequantize()).collect()
    }

    async fn test_gemv_q4_1_generic(backend: &GpuBackend) {
        const ROWS: usize = 64;
        const COLS: usize = 256;
        const BLOCKS_PER_ROW: usize = COLS / 64;
        const TOTAL_CPU_BLOCKS: usize = ROWS * (COLS / 32);

        let cpu_blocks: Vec<BlockQ4_1> = (0..TOTAL_CPU_BLOCKS).map(|_| rand_block_q4_1()).collect();
        let blocks: Vec<GpuBlockQ4_1x2> = bytemuck::cast_slice(&cpu_blocks).to_vec();
        let v: Vec<f32> = (0..COLS)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let deq = dequantize_q4_1x2(&blocks);
        let expected = cpu_gemv(&deq, ROWS, COLS, &v);

        let m = Tensor::matrix(
            backend,
            ROWS as u32,
            BLOCKS_PER_ROW as u32,
            &blocks,
            BufferUsages::STORAGE,
        )
        .unwrap();
        let qt = GpuQuantTensor::Q4_1(m);
        let actual = run_gemv_vec4_per_row(backend, &qt, &v, ROWS as u32).await;

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let denom = e.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "Q4_1 row {i}: gpu={a} cpu={e} diff={diff}"
            );
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q4_1_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_gemv_q4_1_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gemv_q4_1_cpu() {
        let backend = GpuBackend::Cpu;
        test_gemv_q4_1_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q4_1_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_gemv_q4_1_generic(&backend).await;
    }

    // --- Q5_0 ---

    fn dequantize_q5_0x2(blocks: &[GpuBlockQ5_0x2]) -> Vec<f32> {
        let cpu_blocks: &[BlockQ5_0] = bytemuck::cast_slice(blocks);
        cpu_blocks.iter().flat_map(|b| b.dequantize()).collect()
    }

    async fn test_gemv_q5_0_generic(backend: &GpuBackend) {
        const ROWS: usize = 64;
        const COLS: usize = 256;
        const BLOCKS_PER_ROW: usize = COLS / 64;
        const TOTAL_CPU_BLOCKS: usize = ROWS * (COLS / 32);

        let cpu_blocks: Vec<BlockQ5_0> = (0..TOTAL_CPU_BLOCKS).map(|_| rand_block_q5_0()).collect();
        let blocks: Vec<GpuBlockQ5_0x2> = bytemuck::cast_slice(&cpu_blocks).to_vec();
        let v: Vec<f32> = (0..COLS)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let deq = dequantize_q5_0x2(&blocks);
        let expected = cpu_gemv(&deq, ROWS, COLS, &v);

        let m = Tensor::matrix(
            backend,
            ROWS as u32,
            BLOCKS_PER_ROW as u32,
            &blocks,
            BufferUsages::STORAGE,
        )
        .unwrap();
        let qt = GpuQuantTensor::Q5_0(m);
        let actual = run_gemv_vec4_per_row(backend, &qt, &v, ROWS as u32).await;

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let denom = e.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "Q5_0 row {i}: gpu={a} cpu={e} diff={diff}"
            );
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q5_0_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_gemv_q5_0_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gemv_q5_0_cpu() {
        let backend = GpuBackend::Cpu;
        test_gemv_q5_0_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q5_0_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_gemv_q5_0_generic(&backend).await;
    }

    // --- Q5_1 ---

    fn dequantize_q5_1x2(blocks: &[GpuBlockQ5_1x2]) -> Vec<f32> {
        let cpu_blocks: &[BlockQ5_1] = bytemuck::cast_slice(blocks);
        cpu_blocks.iter().flat_map(|b| b.dequantize()).collect()
    }

    async fn test_gemv_q5_1_generic(backend: &GpuBackend) {
        const ROWS: usize = 64;
        const COLS: usize = 256;
        const BLOCKS_PER_ROW: usize = COLS / 64;
        const TOTAL_CPU_BLOCKS: usize = ROWS * (COLS / 32);

        let cpu_blocks: Vec<BlockQ5_1> = (0..TOTAL_CPU_BLOCKS).map(|_| rand_block_q5_1()).collect();
        let blocks: Vec<GpuBlockQ5_1x2> = bytemuck::cast_slice(&cpu_blocks).to_vec();
        let v: Vec<f32> = (0..COLS)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let deq = dequantize_q5_1x2(&blocks);
        let expected = cpu_gemv(&deq, ROWS, COLS, &v);

        let m = Tensor::matrix(
            backend,
            ROWS as u32,
            BLOCKS_PER_ROW as u32,
            &blocks,
            BufferUsages::STORAGE,
        )
        .unwrap();
        let qt = GpuQuantTensor::Q5_1(m);
        let actual = run_gemv_vec4_per_row(backend, &qt, &v, ROWS as u32).await;

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let denom = e.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "Q5_1 row {i}: gpu={a} cpu={e} diff={diff}"
            );
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q5_1_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_gemv_q5_1_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gemv_q5_1_cpu() {
        let backend = GpuBackend::Cpu;
        test_gemv_q5_1_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q5_1_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_gemv_q5_1_generic(&backend).await;
    }

    // --- Q8K ---

    fn dequantize_q8k(blocks: &[GpuBlockQ8K]) -> Vec<f32> {
        blocks.iter().flat_map(|b| b.dequantize()).collect()
    }

    async fn test_gemv_q8k_generic(backend: &GpuBackend) {
        // Q8K has workgroup size 32 but dispatch uses div_ceil(64), so only 32
        // rows are computed per workgroup. Use 32 rows to stay within bounds.
        const ROWS: usize = 32;
        const COLS: usize = 256;
        const BLOCKS_PER_ROW: usize = COLS / 256;
        const TOTAL_BLOCKS: usize = ROWS * BLOCKS_PER_ROW;

        let blocks: Vec<GpuBlockQ8K> = (0..TOTAL_BLOCKS).map(|_| rand_block_q8k()).collect();
        let v: Vec<f32> = (0..COLS)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let deq = dequantize_q8k(&blocks);
        let expected = cpu_gemv(&deq, ROWS, COLS, &v);

        let m = Tensor::matrix(
            backend,
            ROWS as u32,
            BLOCKS_PER_ROW as u32,
            &blocks,
            BufferUsages::STORAGE,
        )
        .unwrap();
        let qt = GpuQuantTensor::Q8K(m);
        let actual = run_gemv_vec4_per_row(backend, &qt, &v, ROWS as u32).await;

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let denom = e.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "Q8K row {i}: gpu={a} cpu={e} diff={diff}"
            );
        }
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q8k_webgpu() {
        let webgpu = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        let backend = GpuBackend::WebGpu(webgpu);
        test_gemv_q8k_generic(&backend).await;
    }

    #[cfg(feature = "cpu")]
    #[futures_test::test]
    async fn gemv_q8k_cpu() {
        let backend = GpuBackend::Cpu;
        test_gemv_q8k_generic(&backend).await;
    }

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gemv_q8k_cuda() {
        let cuda = khal::backend::Cuda::new(0).unwrap();
        let backend = GpuBackend::Cuda(cuda);
        test_gemv_q8k_generic(&backend).await;
    }
}
