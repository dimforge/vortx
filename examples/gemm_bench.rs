//! Benchmark comparing naive vs tiled GEMM implementations.

use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use khal::{BufferUsages, Shader};
use nalgebra::DMatrix;
use vortx::linalg::Gemm;
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};

const WARMUP_ITERS: u32 = 3;
const BENCH_ITERS: u32 = 10;

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let webgpu = WebGpu::new(Features::default(), Limits::default()).await?;
    let backend = GpuBackend::WebGpu(webgpu);

    println!("GEMM Benchmark: Naive vs Tiled");
    println!("==============================");
    println!();
    println!(
        "{:>8} {:>12} {:>12} {:>10}",
        "Size", "Naive (ms)", "Tiled (ms)", "Speedup"
    );
    println!("{:-<8} {:-<12} {:-<12} {:-<10}", "", "", "", "");

    // Test various matrix sizes
    for &dim in &[64, 128, 256, 512, 768, 1024, 1536, 2048, 4096] {
        let (naive_time, tiled_time) = run_benchmark(&backend, dim).await?;
        let speedup = naive_time / tiled_time;
        println!(
            "{:>8} {:>12.3} {:>12.3} {:>10.2}x",
            format!("{}x{}", dim, dim),
            naive_time * 1000.0,
            tiled_time * 1000.0,
            speedup
        );
    }

    println!();
    println!(
        "Times are averages over {} iterations (after {} warmup).",
        BENCH_ITERS, WARMUP_ITERS
    );

    Ok(())
}

async fn run_benchmark(backend: &GpuBackend, dim: u32) -> anyhow::Result<(f32, f32)> {
    let gemm = Gemm::from_backend(backend)?;
    let mut shapes = TensorLayoutBuffers::new(backend);

    // Create random matrices
    let m1_cpu = DMatrix::<f32>::new_random(dim as usize, dim as usize);
    let m2_cpu = DMatrix::<f32>::new_random(dim as usize, dim as usize);
    let result_cpu = DMatrix::<f32>::zeros(dim as usize, dim as usize);

    let m1 = Tensor::matrix_from_na(backend, &m1_cpu, BufferUsages::STORAGE)?;
    let m2 = Tensor::matrix_from_na(backend, &m2_cpu, BufferUsages::STORAGE)?;
    let mut result = Tensor::matrix_from_na(
        backend,
        &result_cpu,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    )?;

    // Benchmark naive kernel
    let naive_time = {
        // Warmup
        for _ in 0..WARMUP_ITERS {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("gemm-warmup", None);
            gemm.dispatch_naive(backend, &mut shapes, &mut pass, &mut result, &m1, &m2)?;
            drop(pass);
            backend.submit(encoder)?;
            backend.synchronize()?;
        }

        // Timed runs
        let t0 = std::time::Instant::now();
        for _ in 0..BENCH_ITERS {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("gemm", None);
            gemm.dispatch_naive(backend, &mut shapes, &mut pass, &mut result, &m1, &m2)?;
            drop(pass);
            backend.submit(encoder)?;
            backend.synchronize()?;
        }
        t0.elapsed().as_secs_f32() / BENCH_ITERS as f32
    };

    // Benchmark tiled kernel
    let tiled_time = {
        // Warmup
        for _ in 0..WARMUP_ITERS {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("gemm-warmup", None);
            gemm.dispatch_tiled(backend, &mut shapes, &mut pass, &mut result, &m1, &m2)?;
            drop(pass);
            backend.submit(encoder)?;
            backend.synchronize()?;
        }

        // Timed runs
        let t0 = std::time::Instant::now();
        for _ in 0..BENCH_ITERS {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("gemm", None);
            gemm.dispatch_tiled(backend, &mut shapes, &mut pass, &mut result, &m1, &m2)?;
            drop(pass);
            backend.submit(encoder)?;
            backend.synchronize()?;
        }
        t0.elapsed().as_secs_f32() / BENCH_ITERS as f32
    };

    Ok((naive_time, tiled_time))
}
