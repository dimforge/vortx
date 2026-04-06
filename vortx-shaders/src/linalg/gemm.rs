//! General matrix multiplication (GEMM).
//!
//! Computes C = A * B where A is MxK and B is KxN.

use super::shape::Shape;
#[cfg(feature = "push_constants")]
use super::shape::Shapes3;
use glamx::UVec3;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

/// Tile size for the optimized GEMM kernel.
pub const TILE_M: u32 = 64;
pub const TILE_N: u32 = 64;
pub const TILE_K: u32 = 16;

/// Workgroup dimensions (16x16 = 256 threads).
pub const WG_M: u32 = 16;
pub const WG_N: u32 = 16;

/// Each thread computes a THREAD_M x THREAD_N tile of outputs.
pub const THREAD_M: u32 = TILE_M / WG_M; // 4
pub const THREAD_N: u32 = TILE_N / WG_N; // 4

/// Shared memory sizes with padding to avoid bank conflicts.
pub const SMEM_A_STRIDE: u32 = TILE_K + 1; // 17
pub const SMEM_B_STRIDE: u32 = TILE_N + 1; // 65
pub const SMEM_A_SIZE: usize = (TILE_M * SMEM_A_STRIDE) as usize; // 64 * 17 = 1088
pub const SMEM_B_SIZE: usize = (TILE_K * SMEM_B_STRIDE) as usize; // 16 * 65 = 1040

/// Legacy workgroup size constant for backward compatibility.
pub const WORKGROUP_SIZE: u32 = WG_M;

/// Optimized tiled GEMM using shared memory.
///
/// Each workgroup computes a TILE_M x TILE_N tile of the output matrix.
/// Tiles of A and B are loaded into shared memory to reduce global memory traffic.
/// Each thread computes a THREAD_M x THREAD_N sub-tile using register blocking.
#[spirv_bindgen]
#[spirv(compute(threads(16, 16, 1)))]
pub fn gemm_tiled(
    #[spirv(global_invocation_id)] _global_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(workgroup)] smem_a: &mut [f32; SMEM_A_SIZE],
    #[spirv(workgroup)] smem_b: &mut [f32; SMEM_B_SIZE],
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes3,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_out: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 1)]
    shape_lhs: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 2)]
    shape_rhs: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] out: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] lhs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] rhs: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_out, shape_lhs, shape_rhs) =
        (&shapes.shape_out, &shapes.shape_lhs, &shapes.shape_rhs);

    let tid_x = local_id.x;
    let tid_y = local_id.y;
    let linear_tid = tid_y * WG_N + tid_x;

    // Output tile start position
    let tile_row = wg_id.y * TILE_M;
    let tile_col = wg_id.x * TILE_N;

    // Matrix dimensions
    let m = shape_out.h;
    let n = shape_out.w;
    let k = shape_lhs.w;
    let mut acc: [f32; 16];

    // Process batch dimension
    for batch in 0..shape_out.n {
        let batch_c = wg_id.z % shape_out.c;

        // Register accumulator for 4x4 outputs per thread
        acc = [0.0; 16];

        // Loop over K dimension in tiles
        let mut k_tile: u32 = 0;
        while k_tile < k {
            // Load tile of A into shared memory (64x16 = 1024 elements, 256 threads = 4 each)
            {
                let mut i = linear_tid;
                while i < TILE_M * TILE_K {
                    let row = i / TILE_K;
                    let col = i % TILE_K;
                    let global_row = tile_row + row;
                    let global_col = k_tile + col;

                    let val = if global_row < m && global_col < k {
                        lhs.read(shape_lhs.it(batch, batch_c, global_row, global_col) as usize)
                    } else {
                        0.0
                    };
                    smem_a.write((row * SMEM_A_STRIDE + col) as usize, val);
                    i += WG_M * WG_N;
                }
            }

            // Load tile of B into shared memory (16x64 = 1024 elements, 256 threads = 4 each)
            {
                let mut i = linear_tid;
                while i < TILE_K * TILE_N {
                    let row = i / TILE_N;
                    let col = i % TILE_N;
                    let global_row = k_tile + row;
                    let global_col = tile_col + col;

                    let val = if global_row < k && global_col < n {
                        rhs.read(shape_rhs.it(batch, batch_c, global_row, global_col) as usize)
                    } else {
                        0.0
                    };
                    smem_b.write((row * SMEM_B_STRIDE + col) as usize, val);
                    i += WG_M * WG_N;
                }
            }

            khal_std::sync::workgroup_memory_barrier_with_group_sync();

            // Compute 4x4 output tile for this thread
            let a_row_base = tid_y * THREAD_M;
            let b_col_base = tid_x * THREAD_N;

            let mut kk: u32 = 0;
            while kk < TILE_K {
                let a0 = smem_a.read((a_row_base * SMEM_A_STRIDE + kk) as usize);
                let a1 = smem_a.read(((a_row_base + 1) * SMEM_A_STRIDE + kk) as usize);
                let a2 = smem_a.read(((a_row_base + 2) * SMEM_A_STRIDE + kk) as usize);
                let a3 = smem_a.read(((a_row_base + 3) * SMEM_A_STRIDE + kk) as usize);

                let b0 = smem_b.read((kk * SMEM_B_STRIDE + b_col_base) as usize);
                let b1 = smem_b.read((kk * SMEM_B_STRIDE + b_col_base + 1) as usize);
                let b2 = smem_b.read((kk * SMEM_B_STRIDE + b_col_base + 2) as usize);
                let b3 = smem_b.read((kk * SMEM_B_STRIDE + b_col_base + 3) as usize);

                acc[0] += a0 * b0;
                acc[1] += a0 * b1;
                acc[2] += a0 * b2;
                acc[3] += a0 * b3;
                acc[4] += a1 * b0;
                acc[5] += a1 * b1;
                acc[6] += a1 * b2;
                acc[7] += a1 * b3;
                acc[8] += a2 * b0;
                acc[9] += a2 * b1;
                acc[10] += a2 * b2;
                acc[11] += a2 * b3;
                acc[12] += a3 * b0;
                acc[13] += a3 * b1;
                acc[14] += a3 * b2;
                acc[15] += a3 * b3;

                kk += 1;
            }

            khal_std::sync::workgroup_memory_barrier_with_group_sync();
            k_tile += TILE_K;
        }

        // Write 4x4 results to global memory
        let out_row = tile_row + tid_y * THREAD_M;
        let out_col = tile_col + tid_x * THREAD_N;

        let mut i: u32 = 0;
        while i < THREAD_M {
            let row = out_row + i;
            if row < m {
                let mut j: u32 = 0;
                while j < THREAD_N {
                    let col = out_col + j;
                    if col < n {
                        let idx = shape_out.it(batch, batch_c, row, col) as usize;
                        out.write(idx, acc[(i * THREAD_N + j) as usize]);
                    }
                    j += 1;
                }
            }
            i += 1;
        }
    }
}

/// Naive GEMM (kept for reference and small matrices)
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gemm_naive(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes3,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_out: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 1)]
    shape_lhs: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 2)]
    shape_rhs: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] out: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] lhs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] rhs: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_out, shape_lhs, shape_rhs) =
        (&shapes.shape_out, &shapes.shape_lhs, &shapes.shape_rhs);

    if invocation_id.x >= shape_out.w {
        return;
    }

    for n in 0..shape_out.n {
        let mut sum = 0.0f32;

        for j in 0..shape_lhs.w {
            let i_lhs = shape_lhs.it(n, invocation_id.z, invocation_id.y, j);
            let i_rhs = shape_rhs.it(n, invocation_id.z, j, invocation_id.x);
            sum += lhs.read(i_lhs as usize) * rhs.read(i_rhs as usize);
        }

        let i_out = shape_out.it(n, invocation_id.z, invocation_id.y, invocation_id.x);
        out.write(i_out as usize, sum);
    }
}
