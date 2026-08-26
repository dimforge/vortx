//! Q8_K quantized GEMV shader.
//!
//! BlockQ8K: f32 delta, 256 x 8-bit signed quants, 16 x 16-bit bsums.

use crate::utils::half::unpack_int4x8;
use khal_std::glamx::{UVec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes1;

const WORKGROUP_SIZE: u32 = 32;

// BlockQ8K structure (repr(C), alignment 4):
// - d: f32 (1 u32)
// - qs: [i8; 256] (64 u32s)
// - bsums: [i16; 16] (8 u32s, no padding — 260 is already 2-byte aligned)
// Total: 73 u32s = 292 bytes
const BLOCK_Q8K_SIZE: u32 = 73;

/// Dequantize a full BlockQ8K block.
#[inline]
fn dequantize_block(data: &[u32], base: usize) -> [Vec4; 64] {
    let mut result = [Vec4::ZERO; 64];

    // d is stored as f32 directly
    let d = f32::from_bits(*data.at(base));

    #[allow(clippy::needless_range_loop)]
    for j in 0..64 {
        let qs = unpack_int4x8(*data.at(base + 1 + j));
        result[j] = Vec4::new(qs.x as f32, qs.y as f32, qs.z as f32, qs.w as f32) * d;
    }

    result
}

#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gemv(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_m: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] m: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] v: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let shape_m = shapes.shape;
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_m = *shape_m.at(0);

    if invocation_id.x < shape_m.h {
        let i_out = invocation_id.x as usize;
        let mut sum = 0.0f32;

        for j in 0..shape_m.w {
            let block_idx = (shape_m.it(0, 0, invocation_id.x, j) * BLOCK_Q8K_SIZE) as usize;
            let dequant = dequantize_block(m, block_idx);

            // Unroll calculation with all block elements
            let i_base = (j * 64) as usize;

            for k in (0u32..64).step_by(16) {
                let k = k as usize;
                sum += dequant[k].dot(*v.at(k + i_base))
                    + dequant[k + 1].dot(*v.at(k + i_base + 1))
                    + dequant[k + 2].dot(*v.at(k + i_base + 2))
                    + dequant[k + 3].dot(*v.at(k + i_base + 3))
                    + dequant[k + 4].dot(*v.at(k + i_base + 4))
                    + dequant[k + 5].dot(*v.at(k + i_base + 5))
                    + dequant[k + 6].dot(*v.at(k + i_base + 6))
                    + dequant[k + 7].dot(*v.at(k + i_base + 7))
                    + dequant[k + 8].dot(*v.at(k + i_base + 8))
                    + dequant[k + 9].dot(*v.at(k + i_base + 9))
                    + dequant[k + 10].dot(*v.at(k + i_base + 10))
                    + dequant[k + 11].dot(*v.at(k + i_base + 11))
                    + dequant[k + 12].dot(*v.at(k + i_base + 12))
                    + dequant[k + 13].dot(*v.at(k + i_base + 13))
                    + dequant[k + 14].dot(*v.at(k + i_base + 14))
                    + dequant[k + 15].dot(*v.at(k + i_base + 15));
            }
        }

        *out.at_mut(i_out) = Vec4::splat(sum);
    }
}
