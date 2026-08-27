//! Q4_1x2 quantized GEMV shader.
//!
//! BlockQ4_1x2 contains two BlockQ4_1 blocks (f16 scale + f16 min + 16 x 4-bit quants each).

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes1;
use crate::utils::half::unpack_half2x16;
use khal_std::glamx::{UVec3, Vec2, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

const WORKGROUP_SIZE: u32 = 64;

/// Dequantize a part of BlockQ4_1 data.
/// Returns two Vec4s: low nibbles and high nibbles, scaled and shifted.
#[inline]
fn dequantize_part(data: u32, scale_mid: Vec2) -> [Vec4; 2] {
    let x0 = data & 0x0F;
    let x1 = (data >> 4) & 0x0F;
    let x2 = (data >> 8) & 0x0F;
    let x3 = (data >> 12) & 0x0F;
    let x4 = (data >> 16) & 0x0F;
    let x5 = (data >> 20) & 0x0F;
    let x6 = (data >> 24) & 0x0F;
    let x7 = (data >> 28) & 0x0F;

    [
        Vec4::new(x0 as f32, x2 as f32, x4 as f32, x6 as f32) * scale_mid.x + scale_mid.y,
        Vec4::new(x1 as f32, x3 as f32, x5 as f32, x7 as f32) * scale_mid.x + scale_mid.y,
    ]
}

/// Dequantize a full BlockQ4_1x2 block.
#[inline]
fn dequantize_block(data: &[u32], base: usize) -> [Vec4; 16] {
    let mut result = [Vec4::ZERO; 16];

    // First block
    let scale_mid_a = unpack_half2x16(*data.at(base));
    for k in 0..4 {
        let parts = dequantize_part(*data.at(base + k + 1), scale_mid_a);
        result[k] = parts[0];
        result[4 + k] = parts[1];
    }

    // Second block
    let scale_mid_b = unpack_half2x16(*data.at(base + 5));
    for k in 0..4 {
        let parts = dequantize_part(*data.at(base + k + 6), scale_mid_b);
        result[8 + k] = parts[0];
        result[12 + k] = parts[1];
    }

    result
}

#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gemv_q4_1x2(
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
            let block_idx = (shape_m.it(0, 0, invocation_id.x, j) * 10) as usize; // BlockQ4_1x2 is 10 u32s
            let dequant = dequantize_block(m, block_idx);

            // Unroll calculation with all block elements
            let i_base = (j * 16) as usize;
            sum += dequant[0].dot(*v.at(i_base))
                + dequant[1].dot(*v.at(i_base + 1))
                + dequant[2].dot(*v.at(i_base + 2))
                + dequant[3].dot(*v.at(i_base + 3))
                + dequant[4].dot(*v.at(i_base + 4))
                + dequant[5].dot(*v.at(i_base + 5))
                + dequant[6].dot(*v.at(i_base + 6))
                + dequant[7].dot(*v.at(i_base + 7))
                + dequant[8].dot(*v.at(i_base + 8))
                + dequant[9].dot(*v.at(i_base + 9))
                + dequant[10].dot(*v.at(i_base + 10))
                + dequant[11].dot(*v.at(i_base + 11))
                + dequant[12].dot(*v.at(i_base + 12))
                + dequant[13].dot(*v.at(i_base + 13))
                + dequant[14].dot(*v.at(i_base + 14))
                + dequant[15].dot(*v.at(i_base + 15));
        }

        *out.at_mut(i_out) = Vec4::splat(sum);
    }
}
