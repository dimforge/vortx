//! Q4_0x2 quantized GEMV shader.
//!
//! BlockQ4_0x2 contains two BlockQ4_0 blocks (f16 scale + 16 x 4-bit quants each).

use crate::utils::half::unpack_half2x16;
use khal_std::glamx::{Mat4, UVec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes1;

const WORKGROUP_SIZE: usize = 32;
const COLS_STEP: u32 = 4;
const BLOCK_Q4_0X2_SIZE: u32 = 9;

#[inline]
fn reduce_sum(index: usize, stride: usize, sketch: &mut [Vec4; WORKGROUP_SIZE]) {
    if index < stride {
        let val = sketch.read(index + stride);
        *sketch.at_mut(index) += val;
    }
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
}

/// Dequantize a part of BlockQ4_0 data.
/// Returns two Vec4s: low nibbles and high nibbles, scaled.
#[inline]
fn dequantize_part(data: u32, scale: f32) -> [Vec4; 2] {
    let x0 = (data & 0x0F) as i32 - 8;
    let x1 = ((data >> 4) & 0x0F) as i32 - 8;
    let x2 = ((data >> 8) & 0x0F) as i32 - 8;
    let x3 = ((data >> 12) & 0x0F) as i32 - 8;
    let x4 = ((data >> 16) & 0x0F) as i32 - 8;
    let x5 = ((data >> 20) & 0x0F) as i32 - 8;
    let x6 = ((data >> 24) & 0x0F) as i32 - 8;
    let x7 = ((data >> 28) & 0x0F) as i32 - 8;

    [
        Vec4::new(x0 as f32, x2 as f32, x4 as f32, x6 as f32) * scale,
        Vec4::new(x1 as f32, x3 as f32, x5 as f32, x7 as f32) * scale,
    ]
}

#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gemv(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup)] sketch: &mut [Vec4; WORKGROUP_SIZE],
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
    #[cfg(not(feature = "push_constants"))]
    let shape_m = shape_m.read(0);

    let mut sum = [Vec4::ZERO];
    let lid = local_id.x;

    for j in 0..(shape_m.w / COLS_STEP) {
        // Calculate block indices for 4 matrix rows
        let quant0 = (shape_m.it(0, 0, workgroup_id.x * 4, j * COLS_STEP + lid / 8)
            * BLOCK_Q4_0X2_SIZE) as usize;
        let quant1 = (shape_m.it(0, 0, workgroup_id.x * 4 + 1, j * COLS_STEP + lid / 8)
            * BLOCK_Q4_0X2_SIZE) as usize;
        let quant2 = (shape_m.it(0, 0, workgroup_id.x * 4 + 2, j * COLS_STEP + lid / 8)
            * BLOCK_Q4_0X2_SIZE) as usize;
        let quant3 = (shape_m.it(0, 0, workgroup_id.x * 4 + 3, j * COLS_STEP + lid / 8)
            * BLOCK_Q4_0X2_SIZE) as usize;

        let j_base = (j * 16 * COLS_STEP) as usize;
        let vj_a = v.read(j_base + lid as usize + (lid / 4) as usize * 4);
        let vj_b = v.read(j_base + lid as usize + (lid / 4) as usize * 4 + 4);

        if (lid / 4).is_multiple_of(2) {
            // Dequantizing block 1
            let llid = (lid % 4) as usize;

            let scale0 = unpack_half2x16(m.read(quant0)).x;
            let data0 = m.read(quant0 + llid) >> 16 | m.read(quant0 + llid + 1) << 16;
            let parts0 = dequantize_part(data0, scale0);

            let scale1 = unpack_half2x16(m.read(quant1)).x;
            let data1 = m.read(quant1 + llid) >> 16 | m.read(quant1 + llid + 1) << 16;
            let parts1 = dequantize_part(data1, scale1);

            let scale2 = unpack_half2x16(m.read(quant2)).x;
            let data2 = m.read(quant2 + llid) >> 16 | m.read(quant2 + llid + 1) << 16;
            let parts2 = dequantize_part(data2, scale2);

            let scale3 = unpack_half2x16(m.read(quant3)).x;
            let data3 = m.read(quant3 + llid) >> 16 | m.read(quant3 + llid + 1) << 16;
            let parts3 = dequantize_part(data3, scale3);

            // Matrix-vector multiply
            let mat0 = Mat4::from_cols(parts0[0], parts1[0], parts2[0], parts3[0]);
            let mat1 = Mat4::from_cols(parts0[1], parts1[1], parts2[1], parts3[1]);
            sum[0] += mat0.transpose() * vj_a + mat1.transpose() * vj_b;
        } else {
            // Dequantizing block 2
            let llid = (lid % 4) as usize;

            let scale0 = unpack_half2x16(m.read(quant0 + 4)).y;
            let data0 = m.read(quant0 + llid + 5);
            let parts0 = dequantize_part(data0, scale0);

            let scale1 = unpack_half2x16(m.read(quant1 + 4)).y;
            let data1 = m.read(quant1 + llid + 5);
            let parts1 = dequantize_part(data1, scale1);

            let scale2 = unpack_half2x16(m.read(quant2 + 4)).y;
            let data2 = m.read(quant2 + llid + 5);
            let parts2 = dequantize_part(data2, scale2);

            let scale3 = unpack_half2x16(m.read(quant3 + 4)).y;
            let data3 = m.read(quant3 + llid + 5);
            let parts3 = dequantize_part(data3, scale3);

            // Matrix-vector multiply
            let mat0 = Mat4::from_cols(parts0[0], parts1[0], parts2[0], parts3[0]);
            let mat1 = Mat4::from_cols(parts0[1], parts1[1], parts2[1], parts3[1]);
            sum[0] += mat0.transpose() * vj_a + mat1.transpose() * vj_b;
        }
    }

    #[cfg(feature = "subgroup_ops")]
    {
        let reduced = Vec4::new(
            khal_std::sync::subgroup_f_add(sum[0].x),
            khal_std::sync::subgroup_f_add(sum[0].y),
            khal_std::sync::subgroup_f_add(sum[0].z),
            khal_std::sync::subgroup_f_add(sum[0].w),
        );
        if lid == 0 {
            *out.at_mut(workgroup_id.x as usize) = reduced;
        }
    }

    #[cfg(not(feature = "subgroup_ops"))]
    {
        *sketch.at_mut(lid as usize) = sum[0];

        khal_std::sync::workgroup_memory_barrier_with_group_sync();

        reduce_sum(lid as usize, 16, sketch);
        reduce_sum(lid as usize, 8, sketch);
        reduce_sum(lid as usize, 4, sketch);
        reduce_sum(lid as usize, 2, sketch);
        reduce_sum(lid as usize, 1, sketch);

        if lid == 0 {
            *out.at_mut(workgroup_id.x as usize) = sketch.read(0);
        }
    }
}
