//! Q8_0x2 quantized GEMV shader.
//!
//! BlockQ8_0x2 contains two BlockQ8_0 blocks (f16 scale + 32 x 8-bit signed quants each).

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes1;
use crate::utils::half::{unpack_half2x16, unpack_int4x8};
use khal_std::glamx::{UVec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

const WORKGROUP_SIZE: usize = 32;
const BLOCK_Q8_0X2_SIZE: u32 = 17; // 17 u32s

#[inline]
fn reduce_sum(index: usize, stride: usize, sketch: &mut [Vec4; WORKGROUP_SIZE]) {
    if index < stride {
        let val = *sketch.at(index + stride);
        *sketch.at_mut(index) += val;
    }
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
}

#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gemv_q8_0x2(
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
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_m = *shape_m.at(0);

    let mut sum = Vec4::ZERO;
    let lid = local_id.x;

    for j in 0..(shape_m.w / 2) {
        // Calculate block indices for 4 matrix rows
        let quant0 =
            (shape_m.it(0, 0, workgroup_id.x * 4, j * 2 + lid / 16) * BLOCK_Q8_0X2_SIZE) as usize;
        let quant1 = (shape_m.it(0, 0, workgroup_id.x * 4 + 1, j * 2 + lid / 16)
            * BLOCK_Q8_0X2_SIZE) as usize;
        let quant2 = (shape_m.it(0, 0, workgroup_id.x * 4 + 2, j * 2 + lid / 16)
            * BLOCK_Q8_0X2_SIZE) as usize;
        let quant3 = (shape_m.it(0, 0, workgroup_id.x * 4 + 3, j * 2 + lid / 16)
            * BLOCK_Q8_0X2_SIZE) as usize;

        let j_base = (j * 32) as usize;
        let vj = *v.at(j_base + lid as usize);

        if (lid / 8).is_multiple_of(2) {
            // Dequantizing block 1
            let llid = (lid % 8) as usize;
            let scale0 = unpack_half2x16(*m.at(quant0)).x;
            let data0 = unpack_int4x8(*m.at(quant0 + llid) >> 16 | *m.at(quant0 + llid + 1) << 16);
            let scale1 = unpack_half2x16(*m.at(quant1)).x;
            let data1 = unpack_int4x8(*m.at(quant1 + llid) >> 16 | *m.at(quant1 + llid + 1) << 16);
            let scale2 = unpack_half2x16(*m.at(quant2)).x;
            let data2 = unpack_int4x8(*m.at(quant2 + llid) >> 16 | *m.at(quant2 + llid + 1) << 16);
            let scale3 = unpack_half2x16(*m.at(quant3)).x;
            let data3 = unpack_int4x8(*m.at(quant3 + llid) >> 16 | *m.at(quant3 + llid + 1) << 16);

            let row0 = Vec4::new(
                data0.x as f32,
                data0.y as f32,
                data0.z as f32,
                data0.w as f32,
            ) * scale0;
            let row1 = Vec4::new(
                data1.x as f32,
                data1.y as f32,
                data1.z as f32,
                data1.w as f32,
            ) * scale1;
            let row2 = Vec4::new(
                data2.x as f32,
                data2.y as f32,
                data2.z as f32,
                data2.w as f32,
            ) * scale2;
            let row3 = Vec4::new(
                data3.x as f32,
                data3.y as f32,
                data3.z as f32,
                data3.w as f32,
            ) * scale3;

            sum += Vec4::new(row0.dot(vj), row1.dot(vj), row2.dot(vj), row3.dot(vj));
        } else {
            // Dequantizing block 2
            let llid = (lid % 8) as usize;
            let scale0 = unpack_half2x16(*m.at(quant0 + 8)).y;
            let data0 = unpack_int4x8(*m.at(quant0 + llid + 9));
            let scale1 = unpack_half2x16(*m.at(quant1 + 8)).y;
            let data1 = unpack_int4x8(*m.at(quant1 + llid + 9));
            let scale2 = unpack_half2x16(*m.at(quant2 + 8)).y;
            let data2 = unpack_int4x8(*m.at(quant2 + llid + 9));
            let scale3 = unpack_half2x16(*m.at(quant3 + 8)).y;
            let data3 = unpack_int4x8(*m.at(quant3 + llid + 9));

            let row0 = Vec4::new(
                data0.x as f32,
                data0.y as f32,
                data0.z as f32,
                data0.w as f32,
            ) * scale0;
            let row1 = Vec4::new(
                data1.x as f32,
                data1.y as f32,
                data1.z as f32,
                data1.w as f32,
            ) * scale1;
            let row2 = Vec4::new(
                data2.x as f32,
                data2.y as f32,
                data2.z as f32,
                data2.w as f32,
            ) * scale2;
            let row3 = Vec4::new(
                data3.x as f32,
                data3.y as f32,
                data3.z as f32,
                data3.w as f32,
            ) * scale3;

            sum += Vec4::new(row0.dot(vj), row1.dot(vj), row2.dot(vj), row3.dot(vj));
        }
    }

    #[cfg(feature = "subgroup_ops")]
    {
        let reduced = Vec4::new(
            khal_std::sync::subgroup_f_add(sum.x),
            khal_std::sync::subgroup_f_add(sum.y),
            khal_std::sync::subgroup_f_add(sum.z),
            khal_std::sync::subgroup_f_add(sum.w),
        );
        if lid == 0 {
            let i_out = workgroup_id.x as usize;
            *out.at_mut(i_out) = reduced;
        }
    }

    #[cfg(not(feature = "subgroup_ops"))]
    {
        *sketch.at_mut(lid as usize) = sum;

        khal_std::sync::workgroup_memory_barrier_with_group_sync();

        // reduce_sum(lid as usize, 32, sketch); // Only 32 threads
        reduce_sum(lid as usize, 16, sketch);
        reduce_sum(lid as usize, 8, sketch);
        reduce_sum(lid as usize, 4, sketch);
        reduce_sum(lid as usize, 2, sketch);
        reduce_sum(lid as usize, 1, sketch);

        if lid == 0 {
            let i_out = workgroup_id.x as usize;
            *out.at_mut(i_out) = *sketch.at(0);
        }
    }
}
