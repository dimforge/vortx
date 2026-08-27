//! Q6_Kx2 quantized GEMV shader.
//!
//! BlockQ6Kx2 contains two BlockQ6K blocks packed together (105 u32s total).
//! Each BlockQ6K: f16 scale, 64 bytes ql (low 4 bits), 32 bytes qh (high 2 bits), 16 bytes scales.

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes1;
use crate::utils::half::{unpack_half2x16, unpack_int4x8, unpack_uint4x8};
use khal_std::glamx::{IVec4, Mat4, UVec3, UVec4, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

const WORKGROUP_SIZE: usize = 32;
// BlockQ6Kx2 size in u32s: 105
const BLOCK_Q6KX2_SIZE: u32 = 105;

#[inline]
fn reduce_sum(index: usize, stride: usize, sketch: &mut [Vec4; WORKGROUP_SIZE]) {
    if index < stride {
        let val = *sketch.at(index + stride);
        *sketch.at_mut(index) += val;
    }
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
}

/// Select element from Vec4 based on index (0-3).
/// Avoids variable indexing which SPIR-V doesn't support well.
#[inline]
fn vec4_select(v: Vec4, idx: usize) -> f32 {
    if idx == 0 {
        v.x
    } else if idx == 1 {
        v.y
    } else if idx == 2 {
        v.z
    } else {
        v.w
    }
}

/// Dequantize Q6_Kx2 block for a workgroup thread.
/// Returns 4 Vec4s for each thread.
#[inline]
fn dequantize_q6_kx2_workgroup(m: &[u32], block_id: u32, k: u32) -> [Vec4; 4] {
    let _0xf = UVec4::splat(0xF);
    let splat_6 = UVec4::splat(6);
    let splat_4 = UVec4::splat(4);
    let splat_3 = UVec4::splat(3);
    let splat_2 = UVec4::splat(2);
    let splat_32 = IVec4::splat(32);

    let data_id = (block_id * BLOCK_Q6KX2_SIZE) as usize;

    if k / 16 == 0 {
        // Block A
        // Its data goes from data[0] to half of data[52]
        let d_a = unpack_half2x16(*m.at(data_id + 52)).x;

        const QL0: usize = 0;
        const QH0: usize = 32;
        const SC0: usize = 48;

        let i = ((k / 8) % 2) as usize;
        let data0 = Vec4::new(1.0, 1.0, 1.0, 1.0)
            * d_a
            * unpack_int4x8(*m.at(data_id + SC0 + i * 2)).as_vec4();
        let data1 = Vec4::new(1.0, 1.0, 1.0, 1.0)
            * d_a
            * unpack_int4x8(*m.at(data_id + SC0 + i * 2 + 1)).as_vec4();

        let l = (k % 8) as usize;
        let is = l / 4; // NOTE: is is either 0 or 1

        let qh = unpack_uint4x8(*m.at(data_id + l + QH0 + i * 8));
        let ql0 = unpack_uint4x8(*m.at(data_id + l + QL0 + i * 16));
        let ql32 = unpack_uint4x8(*m.at(data_id + l + QL0 + i * 16 + 8));

        let q1 = ((ql0 & _0xf) | ((qh & splat_3) << splat_4)).as_ivec4() - splat_32;
        let q2 = ((ql32 & _0xf) | (((qh >> splat_2) & splat_3) << splat_4)).as_ivec4() - splat_32;
        let q3 =
            ((ql0 >> splat_4) | (((qh >> splat_4) & splat_3) << splat_4)).as_ivec4() - splat_32;
        let q4 =
            ((ql32 >> splat_4) | (((qh >> splat_6) & splat_3) << splat_4)).as_ivec4() - splat_32;

        [
            Vec4::splat(vec4_select(data0, is)) * q1.as_vec4(),
            Vec4::splat(vec4_select(data0, is + 2)) * q2.as_vec4(),
            Vec4::splat(vec4_select(data1, is)) * q3.as_vec4(),
            Vec4::splat(vec4_select(data1, is + 2)) * q4.as_vec4(),
        ]
    } else {
        // Block B
        // Its data goes from half of data[52] to data[104].
        // All values are starting with the u16 leftmost bits of the previous index.
        let d_b = unpack_half2x16(*m.at(data_id + 104)).y;

        const QL0: usize = 53;
        const QH0: usize = 53 + 32;
        const SC0: usize = 53 + 48;

        let i = ((k / 8) % 2) as usize;
        let l = (k % 8) as usize;
        let isc0 = SC0 + i * 2;
        let isc1 = SC0 + i * 2 + 1;
        let data0 = Vec4::new(1.0, 1.0, 1.0, 1.0)
            * d_b
            * unpack_int4x8((*m.at(data_id + isc0 - 1) >> 16) | (*m.at(data_id + isc0) << 16))
                .as_vec4();
        let data1 = Vec4::new(1.0, 1.0, 1.0, 1.0)
            * d_b
            * unpack_int4x8((*m.at(data_id + isc1 - 1) >> 16) | (*m.at(data_id + isc1) << 16))
                .as_vec4();

        let is = l / 4; // NOTE: is either 0 or 1

        let iqh = l + QH0 + i * 8;
        let iql0 = l + QL0 + i * 16;
        let iql32 = l + QL0 + i * 16 + 8;

        let qh = unpack_uint4x8((*m.at(data_id + iqh - 1) >> 16) | (*m.at(data_id + iqh) << 16));
        let ql0 = unpack_uint4x8((*m.at(data_id + iql0 - 1) >> 16) | (*m.at(data_id + iql0) << 16));
        let ql32 =
            unpack_uint4x8((*m.at(data_id + iql32 - 1) >> 16) | (*m.at(data_id + iql32) << 16));

        let q1 = ((ql0 & _0xf) | ((qh & splat_3) << splat_4)).as_ivec4() - splat_32;
        let q2 = ((ql32 & _0xf) | (((qh >> splat_2) & splat_3) << splat_4)).as_ivec4() - splat_32;
        let q3 =
            ((ql0 >> splat_4) | (((qh >> splat_4) & splat_3) << splat_4)).as_ivec4() - splat_32;
        let q4 =
            ((ql32 >> splat_4) | (((qh >> splat_6) & splat_3) << splat_4)).as_ivec4() - splat_32;

        [
            Vec4::splat(vec4_select(data0, is)) * q1.as_vec4(),
            Vec4::splat(vec4_select(data0, is + 2)) * q2.as_vec4(),
            Vec4::splat(vec4_select(data1, is)) * q3.as_vec4(),
            Vec4::splat(vec4_select(data1, is + 2)) * q4.as_vec4(),
        ]
    }
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
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_m = *shape_m.at(0);

    let mut sum = Vec4::ZERO;
    let lid = local_id.x;

    for j in 0..shape_m.w {
        let quant0 = shape_m.it(0, 0, workgroup_id.x * 4, j);
        let quant1 = shape_m.it(0, 0, workgroup_id.x * 4 + 1, j);
        let quant2 = shape_m.it(0, 0, workgroup_id.x * 4 + 2, j);
        let quant3 = shape_m.it(0, 0, workgroup_id.x * 4 + 3, j);

        let parts0 = dequantize_q6_kx2_workgroup(m, quant0, lid);
        let parts1 = dequantize_q6_kx2_workgroup(m, quant1, lid);
        let parts2 = dequantize_q6_kx2_workgroup(m, quant2, lid);
        let parts3 = dequantize_q6_kx2_workgroup(m, quant3, lid);

        let j_base = (j * 128) as usize;
        let jj = ((lid / 16) * 64 + ((lid / 8) % 2) * 32 + (lid % 8)) as usize;
        let vj_a = *v.at(j_base + jj);
        let vj_b = *v.at(j_base + jj + 8);
        let vj_c = *v.at(j_base + jj + 16);
        let vj_d = *v.at(j_base + jj + 24);

        let mat0 = Mat4::from_cols(parts0[0], parts1[0], parts2[0], parts3[0]);
        let mat1 = Mat4::from_cols(parts0[1], parts1[1], parts2[1], parts3[1]);
        let mat2 = Mat4::from_cols(parts0[2], parts1[2], parts2[2], parts3[2]);
        let mat3 = Mat4::from_cols(parts0[3], parts1[3], parts2[3], parts3[3]);
        sum += mat0.transpose() * vj_a
            + mat1.transpose() * vj_b
            + mat2.transpose() * vj_c
            + mat3.transpose() * vj_d;
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

        // reduce_sum(lid as usize, 32, sketch);
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
