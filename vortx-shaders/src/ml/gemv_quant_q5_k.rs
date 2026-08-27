//! Q5_K quantized GEMV shader.
//!
//! BlockQ5K: super-block scale, super-block min, 12 bytes scales/mins, 32 bytes high bits, 128 bytes quants.

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes1;
use crate::utils::half::unpack_half2x16;
use khal_std::glamx::{UVec2, UVec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

const WORKGROUP_SIZE: usize = 32;
// BlockQ5K size in u32s: 1 (d_dmin) + 3 (scales) + 8 (qh) + 32 (qs) = 44
const BLOCK_Q5K_SIZE: u32 = 44;

#[inline]
fn reduce_sum(index: usize, stride: usize, sketch: &mut [Vec4; WORKGROUP_SIZE]) {
    if index < stride {
        let val = *sketch.at(index + stride);
        *sketch.at_mut(index) += val;
    }
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
}

/// Unpack scale and min from the packed scales array.
/// Shared with Q4_K.
#[inline]
fn unpack_scale_and_min(j: u32, qj_prev: u32, qj: u32, qj_next: u32) -> UVec2 {
    let shift = (j % 4) * 8;
    let qj_prev_shifted = (qj_prev >> shift) & 0x00ff;
    let qj_shifted = (qj >> shift) & 0x00ff;
    let qj_next_shifted = (qj_next >> shift) & 0x00ff;

    if j < 4 {
        let d = qj_shifted & 63;
        let m = qj_next_shifted & 63;
        UVec2::new(d, m)
    } else {
        let d = (qj_next_shifted & 0xf) | ((qj_prev_shifted >> 6) << 4);
        let m = (qj_next_shifted >> 4) | ((qj_shifted >> 6) << 4);
        UVec2::new(d, m)
    }
}

/// Dequantize Q5_K block for a workgroup thread.
#[inline]
fn dequantize_q5_k_workgroup(m: &[u32], block_id: u32, k: u32) -> [Vec4; 2] {
    let d_dmin_id = (block_id * BLOCK_Q5K_SIZE) as usize;
    let scales_id = d_dmin_id + 1;
    let qh_id = scales_id + 3;
    let qs_id = qh_id + 8;

    let d_dmin = unpack_half2x16(*m.at(d_dmin_id));
    let d = d_dmin.x;
    let min = d_dmin.y;

    let j = k / 8;
    let l = k % 8;
    let is = j * 2;
    let u1 = 1u32 << (j * 2);
    let u2 = 2u32 << (j * 2);

    let qj_prev1 = *m.at(scales_id + (is / 4).max(1) as usize - 1);
    let qj1 = *m.at(scales_id + (is / 4) as usize);
    let qj_next1 = *m.at(scales_id + (is / 4 + 1) as usize);
    let sc_m1 = unpack_scale_and_min(is, qj_prev1, qj1, qj_next1);
    let d1 = d * sc_m1.x as f32;
    let m1 = min * sc_m1.y as f32;

    let qj_prev2 = *m.at(scales_id + ((is + 1) / 4).max(1) as usize - 1);
    let qj2 = *m.at(scales_id + ((is + 1) / 4) as usize);
    let qj_next2 = *m.at(scales_id + ((is + 1) / 4 + 1) as usize);
    let sc_m2 = unpack_scale_and_min(is + 1, qj_prev2, qj2, qj_next2);
    let d2 = d * sc_m2.x as f32;
    let m2 = min * sc_m2.y as f32;

    let qs = *m.at(qs_id + k as usize);
    let qh = *m.at(qh_id + l as usize);

    #[inline]
    fn select_u32(cond: bool, t: u32, f: u32) -> u32 {
        if cond {
            t
        } else {
            f
        }
    }

    let res_a = Vec4::new(
        ((qs & 0xF) + select_u32((qh & u1) != 0, 16, 0)) as f32,
        (((qs >> 8) & 0xF) + select_u32(((qh >> 8) & u1) != 0, 16, 0)) as f32,
        (((qs >> 16) & 0xF) + select_u32(((qh >> 16) & u1) != 0, 16, 0)) as f32,
        (((qs >> 24) & 0xF) + select_u32(((qh >> 24) & u1) != 0, 16, 0)) as f32,
    ) * d1
        - m1;

    let res_b = Vec4::new(
        (((qs >> 4) & 0xF) + select_u32((qh & u2) != 0, 16, 0)) as f32,
        (((qs >> 12) & 0xF) + select_u32(((qh >> 8) & u2) != 0, 16, 0)) as f32,
        (((qs >> 20) & 0xF) + select_u32(((qh >> 16) & u2) != 0, 16, 0)) as f32,
        (((qs >> 28) & 0xF) + select_u32(((qh >> 24) & u2) != 0, 16, 0)) as f32,
    ) * d2
        - m2;

    [res_a, res_b]
}

#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gemv_q5_k(
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

        let parts0 = dequantize_q5_k_workgroup(m, quant0, lid);
        let parts1 = dequantize_q5_k_workgroup(m, quant1, lid);
        let parts2 = dequantize_q5_k_workgroup(m, quant2, lid);
        let parts3 = dequantize_q5_k_workgroup(m, quant3, lid);

        let j_base = (j * 64) as usize;
        let jj = (lid & 0xfffffff8) as usize; // == (lid / 8) * 8
        let vj_a = *v.at(j_base + lid as usize + jj);
        let vj_b = *v.at(j_base + lid as usize + jj + 8);

        sum += Vec4::new(
            parts0[0].dot(vj_a),
            parts1[0].dot(vj_a),
            parts2[0].dot(vj_a),
            parts3[0].dot(vj_a),
        );
        sum += Vec4::new(
            parts0[1].dot(vj_b),
            parts1[1].dot(vj_b),
            parts2[1].dot(vj_b),
            parts3[1].dot(vj_b),
        );
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
