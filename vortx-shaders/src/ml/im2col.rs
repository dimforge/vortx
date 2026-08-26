//! Image to column transformation.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

const WORKGROUP_SIZE: u32 = 32;
const NUM_ITER: u32 = 512 / WORKGROUP_SIZE;

/// Im2Col parameters.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct Im2ColParams {
    pub batch_offset: u32,
    pub offset_delta: u32,
    #[allow(non_snake_case)]
    pub IC: u32,
    #[allow(non_snake_case)]
    pub IW: u32,
    #[allow(non_snake_case)]
    pub IH: u32,
    #[allow(non_snake_case)]
    pub OW: u32,
    #[allow(non_snake_case)]
    pub OH: u32,
    #[allow(non_snake_case)]
    pub KW: u32,
    #[allow(non_snake_case)]
    pub KH: u32,
    pub pelements: u32,
    #[allow(non_snake_case)]
    pub CHW: u32,
    pub s0: i32,
    pub s1: i32,
    pub p0: i32,
    pub p1: i32,
    pub d0: i32,
    pub d1: i32,
}

/// Im2Col transformation.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn im2col(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &[Im2ColParams],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_tensor: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out_tensor: &mut [f32],
) {
    // Load params from storage buffer to local variable (enables LICM)
    let params = *params.at(0);

    let gidx = invocation_id.x;
    let oh = invocation_id.y;
    let batch = invocation_id.z / params.IC;
    let ic = invocation_id.z % params.IC;

    let src_base = ic * params.offset_delta + batch * params.batch_offset;
    let dst_base =
        ((batch * params.OH + oh) * params.OW) * params.CHW + ic * (params.KW * params.KH);
    let oh_s1 = oh as i32 * params.s1;
    let ksize = params.OW * if params.KH > 1 { params.KW } else { 1 };

    let base_linear_idx = gidx * NUM_ITER;

    let max_ky = ksize / params.OW;

    let mut current_kx = base_linear_idx / ksize;
    let rem = base_linear_idx - (current_kx * ksize);
    let mut current_ky = rem / params.OW;
    let mut current_ix = rem % params.OW;

    let mut values = [0.0f32; NUM_ITER as usize];
    let mut offset_dst = [0u32; NUM_ITER as usize];

    for idx in 0..NUM_ITER {
        let linear_idx = base_linear_idx + idx;

        if linear_idx >= params.pelements {
            continue;
        }

        let iiw =
            (current_ix as i32 * params.s0 + current_kx as i32 * params.d0 - params.p0) as u32;
        let iih = (oh_s1 + current_ky as i32 * params.d1 - params.p1) as u32;

        offset_dst[idx as usize] =
            dst_base + current_ix * params.CHW + current_ky * params.KW + current_kx;

        if iih < params.IH && iiw < params.IW {
            values[idx as usize] = *in_tensor.at((src_base + iih * params.IW + iiw) as usize);
        }

        current_ix += 1;
        if current_ix == params.OW {
            current_ix = 0;
            current_ky += 1;
            if current_ky == max_ky {
                current_ky = 0;
                current_kx += 1;
            }
        }
    }

    for idx in 0..NUM_ITER {
        let linear_idx = base_linear_idx + idx;

        if linear_idx >= params.pelements {
            continue;
        }

        *out_tensor.at_mut(offset_dst[idx as usize] as usize) = values[idx as usize];
    }
}
