//! Contiguous memory layout operations.

use super::shape::Shape;
#[cfg(feature = "push_constants")]
use super::shape::Shapes1;
use crate::utils::limits::MAX_NUM_WORKGROUPS;
use glamx::UVec3;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

const WORKGROUP_SIZE: u32 = 128;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Convert to contiguous row-major layout.
#[spirv_bindgen]
#[spirv(compute(threads(128, 1, 1)))]
pub fn contiguous(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_src: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dest: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] src: &[u32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = &shapes.shape;

    contiguous_impl(invocation_id, shape_src, dest, src, 0)
}

/// Convert to contiguous row-major layout with offset.
#[spirv_bindgen]
#[spirv(compute(threads(128, 1, 1)))]
pub fn contiguous_with_offset(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_src: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dest: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] src: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] offset: &u32,
) {
    #[cfg(feature = "push_constants")]
    let shape_src = &shapes.shape;

    contiguous_impl(invocation_id, shape_src, dest, src, *offset)
}

#[inline(always)]
fn contiguous_impl(
    invocation_id: UVec3,
    shape_src: &Shape,
    dest: &mut [u32],
    src: &[u32],
    offset: u32,
) {
    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        // Compute the corresponding (i, j, k, l) indices for the out tensor.
        // A contiguous row-major tensor with dimensions [n,c,h,w] has strides
        // [c * h * w, h * w, w, 1]
        let h_stride = shape_src.w;
        let c_stride = shape_src.h * shape_src.w;
        let n_stride = shape_src.c * shape_src.h * shape_src.w;

        let n = thread_id / n_stride;
        let c = (thread_id % n_stride) / c_stride;
        let h = (thread_id % c_stride) / h_stride;
        let w = thread_id % h_stride;

        // NOTE: `dest` is assumed to have the same size as `src` but contiguous.
        dest.write(
            thread_id as usize,
            src.read((offset + shape_src.it(n, c, h, w)) as usize),
        );
    }
}
