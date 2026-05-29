//! Binary operations for tensors (element-wise add, sub, mul, div, copy).

use super::shape::Shape;
#[cfg(feature = "push_constants")]
use super::shape::Shapes2;
use crate::utils::limits::MAX_NUM_WORKGROUPS;
use glamx::UVec3;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

const WORKGROUP_SIZE: u32 = 256;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Binary operation offsets.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct BinOpOffsets {
    pub a: u32,
    pub b: u32,
    pub pad0: u32,
    pub pad1: u32,
}

/// Element-wise addition: a += b
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_add(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_a: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 1)]
    shape_b: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] b: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_a, shape_b) = (&shapes.shape_a, &shapes.shape_b);

    for thread_id in (invocation_id.x..shape_a.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it_vec(id) as usize;
        let ib = shape_b.it_vec(id) as usize;
        *a.at_mut(ia) += b.read(ib);
    }
}

/// Element-wise subtraction: a -= b
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_sub(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_a: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 1)]
    shape_b: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] b: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_a, shape_b) = (&shapes.shape_a, &shapes.shape_b);

    for thread_id in (invocation_id.x..shape_a.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it_vec(id) as usize;
        let ib = shape_b.it_vec(id) as usize;
        *a.at_mut(ia) -= b.read(ib);
    }
}

/// Element-wise multiplication: a *= b
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_mul(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_a: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 1)]
    shape_b: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] b: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_a, shape_b) = (&shapes.shape_a, &shapes.shape_b);

    for thread_id in (invocation_id.x..shape_a.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it_vec(id) as usize;
        let ib = shape_b.it_vec(id) as usize;
        *a.at_mut(ia) *= b.read(ib);
    }
}

/// Element-wise division: a /= b
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_div(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_a: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 1)]
    shape_b: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] b: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_a, shape_b) = (&shapes.shape_a, &shapes.shape_b);

    for thread_id in (invocation_id.x..shape_a.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it_vec(id) as usize;
        let ib = shape_b.it_vec(id) as usize;
        *a.at_mut(ia) /= b.read(ib);
    }
}

/// Copy operation: a = b
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_copy(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_a: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 1)]
    shape_b: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] b: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_a, shape_b) = (&shapes.shape_a, &shapes.shape_b);

    for thread_id in (invocation_id.x..shape_a.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it_vec(id) as usize;
        let ib = shape_b.it_vec(id) as usize;
        a.write(ia, b.read(ib));
    }
}

// TODO: have the offset in the Shape directly? Note that the offset is only useful when
//       we can't shift the buffer bindings due to platform limitations (like with WebGpu).
/// Copy operation with offsets: a[offset_a + i] = b[offset_b + i]
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_copy_with_offsets(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] offsets: &BinOpOffsets,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] shape_a: &Shape,
    #[spirv(uniform, descriptor_set = 0, binding = 2)] shape_b: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] b: &[f32],
) {
    for thread_id in (invocation_id.x..shape_a.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it_vec(id);
        let ib = shape_b.it_vec(id);
        a.write((offsets.a + ia) as usize, b.read((offsets.b + ib) as usize));
    }
}
