//! Repeat (broadcast) operation for tensors.

use super::shape::Shape;
#[cfg(feature = "push_constants")]
use super::shape::Shapes2;
use glamx::UVec3;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

const _WORKGROUP_SIZE: u32 = 128;

/// Repeat operation: copies source to result with broadcasting.
#[spirv_bindgen]
#[spirv(compute(threads(128, 1, 1)))]
pub fn repeat(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 0)]
    shape_result: &Shape,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(uniform, descriptor_set = 0, binding = 1)]
    shape_source: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] result: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_result, shape_source) = (&shapes.shape_a, &shapes.shape_b);

    if invocation_id.x >= shape_result.len() {
        return;
    }

    let id = shape_result.decompose(invocation_id.x);

    let ia = shape_result.it_vec(id) as usize;
    let ib = shape_source.it_repeating_vec(id) as usize;
    result.write(ia, source.read(ib));
}
