//! Element-wise activation functions (tanh forward/backward).
//!
//! vortx upstream has no activations; these were added for zealot's MLP policy.
//! Uniform-shape bindings only (no push_constants variant), matching the default build.

use super::shape::Shape;
use crate::utils::limits::MAX_NUM_WORKGROUPS;
use crate::utils::trig::stable_tanh;
use glamx::UVec3;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

const WORKGROUP_SIZE: u32 = 256;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Element-wise tanh, in place: `a = tanh(a)`.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_tanh(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] shape_a: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] a: &mut [f32],
) {
    for thread_id in (invocation_id.x..shape_a.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it_vec(id) as usize;
        let slot = a.at_mut(ia);
        *slot = stable_tanh(*slot);
    }
}

/// Backward of tanh, in place: `g *= 1 - y*y`, where `y = tanh(x)` is the forward output.
///
/// `g` and `y` are expected to have the same shape (the per-element local derivative
/// of tanh is `1 - tanh(x)^2`, expressed in terms of the cached output `y`).
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_tanh_backward(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] shape_g: &Shape,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] shape_y: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] g: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] y: &[f32],
) {
    for thread_id in (invocation_id.x..shape_g.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_g.decompose(thread_id);
        let ig = shape_g.it_vec(id) as usize;
        let iy = shape_y.it_vec(id) as usize;
        let yi = y.read(iy);
        *g.at_mut(ig) *= 1.0 - yi * yi;
    }
}
