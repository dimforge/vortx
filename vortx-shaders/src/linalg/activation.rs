//! Element-wise activation functions (tanh forward/backward).
//!
//! vortx upstream has no activations; these were added for zealot's MLP policy.
//! Uniform-shape bindings only (no push_constants variant), matching the default build.

use super::shape::Shape;
use crate::utils::limits::MAX_NUM_WORKGROUPS;
use crate::utils::trig::stable_tanh;
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;
use glamx::{UVec3, Vec4};
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

/// Element-wise ELU (alpha = 1), in place: `a = a if a > 0 else exp(a) - 1`.
///
/// Mirrors `zealot-rl`'s CPU `elu`. Hidden layers of the AGILE/rsl_rl actor/critic
/// stacks use ELU; the output layer stays linear (so this is only applied to the
/// hidden pre-activations).
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_elu(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] shape_a: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] a: &mut [f32],
) {
    for thread_id in (invocation_id.x..shape_a.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it_vec(id) as usize;
        let slot = a.at_mut(ia);
        let x = *slot;
        *slot = if x > 0.0 { x } else { x.exp() - 1.0 };
    }
}

/// Element-wise ELU, **vec4** in place: processes 4 contiguous f32 per thread via
/// 128-bit loads/stores. Assumes a contiguous buffer whose length is a multiple
/// of 4 (true for the dense activation buffers). The buffer is the same bytes as
/// the scalar version — only the binding type differs — so it's a drop-in for
/// contiguous tensors. Memory-bound elementwise kernels win big from the wider
/// transactions.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_elu_vec4(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] shape_a: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] a: &mut [Vec4],
) {
    let n4 = shape_a.len() / 4;
    for thread_id in (invocation_id.x..n4).step_by(MAX_NUM_THREADS as usize) {
        let i = thread_id as usize;
        let v = a.read(i);
        let e = |x: f32| if x > 0.0 { x } else { x.exp() - 1.0 };
        *a.at_mut(i) = Vec4::new(e(v.x), e(v.y), e(v.z), e(v.w));
    }
}

/// Backward of ELU (alpha = 1), in place: `g *= 1 if y > 0 else y + 1`, where
/// `y = elu(x)` is the cached forward output.
///
/// Valid because `elu'(x) = 1` for `x > 0` and `exp(x) = elu(x) + 1` for `x <= 0`,
/// and `y > 0 <=> x > 0`. Same cached-output formulation as `gpu_tanh_backward`,
/// matching `zealot-rl`'s `elu_grad_from_act`.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_elu_backward(
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
        *g.at_mut(ig) *= if yi > 0.0 { 1.0 } else { yi + 1.0 };
    }
}
