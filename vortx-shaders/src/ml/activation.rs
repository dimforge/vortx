//! Backward passes of the element-wise activations.
//!
//! The forward directions live in [`crate::ml::unary`] (`UnaryOp::Tanh`,
//! `UnaryOp::Elu`); only the gradients are provided here, for training.

use crate::linalg::Shape;
use crate::utils::iterators::StepRng;
use crate::utils::limits::MAX_NUM_WORKGROUPS;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

const WORKGROUP_SIZE: u32 = 256;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Backward of tanh, in place: `g *= 1 - y*y`, where `y = tanh(x)` is the forward output.
///
/// `g` and `y` must have the same shape.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_tanh_backward(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] shape_g: &Shape,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] shape_y: &Shape,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] g: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] y: &[f32],
) {
    for thread_id in StepRng::new(invocation_id.x..shape_g.len(), MAX_NUM_THREADS) {
        let id = shape_g.decompose(thread_id);
        let ig = shape_g.it_vec(id) as usize;
        let iy = shape_y.it_vec(id) as usize;
        let yi = y.read(iy);
        *g.at_mut(ig) *= 1.0 - yi * yi;
    }
}
