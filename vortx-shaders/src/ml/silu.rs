//! SiLU (Swish) activation function.

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes2;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;

const WORKGROUP_SIZE: u32 = 64;

/// SwiGLU non-linearity.
#[inline]
fn swish(x: f32, beta: f32) -> f32 {
    // This is the swiglu function from https://youtu.be/Mn_9W1nCFLo?si=LT6puSAfzgpP6ydz&t=3973
    x / (1.0 + (-beta * x).exp())
}

/// SiLU activation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn silu(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_a: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_b: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] in_out_a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] in_b: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_a, shape_b) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_a = *shape_a.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_b = *shape_b.at(0);

    if invocation_id.x < shape_a.w {
        let ia = shape_a.it(0, 0, 0, invocation_id.x) as usize;
        let ib = shape_b.it(0, 0, 0, invocation_id.x) as usize;
        let lhs = *in_out_a.at(ia);
        let rhs = *in_b.at(ib);
        *in_out_a.at_mut(ia) = rhs * swish(lhs, 1.0);
    }
}
