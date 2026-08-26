//! Unary operations for tensors.

use khal_std::glamx::{UVec3, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;
use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::{Shapes1, Shapes2};
use crate::utils::limits::MAX_NUM_WORKGROUPS;
use crate::utils::trig::stable_tanh;

const WORKGROUP_SIZE: u32 = 64;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

// // GELU constants
const GELU_COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;
const GELU_QUICK_COEF: f32 = -1.702;

// Unary operations without arguments

#[inline]
fn abs_op_fn(x: f32) -> f32 {
    x.abs()
}

#[inline]
fn sgn_op_fn(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

#[inline]
fn neg_op_fn(x: f32) -> f32 {
    -x
}

#[inline]
fn step_op_fn(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[inline]
fn elu_op_fn(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        x.exp() - 1.0
    }
}

#[inline]
fn gelu_op_fn(x: f32) -> f32 {
    0.5 * x * (1.0 + stable_tanh(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)))
}

#[inline]
fn gelu_quick_op_fn(x: f32) -> f32 {
    x * (1.0 / (1.0 + (GELU_QUICK_COEF * x).exp()))
}

#[inline]
fn silu_op_fn(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn tanh_op_fn(x: f32) -> f32 {
    stable_tanh(x)
}

#[inline]
fn relu_op_fn(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn sigmoid_op_fn(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn hard_sigmoid_op_fn(x: f32) -> f32 {
    1.0f32.min(0.0f32.max((x + 3.0) / 6.0))
}

#[inline]
fn hard_swish_op_fn(x: f32) -> f32 {
    x * 1.0f32.min(0.0f32.max((x + 3.0) / 6.0))
}

#[inline]
fn sqr_op_fn(x: f32) -> f32 {
    x * x
}

#[inline]
fn sqrt_op_fn(x: f32) -> f32 {
    x.sqrt()
}

#[inline]
fn sin_op_fn(x: f32) -> f32 {
    x.sin()
}

#[inline]
fn cos_op_fn(x: f32) -> f32 {
    x.cos()
}

#[inline]
fn log_op_fn(x: f32) -> f32 {
    x.ln()
}

#[inline]
fn exp_op_fn(x: f32) -> f32 {
    x.exp()
}

#[inline]
fn reciprocal_op_fn(x: f32) -> f32 {
    1.0 / x
}

/// Erf (error function) approximation using Abramowitz and Stegun formula.
#[inline]
#[allow(clippy::excessive_precision)]
fn erf_op_fn(x: f32) -> f32 {
    let a1: f32 = 0.254829592;
    let a2: f32 = -0.284496736;
    let a3: f32 = 1.421413741;
    let a4: f32 = -1.453152027;
    let a5: f32 = 1.061405429;
    let p: f32 = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// Unary operations with arguments

#[inline]
fn leaky_relu_op_fn(x: f32, args: Vec4) -> f32 {
    x.max(0.0) + x.min(0.0) * args.x
}

#[inline]
fn clamp_op_fn(x: f32, args: Vec4) -> f32 {
    args.x.max(x).min(args.y)
}

#[inline]
fn scale_op_fn(x: f32, args: Vec4) -> f32 {
    x * args.x
}

#[inline]
fn add_scalar_op_fn(x: f32, args: Vec4) -> f32 {
    x + args.x
}

#[inline]
fn pow_op_fn(x: f32, args: Vec4) -> f32 {
    x.powf(args.x)
}

// Macro-like helper for generating shader entry points
// Since we can't use actual macros in no_std easily, we'll define each manually

/// Abs operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn abs_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = abs_op_fn(*src.at(isrc));
    }
}

/// Abs operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn abs_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = abs_op_fn(*src.at(isrc));
    }
}

/// Sign operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sgn_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = sgn_op_fn(*src.at(isrc));
    }
}

/// Sign operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sgn_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = sgn_op_fn(*src.at(isrc));
    }
}

/// Negation operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn neg_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = neg_op_fn(*src.at(isrc));
    }
}

/// Negation operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn neg_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = neg_op_fn(*src.at(isrc));
    }
}

/// Step operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn step_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = step_op_fn(*src.at(isrc));
    }
}

/// Step operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn step_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = step_op_fn(*src.at(isrc));
    }
}

/// ELU operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn elu_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = elu_op_fn(*src.at(isrc));
    }
}

/// ELU operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn elu_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = elu_op_fn(*src.at(isrc));
    }
}

/// GELU operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gelu_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = gelu_op_fn(*src.at(isrc));
    }
}

/// GELU operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gelu_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = gelu_op_fn(*src.at(isrc));
    }
}

/// GELU Quick operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gelu_quick_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = gelu_quick_op_fn(*src.at(isrc));
    }
}

/// GELU Quick operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gelu_quick_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = gelu_quick_op_fn(*src.at(isrc));
    }
}

/// SiLU operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn silu_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = silu_op_fn(*src.at(isrc));
    }
}

/// SiLU operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn silu_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = silu_op_fn(*src.at(isrc));
    }
}

/// Tanh operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn tanh_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = tanh_op_fn(*src.at(isrc));
    }
}

/// Tanh operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn tanh_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = tanh_op_fn(*src.at(isrc));
    }
}

/// ReLU operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn relu_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = relu_op_fn(*src.at(isrc));
    }
}

/// ReLU operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn relu_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = relu_op_fn(*src.at(isrc));
    }
}

/// Sigmoid operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sigmoid_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = sigmoid_op_fn(*src.at(isrc));
    }
}

/// Sigmoid operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sigmoid_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = sigmoid_op_fn(*src.at(isrc));
    }
}

/// Hard sigmoid operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn hard_sigmoid_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = hard_sigmoid_op_fn(*src.at(isrc));
    }
}

/// Hard sigmoid operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn hard_sigmoid_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = hard_sigmoid_op_fn(*src.at(isrc));
    }
}

/// Square operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sqr_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = sqr_op_fn(*src.at(isrc));
    }
}

/// Square operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sqr_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = sqr_op_fn(*src.at(isrc));
    }
}

/// Square root operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sqrt_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = sqrt_op_fn(*src.at(isrc));
    }
}

/// Square root operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sqrt_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = sqrt_op_fn(*src.at(isrc));
    }
}

/// Sine operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sin_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = sin_op_fn(*src.at(isrc));
    }
}

/// Sine operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn sin_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = sin_op_fn(*src.at(isrc));
    }
}

/// Cosine operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn cos_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = cos_op_fn(*src.at(isrc));
    }
}

/// Cosine operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn cos_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = cos_op_fn(*src.at(isrc));
    }
}

/// Log operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn log_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = log_op_fn(*src.at(isrc));
    }
}

/// Log operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn log_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = log_op_fn(*src.at(isrc));
    }
}

/// Exp operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn exp_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = exp_op_fn(*src.at(isrc));
    }
}

/// Exp operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn exp_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = exp_op_fn(*src.at(isrc));
    }
}

/// Reciprocal operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn reciprocal_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = reciprocal_op_fn(*src.at(isrc));
    }
}

/// Reciprocal operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn reciprocal_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = reciprocal_op_fn(*src.at(isrc));
    }
}

// Operations with arguments

/// Leaky ReLU operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn leaky_relu_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = leaky_relu_op_fn(*src.at(isrc), args);
    }
}

/// Leaky ReLU operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn leaky_relu_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = leaky_relu_op_fn(*src.at(isrc), args);
    }
}

/// Clamp operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn clamp_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = clamp_op_fn(*src.at(isrc), args);
    }
}

/// Clamp operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn clamp_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = clamp_op_fn(*src.at(isrc), args);
    }
}

/// Scale operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn scale_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = scale_op_fn(*src.at(isrc), args);
    }
}

/// Scale operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn scale_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = scale_op_fn(*src.at(isrc), args);
    }
}

/// Add scalar operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn add_scalar_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = add_scalar_op_fn(*src.at(isrc), args);
    }
}

/// Add scalar operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn add_scalar_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = add_scalar_op_fn(*src.at(isrc), args);
    }
}

/// Erf (error function) operation.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn erf_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = erf_op_fn(*src.at(isrc));
    }
}

/// Erf (error function) operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn erf_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = erf_op_fn(*src.at(isrc));
    }
}

/// Pow operation (x raised to power in args.x).
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn pow_op(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dst: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src, shape_dst) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dst = *shape_dst.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        let idst = shape_dst.it_vec(id) as usize;
        *dst.at_mut(idst) = pow_op_fn(*src.at(isrc), args);
    }
}

/// Pow operation inplace.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn pow_inplace(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] args: &[Vec4],
) {
    #[cfg(feature = "push_constants")]
    let shape_src = shapes.shape;
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);
    let args = *args.at(0);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it_vec(id) as usize;
        *src.at_mut(isrc) = pow_op_fn(*src.at(isrc), args);
    }
}
