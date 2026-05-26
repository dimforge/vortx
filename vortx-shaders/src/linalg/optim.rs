//! Optimizer kernels (Adam). Added for zealot; vortx upstream has no optimizers.

use super::shape::Shape;
use crate::utils::limits::MAX_NUM_WORKGROUPS;
use glamx::UVec3;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;

const WORKGROUP_SIZE: u32 = 256;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Scalar parameters for one Adam step (uniform buffer; padded to 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct AdamParams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// `1 - beta1^t` (bias correction for the first moment).
    pub bias_correction1: f32,
    /// `1 - beta2^t` (bias correction for the second moment).
    pub bias_correction2: f32,
    pub pad0: f32,
    pub pad1: f32,
}

/// One in-place Adam step: updates first/second moments `m`, `v` and parameters
/// `theta` from the gradient `grad`. All buffers share `theta`'s shape.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_adam(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] shape: &Shape,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] params: &AdamParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] theta: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grad: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] m: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] v: &mut [f32],
) {
    for thread_id in (invocation_id.x..shape.len()).step_by(MAX_NUM_THREADS as usize) {
        let id = shape.decompose(thread_id);
        let i = shape.it_vec(id) as usize;
        let g = grad.read(i);
        let m_old = *m.at_mut(i);
        let v_old = *v.at_mut(i);
        let mi = params.beta1 * m_old + (1.0 - params.beta1) * g;
        let vi = params.beta2 * v_old + (1.0 - params.beta2) * g * g;
        *m.at_mut(i) = mi;
        *v.at_mut(i) = vi;
        let mhat = mi / params.bias_correction1;
        let vhat = vi / params.bias_correction2;
        *theta.at_mut(i) -= params.lr * mhat / (vhat.sqrt() + params.eps);
    }
}
