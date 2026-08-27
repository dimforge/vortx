//! RMS normalization kernel.

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes3;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;

#[cfg(feature = "subgroup_ops")]
const WORKGROUP_SIZE: usize = 32;
#[cfg(not(feature = "subgroup_ops"))]
const WORKGROUP_SIZE: usize = 128;

/// RMS normalization configuration.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RmsNormConfig {
    pub nudge_factor: f32,
}

#[inline]
fn reduce_sum(index: usize, stride: usize, workspace: &mut [f32; WORKGROUP_SIZE]) {
    if index < stride {
        *workspace.at_mut(index) += *workspace.at(index + stride);
    }
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
}

fn magnitude_squared(
    thread_id: u32,
    shape_v: Shape,
    v: &[f32],
    workspace: &mut [f32; WORKGROUP_SIZE],
) -> f32 {
    let thread_id_usize = thread_id as usize;
    *workspace.at_mut(thread_id_usize) = 0.0;

    let mut i = thread_id;
    while i < shape_v.w {
        let val_i = v.at(shape_v.it(0, 0, 0, i) as usize);
        *workspace.at_mut(thread_id_usize) += val_i * val_i;
        i += WORKGROUP_SIZE as u32;
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    #[cfg(feature = "subgroup_ops")]
    let sum = khal_std::sync::subgroup_f_add(*workspace.at(thread_id_usize));

    #[cfg(not(feature = "subgroup_ops"))]
    {
        reduce_sum(thread_id_usize, 64, workspace);
        reduce_sum(thread_id_usize, 32, workspace);
        reduce_sum(thread_id_usize, 16, workspace);
        reduce_sum(thread_id_usize, 8, workspace);
        reduce_sum(thread_id_usize, 4, workspace);
        reduce_sum(thread_id_usize, 2, workspace);
        reduce_sum(thread_id_usize, 1, workspace);
    }

    #[cfg(feature = "subgroup_ops")]
    if thread_id_usize == 0 {
        *workspace.at_mut(0) = sum;
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();
    *workspace.at(0)
}

/// RMS normalization.
#[spirv_bindgen]
#[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
#[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
pub fn rms_norm(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(workgroup)] workspace: &mut [f32; WORKGROUP_SIZE],
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes3,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_v: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_w: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_out: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] v: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] w: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] out: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] config: &[RmsNormConfig],
) {
    #[cfg(feature = "push_constants")]
    let (shape_v, shape_w, shape_out) = (shapes.shape_out, shapes.shape_lhs, shapes.shape_rhs);
    // Load shapes and config from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_v = *shape_v.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_w = *shape_w.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_out = *shape_out.at(0);
    let config = *config.at(0);

    let magnitude_sq = magnitude_squared(invocation_id.x, shape_v, v, workspace);

    let len = shape_v.w;
    let rms = 1.0 / ((magnitude_sq / len as f32) + config.nudge_factor).sqrt();

    for i in (invocation_id.x..len).step_by(WORKGROUP_SIZE) {
        *out.at_mut(shape_out.it(0, 0, 0, i) as usize) =
            (*v.at(shape_v.it(0, 0, 0, i) as usize) * rms) * *w.at(shape_w.it(0, 0, 0, i) as usize);
    }
}
