//! Layer normalization kernels.

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes2;
use crate::utils::iterators::StepRng;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;

#[cfg(feature = "subgroup_ops")]
const WORKGROUP_SIZE: usize = 32;
#[cfg(not(feature = "subgroup_ops"))]
const WORKGROUP_SIZE: usize = 128;
const NUDGE_FACTOR: f32 = 1.0e-6;

#[inline]
fn reduce_sum(index: usize, stride: usize, workspace: &mut [f32; WORKGROUP_SIZE]) {
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
    if index < stride {
        let val = *workspace.at(index + stride);
        *workspace.at_mut(index) += val;
    }
}

/// Layer normalization on columns.
#[spirv_bindgen]
#[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
#[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
pub fn layernorm_cols(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup)] workspace: &mut [f32; WORKGROUP_SIZE],
    #[spirv(workgroup)] the_mean: &mut f32,
    #[spirv(workgroup)] scale: &mut f32,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    in_shape: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    out_shape: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] output: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (in_shape, out_shape) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let in_shape = *in_shape.at(0);
    #[cfg(not(feature = "push_constants"))]
    let out_shape = *out_shape.at(0);

    let thread_id = local_id.x as usize;

    // Compute the MEAN
    let data_len = in_shape.h;
    *workspace.at_mut(thread_id) = 0.0;
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let val_i = *input.at(in_shape.it(wid.z, wid.y, i, wid.x) as usize);
        *workspace.at_mut(thread_id) = *workspace.at(thread_id) + val_i;
    }

    #[cfg(feature = "subgroup_ops")]
    let sum = khal_std::sync::subgroup_f_add(*workspace.at(thread_id));

    #[cfg(not(feature = "subgroup_ops"))]
    {
        reduce_sum(thread_id, 64, workspace);
        reduce_sum(thread_id, 32, workspace);
        reduce_sum(thread_id, 16, workspace);
        reduce_sum(thread_id, 8, workspace);
        reduce_sum(thread_id, 4, workspace);
        reduce_sum(thread_id, 2, workspace);
        reduce_sum(thread_id, 1, workspace);
    }

    if thread_id == 0 {
        #[cfg(feature = "subgroup_ops")]
        {
            *the_mean = sum / data_len as f32;
        }
        #[cfg(not(feature = "subgroup_ops"))]
        {
            *the_mean = *workspace.at(0) / data_len as f32;
        }
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // Compute the SQUARED NORM
    *workspace.at_mut(thread_id) = 0.0;
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let val_i = *input.at(in_shape.it(wid.z, wid.y, i, wid.x) as usize) - *the_mean;
        *workspace.at_mut(thread_id) = *workspace.at(thread_id) + val_i * val_i;
    }

    #[cfg(feature = "subgroup_ops")]
    let sum = khal_std::sync::subgroup_f_add(*workspace.at(thread_id));

    #[cfg(not(feature = "subgroup_ops"))]
    {
        reduce_sum(thread_id, 64, workspace);
        reduce_sum(thread_id, 32, workspace);
        reduce_sum(thread_id, 16, workspace);
        reduce_sum(thread_id, 8, workspace);
        reduce_sum(thread_id, 4, workspace);
        reduce_sum(thread_id, 2, workspace);
        reduce_sum(thread_id, 1, workspace);
    }

    if thread_id == 0 {
        #[cfg(feature = "subgroup_ops")]
        let variance = sum / data_len as f32;
        #[cfg(not(feature = "subgroup_ops"))]
        let variance = *workspace.at(0) / data_len as f32;

        *scale = 1.0 / (variance + NUDGE_FACTOR).sqrt();
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // Apply the scale.
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let ii = in_shape.it(wid.z, wid.y, i, wid.x) as usize;
        let iout = out_shape.it(wid.z, wid.y, i, wid.x) as usize;
        *output.at_mut(iout) = (*input.at(ii) - *the_mean) * *scale;
    }
}

/// Layer normalization on rows.
#[spirv_bindgen]
#[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
#[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
pub fn layernorm_rows(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup)] workspace: &mut [f32; WORKGROUP_SIZE],
    #[spirv(workgroup)] the_mean: &mut f32,
    #[spirv(workgroup)] scale: &mut f32,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    in_shape: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    out_shape: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] output: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (in_shape, out_shape) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let in_shape = *in_shape.at(0);
    #[cfg(not(feature = "push_constants"))]
    let out_shape = *out_shape.at(0);

    let thread_id = local_id.x as usize;

    // Compute the MEAN
    let data_len = in_shape.w;
    *workspace.at_mut(thread_id) = 0.0;
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let val_i = *input.at(in_shape.it(wid.z, wid.y, wid.x, i) as usize);
        *workspace.at_mut(thread_id) = *workspace.at(thread_id) + val_i;
    }

    #[cfg(feature = "subgroup_ops")]
    let sum = khal_std::sync::subgroup_f_add(*workspace.at(thread_id));

    #[cfg(not(feature = "subgroup_ops"))]
    {
        reduce_sum(thread_id, 64, workspace);
        reduce_sum(thread_id, 32, workspace);
        reduce_sum(thread_id, 16, workspace);
        reduce_sum(thread_id, 8, workspace);
        reduce_sum(thread_id, 4, workspace);
        reduce_sum(thread_id, 2, workspace);
        reduce_sum(thread_id, 1, workspace);
    }

    if thread_id == 0 {
        #[cfg(feature = "subgroup_ops")]
        {
            *the_mean = sum / data_len as f32;
        }
        #[cfg(not(feature = "subgroup_ops"))]
        {
            *the_mean = *workspace.at(0) / data_len as f32;
        }
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // Compute the SQUARED NORM
    *workspace.at_mut(thread_id) = 0.0;
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let val_i = *input.at(in_shape.it(wid.z, wid.y, wid.x, i) as usize) - *the_mean;
        *workspace.at_mut(thread_id) = *workspace.at(thread_id) + val_i * val_i;
    }

    #[cfg(feature = "subgroup_ops")]
    let sum = khal_std::sync::subgroup_f_add(*workspace.at(thread_id));

    #[cfg(not(feature = "subgroup_ops"))]
    {
        reduce_sum(thread_id, 64, workspace);
        reduce_sum(thread_id, 32, workspace);
        reduce_sum(thread_id, 16, workspace);
        reduce_sum(thread_id, 8, workspace);
        reduce_sum(thread_id, 4, workspace);
        reduce_sum(thread_id, 2, workspace);
        reduce_sum(thread_id, 1, workspace);
    }

    if thread_id == 0 {
        #[cfg(feature = "subgroup_ops")]
        let variance = sum / data_len as f32;
        #[cfg(not(feature = "subgroup_ops"))]
        let variance = *workspace.at(0) / data_len as f32;

        *scale = 1.0 / (variance + NUDGE_FACTOR).sqrt();
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // Apply the scale.
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let ii = in_shape.it(wid.z, wid.y, wid.x, i) as usize;
        let iout = out_shape.it(wid.z, wid.y, wid.x, i) as usize;
        *output.at_mut(iout) = (*input.at(ii) - *the_mean) * *scale;
    }
}
