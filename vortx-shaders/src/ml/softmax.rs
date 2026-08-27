//! Softmax and log-softmax kernels.

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes1;
use crate::utils::iterators::StepRng;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;

#[cfg(feature = "subgroup_ops")]
const WORKGROUP_SIZE: usize = 32;
#[cfg(not(feature = "subgroup_ops"))]
const WORKGROUP_SIZE: usize = 64;

#[inline]
fn reduce_max(index: usize, stride: usize, workspace: &mut [f32; WORKGROUP_SIZE]) {
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
    if index < stride {
        *workspace.at_mut(index) = workspace.at(index).max(*workspace.at(index + stride));
    }
}

#[inline]
fn reduce_sum(index: usize, stride: usize, workspace: &mut [f32; WORKGROUP_SIZE]) {
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
    if index < stride {
        *workspace.at_mut(index) = *workspace.at(index) + *workspace.at(index + stride);
    }
}

/// Softmax on columns.
#[spirv_bindgen]
#[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
#[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(64, 1, 1))))]
pub fn softmax(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup)] workspace: &mut [f32; WORKGROUP_SIZE],
    #[spirv(workgroup)] the_max: &mut f32,
    #[spirv(workgroup)] denominator: &mut f32,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_out_mat: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape = *shape.at(0);

    let j = workgroup_id.x;
    let k = workgroup_id.y;
    let l = workgroup_id.z;
    let thread_id = local_id.x as usize;

    // Compute the max.
    let data_len = shape.w;
    let mut my_max = [-1.0e38f32];

    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let val_i = *in_out_mat.at(shape.it(l, k, j, i) as usize);
        my_max[0] = my_max[0].max(val_i);
    }

    #[cfg(not(feature = "subgroup_ops"))]
    {
        *workspace.at_mut(thread_id) = my_max[0];
    }

    #[cfg(feature = "subgroup_ops")]
    let max_val = khal_std::sync::subgroup_f_max(my_max[0]);

    #[cfg(not(feature = "subgroup_ops"))]
    {
        reduce_max(thread_id, 32, workspace);
        reduce_max(thread_id, 16, workspace);
        reduce_max(thread_id, 8, workspace);
        reduce_max(thread_id, 4, workspace);
        reduce_max(thread_id, 2, workspace);
        reduce_max(thread_id, 1, workspace);
    }

    if thread_id == 0 {
        #[cfg(feature = "subgroup_ops")]
        {
            *the_max = max_val;
        }
        #[cfg(not(feature = "subgroup_ops"))]
        {
            *the_max = *workspace.at(0);
        }
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // Compute the denominator (sum of exponential).
    let mut my_denominator = [0.0f32];
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let ii = shape.it(l, k, j, i) as usize;
        let val_i = *in_out_mat.at(ii);
        let centered_val = val_i - *the_max;
        let exp_i = centered_val.exp();
        my_denominator[0] += exp_i;
        *in_out_mat.at_mut(ii) = exp_i;
    }

    #[cfg(not(feature = "subgroup_ops"))]
    {
        *workspace.at_mut(thread_id) = my_denominator[0];
    }

    #[cfg(feature = "subgroup_ops")]
    let sum = khal_std::sync::subgroup_f_add(my_denominator[0]);

    #[cfg(not(feature = "subgroup_ops"))]
    {
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
            *denominator = sum;
        }
        #[cfg(not(feature = "subgroup_ops"))]
        {
            *denominator = *workspace.at(0);
        }
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // Divide by the denominator.
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let ii = shape.it(l, k, j, i) as usize;
        let val_i = *in_out_mat.at(ii);
        *in_out_mat.at_mut(ii) = val_i / *denominator;
    }
}

/// Log-softmax on columns.
#[spirv_bindgen]
#[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
#[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(64, 1, 1))))]
pub fn log_softmax(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup)] workspace: &mut [f32; WORKGROUP_SIZE],
    #[spirv(workgroup)] the_max: &mut f32,
    #[spirv(workgroup)] denominator: &mut f32,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_out_mat: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape = *shape.at(0);

    let j = workgroup_id.x;
    let k = workgroup_id.y;
    let l = workgroup_id.z;
    let thread_id = local_id.x as usize;

    // Compute the max.
    let data_len = shape.w;
    let mut my_max = [-1.0e38f32];

    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let val_i = *in_out_mat.at(shape.it(l, k, j, i) as usize);
        my_max[0] = my_max[0].max(val_i);
    }

    *workspace.at_mut(thread_id) = my_max[0];

    #[cfg(feature = "subgroup_ops")]
    let max_val = khal_std::sync::subgroup_f_max(my_max[0]);

    #[cfg(not(feature = "subgroup_ops"))]
    {
        reduce_max(thread_id, 32, workspace);
        reduce_max(thread_id, 16, workspace);
        reduce_max(thread_id, 8, workspace);
        reduce_max(thread_id, 4, workspace);
        reduce_max(thread_id, 2, workspace);
        reduce_max(thread_id, 1, workspace);
    }

    if thread_id == 0 {
        #[cfg(feature = "subgroup_ops")]
        {
            *the_max = max_val;
        }
        #[cfg(not(feature = "subgroup_ops"))]
        {
            *the_max = *workspace.at(0);
        }
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // Compute the denominator (sum of exponentials).
    let mut my_denominator = [0.0f32];
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let ii = shape.it(l, k, j, i) as usize;
        let val_i = *in_out_mat.at(ii);
        let centered_val_i = val_i - *the_max;
        let exp_i = centered_val_i.exp();
        my_denominator[0] += exp_i;
        *in_out_mat.at_mut(ii) = centered_val_i;
    }

    *workspace.at_mut(thread_id) = my_denominator[0];

    #[cfg(feature = "subgroup_ops")]
    let sum = khal_std::sync::subgroup_f_add(my_denominator[0]);

    #[cfg(not(feature = "subgroup_ops"))]
    {
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
            *denominator = sum;
        }
        #[cfg(not(feature = "subgroup_ops"))]
        {
            *denominator = *workspace.at(0);
        }
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // Subtract log(denominator).
    for i in StepRng::new(thread_id as u32..data_len, WORKGROUP_SIZE as u32) {
        let ii = shape.it(l, k, j, i) as usize;
        let val_i = *in_out_mat.at(ii);
        *in_out_mat.at_mut(ii) = val_i - *denominator;
    }
}
