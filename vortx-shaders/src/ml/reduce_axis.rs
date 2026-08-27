//! Axis-based reduction operations (ReduceSum, ReduceMean, etc.)

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes2;
use crate::utils::limits::MAX_NUM_WORKGROUPS;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

const WORKGROUP_SIZE: u32 = 64;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Reduce sum along a single axis.
///
/// Each thread handles one element in the output. It iterates over all
/// elements along the reduce axis in the input and sums them.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn reduce_sum_axis(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_dest: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dest: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] src: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] params: &[u32], // [axis, reduce_size]
) {
    #[cfg(feature = "push_constants")]
    let (shape_dest, shape_src) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    let axis = *params.at(0);
    let reduce_size = *params.at(1);

    for thread_id in (invocation_id.x..shape_dest.len()).step_by(MAX_NUM_THREADS as usize) {
        // Decompose linear index in output
        let id_dest = shape_dest.decompose(thread_id);

        // Build the source coordinates, starting from the output coords.
        let mut id_src = id_dest;

        // Sum over all elements along the reduce axis
        let mut sum: f32 = 0.0;
        for i in 0..reduce_size {
            match axis {
                0 => id_src.x = i,
                1 => id_src.y = i,
                2 => id_src.z = i,
                _ => id_src.w = i,
            }
            let i_src = shape_src.it_vec(id_src) as usize;
            sum += *src.at(i_src);
        }

        let i_dest = shape_dest.it_vec(id_dest) as usize;
        *dest.at_mut(i_dest) = sum;
    }
}

/// Reduce mean along a single axis.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn reduce_mean_axis(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_dest: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dest: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] src: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] params: &[u32], // [axis, reduce_size]
) {
    #[cfg(feature = "push_constants")]
    let (shape_dest, shape_src) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    let axis = *params.at(0);
    let reduce_size = *params.at(1);

    for thread_id in (invocation_id.x..shape_dest.len()).step_by(MAX_NUM_THREADS as usize) {
        // Decompose linear index in output
        let id_dest = shape_dest.decompose(thread_id);

        // Build the source coordinates, starting from the output coords.
        let mut id_src = id_dest;

        // Sum over all elements along the reduce axis
        let mut sum: f32 = 0.0;
        for i in 0..reduce_size {
            match axis {
                0 => id_src.x = i,
                1 => id_src.y = i,
                2 => id_src.z = i,
                _ => id_src.w = i,
            }
            let i_src = shape_src.it_vec(id_src) as usize;
            sum += *src.at(i_src);
        }

        let i_dest = shape_dest.it_vec(id_dest) as usize;
        *dest.at_mut(i_dest) = sum / (reduce_size as f32);
    }
}

/// Reduce max along a single axis.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn reduce_max_axis(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_dest: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dest: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] src: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] params: &[u32], // [axis, reduce_size]
) {
    #[cfg(feature = "push_constants")]
    let (shape_dest, shape_src) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    let axis = *params.at(0);
    let reduce_size = *params.at(1);

    for thread_id in (invocation_id.x..shape_dest.len()).step_by(MAX_NUM_THREADS as usize) {
        // Decompose linear index in output
        let id_dest = shape_dest.decompose(thread_id);

        // Build the source coordinates, starting from the output coords.
        let mut id_src = id_dest;

        // Find max over all elements along the reduce axis
        id_src.x = 0;
        id_src.y = 0;
        id_src.z = 0;
        id_src.w = 0;
        match axis {
            0 => id_src.x = 0,
            1 => id_src.y = 0,
            2 => id_src.z = 0,
            _ => id_src.w = 0,
        }
        // Restore non-axis coordinates from dest
        match axis {
            0 => {
                id_src.y = id_dest.y;
                id_src.z = id_dest.z;
                id_src.w = id_dest.w;
            }
            1 => {
                id_src.x = id_dest.x;
                id_src.z = id_dest.z;
                id_src.w = id_dest.w;
            }
            2 => {
                id_src.x = id_dest.x;
                id_src.y = id_dest.y;
                id_src.w = id_dest.w;
            }
            _ => {
                id_src.x = id_dest.x;
                id_src.y = id_dest.y;
                id_src.z = id_dest.z;
            }
        }

        let i_src_init = shape_src.it_vec(id_src) as usize;
        let mut max_val = *src.at(i_src_init);

        for i in 1..reduce_size {
            match axis {
                0 => id_src.x = i,
                1 => id_src.y = i,
                2 => id_src.z = i,
                _ => id_src.w = i,
            }
            let i_src = shape_src.it_vec(id_src) as usize;
            let val = *src.at(i_src);
            if val > max_val {
                max_val = val;
            }
        }

        let i_dest = shape_dest.it_vec(id_dest) as usize;
        *dest.at_mut(i_dest) = max_val;
    }
}

/// Reduce min along a single axis.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn reduce_min_axis(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_dest: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_src: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dest: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] src: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] params: &[u32], // [axis, reduce_size]
) {
    #[cfg(feature = "push_constants")]
    let (shape_dest, shape_src) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    let axis = *params.at(0);
    let reduce_size = *params.at(1);

    for thread_id in (invocation_id.x..shape_dest.len()).step_by(MAX_NUM_THREADS as usize) {
        // Decompose linear index in output
        let id_dest = shape_dest.decompose(thread_id);

        // Build source coordinates
        let mut id_src = id_dest;
        match axis {
            0 => {
                id_src.y = id_dest.y;
                id_src.z = id_dest.z;
                id_src.w = id_dest.w;
                id_src.x = 0;
            }
            1 => {
                id_src.x = id_dest.x;
                id_src.z = id_dest.z;
                id_src.w = id_dest.w;
                id_src.y = 0;
            }
            2 => {
                id_src.x = id_dest.x;
                id_src.y = id_dest.y;
                id_src.w = id_dest.w;
                id_src.z = 0;
            }
            _ => {
                id_src.x = id_dest.x;
                id_src.y = id_dest.y;
                id_src.z = id_dest.z;
                id_src.w = 0;
            }
        }

        let i_src_init = shape_src.it_vec(id_src) as usize;
        let mut min_val = *src.at(i_src_init);

        for i in 1..reduce_size {
            match axis {
                0 => id_src.x = i,
                1 => id_src.y = i,
                2 => id_src.z = i,
                _ => id_src.w = i,
            }
            let i_src = shape_src.it_vec(id_src) as usize;
            let val = *src.at(i_src);
            if val < min_val {
                min_val = val;
            }
        }

        let i_dest = shape_dest.it_vec(id_dest) as usize;
        *dest.at_mut(i_dest) = min_val;
    }
}
