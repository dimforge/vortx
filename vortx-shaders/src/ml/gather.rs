//! Gather operation: gathers elements from source tensor based on indices along a given axis.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes2;
use crate::utils::limits::MAX_NUM_WORKGROUPS;

const WORKGROUP_SIZE: u32 = 64;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Gather elements from source based on indices along axis.
///
/// For axis=0:
///   output\[i, j, k\] = input\[indices\[i\], j, k\]
/// For axis=1:
///   output\[i, j, k\] = input\[i, indices\[j\], k\]
/// etc.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gather(
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] indices: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] axis_buf: &[u32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_dest, shape_src) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    let axis = *axis_buf.at(0);

    for thread_id in (invocation_id.x..shape_dest.len()).step_by(MAX_NUM_THREADS as usize) {
        // Decompose linear index to 4D coordinates in output
        let id_dest = shape_dest.decompose(thread_id);

        // Get the coordinate along the gather axis
        let idx_coord = match axis {
            0 => id_dest.x,
            1 => id_dest.y,
            2 => id_dest.z,
            _ => id_dest.w,
        };

        // Look up the index value
        let gathered_idx = *indices.at(idx_coord as usize);

        // Build source coordinates by replacing the axis coordinate with the gathered index
        let mut id_src = id_dest;
        match axis {
            0 => id_src.x = gathered_idx as u32,
            1 => id_src.y = gathered_idx as u32,
            2 => id_src.z = gathered_idx as u32,
            _ => id_src.w = gathered_idx as u32,
        }

        let i_dest = shape_dest.it_vec(id_dest) as usize;
        let i_src = shape_src.it_vec(id_src) as usize;

        *dest.at_mut(i_dest) = *src.at(i_src);
    }
}
