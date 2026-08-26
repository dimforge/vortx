//! Concat operation: concatenates tensors along a given axis.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes2;
use crate::utils::limits::MAX_NUM_WORKGROUPS;

const WORKGROUP_SIZE: u32 = 64;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Copy a source tensor into a slice of the destination tensor along a given axis.
///
/// This is used to implement concat by calling it once per input tensor,
/// with offset indicating where this tensor's data should be placed in the output.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn concat_copy(
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] params: &[u32], // [axis, offset_along_axis]
) {
    #[cfg(feature = "push_constants")]
    let (shape_dest, shape_src) = (shapes.shape_a, shapes.shape_b);
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    let axis = *params.at(0);
    let offset = *params.at(1);

    for thread_id in (invocation_id.x..shape_src.len()).step_by(MAX_NUM_THREADS as usize) {
        // Decompose linear index in source
        let id_src = shape_src.decompose(thread_id);

        // Build destination coordinates by adding offset along the axis
        let mut id_dest = id_src;
        match axis {
            0 => id_dest.x += offset,
            1 => id_dest.y += offset,
            2 => id_dest.z += offset,
            _ => id_dest.w += offset,
        }

        let i_src = shape_src.it_vec(id_src) as usize;
        let i_dest = shape_dest.it_vec(id_dest) as usize;

        *dest.at_mut(i_dest) = *src.at(i_src);
    }
}
