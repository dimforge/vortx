//! Select operation: selects elements from a source tensor based on indices.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes2;

/// Select elements from source based on indices and write to destination.
///
/// For each element in dest at position (i, j, k, l), this reads the index
/// from idx\[i\] and then copies src\[idx\[i\], j, k, l\] to dest\[i, j, k, l\].
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn select(
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] idx: &[u32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_dest, shape_src) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src = *shape_src.at(0);

    let thread_id = invocation_id.x;
    if thread_id < shape_dest.len() {
        // Decompose linear index to 4D coordinates
        let id = shape_dest.decompose(thread_id);

        // Compute destination index
        let i_dest = shape_dest.it_vec(id) as usize;

        // Replace x coordinate with the index lookup
        let mut id_src = id;
        id_src.x = *idx.at(id.x as usize);

        // Compute source index with wrapping
        let i_src = shape_src.it_repeating_vec(id_src) as usize;

        *dest.at_mut(i_dest) = *src.at(i_src);
    }
}
