//! Relative position computation.

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::{Shapes1, Shapes2};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

const WORKGROUP_SIZE: u32 = 128;

/// Get relative position.
#[spirv_bindgen]
#[spirv(compute(threads(128, 1, 1)))]
pub fn get_rel_pos(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_result: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_source: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] result: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_result, shape_source) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_result = *shape_result.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_source = *shape_source.at(0);

    if invocation_id.x >= shape_result.len() {
        return;
    }

    let id = shape_result.decompose(invocation_id.x);
    let i = shape_result.it_vec(id) as usize;
    let w = shape_result.h;
    let pos = (w - id.x - 1) + id.z;
    let j = shape_source.it(0, 0, pos, id.y) as usize;

    *result.at_mut(i) = *source.at(j);
}

/// Add relative position phase 2.
#[spirv_bindgen]
#[spirv(compute(threads(128, 1, 1)))]
pub fn add_rel_pos_phase_b(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src1: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dst: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] src2: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src1 = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src1 = *shape_src1.at(0);

    if invocation_id.x >= shape_src1.len() {
        return;
    }

    let id = shape_src1.decompose(invocation_id.x);
    let jp0 = shape_src1.it_vec(id);

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359
    let src2_e = *src2.at(jp0 as usize);
    let ne10 = shape_src1.w;

    let jdh = jp0 * ne10;

    for j in 0..ne10 {
        *dst.at_mut((jdh + j) as usize) = *dst.at((jdh + j) as usize) + src2_e;
    }
}

/// Add relative position phase 1.
#[spirv_bindgen]
#[spirv(compute(threads(128, 1, 1)))]
pub fn add_rel_pos_phase_a(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src1: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dst: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] src1: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src1 = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src1 = *shape_src1.at(0);

    if invocation_id.x >= shape_src1.len() {
        return;
    }

    let id = shape_src1.decompose(invocation_id.x);
    let jp0 = shape_src1.it_vec(id);

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359
    let src1_e = *src1.at(jp0 as usize);
    let ne10 = shape_src1.w;

    let jdh = jp0 * ne10;
    let jdw = jdh - (ne10 - 1) * id.y;

    for j in 0..ne10 {
        *dst.at_mut((jdw + j * ne10) as usize) = *dst.at((jdw + j * ne10) as usize) + src1_e;
    }
}
