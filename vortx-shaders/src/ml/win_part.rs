//! Window partitioning.

use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes2;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

const WORKGROUP_SIZE: u32 = 128;

/// Window partition.
/// source: [R1, C2, M1, T1]
/// result: [R1, W, W, _]
#[spirv_bindgen]
#[spirv(compute(threads(128, 1, 1)))]
pub fn win_part(
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

    let w = shape_result.c;
    let pad_x = (w - shape_source.h % w) % w;
    // NOTE: notation nep0 === npx
    let nep0 = (pad_x + shape_source.h) / w;

    // NOTE: id[3] spans [0..nep0*nep1[ by definition of the result tensor.
    let py = id.w / nep0;
    let px = id.w - py * nep0;
    let i02 = py * w + id.z;
    let i01 = px * w + id.x;
    let i00 = id.y;

    if py * w + id.z >= shape_source.c || px * w + id.x >= shape_source.h {
        *result.at_mut(i) = 0.0;
    } else {
        let j = shape_source.it(0, i02, i01, i00) as usize;
        *result.at_mut(i) = *source.at(j);
    }
}

/// Window unpartition.
/// source: [R1, C2, M1, T1]
/// result: [R1, W, W, _]
#[spirv_bindgen]
#[spirv(compute(threads(128, 1, 1)))]
pub fn win_unpart(
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] w: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] result: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] source: &[f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_result, shape_source) = (shapes.shape_a, shapes.shape_b);
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_result = *shape_result.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_source = *shape_source.at(0);
    let w = *w.at(0);

    if invocation_id.x >= shape_result.len() {
        return;
    }

    let id = shape_result.decompose(invocation_id.x);
    let j = shape_result.it_vec(id) as usize;

    let px = (w - shape_result.h % w) % w;
    let npx = (px + shape_result.h) / w;
    let ip2 = id.z / w;
    let ip1 = id.x / w;
    let i = shape_source.it(ip2 * npx + ip1, id.z % w, id.x % w, id.y) as usize;

    *result.at_mut(j) = *source.at(i);
}
