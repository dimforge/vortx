//! Transposed 2D convolution.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::{Shapes1, Shapes2, Shapes3};

const WORKGROUP_SIZE: u32 = 64;

/// Initialize destination buffer to zero.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn init_dest(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_dest: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dest: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_dest = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);

    if invocation_id.x >= shape_dest.len() {
        return;
    }

    *dest.at_mut(invocation_id.x as usize) = 0.0;
}

/// Initialize working data buffer to zero.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn init_wdata(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_wdata: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] wdata: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_wdata = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_wdata = *shape_wdata.at(0);

    if invocation_id.x >= shape_wdata.len() {
        return;
    }

    *wdata.at_mut(invocation_id.x as usize) = 0.0;
}

/// Initialize src0 (permute kernel data).
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn init_src_a(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes1,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src0: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src0: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wdata: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let shape_src0 = shapes.shape;
    // Load shape from storage buffer to local variable (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src0 = *shape_src0.at(0);

    if invocation_id.x >= shape_src0.len() {
        return;
    }

    // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
    let id = shape_src0.decompose(invocation_id.x);
    let id_wdata = id.z * shape_src0.h * shape_src0.w * shape_src0.n
        + id.x * shape_src0.w * shape_src0.n
        + id.y * shape_src0.n
        + id.w;
    *wdata.at_mut(id_wdata as usize) = *src0.at(invocation_id.x as usize);
}

/// Initialize src1 (permute source data).
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn init_src_b(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src0: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_src1: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] src1: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wdata: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src0, shape_src1) = (shapes.shape_a, shapes.shape_b);
    // Load shapes from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src0 = *shape_src0.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src1 = *shape_src1.at(0);

    if invocation_id.x >= shape_src1.len() {
        return;
    }

    // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
    let nk = shape_src0.len();
    let id = shape_src1.decompose(invocation_id.x);
    let id_wdata = nk + id.x * shape_src1.w * shape_src1.c + id.y * shape_src1.c + id.z;
    *wdata.at_mut(id_wdata as usize) = *src1.at(invocation_id.x as usize);
}

/// Reference implementation of transposed 2D convolution.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn conv_transpose_2d_ref(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes3,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src0: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_src1: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dest: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] stride: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] wdata: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dest: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src0, shape_src1, shape_dest) =
        (shapes.shape_out, shapes.shape_lhs, shapes.shape_rhs);
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src0 = *shape_src0.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src1 = *shape_src1.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);
    let stride = *stride.at(0);

    let i2 = invocation_id.x;

    if i2 >= shape_dest.c {
        return;
    }

    let nk = shape_src0.len();
    let ne0 = shape_dest.w;
    let nb2 = shape_dest.c_stride;

    let ne00 = shape_src0.w;
    let ne01 = shape_src0.h;
    let ne03 = shape_src0.n;
    let ne10 = shape_src1.w;
    let ne11 = shape_src1.h;
    let ne12 = shape_src1.c;

    for i11 in 0..ne11 as i32 {
        for i10 in 0..ne10 as i32 {
            let i1n = i11 * ne10 as i32 * ne12 as i32 + i10 * ne12 as i32;
            for i01 in 0..ne01 as i32 {
                for i00 in 0..ne00 as i32 {
                    let mut v = 0.0f32;

                    for k in 0..ne03 as i32 {
                        v += *wdata.at((nk as i32 + i1n + k) as usize)
                            * *wdata.at((i2 as i32 * ne01 as i32 * ne00 as i32 * ne03 as i32
                                + i01 * ne00 as i32 * ne03 as i32
                                + i00 * ne03 as i32
                                + k) as usize);
                    }
                    let dest_idx = (i2 * nb2
                        + (i11 * stride as i32 + i01) as u32 * ne0
                        + (i10 * stride as i32 + i00) as u32)
                        as usize;
                    *dest.at_mut(dest_idx) = *dest.at(dest_idx) + v;
                }
            }
        }
    }
}

/// Transposed 2D convolution.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn conv_transpose_2d(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes3,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_src1: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_src0: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    shape_dest: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] stride: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] src1: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] src0: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dest: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_src1, shape_src0, shape_dest) =
        (shapes.shape_out, shapes.shape_lhs, shapes.shape_rhs);
    // Load from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_src1 = *shape_src1.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_src0 = *shape_src0.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_dest = *shape_dest.at(0);
    let stride = *stride.at(0);

    let i2 = invocation_id.x;

    if i2 >= shape_dest.c {
        return;
    }

    for k in 0..(shape_dest.h * shape_dest.w) as i32 {
        *dest.at_mut((i2 * shape_dest.c + k as u32) as usize) = 0.0;
    }

    for i11 in 0..shape_src1.h as i32 {
        for i10 in 0..shape_src1.w as i32 {
            for i01 in 0..shape_src0.h as i32 {
                for i00 in 0..shape_src0.w as i32 {
                    let mut v = 0.0f32;

                    for k in 0..shape_src0.c as i32 {
                        v += *src1.at(shape_src1.it(0, i11 as u32, i10 as u32, k as u32) as usize)
                            * *src0
                                .at(shape_src0.it(i2, i01 as u32, i00 as u32, k as u32) as usize);
                    }

                    let dest_id = shape_dest.it(
                        0,
                        i2,
                        (i11 * stride as i32 + i01) as u32,
                        (i10 * stride as i32 + i00) as u32,
                    ) as usize;
                    *dest.at_mut(dest_id) = *dest.at(dest_id) + v;
                }
            }
        }
    }
}
