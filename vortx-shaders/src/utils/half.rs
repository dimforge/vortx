//! Half-precision floating point utilities.

use khal_std::glamx::{IVec4, UVec4, Vec2};

/// Unpack a u32 containing two f16 values into a Vec2 of f32.
/// The low 16 bits are the first half, high 16 bits are the second half.
#[inline]
pub fn unpack_half2x16(v: u32) -> Vec2 {
    khal_std::float::unpack_half2x16(v)
}

/// Unpack a u32 containing 4 signed 8-bit integers into an IVec4.
/// Extracts bytes and sign-extends them to i32.
#[inline]
pub fn unpack_int4x8(v: u32) -> IVec4 {
    // Extract each byte and sign-extend from i8 to i32
    let b0 = ((v & 0xFF) as i32) << 24 >> 24;
    let b1 = (((v >> 8) & 0xFF) as i32) << 24 >> 24;
    let b2 = (((v >> 16) & 0xFF) as i32) << 24 >> 24;
    let b3 = (((v >> 24) & 0xFF) as i32) << 24 >> 24;
    IVec4::new(b0, b1, b2, b3)
}

/// Unpack a u32 containing 4 unsigned 8-bit integers into a UVec4.
#[inline]
pub fn unpack_uint4x8(v: u32) -> UVec4 {
    let b0 = v & 0xFF;
    let b1 = (v >> 8) & 0xFF;
    let b2 = (v >> 16) & 0xFF;
    let b3 = (v >> 24) & 0xFF;
    UVec4::new(b0, b1, b2, b3)
}
