//! Matrix utility functions.

use glamx::{Mat2, Mat3, Mat4, Vec2, Vec3, Vec4};

/*
 * Trace of a matrix.
 */

/// The trace of a 2x2 matrix.
#[inline]
pub fn trace2(m: Mat2) -> f32 {
    m.col(0).x + m.col(1).y
}

/// The trace of a 3x3 matrix.
#[inline]
pub fn trace3(m: Mat3) -> f32 {
    m.col(0).x + m.col(1).y + m.col(2).z
}

/// The trace of a 4x4 matrix.
#[inline]
pub fn trace4(m: Mat4) -> f32 {
    m.col(0).x + m.col(1).y + m.col(2).z + m.col(3).w
}

/*
 * Diagonal extraction and diagonal matrix init.
 */

/// Initializes a diagonal 2x2 matrix.
#[inline]
pub fn diag2_from_vec(d: Vec2) -> Mat2 {
    Mat2::from_cols(Vec2::new(d.x, 0.0), Vec2::new(0.0, d.y))
}

/// Initializes a diagonal 3x3 matrix.
#[inline]
pub fn diag3_from_vec(d: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(d.x, 0.0, 0.0),
        Vec3::new(0.0, d.y, 0.0),
        Vec3::new(0.0, 0.0, d.z),
    )
}

/// Initializes a diagonal 4x4 matrix.
#[inline]
pub fn diag4_from_vec(d: Vec4) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(d.x, 0.0, 0.0, 0.0),
        Vec4::new(0.0, d.y, 0.0, 0.0),
        Vec4::new(0.0, 0.0, d.z, 0.0),
        Vec4::new(0.0, 0.0, 0.0, d.w),
    )
}

/// Return the diagonal of a 2x2 matrix.
#[inline]
pub fn diag2(m: Mat2) -> Vec2 {
    Vec2::new(m.col(0).x, m.col(1).y)
}

/// Return the diagonal of a 3x3 matrix.
#[inline]
pub fn diag3(m: Mat3) -> Vec3 {
    Vec3::new(m.col(0).x, m.col(1).y, m.col(2).z)
}

/// Return the diagonal of a 4x4 matrix.
#[inline]
pub fn diag4(m: Mat4) -> Vec4 {
    Vec4::new(m.col(0).x, m.col(1).y, m.col(2).z, m.col(3).w)
}

pub fn frobenius_norm_squared2(m: Mat2) -> f32 {
    m.x_axis.length_squared() + m.y_axis.length_squared()
}

pub fn frobenius_norm_squared3(m: Mat3) -> f32 {
    m.x_axis.length_squared() + m.y_axis.length_squared() + m.z_axis.length_squared()
}

pub fn frobenius_norm_squared4(m: Mat4) -> f32 {
    m.x_axis.length_squared()
        + m.y_axis.length_squared()
        + m.z_axis.length_squared()
        + m.w_axis.length_squared()
}
