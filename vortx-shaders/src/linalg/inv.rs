//! Matrix inverse functions.
//!
//! These inverse functions were copied from <https://github.com/gfx-rs/wgpu/tree/trunk/naga/src/back/wgsl/polyfill/inverse> (MIT/Apache 2 license)

use glamx::{Mat2, Mat3, Mat4, Vec2, Vec3, Vec4};

/// The inverse of a 2x2 matrix.
///
/// Returns an invalid result if the matrix is not invertible.
#[inline]
pub fn inv2(m: Mat2) -> Mat2 {
    let adj = Mat2::from_cols(
        Vec2::new(m.col(1).y, -m.col(0).y),
        Vec2::new(-m.col(1).x, m.col(0).x),
    );

    let det = m.col(0).x * m.col(1).y - m.col(1).x * m.col(0).y;
    adj * (1.0 / det)
}

/// The inverse of a 3x3 matrix.
///
/// Returns an invalid result if the matrix is not invertible.
#[inline]
pub fn inv3(m: Mat3) -> Mat3 {
    let adj = Mat3::from_cols(
        Vec3::new(
            m.col(1).y * m.col(2).z - m.col(2).y * m.col(1).z,
            -(m.col(0).y * m.col(2).z - m.col(2).y * m.col(0).z),
            m.col(0).y * m.col(1).z - m.col(1).y * m.col(0).z,
        ),
        Vec3::new(
            -(m.col(1).x * m.col(2).z - m.col(2).x * m.col(1).z),
            m.col(0).x * m.col(2).z - m.col(2).x * m.col(0).z,
            -(m.col(0).x * m.col(1).z - m.col(1).x * m.col(0).z),
        ),
        Vec3::new(
            m.col(1).x * m.col(2).y - m.col(2).x * m.col(1).y,
            -(m.col(0).x * m.col(2).y - m.col(2).x * m.col(0).y),
            m.col(0).x * m.col(1).y - m.col(1).x * m.col(0).y,
        ),
    );

    let det = m.col(0).x * (m.col(1).y * m.col(2).z - m.col(1).z * m.col(2).y)
        - m.col(0).y * (m.col(1).x * m.col(2).z - m.col(1).z * m.col(2).x)
        + m.col(0).z * (m.col(1).x * m.col(2).y - m.col(1).y * m.col(2).x);

    adj * (1.0 / det)
}

/// The inverse of a 4x4 matrix.
///
/// Returns an invalid result if the matrix is not invertible.
#[inline]
pub fn inv4(m: Mat4) -> Mat4 {
    let sub_factor00 = m.col(2).z * m.col(3).w - m.col(3).z * m.col(2).w;
    let sub_factor01 = m.col(2).y * m.col(3).w - m.col(3).y * m.col(2).w;
    let sub_factor02 = m.col(2).y * m.col(3).z - m.col(3).y * m.col(2).z;
    let sub_factor03 = m.col(2).x * m.col(3).w - m.col(3).x * m.col(2).w;
    let sub_factor04 = m.col(2).x * m.col(3).z - m.col(3).x * m.col(2).z;
    let sub_factor05 = m.col(2).x * m.col(3).y - m.col(3).x * m.col(2).y;
    let sub_factor06 = m.col(1).z * m.col(3).w - m.col(3).z * m.col(1).w;
    let sub_factor07 = m.col(1).y * m.col(3).w - m.col(3).y * m.col(1).w;
    let sub_factor08 = m.col(1).y * m.col(3).z - m.col(3).y * m.col(1).z;
    let sub_factor09 = m.col(1).x * m.col(3).w - m.col(3).x * m.col(1).w;
    let sub_factor10 = m.col(1).x * m.col(3).z - m.col(3).x * m.col(1).z;
    let sub_factor11 = m.col(1).y * m.col(3).w - m.col(3).y * m.col(1).w;
    let sub_factor12 = m.col(1).x * m.col(3).y - m.col(3).x * m.col(1).y;
    let sub_factor13 = m.col(1).z * m.col(2).w - m.col(2).z * m.col(1).w;
    let sub_factor14 = m.col(1).y * m.col(2).w - m.col(2).y * m.col(1).w;
    let sub_factor15 = m.col(1).y * m.col(2).z - m.col(2).y * m.col(1).z;
    let sub_factor16 = m.col(1).x * m.col(2).w - m.col(2).x * m.col(1).w;
    let sub_factor17 = m.col(1).x * m.col(2).z - m.col(2).x * m.col(1).z;
    let sub_factor18 = m.col(1).x * m.col(2).y - m.col(2).x * m.col(1).y;

    let adj = Mat4::from_cols(
        Vec4::new(
            m.col(1).y * sub_factor00 - m.col(1).z * sub_factor01 + m.col(1).w * sub_factor02,
            -(m.col(1).x * sub_factor00 - m.col(1).z * sub_factor03 + m.col(1).w * sub_factor04),
            m.col(1).x * sub_factor01 - m.col(1).y * sub_factor03 + m.col(1).w * sub_factor05,
            -(m.col(1).x * sub_factor02 - m.col(1).y * sub_factor04 + m.col(1).z * sub_factor05),
        ),
        Vec4::new(
            -(m.col(0).y * sub_factor00 - m.col(0).z * sub_factor01 + m.col(0).w * sub_factor02),
            m.col(0).x * sub_factor00 - m.col(0).z * sub_factor03 + m.col(0).w * sub_factor04,
            -(m.col(0).x * sub_factor01 - m.col(0).y * sub_factor03 + m.col(0).w * sub_factor05),
            m.col(0).x * sub_factor02 - m.col(0).y * sub_factor04 + m.col(0).z * sub_factor05,
        ),
        Vec4::new(
            m.col(0).y * sub_factor06 - m.col(0).z * sub_factor07 + m.col(0).w * sub_factor08,
            -(m.col(0).x * sub_factor06 - m.col(0).z * sub_factor09 + m.col(0).w * sub_factor10),
            m.col(0).x * sub_factor11 - m.col(0).y * sub_factor09 + m.col(0).w * sub_factor12,
            -(m.col(0).x * sub_factor08 - m.col(0).y * sub_factor10 + m.col(0).z * sub_factor12),
        ),
        Vec4::new(
            -(m.col(0).y * sub_factor13 - m.col(0).z * sub_factor14 + m.col(0).w * sub_factor15),
            m.col(0).x * sub_factor13 - m.col(0).z * sub_factor16 + m.col(0).w * sub_factor17,
            -(m.col(0).x * sub_factor14 - m.col(0).y * sub_factor16 + m.col(0).w * sub_factor18),
            m.col(0).x * sub_factor15 - m.col(0).y * sub_factor17 + m.col(0).z * sub_factor18,
        ),
    );

    let det = m.col(0).x * adj.col(0).x
        + m.col(0).y * adj.col(0).y
        + m.col(0).z * adj.col(0).z
        + m.col(0).w * adj.col(0).w;

    adj * (1.0 / det)
}
