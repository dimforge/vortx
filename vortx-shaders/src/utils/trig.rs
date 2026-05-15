//! Trigonometric utility functions.

#[cfg(target_arch_is_gpu)]
use khal_std::num_traits::Float;

/// The value of pi.
pub const PI: f32 = core::f32::consts::PI;

/// A numerically stable implementation of tanh.
///
/// Metal's implementation of tanh returns NaN for large values.
/// This function is more numerically stable and should be used as a
/// drop-in replacement.
// Inspired from https://github.com/apache/tvm/pull/16438 (Apache 2.0 license).
#[inline]
pub fn stable_tanh(x: f32) -> f32 {
    let exp_neg2x = (-2.0 * x).exp();
    let exp_pos2x = (2.0 * x).exp();
    let tanh_pos = (1.0 - exp_neg2x) / (1.0 + exp_neg2x);
    let tanh_neg = (exp_pos2x - 1.0) / (exp_pos2x + 1.0);
    if x >= 0.0 {
        tanh_pos
    } else {
        tanh_neg
    }
}

/// In some platforms, atan2 has unusable edge cases, e.g., returning NaN when y = 0 and x = 0.
///
/// This is for example the case in Metal/MSL: <https://github.com/gfx-rs/wgpu/issues/4319>
/// So we need to implement it ourselves to ensure svd always returns reasonable results on some
/// edge cases like the identity.
#[inline]
pub fn stable_atan2(y: f32, x: f32) -> f32 {
    let ang = (y / x).atan();
    if x > 0.0 {
        return ang;
    }
    if x < 0.0 && y > 0.0 {
        return ang + PI;
    }
    if x < 0.0 && y < 0.0 {
        return ang - PI;
    }

    // Force the other unbounded cases to 0.
    0.0
}
