//! Shader limits and constants.

/// Maximum number of workgroups.
pub const MAX_NUM_WORKGROUPS: u32 = 65535;

/// Dummy function for module inclusion.
#[inline]
pub fn limits_dummy_fn(x: f32) -> f32 {
    x
}
