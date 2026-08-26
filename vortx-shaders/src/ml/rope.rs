//! Rotary Positional Encoding (RoPE).

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;
use crate::linalg::Shape;
#[cfg(feature = "push_constants")]
use crate::linalg::Shapes2;

const WORKGROUP_SIZE: u32 = 64;

/// RoPE configuration.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RoPEConfig {
    pub head_size: u32,
    pub kv_dim: u32,
    pub pos: u32,
    pub base_freq: f32,
}

/// 2D rotation.
#[derive(Clone, Copy)]
struct Rotation2 {
    cos: f32,
    sin: f32,
}

#[inline]
fn rot2(angle: f32) -> Rotation2 {
    Rotation2 {
        cos: angle.cos(),
        sin: angle.sin(),
    }
}

#[inline]
fn rotate2(r: Rotation2, vx: f32, vy: f32) -> (f32, f32) {
    (r.cos * vx - r.sin * vy, r.sin * vx + r.cos * vy)
}

/// Standard RoPE.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn rope(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_q: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_k: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] config: &[RoPEConfig],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] in_out_q: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] in_out_k: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_q, shape_k) = (shapes.shape_a, shapes.shape_b);
    // Load shapes and config from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_q = *shape_q.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_k = *shape_k.at(0);
    let config = *config.at(0);

    let i = invocation_id.x;
    let head_dim = ((i * 2) % config.head_size) as f32;
    let theta = config.base_freq.powf(-head_dim / config.head_size as f32);
    let m_theta = config.pos as f32 * theta;
    let rot = rot2(m_theta);

    let iq = shape_q.it(0, 0, i * 2, 0) as usize;
    let q_rotated = rotate2(rot, *in_out_q.at(iq), *in_out_q.at(iq + 1));
    *in_out_q.at_mut(iq) = q_rotated.0;
    *in_out_q.at_mut(iq + 1) = q_rotated.1;

    if i * 2 < config.kv_dim {
        let ik = shape_k.it(0, 0, i * 2, 0) as usize;
        let k_rotated = rotate2(rot, *in_out_k.at(ik), *in_out_k.at(ik + 1));
        *in_out_k.at_mut(ik) = k_rotated.0;
        *in_out_k.at_mut(ik + 1) = k_rotated.1;
    }
}

/// NeoX-style RoPE.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn rope_neox(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[cfg(feature = "push_constants")]
    #[spirv(push_constant)]
    shapes: &Shapes2,
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    shape_q: &[Shape],
    #[cfg(not(feature = "push_constants"))]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    shape_k: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] config: &[RoPEConfig],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] in_out_q: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] in_out_k: &mut [f32],
) {
    #[cfg(feature = "push_constants")]
    let (shape_q, shape_k) = (shapes.shape_a, shapes.shape_b);
    // Load shapes and config from storage buffer to local variables (enables LICM)
    #[cfg(not(feature = "push_constants"))]
    let shape_q = *shape_q.at(0);
    #[cfg(not(feature = "push_constants"))]
    let shape_k = *shape_k.at(0);
    let config = *config.at(0);

    let i = invocation_id.x;
    let head_dim = ((i * 2) % config.head_size) as f32;
    let theta = config.base_freq.powf(-head_dim / config.head_size as f32);
    let m_theta = config.pos as f32 * theta;
    let rot = rot2(m_theta);

    let head_id = (i * 2) / config.head_size;
    let shift = config.head_size / 2;

    let iq = shape_q.it(0, 0, i + head_id * config.head_size / 2, 0) as usize;
    let q_rotated = rotate2(rot, *in_out_q.at(iq), *in_out_q.at(iq + shift as usize));
    *in_out_q.at_mut(iq) = q_rotated.0;
    *in_out_q.at_mut(iq + shift as usize) = q_rotated.1;

    if i * 2 < config.kv_dim {
        let ik = shape_k.it(0, 0, i + head_id * config.head_size / 2, 0) as usize;
        let k_rotated = rotate2(rot, *in_out_k.at(ik), *in_out_k.at(ik + shift as usize));
        *in_out_k.at_mut(ik) = k_rotated.0;
        *in_out_k.at_mut(ik + shift as usize) = k_rotated.1;
    }
}
