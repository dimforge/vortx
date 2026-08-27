//! Batched multi-query attention.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;

const WORKGROUP_SIZE: u32 = 64;

/// Attention parameters.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct AttentionParams {
    /// Maximum sequence length (for KV cache sizing).
    pub seq_len: u32,
    /// KV dimension (n_kv_heads * head_size).
    pub kv_dim: u32,
    /// Number of query heads per KV head (for grouped-query attention).
    pub kv_mul: u32,
    /// Total number of query heads.
    pub n_heads: u32,
    /// Size of each attention head.
    pub head_size: u32,
    /// Current position in sequence (0-indexed).
    pub pos: u32,
}

#[inline]
fn div_ceil4(a: u32) -> u32 {
    a.div_ceil(4)
}

/// Multiply and mask attention scores.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn mult_mask_attn(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &[AttentionParams],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] attn: &mut [f32],
) {
    // Load params from storage buffer to local variable (enables LICM)
    let params = *params.at(0);

    let nonzero_len = params.pos + 1;
    let aligned_len = div_ceil4(params.pos + 1) * 4;
    if invocation_id.x % aligned_len < nonzero_len {
        *attn.at_mut(invocation_id.x as usize) =
            *attn.at(invocation_id.x as usize) / (params.head_size as f32).sqrt();
    } else {
        *attn.at_mut(invocation_id.x as usize) = 0.0;
    }
}
