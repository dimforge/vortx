//! Fused multi-query attention kernel.
//!
//! This kernel fuses the following operations into a single dispatch:
//! 1. Q × K^T (dot products)
//! 2. Scale by 1/sqrt(head_size)
//! 3. Causal masking
//! 4. Softmax
//! 5. Attention × V (weighted sum)
//!
//! Each workgroup processes one query head.

use crate::ml::batched_multiquery_attention::AttentionParams;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;

/// Workgroup size - should be >= head_size for efficient V accumulation.
#[cfg(feature = "subgroup_ops")]
const WORKGROUP_SIZE: usize = 32;
#[cfg(not(feature = "subgroup_ops"))]
const WORKGROUP_SIZE: usize = 128;

/// Maximum sequence length we can handle in shared memory for attention scores.
/// For longer sequences, we use online softmax to avoid storing all scores.
const MAX_SEQ_LEN: usize = 2048;

/// Block size for Flash Attention - number of KV tokens processed per iteration.
/// Must be tuned to fit shared memory: kv_tile uses BLOCK_KV * WORKGROUP_SIZE * 4 bytes.
/// With BLOCK_KV=32 and WORKGROUP_SIZE=128: 32 * 128 * 4 = 16KB for kv_tile alone.
const BLOCK_KV: usize = 32;

#[inline]
fn reduce_max(index: usize, stride: usize, workspace: &mut [f32; WORKGROUP_SIZE]) {
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
    if index < stride {
        workspace.write(
            index,
            workspace.read(index).max(workspace.read(index + stride)),
        );
    }
}

#[inline]
fn reduce_sum(index: usize, stride: usize, workspace: &mut [f32; WORKGROUP_SIZE]) {
    khal_std::sync::workgroup_memory_barrier_with_group_sync();
    if index < stride {
        workspace.write(
            index,
            workspace.read(index) + workspace.read(index + stride),
        );
    }
}

/// Fused attention kernel for single-token inference.
///
/// Workgroup layout: [WORKGROUP_SIZE, 1, 1]
/// Dispatch: [n_kv_heads * kv_mul, 1, 1] workgroups
///
/// Each workgroup computes attention for one query head:
/// out\[head\] = softmax(Q\[head\] · K^T / sqrt(d)) · V
#[spirv_bindgen]
#[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
#[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
pub fn fused_attention(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup)] workspace: &mut [f32; WORKGROUP_SIZE],
    #[spirv(workgroup)] attn_scores: &mut [f32; MAX_SEQ_LEN],
    #[spirv(workgroup)] max_score: &mut f32,
    #[spirv(workgroup)] sum_exp: &mut f32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &[AttentionParams],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] q: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] key_cache: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] value_cache: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] out: &mut [f32],
) {
    let params = params.read(0);
    let tid = local_id.x as usize;
    let head_idx = wg_id.x; // Which query head we're processing

    let head_size = params.head_size as usize;
    let kv_mul = params.kv_mul;
    let n_kv_heads = params.n_heads / kv_mul;
    let seq_len = (params.pos + 1) as usize; // Number of tokens to attend to

    // Which KV head this query head uses
    let kv_head = head_idx / kv_mul;

    // Base offsets for this head
    let q_base = (head_idx * params.head_size) as usize;
    let kv_base = (kv_head * params.head_size) as usize;

    // ==========================================================================
    // Phase 1: Compute Q · K^T for all positions, scale, and find max
    // ==========================================================================

    // Each thread computes dot products for a subset of positions
    // Max iterations: ceil(MAX_SEQ_LEN / WORKGROUP_SIZE) = ceil(2048/128) = 16
    let mut my_max = -1.0e38f32;

    for iter in 0..16 {
        let t = tid + iter * WORKGROUP_SIZE;
        if t < seq_len {
            // Compute Q · K[t] for position t
            // Key cache layout: [seq_len, kv_dim] where kv_dim = n_kv_heads * head_size
            let k_base = t * (n_kv_heads * params.head_size) as usize + kv_base;

            let mut dot = 0.0f32;
            for d in 0..head_size {
                let q_val = q.read(q_base + d);
                let k_val = key_cache.read(k_base + d);
                dot += q_val * k_val;
            }

            // Scale by 1/sqrt(head_size)
            let score = dot / (head_size as f32).sqrt();

            // Store score in shared memory
            attn_scores.write(t, score);
            my_max = my_max.max(score);
        }
    }

    // Reduce to find global max
    workspace.write(tid, my_max);

    #[cfg(feature = "subgroup_ops")]
    let max_val = khal_std::sync::subgroup_f_max(my_max);

    #[cfg(not(feature = "subgroup_ops"))]
    {
        reduce_max(tid, 64, workspace);
        reduce_max(tid, 32, workspace);
        reduce_max(tid, 16, workspace);
        reduce_max(tid, 8, workspace);
        reduce_max(tid, 4, workspace);
        reduce_max(tid, 2, workspace);
        reduce_max(tid, 1, workspace);
    }

    if tid == 0 {
        #[cfg(feature = "subgroup_ops")]
        {
            *max_score = max_val;
        }
        #[cfg(not(feature = "subgroup_ops"))]
        {
            *max_score = workspace.read(0);
        }
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // ==========================================================================
    // Phase 2: Compute exp(score - max) and sum
    // ==========================================================================

    let the_max = *max_score;
    let mut my_sum = 0.0f32;

    for iter in 0..16 {
        let t = tid + iter * WORKGROUP_SIZE;
        if t < seq_len {
            let score = attn_scores.read(t);
            let exp_score = (score - the_max).exp();
            attn_scores.write(t, exp_score);
            my_sum += exp_score;
        }
    }

    // Reduce to find sum
    workspace.write(tid, my_sum);

    #[cfg(feature = "subgroup_ops")]
    let sum = khal_std::sync::subgroup_f_add(my_sum);

    #[cfg(not(feature = "subgroup_ops"))]
    {
        reduce_sum(tid, 64, workspace);
        reduce_sum(tid, 32, workspace);
        reduce_sum(tid, 16, workspace);
        reduce_sum(tid, 8, workspace);
        reduce_sum(tid, 4, workspace);
        reduce_sum(tid, 2, workspace);
        reduce_sum(tid, 1, workspace);
    }

    if tid == 0 {
        #[cfg(feature = "subgroup_ops")]
        {
            *sum_exp = sum;
        }
        #[cfg(not(feature = "subgroup_ops"))]
        {
            *sum_exp = workspace.read(0);
        }
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // ==========================================================================
    // Phase 3: Normalize attention weights (divide by sum)
    // ==========================================================================

    let the_sum = *sum_exp;
    let inv_sum = 1.0 / the_sum;

    for iter in 0..16 {
        let t = tid + iter * WORKGROUP_SIZE;
        if t < seq_len {
            let exp_score = attn_scores.read(t);
            attn_scores.write(t, exp_score * inv_sum);
        }
    }

    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // ==========================================================================
    // Phase 4: Compute weighted sum of values
    // ==========================================================================

    // Each thread computes output for a subset of head dimensions
    let out_base = (head_idx * params.head_size) as usize;

    // Since head_size <= WORKGROUP_SIZE typically, each thread handles at most one dimension
    if tid < head_size {
        let mut weighted_sum = 0.0f32;

        for t in 0..seq_len {
            // Value cache layout: [seq_len, kv_dim]
            let v_base = t * (n_kv_heads * params.head_size) as usize + kv_base;
            let v_val = value_cache.read(v_base + tid);
            let attn_weight = attn_scores.read(t);
            weighted_sum += attn_weight * v_val;
        }

        out.write(out_base + tid, weighted_sum);
    }
}

/// Fused attention with online softmax for long sequences.
///
/// This variant uses online softmax to avoid storing all attention scores,
/// making it memory-efficient for arbitrarily long sequences.
#[spirv_bindgen]
#[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
#[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
pub fn fused_attention_online(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup)] workspace: &mut [f32; WORKGROUP_SIZE],
    #[spirv(workgroup)] out_accum: &mut [f32; WORKGROUP_SIZE], // Accumulator for output
    #[spirv(workgroup)] max_score: &mut f32,
    #[spirv(workgroup)] sum_exp: &mut f32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &[AttentionParams],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] q: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] key_cache: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] value_cache: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] out: &mut [f32],
) {
    let params = params.read(0);
    let tid = local_id.x as usize;
    let head_idx = wg_id.x;

    let head_size = params.head_size as usize;
    let kv_mul = params.kv_mul;
    let n_kv_heads = params.n_heads / kv_mul;
    let seq_len = (params.pos + 1) as usize;

    let kv_head = head_idx / kv_mul;
    let q_base = (head_idx * params.head_size) as usize;
    let kv_base = (kv_head * params.head_size) as usize;
    let out_base = (head_idx * params.head_size) as usize;

    // Initialize accumulators
    if tid < head_size {
        out_accum.write(tid, 0.0);
    }
    if tid == 0 {
        *max_score = -1.0e38f32;
        *sum_exp = 0.0;
    }
    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // Process sequence one token at a time, using online softmax
    for t in 0..seq_len {
        // Compute Q · K[t]
        let k_base = t * (n_kv_heads * params.head_size) as usize + kv_base;

        // Collaborative dot product - each thread handles part of the dimensions
        let mut partial_dot = 0.0f32;
        if tid < head_size {
            let q_val = q.read(q_base + tid);
            let k_val = key_cache.read(k_base + tid);
            partial_dot = q_val * k_val;
        }
        workspace.write(tid, partial_dot);

        // Reduce dot product
        #[cfg(feature = "subgroup_ops")]
        let sum = khal_std::sync::subgroup_f_add(partial_dot);

        #[cfg(not(feature = "subgroup_ops"))]
        {
            reduce_sum(tid, 64, workspace);
            reduce_sum(tid, 32, workspace);
            reduce_sum(tid, 16, workspace);
            reduce_sum(tid, 8, workspace);
            reduce_sum(tid, 4, workspace);
            reduce_sum(tid, 2, workspace);
            reduce_sum(tid, 1, workspace);
        }

        if tid == 0 {
            #[cfg(feature = "subgroup_ops")]
            {
                workspace.write(0, sum);
            }
        }

        khal_std::sync::workgroup_memory_barrier_with_group_sync();

        // Scale score
        let score = workspace.read(0) / (head_size as f32).sqrt();

        // Online softmax update
        let old_max = *max_score;
        let new_max = old_max.max(score);
        let old_sum = *sum_exp;

        // Rescale old accumulator and sum
        let rescale = (old_max - new_max).exp();
        let new_weight = (score - new_max).exp();

        if tid == 0 {
            *max_score = new_max;
            *sum_exp = old_sum * rescale + new_weight;
        }

        // Update output accumulator with rescaling
        let v_base = t * (n_kv_heads * params.head_size) as usize + kv_base;
        if tid < head_size {
            let old_val = out_accum.read(tid);
            let v_val = value_cache.read(v_base + tid);
            out_accum.write(tid, old_val * rescale + new_weight * v_val);
        }

        khal_std::sync::workgroup_memory_barrier_with_group_sync();
    }

    // Final normalization and write output
    let final_sum = *sum_exp;
    if tid < head_size {
        let val = out_accum.read(tid) / final_sum;
        out.write(out_base + tid, val);
    }
}

/// Flash Attention kernel with tiled/block-wise processing.
///
/// This kernel processes KV in blocks of BLOCK_KV tokens, using online softmax
/// to maintain O(1) memory per softmax row. This is ~100x more efficient than
/// the fused_attention_online kernel which processes one token at a time.
///
/// Workgroup layout: [WORKGROUP_SIZE, 1, 1]
/// Dispatch: [n_heads, 1, 1] workgroups
///
/// Each workgroup computes attention for one query head using Flash Attention:
/// - Processes KV cache in blocks of BLOCK_KV tokens
/// - Uses online softmax with rescaling between blocks
/// - Accumulates weighted V values incrementally
#[spirv_bindgen]
#[cfg_attr(feature = "subgroup_ops", spirv(compute(threads(32, 1, 1))))]
#[cfg_attr(not(feature = "subgroup_ops"), spirv(compute(threads(128, 1, 1))))]
pub fn flash_attention(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup)] q_shared: &mut [f32; WORKGROUP_SIZE],
    #[spirv(workgroup)] kv_tile: &mut [f32; BLOCK_KV * WORKGROUP_SIZE],
    #[spirv(workgroup)] scores: &mut [f32; BLOCK_KV],
    #[spirv(workgroup)] workspace: &mut [f32; WORKGROUP_SIZE],
    #[spirv(workgroup)] out_accum: &mut [f32; WORKGROUP_SIZE],
    #[spirv(workgroup)] running_max: &mut f32,
    #[spirv(workgroup)] running_sum: &mut f32,
    #[spirv(workgroup)] block_max_shared: &mut f32,
    #[spirv(workgroup)] block_sum_shared: &mut f32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &[AttentionParams],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] q: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] key_cache: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] value_cache: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] out: &mut [f32],
) {
    let params = params.read(0);
    let tid = local_id.x as usize;
    let head_idx = wg_id.x;

    let head_size = params.head_size as usize;
    let kv_mul = params.kv_mul;
    let n_kv_heads = params.n_heads / kv_mul;
    let seq_len = (params.pos + 1) as usize;

    let kv_head = head_idx / kv_mul;
    let q_base = (head_idx * params.head_size) as usize;
    let kv_base = (kv_head * params.head_size) as usize;
    let kv_stride = (n_kv_heads * params.head_size) as usize;
    let out_base = (head_idx * params.head_size) as usize;

    // ==========================================================================
    // Phase 0: Load Q into shared memory and initialize accumulators
    // ==========================================================================
    if tid < head_size {
        q_shared.write(tid, q.read(q_base + tid));
        out_accum.write(tid, 0.0);
    }
    if tid == 0 {
        *running_max = -1.0e38f32;
        *running_sum = 0.0;
    }
    khal_std::sync::workgroup_memory_barrier_with_group_sync();

    // ==========================================================================
    // Main loop: Process KV in blocks of BLOCK_KV tokens
    // ==========================================================================
    // Maximum number of blocks we might process
    let num_blocks = seq_len.div_ceil(BLOCK_KV);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_KV;
        let block_end = if block_start + BLOCK_KV < seq_len {
            block_start + BLOCK_KV
        } else {
            seq_len
        };
        let block_len = block_end - block_start;

        // ----------------------------------------------------------------------
        // Step 1: Load K block into shared memory
        // ----------------------------------------------------------------------
        // Layout: kv_tile[pos_in_block * head_size + dim]
        // Max elements = BLOCK_KV * head_size, max iterations = ceil(BLOCK_KV * head_size / WORKGROUP_SIZE)
        for iter in 0..BLOCK_KV {
            let load_idx = tid + iter * WORKGROUP_SIZE;
            if load_idx < block_len * head_size {
                let pos_in_block = load_idx / head_size;
                let dim = load_idx % head_size;
                let global_pos = block_start + pos_in_block;
                let k_idx = global_pos * kv_stride + kv_base + dim;
                kv_tile.write(load_idx, key_cache.read(k_idx));
            }
        }
        khal_std::sync::workgroup_memory_barrier_with_group_sync();

        // ----------------------------------------------------------------------
        // Step 2: Compute Q · K[t] for all positions in block
        // ----------------------------------------------------------------------
        // Each thread handles at most one position (since BLOCK_KV <= WORKGROUP_SIZE)
        if tid < block_len {
            let mut dot = 0.0f32;
            for d in 0..head_size {
                let q_val = q_shared.read(d);
                let k_val = kv_tile.read(tid * head_size + d);
                dot += q_val * k_val;
            }

            // Scale by 1/sqrt(head_size) and apply causal mask
            let global_pos = block_start + tid;
            let score = if global_pos <= params.pos as usize {
                dot / (head_size as f32).sqrt()
            } else {
                -1.0e38f32 // Causal mask: future positions masked out
            };
            scores.write(tid, score);
        }
        khal_std::sync::workgroup_memory_barrier_with_group_sync();

        // ----------------------------------------------------------------------
        // Step 3: Find block max via parallel reduction
        // ----------------------------------------------------------------------
        let my_max = if tid < block_len {
            scores.read(tid)
        } else {
            -1.0e38f32
        };
        workspace.write(tid, my_max);

        #[cfg(feature = "subgroup_ops")]
        let max_val = khal_std::sync::subgroup_f_max(my_max);

        #[cfg(not(feature = "subgroup_ops"))]
        {
            reduce_max(tid, 64, workspace);
            reduce_max(tid, 32, workspace);
            reduce_max(tid, 16, workspace);
            reduce_max(tid, 8, workspace);
            reduce_max(tid, 4, workspace);
            reduce_max(tid, 2, workspace);
            reduce_max(tid, 1, workspace);
        }

        // Store result in shared variable so all threads can read it after barrier
        if tid == 0 {
            #[cfg(feature = "subgroup_ops")]
            {
                *block_max_shared = max_val;
            }
            #[cfg(not(feature = "subgroup_ops"))]
            {
                *block_max_shared = workspace.read(0);
            }
        }

        khal_std::sync::workgroup_memory_barrier_with_group_sync();

        let block_max = *block_max_shared;

        // ----------------------------------------------------------------------
        // Step 4: Compute exp(score - block_max) and sum
        // ----------------------------------------------------------------------
        let my_sum = if tid < block_len {
            let exp_score = (scores.read(tid) - block_max).exp();
            scores.write(tid, exp_score); // Overwrite with exp values
            exp_score
        } else {
            0.0f32
        };
        workspace.write(tid, my_sum);

        #[cfg(feature = "subgroup_ops")]
        let sum = khal_std::sync::subgroup_f_add(my_sum);

        #[cfg(not(feature = "subgroup_ops"))]
        {
            reduce_sum(tid, 64, workspace);
            reduce_sum(tid, 32, workspace);
            reduce_sum(tid, 16, workspace);
            reduce_sum(tid, 8, workspace);
            reduce_sum(tid, 4, workspace);
            reduce_sum(tid, 2, workspace);
            reduce_sum(tid, 1, workspace);
        }

        // Store result in shared variable so all threads can read it after barrier
        if tid == 0 {
            #[cfg(feature = "subgroup_ops")]
            {
                *block_sum_shared = sum;
            }
            #[cfg(not(feature = "subgroup_ops"))]
            {
                *block_sum_shared = workspace.read(0);
            }
        }
        khal_std::sync::workgroup_memory_barrier_with_group_sync();

        let block_sum = *block_sum_shared;

        // ----------------------------------------------------------------------
        // Step 5: Update running statistics with rescaling
        // ----------------------------------------------------------------------
        let old_max = *running_max;
        let old_sum = *running_sum;
        let new_max = old_max.max(block_max);
        let rescale_old = (old_max - new_max).exp();
        let rescale_new = (block_max - new_max).exp();

        if tid == 0 {
            *running_max = new_max;
            *running_sum = old_sum * rescale_old + block_sum * rescale_new;
        }
        khal_std::sync::workgroup_memory_barrier_with_group_sync();

        // ----------------------------------------------------------------------
        // Step 6: Load V block and accumulate weighted values
        // ----------------------------------------------------------------------
        // Reuse kv_tile for V block
        for iter in 0..BLOCK_KV {
            let load_idx = tid + iter * WORKGROUP_SIZE;
            if load_idx < block_len * head_size {
                let pos_in_block = load_idx / head_size;
                let dim = load_idx % head_size;
                let global_pos = block_start + pos_in_block;
                let v_idx = global_pos * kv_stride + kv_base + dim;
                kv_tile.write(load_idx, value_cache.read(v_idx));
            }
        }
        khal_std::sync::workgroup_memory_barrier_with_group_sync();

        // Update output accumulator: O = O * rescale_old + (S_exp @ V) * rescale_new
        // Each thread handles one output dimension
        if tid < head_size {
            // Rescale old accumulator
            let old_val = out_accum.read(tid) * rescale_old;

            // Accumulate weighted V values for this dimension
            let mut new_contrib = 0.0f32;
            for pos in 0..block_len {
                let weight = scores.read(pos) * rescale_new;
                let v_val = kv_tile.read(pos * head_size + tid);
                new_contrib += weight * v_val;
            }

            out_accum.write(tid, old_val + new_contrib);
        }
        khal_std::sync::workgroup_memory_barrier_with_group_sync();
    }

    // ==========================================================================
    // Final: Normalize by running sum and write output
    // ==========================================================================
    let final_sum = *running_sum;
    if tid < head_size {
        let val = out_accum.read(tid) / final_sum;
        out.write(out_base + tid, val);
    }
}
