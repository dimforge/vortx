//! 2D Pooling operations (MaxPool2d, AvgPool2d).
//!
//! Input shape: [N, C, H, W] (NCHW format)
//! Output shape: [N, C, H_out, W_out]
//!
//! Parameters in params buffer:
//! \[0\] input_height
//! \[1\] input_width
//! \[2\] output_height
//! \[3\] output_width
//! \[4\] kernel_h
//! \[5\] kernel_w
//! \[6\] stride_h
//! \[7\] stride_w
//! \[8\] pad_h
//! \[9\] pad_w
//! \[10\] channels
//! \[11\] batch_size

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::utils::limits::MAX_NUM_WORKGROUPS;

const WORKGROUP_SIZE: u32 = 64;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// MaxPool2d - compute max over a 2D window.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn max_pool_2d(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dest: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] params: &[u32],
) {
    let input_h = *params.at(0);
    let input_w = *params.at(1);
    let output_h = *params.at(2);
    let output_w = *params.at(3);
    let kernel_h = *params.at(4);
    let kernel_w = *params.at(5);
    let stride_h = *params.at(6);
    let stride_w = *params.at(7);
    let pad_h = *params.at(8);
    let pad_w = *params.at(9);
    let channels = *params.at(10);
    let batch_size = *params.at(11);

    let output_len = batch_size * channels * output_h * output_w;

    for thread_id in (invocation_id.x..output_len).step_by(MAX_NUM_THREADS as usize) {
        // Decompose output index into [n, c, oh, ow]
        let ow = thread_id % output_w;
        let oh = (thread_id / output_w) % output_h;
        let c = (thread_id / (output_w * output_h)) % channels;
        let n = thread_id / (output_w * output_h * channels);

        // Find the input window
        let h_start_signed = (oh * stride_h) as i32 - pad_h as i32;
        let w_start_signed = (ow * stride_w) as i32 - pad_w as i32;

        // Clamp to valid input range
        let h_start = if h_start_signed < 0 {
            0u32
        } else {
            h_start_signed as u32
        };
        let w_start = if w_start_signed < 0 {
            0u32
        } else {
            w_start_signed as u32
        };
        let h_end_unclamped = (h_start_signed + kernel_h as i32) as u32;
        let w_end_unclamped = (w_start_signed + kernel_w as i32) as u32;
        let h_end = if h_end_unclamped > input_h {
            input_h
        } else {
            h_end_unclamped
        };
        let w_end = if w_end_unclamped > input_w {
            input_w
        } else {
            w_end_unclamped
        };

        // Initialize max with a very small value (SPIR-V doesn't support infinity literals)
        let mut max_val = -3.4028235e+38_f32; // Close to f32::MIN

        // Iterate over the pooling window
        for ih in h_start..h_end {
            for iw in w_start..w_end {
                // Input index: n * (C * H * W) + c * (H * W) + ih * W + iw
                let i_src =
                    (n * channels * input_h * input_w + c * input_h * input_w + ih * input_w + iw)
                        as usize;
                let val = *src.at(i_src);
                if val > max_val {
                    max_val = val;
                }
            }
        }

        // Handle edge case where window was entirely in padding
        if max_val < -3.4e+38_f32 {
            max_val = 0.0;
        }

        *dest.at_mut(thread_id as usize) = max_val;
    }
}

/// AvgPool2d - compute average over a 2D window.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn avg_pool_2d(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dest: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] params: &[u32],
) {
    let input_h = *params.at(0);
    let input_w = *params.at(1);
    let output_h = *params.at(2);
    let output_w = *params.at(3);
    let kernel_h = *params.at(4);
    let kernel_w = *params.at(5);
    let stride_h = *params.at(6);
    let stride_w = *params.at(7);
    let pad_h = *params.at(8);
    let pad_w = *params.at(9);
    let channels = *params.at(10);
    let batch_size = *params.at(11);
    // params[12] = count_include_pad (0 or 1)
    let count_include_pad = *params.at(12);

    let output_len = batch_size * channels * output_h * output_w;

    for thread_id in (invocation_id.x..output_len).step_by(MAX_NUM_THREADS as usize) {
        // Decompose output index into [n, c, oh, ow]
        let ow = thread_id % output_w;
        let oh = (thread_id / output_w) % output_h;
        let c = (thread_id / (output_w * output_h)) % channels;
        let n = thread_id / (output_w * output_h * channels);

        // Find the input window
        let h_start_signed = (oh * stride_h) as i32 - pad_h as i32;
        let w_start_signed = (ow * stride_w) as i32 - pad_w as i32;

        // Clamp to valid input range
        let h_start = if h_start_signed < 0 {
            0u32
        } else {
            h_start_signed as u32
        };
        let w_start = if w_start_signed < 0 {
            0u32
        } else {
            w_start_signed as u32
        };
        let h_end_unclamped = (h_start_signed + kernel_h as i32) as u32;
        let w_end_unclamped = (w_start_signed + kernel_w as i32) as u32;
        let h_end = if h_end_unclamped > input_h {
            input_h
        } else {
            h_end_unclamped
        };
        let w_end = if w_end_unclamped > input_w {
            input_w
        } else {
            w_end_unclamped
        };

        // Sum over the pooling window
        let mut sum: f32 = 0.0;
        let mut count: u32 = 0;

        for ih in h_start..h_end {
            for iw in w_start..w_end {
                // Input index: n * (C * H * W) + c * (H * W) + ih * W + iw
                let i_src =
                    (n * channels * input_h * input_w + c * input_h * input_w + ih * input_w + iw)
                        as usize;
                sum += *src.at(i_src);
                count += 1;
            }
        }

        // Compute average
        let divisor = if count_include_pad != 0 {
            kernel_h * kernel_w
        } else {
            count
        };

        let avg = if divisor > 0 {
            sum / (divisor as f32)
        } else {
            0.0
        };
        *dest.at_mut(thread_id as usize) = avg;
    }
}

/// GlobalAvgPool2d - average over entire spatial dimensions.
/// Input: [N, C, H, W], Output: [N, C, 1, 1]
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn global_avg_pool_2d(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dest: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] params: &[u32], // [input_h, input_w, channels, batch_size]
) {
    let input_h = *params.at(0);
    let input_w = *params.at(1);
    let channels = *params.at(2);
    let batch_size = *params.at(3);

    let output_len = batch_size * channels;
    let spatial_size = input_h * input_w;

    for thread_id in (invocation_id.x..output_len).step_by(MAX_NUM_THREADS as usize) {
        // Output index: [n, c]
        let c = thread_id % channels;
        let n = thread_id / channels;

        // Sum over all spatial elements
        let mut sum: f32 = 0.0;
        for ih in 0..input_h {
            for iw in 0..input_w {
                let i_src =
                    (n * channels * input_h * input_w + c * input_h * input_w + ih * input_w + iw)
                        as usize;
                sum += *src.at(i_src);
            }
        }

        *dest.at_mut(thread_id as usize) = sum / (spatial_size as f32);
    }
}

/// GlobalMaxPool2d - max over entire spatial dimensions.
/// Input: [N, C, H, W], Output: [N, C, 1, 1]
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn global_max_pool_2d(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dest: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] params: &[u32], // [input_h, input_w, channels, batch_size]
) {
    let input_h = *params.at(0);
    let input_w = *params.at(1);
    let channels = *params.at(2);
    let batch_size = *params.at(3);

    let output_len = batch_size * channels;

    for thread_id in (invocation_id.x..output_len).step_by(MAX_NUM_THREADS as usize) {
        // Output index: [n, c]
        let c = thread_id % channels;
        let n = thread_id / channels;

        // Find max over all spatial elements
        let i_src_init = (n * channels * input_h * input_w + c * input_h * input_w) as usize;
        let mut max_val = *src.at(i_src_init);

        for ih in 0..input_h {
            for iw in 0..input_w {
                let i_src =
                    (n * channels * input_h * input_w + c * input_h * input_w + ih * input_w + iw)
                        as usize;
                let val = *src.at(i_src);
                if val > max_val {
                    max_val = val;
                }
            }
        }

        *dest.at_mut(thread_id as usize) = max_val;
    }
}
