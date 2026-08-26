//! 2D Convolution operation (NCHW format).
//!
//! Input X shape: [N, C_in, H, W]
//! Weight W shape: [C_out, C_in/groups, K_H, K_W]
//! Output Y shape: [N, C_out, H_out, W_out]
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
//! \[10\] dilation_h
//! \[11\] dilation_w
//! \[12\] in_channels
//! \[13\] out_channels
//! \[14\] batch_size
//! \[15\] groups (must be 1 for now)

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::utils::limits::MAX_NUM_WORKGROUPS;

const WORKGROUP_SIZE: u32 = 64;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// Conv2d - compute 2D convolution.
///
/// This is a straightforward implementation, not optimized for performance.
/// For each output element, iterate over the kernel and compute the convolution.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn conv_2d_nchw(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] output: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] weight: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] params: &[u32],
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
    let dilation_h = *params.at(10);
    let dilation_w = *params.at(11);
    let in_channels = *params.at(12);
    let out_channels = *params.at(13);
    let batch_size = *params.at(14);
    let groups = *params.at(15);

    // For grouped convolution:
    // - Input channels per group: in_channels / groups
    // - Output channels per group: out_channels / groups
    // - Weight shape: [out_channels, in_channels/groups, kernel_h, kernel_w]
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;

    let output_len = batch_size * out_channels * output_h * output_w;

    for thread_id in (invocation_id.x..output_len).step_by(MAX_NUM_THREADS as usize) {
        // Decompose output index into [n, oc, oh, ow]
        let ow = thread_id % output_w;
        let oh = (thread_id / output_w) % output_h;
        let oc = (thread_id / (output_w * output_h)) % out_channels;
        let n = thread_id / (output_w * output_h * out_channels);

        // Determine which group this output channel belongs to
        let group = oc / out_channels_per_group;

        // Input channels for this group: [group * in_channels_per_group, (group + 1) * in_channels_per_group)
        let ic_start = group * in_channels_per_group;

        let mut sum: f32 = 0.0;

        // Iterate over input channels in this group and kernel
        for ic_local in 0..in_channels_per_group {
            let ic = ic_start + ic_local;

            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    // Compute input position with dilation
                    let ih_signed = (oh * stride_h + kh * dilation_h) as i32 - pad_h as i32;
                    let iw_signed = (ow * stride_w + kw * dilation_w) as i32 - pad_w as i32;

                    // Check if within bounds
                    if ih_signed >= 0
                        && ih_signed < input_h as i32
                        && iw_signed >= 0
                        && iw_signed < input_w as i32
                    {
                        let ih = ih_signed as u32;
                        let iw = iw_signed as u32;

                        // Input index: [n, ic, ih, iw] in NCHW
                        let i_input = (n * in_channels * input_h * input_w
                            + ic * input_h * input_w
                            + ih * input_w
                            + iw) as usize;

                        // Weight index: [oc, ic_local, kh, kw]
                        // Note: ic_local is used because weight has shape [out_channels, in_channels/groups, kh, kw]
                        let i_weight = (oc * in_channels_per_group * kernel_h * kernel_w
                            + ic_local * kernel_h * kernel_w
                            + kh * kernel_w
                            + kw) as usize;

                        sum += *input.at(i_input) * *weight.at(i_weight);
                    }
                }
            }
        }

        *output.at_mut(thread_id as usize) = sum;
    }
}
