//! 2D Pooling operations (MaxPool2d, AvgPool2d, GlobalAvgPool2d, GlobalMaxPool2d).

use khal::backend::{GpuBackendError, GpuBuffer, GpuPass};
use khal::Shader;
use crate::tensor::{AsTensorMut, AsTensorRef};

/// Pool2d configuration parameters.
#[derive(Copy, Clone, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
#[repr(C)]
pub struct Pool2dConfig {
    pub input_h: u32,
    pub input_w: u32,
    pub output_h: u32,
    pub output_w: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub channels: u32,
    pub batch_size: u32,
    pub count_include_pad: u32, // For avg pool only
    pub _padding: [u32; 3],
}

/// Global pooling configuration.
#[derive(Copy, Clone, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
#[repr(C)]
pub struct GlobalPool2dConfig {
    pub input_h: u32,
    pub input_w: u32,
    pub channels: u32,
    pub batch_size: u32,
}

#[derive(Shader)]
pub struct Pool2d {
    pub max_pool_2d: vortx_shaders::ml::pool2d::MaxPool2d,
    pub avg_pool_2d: vortx_shaders::ml::pool2d::AvgPool2d,
    pub global_avg_pool_2d: vortx_shaders::ml::pool2d::GlobalAvgPool2d,
    pub global_max_pool_2d: vortx_shaders::ml::pool2d::GlobalMaxPool2d,
}

impl Pool2d {
    /// Launch MaxPool2d operation.
    ///
    /// Input: [N, C, H, W], Output: [N, C, H_out, W_out]
    pub fn launch_max_pool(
        &self,
        pass: &mut GpuPass,
        params: &GpuBuffer<u32>,
        input: impl AsTensorRef<f32>,
        mut output: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut output = output.as_tensor_mut();
        let input = input.as_tensor_ref();

        let output_len = output.len() as u32;
        let mut buf_output = output.buffer_mut();

        self.max_pool_2d.call(
            pass,
            [output_len, 1, 1],
            &mut buf_output,
            &input.buffer(),
            &params.as_slice(),
        )?;

        Ok(())
    }

    /// Launch AvgPool2d operation.
    ///
    /// Input: [N, C, H, W], Output: [N, C, H_out, W_out]
    pub fn launch_avg_pool(
        &self,
        pass: &mut GpuPass,
        params: &GpuBuffer<u32>,
        input: impl AsTensorRef<f32>,
        mut output: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut output = output.as_tensor_mut();
        let input = input.as_tensor_ref();

        let output_len = output.len() as u32;
        let mut buf_output = output.buffer_mut();

        self.avg_pool_2d.call(
            pass,
            [output_len, 1, 1],
            &mut buf_output,
            &input.buffer(),
            &params.as_slice(),
        )?;

        Ok(())
    }

    /// Launch GlobalAvgPool2d operation.
    ///
    /// Input: [N, C, H, W], Output: [N, C, 1, 1] (stored as [N, C])
    pub fn launch_global_avg_pool(
        &self,
        pass: &mut GpuPass,
        params: &GpuBuffer<u32>,
        input: impl AsTensorRef<f32>,
        mut output: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut output = output.as_tensor_mut();
        let input = input.as_tensor_ref();

        let output_len = output.len() as u32;
        let mut buf_output = output.buffer_mut();

        self.global_avg_pool_2d.call(
            pass,
            [output_len, 1, 1],
            &mut buf_output,
            &input.buffer(),
            &params.as_slice(),
        )?;

        Ok(())
    }

    /// Launch GlobalMaxPool2d operation.
    ///
    /// Input: [N, C, H, W], Output: [N, C, 1, 1] (stored as [N, C])
    pub fn launch_global_max_pool(
        &self,
        pass: &mut GpuPass,
        params: &GpuBuffer<u32>,
        input: impl AsTensorRef<f32>,
        mut output: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut output = output.as_tensor_mut();
        let input = input.as_tensor_ref();

        let output_len = output.len() as u32;
        let mut buf_output = output.buffer_mut();

        self.global_max_pool_2d.call(
            pass,
            [output_len, 1, 1],
            &mut buf_output,
            &input.buffer(),
            &params.as_slice(),
        )?;

        Ok(())
    }
}

/// Compute output dimensions for pooling.
pub fn pool_output_size(
    input_size: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    ceil_mode: bool,
) -> u32 {
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let numerator = input_size + 2 * padding - effective_kernel;

    if ceil_mode {
        numerator.div_ceil(stride) + 1
    } else {
        numerator / stride + 1
    }
}
