//! 2D Convolution operation (NCHW format).
//!
//! This implementation works with ONNX tensor format directly.

use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuBuffer, GpuPass};

#[derive(Shader)]
pub struct Conv2dNchw {
    pub conv_2d_nchw: vortx_shaders::ml::Conv2dNchw,
}

impl Conv2dNchw {
    /// Launch Conv2d operation.
    ///
    /// Input: [N, C_in, H, W]
    /// Weight: [C_out, C_in, K_H, K_W]
    /// Output: [N, C_out, H_out, W_out]
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        params: &GpuBuffer<u32>,
        input: impl AsTensorRef<f32>,
        weight: impl AsTensorRef<f32>,
        mut output: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let mut output = output.as_tensor_mut();
        let input = input.as_tensor_ref();
        let weight = weight.as_tensor_ref();

        let output_len = output.len() as u32;
        let mut buf_output = output.buffer_mut();

        self.conv_2d_nchw.call(
            pass,
            [output_len, 1, 1],
            &mut buf_output,
            &input.buffer(),
            &weight.buffer(),
            &params.as_slice(),
        )?;

        Ok(())
    }
}

/// Compute output dimensions for convolution.
pub fn conv_output_size(
    input_size: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
) -> u32 {
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    (input_size + 2 * padding - effective_kernel) / stride + 1
}
