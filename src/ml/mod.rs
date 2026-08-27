//! Primitives for building LLM inferences.

mod activation;
mod batched_multiquery_attention;
mod concat;
mod conv2d_nchw;
mod conv_transpose_2d;
mod gather;
mod gemv_quant;
mod get_rel_pos;
mod im2col;
mod layernorm;
mod optim;
mod pool2d;
pub mod quantization;
mod quantized_matrix;
mod reduce_axis;
mod rms_norm;
mod rope;
mod select;
mod silu;
mod softmax;
mod unary;
mod win_part;

pub use activation::ActivationBackward;
pub use batched_multiquery_attention::FusedAttention;
pub use concat::Concat;
pub use conv_transpose_2d::ConvTranspose2d;
pub use conv2d_nchw::{Conv2dNchw, conv_output_size};
pub use gather::Gather;
pub use gemv_quant::{
    GemvQuant, GpuBlockQ4_0x2, GpuBlockQ4_1x2, GpuBlockQ4K, GpuBlockQ5_0x2, GpuBlockQ5_1x2,
    GpuBlockQ5K, GpuBlockQ6Kx2, GpuBlockQ8_0x2, GpuBlockQ8K, QuantizedValue,
};
pub use get_rel_pos::GetRelPos;
pub use im2col::{Im2Col, Im2ColConfig};
pub use layernorm::LayerNorm;
pub use optim::{Adam, AdamParams};
pub use pool2d::{GlobalPool2dConfig, Pool2d, Pool2dConfig, pool_output_size};
pub use quantized_matrix::*;
pub use reduce_axis::{ReduceAxis, ReduceOp};
pub use rms_norm::{RmsNorm, RmsNormConfig};
pub use rope::{RoPE, RoPEConfig, RoPEVariant};
pub use select::Select;
pub use silu::Silu;
pub use softmax::SoftMax;
pub use unary::{Unary, UnaryOp};
pub use win_part::WinPart;
