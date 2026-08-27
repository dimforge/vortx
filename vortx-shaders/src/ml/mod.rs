// #![allow(clippy::too_many_arguments)]
// // `spirv_bindgen` generates host-side dispatch code that performs `% workgroup_size`,
// // which triggers this lint when a workgroup dimension is 1.
// #![allow(clippy::modulo_one)]
// #![allow(unexpected_cfgs)]
// // Shader entry points and their constants appear dead on host but are used on GPU.
// #![allow(dead_code, non_snake_case)]

// TODO: keep the modules private?
pub mod activation;
pub mod batched_multiquery_attention;
pub mod concat;
pub mod conv2d;
pub mod conv_transpose_2d;
pub mod fused_attention;
pub mod gather;
pub mod gemv_quant_q4_0x2;
pub mod gemv_quant_q4_1x2;
pub mod gemv_quant_q4_k;
pub mod gemv_quant_q5_0x2;
pub mod gemv_quant_q5_1x2;
pub mod gemv_quant_q5_k;
pub mod gemv_quant_q6_kx2;
pub mod gemv_quant_q8_0x2;
pub mod gemv_quant_q8_k;
pub mod get_rel_pos;
pub mod im2col;
pub mod layernorm;
pub mod optim;
pub mod pool2d;
pub mod reduce_axis;
pub mod rms_norm;
pub mod rope;
pub mod select;
pub mod silu;
pub mod softmax;
pub mod unary;
pub mod win_part;

pub use activation::*;
pub use batched_multiquery_attention::*;
pub use concat::*;
pub use conv2d::*;
pub use conv_transpose_2d::*;
pub use fused_attention::*;
pub use gather::*;
pub use get_rel_pos::*;
pub use im2col::*;
pub use layernorm::*;
pub use optim::*;
pub use pool2d::*;
pub use reduce_axis::*;
pub use rms_norm::*;
pub use rope::*;
pub use select::*;
pub use silu::*;
pub use softmax::*;
pub use unary::*;
pub use win_part::*;
