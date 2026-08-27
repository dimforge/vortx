//! Machine-learning kernels for shaders.

// Shader entry points and their constants look dead on the host, but are used on GPU.
#![allow(dead_code, non_snake_case)]

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
pub mod ppo;
pub mod reduce_axis;
pub mod rms_norm;
pub mod rope;
pub mod select;
pub mod silu;
pub mod softmax;
pub mod unary;
pub mod win_part;

// Parameter and configuration structs, shared by the host and the GPU.
pub use batched_multiquery_attention::AttentionParams;
pub use im2col::Im2ColParams;
pub use optim::AdamParams;
pub use ppo::{PpoActorParams, PpoStageParams, PpoValueParams};
pub use rms_norm::RmsNormConfig;
pub use rope::RoPEConfig;

// Generated ShaderArgs structs (host only). The per-module `MAX_NUM_THREADS`
// constants are not re-exported: callers reach those through the module path.
#[cfg(not(target_arch_is_gpu))]
pub use activation::{GpuEluBackward, GpuTanhBackward};
#[cfg(not(target_arch_is_gpu))]
pub use batched_multiquery_attention::MultMaskAttn;
#[cfg(not(target_arch_is_gpu))]
pub use concat::ConcatCopy;
#[cfg(not(target_arch_is_gpu))]
pub use conv2d::Conv2dNchw;
#[cfg(not(target_arch_is_gpu))]
pub use conv_transpose_2d::{
    ConvTranspose2d, ConvTranspose2dRef, InitDest, InitSrcA, InitSrcB, InitWdata,
};
#[cfg(not(target_arch_is_gpu))]
pub use fused_attention::{FlashAttention, FusedAttention, FusedAttentionOnline};
#[cfg(not(target_arch_is_gpu))]
pub use gather::Gather;
#[cfg(not(target_arch_is_gpu))]
pub use gemv_quant_q4_0x2::GemvQ40x2;
#[cfg(not(target_arch_is_gpu))]
pub use gemv_quant_q4_1x2::GemvQ41x2;
#[cfg(not(target_arch_is_gpu))]
pub use gemv_quant_q4_k::GemvQ4K;
#[cfg(not(target_arch_is_gpu))]
pub use gemv_quant_q5_0x2::GemvQ50x2;
#[cfg(not(target_arch_is_gpu))]
pub use gemv_quant_q5_1x2::GemvQ51x2;
#[cfg(not(target_arch_is_gpu))]
pub use gemv_quant_q5_k::GemvQ5K;
#[cfg(not(target_arch_is_gpu))]
pub use gemv_quant_q6_kx2::GemvQ6Kx2;
#[cfg(not(target_arch_is_gpu))]
pub use gemv_quant_q8_0x2::GemvQ80x2;
#[cfg(not(target_arch_is_gpu))]
pub use gemv_quant_q8_k::GemvQ8K;
#[cfg(not(target_arch_is_gpu))]
pub use get_rel_pos::{AddRelPosPhaseA, AddRelPosPhaseB, GetRelPos};
#[cfg(not(target_arch_is_gpu))]
pub use im2col::Im2col;
#[cfg(not(target_arch_is_gpu))]
pub use layernorm::{LayernormCols, LayernormRows};
#[cfg(not(target_arch_is_gpu))]
pub use optim::GpuAdam;
#[cfg(not(target_arch_is_gpu))]
pub use pool2d::{AvgPool2d, GlobalAvgPool2d, GlobalMaxPool2d, MaxPool2d};
#[cfg(not(target_arch_is_gpu))]
pub use ppo::{GpuPpoActorGrad, GpuPpoStageBatch, GpuPpoValueGrad};
#[cfg(not(target_arch_is_gpu))]
pub use reduce_axis::{ReduceMaxAxis, ReduceMeanAxis, ReduceMinAxis, ReduceSumAxis};
#[cfg(not(target_arch_is_gpu))]
pub use rms_norm::RmsNorm;
#[cfg(not(target_arch_is_gpu))]
pub use rope::{Rope, RopeNeox};
#[cfg(not(target_arch_is_gpu))]
pub use select::Select;
#[cfg(not(target_arch_is_gpu))]
pub use silu::Silu;
#[cfg(not(target_arch_is_gpu))]
pub use softmax::{LogSoftmax, Softmax};
#[cfg(not(target_arch_is_gpu))]
pub use unary::{
    AbsInplace, AbsOp, AddScalarInplace, AddScalarOp, ClampInplace, ClampOp, CosInplace, CosOp,
    EluInplace, EluOp, ErfInplace, ErfOp, ExpInplace, ExpOp, GeluInplace, GeluOp, GeluQuickInplace,
    GeluQuickOp, HardSigmoidInplace, HardSigmoidOp, LeakyReluInplace, LeakyReluOp, LogInplace,
    LogOp, NegInplace, NegOp, PowInplace, PowOp, ReciprocalInplace, ReciprocalOp, ReluInplace,
    ReluOp, ScaleInplace, ScaleOp, SgnInplace, SgnOp, SigmoidInplace, SigmoidOp, SiluInplace,
    SiluOp, SinInplace, SinOp, SqrInplace, SqrOp, SqrtInplace, SqrtOp, StepInplace, StepOp,
    TanhInplace, TanhOp,
};
#[cfg(not(target_arch_is_gpu))]
pub use win_part::{WinPart, WinUnpart};
