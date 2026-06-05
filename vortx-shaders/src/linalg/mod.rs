//! Linear algebra modules for shaders.

pub mod activation;
pub mod contiguous;
pub mod gemm;
pub mod inv;
pub mod op_assign;
pub mod optim;
pub mod ppo;
pub mod reduce;
pub mod repeat;
pub mod shape;

pub use shape::Shape;
#[cfg(feature = "push_constants")]
pub use shape::{Shapes1, Shapes2, Shapes3};

// Re-export generated ShaderArgs structs (only available on host)
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use activation::{GpuTanh, GpuTanhBackward};
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use contiguous::{Contiguous, ContiguousWithOffset};
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use gemm::{GemmNaive, GemmTiled};
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use op_assign::{GpuAdd, GpuCopy, GpuCopyWithOffsets, GpuDiv, GpuMul, GpuSub};
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use optim::GpuAdam;
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use ppo::{GpuPpoActorGrad, GpuPpoValueGrad};
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use reduce::{ReduceAdd, ReduceMax, ReduceMin, ReduceMul, ReduceSqNorm};
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use repeat::Repeat;
