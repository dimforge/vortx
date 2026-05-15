//! Linear algebra modules for shaders.

pub mod contiguous;
pub mod gemm;
pub mod inv;
pub mod op_assign;
pub mod reduce;
pub mod repeat;
pub mod shape;

pub use shape::Shape;
#[cfg(feature = "push_constants")]
pub use shape::{Shapes1, Shapes2, Shapes3};

// Re-export generated ShaderArgs structs (only available on host)
#[cfg(not(target_arch_is_gpu))]
pub use contiguous::{Contiguous, ContiguousWithOffset};
#[cfg(not(target_arch_is_gpu))]
pub use gemm::{GemmNaive, GemmTiled};
#[cfg(not(target_arch_is_gpu))]
pub use op_assign::{GpuAdd, GpuCopy, GpuCopyWithOffsets, GpuDiv, GpuMul, GpuSub};
#[cfg(not(target_arch_is_gpu))]
pub use reduce::{ReduceAdd, ReduceMax, ReduceMin, ReduceMul, ReduceSqNorm};
#[cfg(not(target_arch_is_gpu))]
pub use repeat::Repeat;
