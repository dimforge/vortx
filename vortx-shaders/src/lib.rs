//! vortx shaders for rust-gpu.
//!
//! This crate contains GPU shaders for tensor operations, geometry transformations,
//! and linear algebra primitives, written for rust-gpu.

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(unexpected_cfgs)]
#![allow(clippy::too_many_arguments)]

// Enable std on host for generated ShaderArgs structs (not on GPU targets).
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
extern crate std;

pub mod linalg;
pub mod utils;
