#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::result_large_err)]
#![allow(missing_docs)] // TODO: warn

// Re-export the shader crate for access to generated ShaderArgs structs
pub use vortx_shaders as shaders;

use khal::re_exports::include_dir::{Dir, include_dir};

/// Embedded SPIR-V shader directory.
pub static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

pub use linalg::*;

pub mod linalg;
pub mod shapes;
pub mod tensor;
