pub use self::tensor_impl::{Tensor, TensorBuilder};
pub use tensor_mut::{AsTensorMut, TensorMut};
pub use tensor_ref::{AsTensorRef, TensorRef};

pub(crate) mod tensor_macro;

#[allow(clippy::module_inception)]
mod tensor_impl;
mod tensor_mut;
mod tensor_ref;
