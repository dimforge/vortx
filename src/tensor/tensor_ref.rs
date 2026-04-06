use crate::shapes::{GGML_IDS, TensorLayout};
use crate::tensor::Tensor;
use bytemuck::NoUninit;
use khal::backend::{Buffer, DeviceValue, GpuBuffer, GpuBufferSlice};
use std::sync::Arc;

/// A view over a tensor.
///
/// This is a view over an entier tensor, or only part of it, with a shape that doesn't necessarily
/// match the original tensor's shape.
pub struct TensorRef<'a, T: DeviceValue> {
    layout: TensorLayout,
    buffer: &'a GpuBuffer<T>,
}

impl<'a, T: DeviceValue> Clone for TensorRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: DeviceValue> Copy for TensorRef<'a, T> {}

pub trait AsTensorRef<T: DeviceValue> {
    fn as_tensor_ref(&self) -> TensorRef<'_, T>;
}

impl<'a, T: DeviceValue> From<&'a Arc<Tensor<T>>> for TensorRef<'a, T> {
    fn from(val: &'a Arc<Tensor<T>>) -> Self {
        val.as_view()
    }
}

impl<'a, T: DeviceValue> From<&'a Tensor<T>> for TensorRef<'a, T> {
    fn from(val: &'a Tensor<T>) -> Self {
        val.as_view()
    }
}

impl<T: DeviceValue> AsTensorRef<T> for Tensor<T> {
    #[inline]
    fn as_tensor_ref(&self) -> TensorRef<'_, T> {
        self.as_view()
    }
}

impl<T: DeviceValue> AsTensorRef<T> for &Tensor<T> {
    #[inline]
    fn as_tensor_ref(&self) -> TensorRef<'_, T> {
        self.as_view()
    }
}

impl<'a, T: DeviceValue> AsTensorRef<T> for TensorRef<'a, T> {
    #[inline]
    fn as_tensor_ref(&self) -> TensorRef<'_, T> {
        *self
    }
}

impl<'a, 'b, T: DeviceValue> AsTensorRef<T> for &'b TensorRef<'a, T> {
    #[inline]
    fn as_tensor_ref(&self) -> TensorRef<'_, T> {
        **self
    }
}

impl<'a, T: DeviceValue> TensorRef<'a, T> {
    pub(crate) fn new(layout: TensorLayout, buffer: &'a GpuBuffer<T>) -> Self {
        TensorRef { layout, buffer }
    }

    pub(crate) fn contiguous(dims: &[u32], buffer: &'a GpuBuffer<T>) -> Self {
        Self::new(TensorLayout::contiguous(dims), buffer)
    }

    /// Checks if this tensor is contiguous in memory (in row-major order).
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Checks if `self` contains the same number of elements and matches exactly the layout of
    /// its underlying `GpuTensor`.
    pub fn is_entire_tensor(&self) -> bool
    where
        T: NoUninit,
    {
        self.buffer.len() == self.len() as usize && self.layout.offset == 0 && self.is_contiguous()
    }

    pub fn rank(&self) -> u32 {
        self.layout().rank
    }

    /// The view's shape.
    pub fn layout(&self) -> TensorLayout {
        self.layout
    }

    /// The view's buffer.
    pub fn buffer(&self) -> GpuBufferSlice<'_, T> {
        self.buffer.slice(self.layout.offset as usize..)
    }

    /// The view's underlying buffer without any offset.
    pub fn raw_buffer(&self) -> &GpuBuffer<T> {
        self.buffer
    }

    /// Is this view empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of elements in this view.
    pub fn len(&self) -> u64 {
        self.layout.len()
    }

    /// Size of this tensor along the dimension `i`.
    pub fn size(&self, i: usize) -> u32 {
        self.layout.size[i]
    }

    /// Size of this tensor along the dimension `i`.
    pub fn size_ggml(&self, i: usize) -> u32 {
        self.layout.size[GGML_IDS[i]]
    }

    /// Stride of this tensor along the dimension `i`.
    pub fn stride(&self, i: usize) -> u32 {
        self.layout.stride[i]
    }

    /// Stride of this tensor along the dimension `i`.
    pub fn stride_ggml(&self, i: usize) -> u32 {
        self.layout.stride[GGML_IDS[i]]
    }

    fn with_layout(&self, layout: TensorLayout) -> Self {
        Self {
            layout,
            buffer: self.buffer,
        }
    }

    /*
     * Layout manipulation functions.
     */
    super::tensor_macro::impl_layout_modifiers! {}
}
