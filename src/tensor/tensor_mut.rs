use crate::shapes::{GGML_IDS, TensorLayout};
use crate::tensor::{AsTensorRef, Tensor, TensorRef};
use bytemuck::NoUninit;
use khal::backend::{Buffer, DeviceValue, GpuBuffer, GpuBufferSlice, GpuBufferSliceMut};

/// A view over a mutable vortx.
///
/// This is a view over an entier tensor, or only part of it, with a shape that doesn't necessarily
/// match the original tensor's shape.
pub struct TensorMut<'a, T: DeviceValue> {
    layout: TensorLayout,
    buffer: &'a mut GpuBuffer<T>,
}

pub trait AsTensorMut<T: DeviceValue> {
    fn as_tensor_mut(&mut self) -> TensorMut<'_, T>;
}

impl<'a, T: DeviceValue> From<&'a mut Tensor<T>> for TensorMut<'a, T> {
    fn from(val: &'a mut Tensor<T>) -> TensorMut<'a, T> {
        val.as_view_mut()
    }
}

impl<T: DeviceValue> AsTensorMut<T> for Tensor<T> {
    #[inline]
    fn as_tensor_mut(&mut self) -> TensorMut<'_, T> {
        self.as_view_mut()
    }
}

impl<T: DeviceValue> AsTensorMut<T> for &mut Tensor<T> {
    #[inline]
    fn as_tensor_mut(&mut self) -> TensorMut<'_, T> {
        self.as_view_mut()
    }
}

impl<'a, T: DeviceValue> AsTensorMut<T> for TensorMut<'a, T> {
    #[inline]
    fn as_tensor_mut(&mut self) -> TensorMut<'_, T> {
        TensorMut {
            layout: self.layout,
            buffer: &mut *self.buffer,
        }
    }
}

impl<'a, 'b, T: DeviceValue> AsTensorMut<T> for &'b mut TensorMut<'a, T> {
    #[inline]
    fn as_tensor_mut(&mut self) -> TensorMut<'_, T> {
        TensorMut {
            layout: self.layout,
            buffer: &mut *self.buffer,
        }
    }
}

impl<'a, T: DeviceValue> AsTensorRef<T> for TensorMut<'a, T> {
    #[inline]
    fn as_tensor_ref(&self) -> TensorRef<'_, T> {
        self.as_ref()
    }
}

impl<'a, 'b, T: DeviceValue> AsTensorRef<T> for &'b TensorMut<'a, T> {
    #[inline]
    fn as_tensor_ref(&self) -> TensorRef<'_, T> {
        self.as_ref()
    }
}

impl<'a, T: DeviceValue> TensorMut<'a, T> {
    pub(crate) fn new(layout: TensorLayout, buffer: &'a mut GpuBuffer<T>) -> TensorMut<'a, T> {
        TensorMut { layout, buffer }
    }

    pub(crate) fn contiguous(dims: &[u32], buffer: &'a mut GpuBuffer<T>) -> TensorMut<'a, T> {
        Self::new(TensorLayout::contiguous(dims), buffer)
    }

    /// Converts this mutable view into an immutable view.
    pub fn as_ref(&self) -> TensorRef<'_, T> {
        TensorRef::new(self.layout, &*self.buffer)
    }

    /// Checks if this tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Checks if `self` contains the same number oof elements and matches exactly the layout of
    /// its underlying `GpuTensor`.
    pub fn is_entire_tensor(&self) -> bool
    where
        T: NoUninit,
    {
        self.as_ref().is_entire_tensor()
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

    /// The view's buffer.
    pub fn buffer_mut(&mut self) -> GpuBufferSliceMut<'_, T> {
        self.buffer.slice_mut(self.layout.offset as usize..)
    }

    /// The view's underlying buffer without any offset.
    pub fn raw_buffer(&mut self) -> &mut GpuBuffer<T> {
        self.buffer
    }

    /// Is this view empty?
    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
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

    fn with_layout(self, layout: TensorLayout) -> Self {
        Self {
            layout,
            buffer: self.buffer,
        }
    }

    super::tensor_macro::impl_layout_modifiers! {}
}
