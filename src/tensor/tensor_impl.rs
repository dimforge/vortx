//! Utilities for initializing and slicing tensors, matrices, vectors, and scalars gpu storage
//! buffers.

use crate::shapes::{GGML_IDS, TensorLayout};
use bytemuck::NoUninit;
use khal::backend::{
    Backend, Buffer, DeviceValue, Encoder, GpuBackend, GpuBackendError, GpuBuffer, GpuBufferSlice,
    GpuBufferSliceMut, GpuDispatch, GpuEncoder, ShaderBinding,
};
use nalgebra::DMatrix;
use std::ops::{Bound, RangeBounds};

use crate::tensor::{AsTensorRef, TensorMut, TensorRef};
use khal::shader::ShaderArgsError;
use khal::{BufferUsages, ShaderArgs};

/// Helper struct for creating gpu storage buffers (scalars, vectors, matrices, tensors).
pub struct TensorBuilder {
    shape: [u32; 4],
    rank: u32,
    usage: BufferUsages,
    label: Option<String>,
}

impl TensorBuilder {
    /// Starts building a storage buffer containing a single scalar value.
    pub fn scalar(usage: BufferUsages) -> Self {
        Self::tensor(&[], usage)
    }

    /// Starts building a storage buffer containing a vector.
    pub fn vector(dim: u32, usage: BufferUsages) -> Self {
        Self::tensor(&[dim], usage)
    }

    /// Starts building a storage buffer containing a single matrix with `nrows` rows and
    /// `ncols` columns.
    pub fn matrix(nrows: u32, ncols: u32, usage: BufferUsages) -> Self {
        Self::tensor(&[nrows, ncols], usage)
    }

    /// Starts building a storage buffer containing a tensor with the specified `shape`.
    pub fn tensor(shape: &[u32], usage: BufferUsages) -> Self {
        let (shape, rank) = TensorLayout::append_ones(shape);
        Self {
            shape,
            rank,
            usage,
            label: None,
        }
    }

    /// The number of elements in this tensor.
    fn len(&self) -> u64 {
        self.shape.into_iter().map(|s| s as u64).product()
    }

    /// Sets the debug label of this tensor.
    pub fn label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Builds the uninitialized gpu tensor.
    pub fn build_uninit<T: DeviceValue + NoUninit>(
        self,
        backend: &GpuBackend,
    ) -> Result<Tensor<T>, GpuBackendError> {
        let buffer = backend.uninit_buffer(self.len() as usize, self.usage)?;
        Ok(Tensor {
            shape: self.shape,
            rank: self.rank,
            buffer,
        })
    }

    // /// Builds this tensor with raw bytes given for its initial value.
    // pub fn build_bytes<T: DeviceValue>(self, device: &Device, data: &[u8]) -> WgpuTensor<T, DIM> {
    //     let buffer = device.create_buffer_init(&BufferInitDescriptor {
    //         label: self.label.as_deref(),
    //         contents: bytemuck::cast_slice(data),
    //         usage: self.usage,
    //     });
    //
    //     GpuTensor {
    //         shape: self.shape,
    //         buffer,
    //     }
    // }

    /// Builds this tensor with an array of values given for its initial value.
    pub fn build_init<T: DeviceValue + NoUninit>(
        self,
        backend: &GpuBackend,
        data: &[T],
    ) -> Result<Tensor<T>, GpuBackendError> {
        assert!(
            data.len() as u64 >= self.len(),
            "Incorrect number of elements provided for initializing Tensor.\
            Expected at least {}, found {}",
            self.len(),
            data.len()
        );

        let buffer = backend.init_buffer(data, self.usage)?;
        Ok(Tensor {
            shape: self.shape,
            rank: self.rank,
            buffer,
        })
    }
}

/// A tensor stored in the GPU.
///
/// When the tensor is a matrix, they are generally seen as being column-major.
pub struct Tensor<T: DeviceValue> {
    shape: [u32; 4],
    rank: u32,
    buffer: GpuBuffer<T>,
}

impl<T: DeviceValue> Tensor<T> {
    /// Does this tensor contain zero elements?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of elements in this tensor.
    pub fn len(&self) -> u64 {
        self.shape.into_iter().map(|s| s as u64).product()
    }

    // /// The tensor's rank.
    // pub fn rank(&self) -> u64 {
    //     self.shape.iter().filter(|i| **i != 1).count() as u64
    // }

    /// The maximum number of elements this tensor can hold without needing a resize of the
    /// underlying GPU buffer.
    pub fn capacity(&self) -> u64
    where
        T: NoUninit,
    {
        self.buffer.len() as u64
    }

    /// The tensor’s order (i.e. the number of dimensions with a size > 1).
    pub fn order(&self) -> u8 {
        self.shape.iter().map(|s| (*s > 1) as u8).sum()
    }

    /// Size of this tensor along the dimension `i`.
    pub fn size(&self, i: usize) -> u32 {
        self.shape[i]
    }

    /// Size of this tensor along the dimension `i`.
    pub fn size_ggml(&self, i: usize) -> u32 {
        self.size(GGML_IDS[i])
    }

    /// Size of this tensor along the dimension `i`.
    pub fn stride(&self, i: usize) -> u32 {
        self.layout().stride[i]
    }

    /// Size of this tensor along the dimension `i`.
    pub fn stride_ggml(&self, i: usize) -> u32 {
        self.stride(GGML_IDS[i])
    }

    /// The size, in bytes, of this tensor’s content.
    pub fn bytes_len(&self) -> u64
    where
        T: DeviceValue,
    {
        std::mem::size_of::<T>() as u64 * self.len()
    }

    // /// Queues a buffer-to-buffer copy from `source` to `self`.
    // ///
    // /// Panics if the lengths do not match.
    // pub fn copy_from(&self, encoder: &mut CommandEncoder, source: &GpuTensor<T, B>)
    // where
    //     T: DeviceValue,
    // {
    //     assert_eq!(self.len(), source.len());
    //     encoder.copy_buffer_to_buffer(&source.buffer, 0, &self.buffer, 0, self.bytes_len())
    // }

    /// Queues a buffer-to-buffer copy from `source` to `self`.
    pub fn copy_from_view(
        &mut self,
        encoder: &mut GpuEncoder,
        source: impl AsTensorRef<T>,
    ) -> Result<(), GpuBackendError>
    where
        T: DeviceValue + NoUninit,
    {
        let source = source.as_tensor_ref();
        let copy_len = self.len();
        assert!(self.len() <= source.len());

        // FIXME: assert that the source view is contiguous in a way that is
        //        compatible with `self`.
        encoder.copy_buffer_to_buffer(
            source.raw_buffer(),
            source.layout().offset as usize,
            &mut self.buffer,
            0,
            copy_len as usize,
        )
    }

    /// The tensor’s shape (typically `[num_rows, num_cols, ...]`).
    pub fn shape(&self) -> [u32; 4] {
        self.shape
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// The tensor's underlying GPU buffer.
    pub fn buffer(&self) -> &GpuBuffer<T> {
        &self.buffer
    }

    /// The tensor's underlying GPU buffer.
    pub fn buffer_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.buffer
    }

    /// The tensor's underlying GPU buffer as a full slice.
    pub fn buffer_slice(&self) -> GpuBufferSlice<'_, T> {
        self.buffer.slice(..)
    }

    /// The tensor's underlying GPU buffer as a full slice.
    pub fn buffer_slice_mut(&mut self) -> GpuBufferSliceMut<'_, T> {
        self.buffer.slice_mut(..)
    }

    /// Extracts the underlying GPU buffer.
    pub fn into_inner(self) -> GpuBuffer<T> {
        self.buffer
    }

    /// Builds a tensor view sharing the same shape, stride, and buffer, as `self`.
    pub fn as_view(&self) -> TensorRef<'_, T> {
        TensorRef::contiguous(self.dims(), &self.buffer)
    }

    /// Builds a mutable tensor view sharing the same shape, stride, and buffer, as `self`.
    pub fn as_view_mut(&mut self) -> TensorMut<'_, T> {
        TensorMut::contiguous(&self.shape[..self.rank as usize], &mut self.buffer)
    }

    fn dims(&self) -> &[u32] {
        &self.shape[..self.rank as usize]
    }

    // fn vector_dim(&self) -> u32 {
    //     assert!(self.rank == 1, "This tensor isn’t a 1D tensor.");
    //     self.shape[0]
    // }

    // /// Reads the buffer’s content into a vector.
    // pub async fn read_bytes<'a>(&'a self, device: &'a Device) -> anyhow::Result<BufferView<'a>> {
    //     // TODO: could probably be optimized?
    //     let buffer_slice = self.buffer.slice(..);
    //
    //     #[cfg(not(target_arch = "wasm32"))]
    //     {
    //         let (sender, receiver) = async_channel::bounded(1);
    //         buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
    //             sender.send_blocking(v).unwrap()
    //         });
    //         device.poll(wgpu::PollType::wait());
    //         receiver.recv().await?.unwrap();
    //     }
    //     #[cfg(target_arch = "wasm32")]
    //     {
    //         let (sender, receiver) = async_channel::bounded(1);
    //         buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
    //             let _ = sender.force_send(v).unwrap();
    //         });
    //         device.poll(wgpu::PollType::wait());
    //         receiver.recv().await?.unwrap();
    //     }
    //
    //     let data = buffer_slice.get_mapped_range();
    //     Ok(data)
    // }
    //
    // /// Reads the buffer’s content into a slice.
    // pub async fn read_to(&self, device: &Device, out: &mut [T]) -> anyhow::Result<()>
    // where
    //     T: DeviceValue,
    // {
    //     let data = self.read_bytes(device).await?;
    //     let result = bytemuck::try_cast_slice(&data)?;
    //     out.copy_from_slice(result);
    //     drop(data);
    //     self.buffer.unmap();
    //     Ok(())
    // }
    //
    // /// Reads the buffer’s content into a vector.
    // pub async fn read(&self, device: &Device) -> anyhow::Result<Vec<T>>
    // where
    //     T: DeviceValue,
    // {
    //     let data = self.read_bytes(device).await?;
    //     let result = bytemuck::try_cast_slice(&data)?.to_vec();
    //     drop(data);
    //     self.buffer.unmap();
    //     Ok(result)
    // }
}

impl<T: DeviceValue> Tensor<T> {
    pub fn layout(&self) -> TensorLayout {
        TensorLayout::contiguous(&self.shape[..self.rank as usize])
    }

    /// Removes all the 1 from the shape of `self`.
    ///
    /// This reduces the rank of `self` by the number of 1 on its current shape.
    pub fn squeeze(&mut self) {
        let new_layout = self.layout().squeeze();
        self.shape = new_layout.size;
        self.rank = new_layout.rank;
    }

    /// Inserts a dimension of size 1 at the given `axis` of this tensor.
    ///
    /// This operation increases the rank of `self` by 1.
    pub fn unsqueeze(&mut self, axis: u32) {
        assert!(axis <= self.rank, "unsqueeze axis out of bounds");
        assert!(
            self.rank < 4,
            "unsqueezing would exceed the max supported tensor rank"
        );
        let new_layout = self
            .layout()
            .unsqueeze(axis)
            .unwrap_or_else(|| unreachable!());
        self.shape = new_layout.size;
        self.rank = new_layout.rank;
    }

    /// Reshapes this tensor to the specified shape.
    pub fn reshape(&self, shape: &[u32]) -> TensorRef<'_, T> {
        self.as_view().reshape(shape)
    }

    /// Reshapes this tensor using GGML's dimension ordering convention.
    pub fn reshape_ggml(&self, shape: &[u32]) -> TensorRef<'_, T> {
        self.as_view().reshape_ggml(shape)
    }

    /// Permutes the dimensions of this tensor according to the given permutation array.
    pub fn permute(&self, permutations: [usize; 4]) -> TensorRef<'_, T> {
        self.as_view().permute(permutations)
    }

    /// Permutes the dimensions according to GGML's dimension ordering convention.
    pub fn permute_ggml(&self, permutations: [usize; 4]) -> TensorRef<'_, T> {
        self.as_view().permute_ggml(permutations)
    }

    /// Creates a view of a sub-tensor with the specified offset, shape, and optional strides.
    pub fn view(&self, offset: u32, shape: &[u32], stride: &[Option<u32>]) -> TensorRef<'_, T> {
        self.as_view().view(offset, shape, stride)
    }

    /// Creates a view using GGML's dimension ordering convention.
    pub fn view_ggml(
        &self,
        offset: u32,
        shape: &[u32],
        stride: &[Option<u32>],
    ) -> TensorRef<'_, T> {
        self.as_view().view_ggml(offset, shape, stride)
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> TensorRef<'_, T> {
        self.as_view().transpose(axis_a, axis_b)
    }

    pub fn transpose_last_dims(&self) -> TensorRef<'_, T> {
        self.as_view().transpose_last_dims()
    }

    /// Returns a view containing `nelts` elements along the axis `axis`, starting from `first_elt`.
    pub fn narrow(&self, axis: u32, first_elt: u32, nelts: u32) -> TensorRef<'_, T> {
        self.as_view().narrow(axis, first_elt, nelts)
    }

    /// Takes a view over the `i`-th column of `self`.
    pub fn column(&self, i: u32) -> TensorRef<'_, T> {
        self.as_view().column(i)
    }

    /// Returns a view containing `ncols` columns starting from `first_col`.
    pub fn columns(&self, first_col: u32, ncols: u32) -> TensorRef<'_, T> {
        self.as_view().columns(first_col, ncols)
    }

    /// Returns a view of the specified row.
    pub fn row(&self, i: u32) -> TensorRef<'_, T> {
        self.as_view().row(i)
    }

    /// Returns a view containing `nrows` rows starting from `first_row`.
    pub fn rows(&self, first_row: u32, nrows: u32) -> TensorRef<'_, T> {
        self.as_view().rows(first_row, nrows)
    }

    /*
     * Same methods, for for mutable views.
     */

    /// Reshapes this tensor to the specified shape.
    pub fn reshape_mut(&mut self, shape: &[u32]) -> TensorMut<'_, T> {
        self.as_view_mut().reshape(shape)
    }

    /// Reshapes this tensor using GGML's dimension ordering convention.
    pub fn reshape_ggml_mut(&mut self, shape: &[u32]) -> TensorMut<'_, T> {
        self.as_view_mut().reshape_ggml(shape)
    }

    /// Permutes the dimensions of this tensor according to the given permutation array.
    pub fn permute_mut(&mut self, permutations: [usize; 4]) -> TensorMut<'_, T> {
        self.as_view_mut().permute(permutations)
    }

    /// Permutes the dimensions according to GGML's dimension ordering convention.
    pub fn permute_ggml_mut(&mut self, permutations: [usize; 4]) -> TensorMut<'_, T> {
        self.as_view_mut().permute_ggml(permutations)
    }

    /// Creates a view of a sub-tensor with the specified offset, shape, and optional strides.
    pub fn view_mut(
        &mut self,
        offset: u32,
        shape: &[u32],
        stride: &[Option<u32>],
    ) -> TensorMut<'_, T> {
        self.as_view_mut().view(offset, shape, stride)
    }

    /// Creates a view using GGML's dimension ordering convention.
    pub fn view_ggml_mut(
        &mut self,
        offset: u32,
        shape: &[u32],
        stride: &[Option<u32>],
    ) -> TensorMut<'_, T> {
        self.as_view_mut().view_ggml(offset, shape, stride)
    }

    pub fn transpose_mut(&mut self, axis_a: usize, axis_b: usize) -> TensorMut<'_, T> {
        self.as_view_mut().transpose(axis_a, axis_b)
    }

    pub fn transpose_last_dims_mut(&mut self) -> TensorMut<'_, T> {
        self.as_view_mut().transpose_last_dims()
    }

    /// Returns a view containing `nelts` elements along the axis `axis`, starting from `first_elt`.
    pub fn narrow_mut(&mut self, axis: u32, first_elt: u32, nelts: u32) -> TensorMut<'_, T> {
        self.as_view_mut().narrow(axis, first_elt, nelts)
    }

    /// Takes a view over the `i`-th column of `self`.
    pub fn column_mut(&mut self, i: u32) -> TensorMut<'_, T> {
        self.as_view_mut().column(i)
    }

    /// Returns a view containing `ncols` columns starting from `first_col`.
    pub fn columns_mut(&mut self, first_col: u32, ncols: u32) -> TensorMut<'_, T> {
        self.as_view_mut().columns(first_col, ncols)
    }

    /// Returns a view of the specified row.
    pub fn row_mut(&mut self, i: u32) -> TensorMut<'_, T> {
        self.as_view_mut().row(i)
    }

    /// Returns a view containing `nrows` rows starting from `first_row`.
    pub fn rows_mut(&mut self, first_row: u32, nrows: u32) -> TensorMut<'_, T> {
        self.as_view_mut().rows(first_row, nrows)
    }
}

impl<T: DeviceValue + NoUninit> Tensor<T> {
    /// Allocates a new matrix on the gpu with uninitialized elements.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub fn matrix_uninit(
        backend: &GpuBackend,
        nrows: u32,
        ncols: u32,
        usage: BufferUsages,
    ) -> Result<Self, GpuBackendError>
    where
        T: DeviceValue,
    {
        TensorBuilder::matrix(nrows, ncols, usage).build_uninit(backend)
    }

    /// Allocates a new matrix on the gpu initialized from the given nalgebra matrix.
    ///
    /// Note that this is particularly slow since the nalgebra matrix will be transposed to fit
    /// a row-major storage format (nalgebra is column-major).
    pub fn matrix_from_na(
        backend: &GpuBackend,
        matrix: &DMatrix<T>,
        usage: BufferUsages,
    ) -> Result<Self, GpuBackendError>
    where
        T: DeviceValue + nalgebra::Scalar,
    {
        let matrix_tr = matrix.transpose();
        // NOTE: we use the original shape, but with the transposed data-buffer (which is the same
        //       as the un-transposed data but in row-major format).
        Self::matrix(
            backend,
            matrix.nrows() as u32,
            matrix.ncols() as u32,
            matrix_tr.as_slice(),
            usage,
        )
    }

    /// Allocates a new matrix on the gpu initialized from the `data` vector, in row-major order.
    pub fn matrix(
        backend: &GpuBackend,
        nrows: u32,
        ncols: u32,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self, GpuBackendError>
    where
        T: DeviceValue + nalgebra::Scalar,
    {
        assert_eq!(data.len(), nrows as usize * ncols as usize);
        TensorBuilder::matrix(nrows, ncols, usage).build_init(backend, data)
    }
}

// impl<T: DeviceValue, B: Backend> GpuMatrix<T, B> {
//     pub fn slice(&self, (i, j): (u32, u32), (nrows, ncols): (u32, u32)) -> GpuTensorView<'_, T, B> {
//         GpuTensorView {
//             view_shape: TensorLayout {
//                 size: [nrows, ncols, 1, 1],
//                 stride: [1, self.shape[0], self.shape[0] * self.shape[1], 1],
//             },
//             offset: i + j * nrows,
//             buffer: &self.buffer,
//         }
//     }
// }

impl<T: DeviceValue> Tensor<T> {
    /// Allocates an empty (len = 0) vector (rank = 1) with pre-allocated room for `capacity` elements.
    pub fn with_capacity(backend: &GpuBackend, capacity: u32, usage: BufferUsages) -> Result<Self, GpuBackendError>
    where T: DeviceValue + NoUninit {
        let mut t = Self::vector_uninit(backend, capacity, usage)?;
        t.shape = [0, 1, 1, 1];
        Ok(t)
    }

    /// Allocates a new uninitialized vector on the gpu for `len` elements of type `T`.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub fn vector_uninit(
        backend: &GpuBackend,
        len: u32,
        usage: BufferUsages,
    ) -> Result<Self, GpuBackendError>
    where
        T: DeviceValue + NoUninit,
    {
        TensorBuilder::vector(len, usage).build_uninit(backend)
    }

    /// Allocates a new vector on the gpu initialized from `vector`.
    pub fn vector(
        backend: &GpuBackend,
        vector: impl AsRef<[T]>,
        usage: BufferUsages,
    ) -> Result<Self, GpuBackendError>
    where
        T: DeviceValue + NoUninit,
    {
        let v = vector.as_ref();
        TensorBuilder::vector(v.len() as u32, usage).build_init(backend, v.as_ref())
    }
}

impl<T: DeviceValue> Tensor<T> {
    /// Allocates a new gpu storage buffer with a single uninitialized element.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub fn scalar_uninit(backend: &GpuBackend, usage: BufferUsages) -> Result<Self, GpuBackendError>
    where
        T: DeviceValue + NoUninit,
    {
        TensorBuilder::scalar(usage).build_uninit(backend)
    }

    /// Allocates a new gpu storage buffer with a single element initialized to `value`.
    pub fn scalar(
        backend: &GpuBackend,
        value: T,
        usage: BufferUsages,
    ) -> Result<Self, GpuBackendError>
    where
        T: DeviceValue + NoUninit,
    {
        TensorBuilder::scalar(usage).build_init(backend, &[value])
    }
}

impl<T: DeviceValue> khal::AsGpuSlice<T> for Tensor<T> {
    fn as_gpu_slice(&self) -> GpuBufferSlice<'_, T> {
        self.buffer_slice()
    }
}

impl<T: DeviceValue> khal::AsGpuSliceMut<T> for Tensor<T> {
    fn as_gpu_slice_mut(&mut self) -> GpuBufferSliceMut<'_, T> {
        self.buffer_slice_mut()
    }
}

impl<'a> From<&'a Tensor<[u32; 3]>> for khal::backend::DispatchGrid<'a, GpuBackend> {
    fn from(tensor: &'a Tensor<[u32; 3]>) -> Self {
        khal::backend::DispatchGrid::Indirect(tensor.buffer())
    }
}

impl<'b, T: DeviceValue> ShaderArgs<'b> for Tensor<T> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        dispatch: &mut GpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        self.buffer.write_arg(binding, dispatch)
    }
}

macro_rules! append_and_remove(
    ($append: ident, $shift_remove: ident, $TraitBound: ident, $capacity: ident, $copy_buffer_to_buffer: ident, $uninit_buffer: ident, $write_buffer: ident) => {
        /// Append the `data` elements at the end of this tensor.
        ///
        /// Panics if `self` isn’t a rank-1 tensor.
        ///
        /// If the underlying GPU buffer is too small to contain the extra elements, it is automatically
        /// resized. If a resize happens, the tensor's capacity is the next power of two sufficient
        /// to contain the appended data.
        // TODO: broadcast automatically to generalize to any tensor order.
        pub fn $append(&mut self, backend: &GpuBackend, data: &[T]) -> Result<(), GpuBackendError>
        where
            T: $TraitBound,
        {
            assert_eq!(self.rank(), 1, "Appending is curretnly only supported on rank-1 tensors.");
            let dim_to_grow = 0;
            let num_added = data.len();
            let curr_len = self.shape[dim_to_grow as usize];
            let new_len = curr_len + num_added as u32;

            let mut encoder = backend.begin_encoding();


            if new_len as u64 >= self.$capacity() {
                // We need to grow the buffer.
                let new_capacity = new_len.next_power_of_two();
                // SAFETY: will be initialized by the buffer init.
                let mut new_buffer = backend.$uninit_buffer(
                    new_capacity as usize,
                    self.buffer().usage() | BufferUsages::COPY_DST
                )?;

                encoder.$copy_buffer_to_buffer(
                    &self.buffer,
                    0,
                    &mut new_buffer,
                    0,
                    curr_len as usize,
                )?;
                self.buffer = new_buffer;
            }

            backend.$write_buffer(&mut self.buffer, curr_len as u64, data)?;
            backend.submit(encoder)?;
            self.shape[dim_to_grow as usize] = new_len;
            Ok(())
        }

        /// Removes a `range` of elements from this tensor if it is a vector, shifting back elements to
        /// fill the gap.
        ///
        /// This method doesn't change the tensor's capacity so the internal GPU buffer isn't resized.
        ///
        /// # Performance note
        ///
        /// This method is currently fairly expensive as it always involves the creation of a staging
        /// buffer for copying the data being moved. The staging buffer size is equal to the number of
        /// moved elements.
        ///
        /// # Panic
        ///
        /// Panics if `self` wasn't created with the `BufferUsages::COPY_SRC | BufferUsages::COPY_DST` flags.
        /// Panics if `self` isn’t a rank-1 tensor.
        /// Panics if the range is out of the bounds of `self`.
        ///
        /// # Return
        ///
        /// If the operation suceeded, returns the number of removed elements.
        // TODO: add a special case for targets capable of copying slices within the same buffer.
        // TODO: it would be worth benchmarking with doing the shift with a compute shader instead.
        pub fn $shift_remove(
            &mut self,
            backend: &GpuBackend,
            range: impl RangeBounds<usize>,
        ) -> Result<usize, GpuBackendError>
        where T: $TraitBound {
            assert_eq!(self.rank(), 1, "Remove is currently only supported on rank-1 tensors.");
            let dim_to_shrink = 0;
            let curr_len = self.shape[dim_to_shrink as usize] as usize;
            let range_start = match range.start_bound() {
                Bound::Included(i) => *i,
                Bound::Excluded(i) => *i + 1,
                Bound::Unbounded => 0,
            };
            let range_end = match range.end_bound() {
                Bound::Included(i) => *i + 1,
                Bound::Excluded(i) => *i,
                Bound::Unbounded => curr_len,
            };

            if range_end <= range_start {
                // The range to remove is empty.
                return Ok(0);
            }

            assert!(range_end <= curr_len, "Range index out of bounds.");
            let num_elements_to_move = curr_len - range_end;

            // NOTE: if `curr_end == range_end` we don't actually need to move any data, shrinking
            //       the shape is sufficient.
            if num_elements_to_move > 0 {
                // SAFETY: will be initialized with a buffer-to-buffer copy.
                let mut staging = backend.$uninit_buffer(
                    num_elements_to_move,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                )?;

                let mut encoder = backend.begin_encoding();
                encoder.$copy_buffer_to_buffer(
                    &self.buffer,
                    range_end,
                    &mut staging,
                    0,
                    num_elements_to_move,
                )?;
                encoder.$copy_buffer_to_buffer(
                    &staging,
                    0,
                    &mut self.buffer,
                    range_start,
                    num_elements_to_move,
                )?;
                backend.submit(encoder)?;
            }

            let num_removed = range_end - range_start;
            self.shape[dim_to_shrink as usize] -= num_removed as u32;
            Ok(num_removed)
        }
    }
);

impl<T: DeviceValue> Tensor<T> {
    append_and_remove!(
        append,
        shift_remove,
        NoUninit,
        capacity,
        copy_buffer_to_buffer,
        uninit_buffer,
        write_buffer
    );
}
