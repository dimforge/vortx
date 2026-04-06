//! Tensor shape definition.

use khal::BufferUsages;
use khal::backend::{Backend, GpuBackend, GpuBackendError, GpuBuffer};
#[cfg(feature = "push_constants")]
use khal::backend::{Dispatch, GpuDispatch};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;

// Re-export Shape from shaders as GpuTensorLayout for compatibility with generated ShaderArgs.
// Shape and the old GpuTensorLayout have identical memory layouts (8 u32s in the same order).
pub use vortx_shaders::linalg::Shape as GpuTensorLayout;

/// GGML dimension index mapping: converts between GGML and vortx dimension ordering.
pub const GGML_IDS: [usize; 4] = [1, 0, 2, 3];
/// GGML dimension index mapping (u32 version).
pub const GGML_IDS_U32: [u32; 4] = [1, 0, 2, 3];

impl From<TensorLayout> for GpuTensorLayout {
    fn from(shape: TensorLayout) -> Self {
        let mut size = shape.size;
        let mut stride = shape.stride;

        // Ensure we don't send garbage data on dimensions
        // beyond the tensor rank.
        for k in shape.rank..4 {
            size[k as usize] = 1;
            if shape.rank > 0 {
                stride[k as usize] = stride[shape.rank as usize - 1];
            } else {
                stride[k as usize] = 1;
            }
        }

        Self {
            n: size[0],
            c: size[1],
            h: size[2],
            w: size[3],
            n_stride: stride[0],
            c_stride: stride[1],
            h_stride: stride[2],
            w_stride: stride[3],
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
/// The shape of a matrix view over a GPU tensor.
pub struct TensorLayout {
    /// The tensor view’s number of rows, columns, matrices, and 3-tensors.
    pub size: [u32; 4],
    /// The stride along each dimension.
    pub stride: [u32; 4],
    /// The tensor rank.
    pub rank: u32,
    /// The starting index in the underlying buffer.
    pub offset: u32,
}

impl TensorLayout {
    pub fn dims(&self) -> &[u32] {
        &self.size[..self.rank as usize]
    }

    pub fn stride(&self) -> &[u32] {
        &self.stride[..self.rank as usize]
    }

    pub fn into_gpu(self) -> GpuTensorLayout {
        self.into()
    }

    pub(crate) fn append_ones(size: &[u32]) -> ([u32; 4], u32) {
        assert!(size.len() <= 4, "tensors of ranks > 4 not supported");
        let mut result = [1; 4];
        result[..size.len()].copy_from_slice(size);
        (result, size.len() as u32)
    }

    /// Creates a contiguous view shape with the given size and ordering.
    ///
    /// The `size.len()` is the tensor rank.
    pub fn contiguous(size: &[u32]) -> Self {
        let (size, rank) = Self::append_ones(size);
        let mut stride = [1; 4];
        let mut curr_stride = 1;
        for k in (0..rank as usize).rev() {
            stride[k] = curr_stride;
            curr_stride *= size[k];
        }
        Self {
            size,
            stride,
            rank,
            offset: 0,
        }
    }

    pub fn contiguous_strides<const RANK: usize>(size: [u32; RANK]) -> [u32; RANK] {
        let cont = Self::contiguous(&size);
        let mut result = [0; RANK];
        result.copy_from_slice(&cont.stride[..RANK]);
        result
    }

    pub fn reshape(&self, shape: &[u32]) -> Self {
        self.view(0, shape, &[])
    }

    /// Adds enough dimensions of size 1 at the end of this tensor so it has the desired rank.
    ///
    /// Returns `None` if `rank > 4 || rank < self.rank`.
    pub fn unsqueeze_to_rank(mut self, rank: u32) -> Option<Self> {
        if self.rank > 4 || rank < self.rank {
            return None;
        }

        if self.rank == rank {
            return Some(self);
        }

        if self.rank == 0 {
            Some(Self {
                size: [1; 4],
                stride: [1; 4],
                rank,
                offset: self.offset,
            })
        } else {
            let default_stride =
                self.stride[self.rank as usize - 1] * self.size[self.rank as usize - 1];
            for k in self.rank..4 {
                self.size[k as usize] = 1;
                self.stride[k as usize] = default_stride;
            }
            self.rank = rank;
            Some(self)
        }
    }

    /// Insert dimensions of size 1 on the left of the tensor until
    /// it reaches the maximum rank (4).
    pub fn canonicalize(self) -> Self {
        if self.rank == 4 {
            return self;
        }
        let rank_diff = (4 - self.rank) as usize;
        let mut size = [1; 4];
        size[rank_diff..].copy_from_slice(self.dims());
        let mut result = Self::contiguous(&size);
        result.stride[rank_diff..].copy_from_slice(self.stride());
        result
    }

    /// Inserts a dimension of size 1 at the given `axis` of this tensor.
    ///
    /// Return `None` if unsqueezing is not possible either because it would overflow the
    /// maximum tensor rank supported (4), or if the axis is > to the current tensor rank.
    pub fn unsqueeze(mut self, axis: u32) -> Option<Self> {
        if self.rank >= 4 || axis > self.rank {
            return None;
        }

        for i in (axis as usize..3).rev() {
            self.size[i + 1] = self.size[i];
            self.stride[i + 1] = self.stride[i];
        }
        self.size[axis as usize] = 1;
        // Stride value selected to keep the `is_contiguous` check happy.
        self.stride[axis as usize] = self.stride[axis as usize + 1];
        self.rank += 1;
        Some(self)
    }

    pub fn squeeze_axis(mut self, axis: u32) -> Self {
        assert!(axis < self.rank, "axis out of bounds");
        assert_eq!(
            self.size[axis as usize], 1,
            "can only squeeze an axis with length 1"
        );

        for k in axis..self.rank - 1 {
            self.size[k as usize] = self.size[k as usize + 1];
            self.stride[k as usize] = self.stride[k as usize + 1];
        }
        self.rank -= 1;

        // Doesn’t really mater but good to know it contains sane values.
        self.size[self.rank as usize] = 1;
        if self.rank > 0 {
            self.stride[self.rank as usize] = self.size[..self.rank as usize - 1]
                .iter()
                .copied()
                .product();
        }

        self
    }

    pub fn squeeze(mut self) -> Self {
        let mut new_rank = 0;
        for k in 0..self.rank as usize {
            if self.size[k] != 1 {
                self.size[new_rank] = self.size[k];
                self.stride[new_rank] = self.stride[k];
                new_rank += 1;
            }
        }

        if new_rank == 0 {
            Self {
                size: [1; 4],
                stride: [1; 4],
                rank: 0,
                offset: self.offset,
            }
        } else {
            for k in new_rank..4 {
                self.size[k] = 1;
                // Not very important, but we set strides that maximizes changes of
                // the tensor being contiguous.
                self.stride[k] = self.stride[new_rank - 1] * self.size[new_rank - 1];
            }

            self.rank = new_rank as u32;
            self
        }
    }

    pub fn index(self, i: u32) -> Self {
        self.narrow(0, i, 1).squeeze_axis(0)
    }

    /// Returns a transposed view of this shape.
    #[must_use]
    pub fn transpose(mut self, axis_a: usize, axis_b: usize) -> Self {
        assert!(
            axis_a < self.rank as usize,
            "transpose axis index out of bounds: {} >= {}",
            axis_a,
            self.rank
        );
        assert!(
            axis_b < self.rank as usize,
            "transpose axis index out of bounds: {} >= {}",
            axis_b,
            self.rank
        );
        self.stride.swap(axis_a, axis_b);
        self.size.swap(axis_a, axis_b);
        self
    }

    pub fn transpose_last_dims(self) -> Self {
        assert!(self.rank >= 2);
        self.transpose(self.rank as usize - 2, self.rank as usize - 1)
    }

    /// Permutes the dimensions according to GGML's dimension ordering convention.
    pub fn permute_ggml(&self, mut permutations: [usize; 4]) -> Self {
        permutations.swap(0, 1);
        self.permute(permutations.map(|i| GGML_IDS[i]))
    }

    /// Permutes the dimensions according to the given permutation array.
    pub fn permute(&self, permutations: [usize; 4]) -> Self {
        // Check all the permutation indices are valid and without
        // duplicate.
        assert_ne!(
            permutations[0], permutations[1],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[0], permutations[2],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[0], permutations[3],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[1], permutations[2],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[1], permutations[3],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[2], permutations[3],
            "Permutation indices must not overlap."
        );

        #[allow(clippy::needless_range_loop)]
        for k in 0..self.rank as usize {
            assert!(
                permutations[k] < self.rank as usize,
                "permutation index {} exceeds this matrix rank {}",
                permutations[k],
                self.rank
            );
        }

        #[allow(clippy::needless_range_loop)]
        for k in self.rank as usize..4 {
            assert_eq!(
                permutations[k], k,
                "Indices exceeding the tensor rank {} must be identity",
                self.rank
            );
        }

        let mut size = [0; 4];
        let mut stride = [0; 4];

        for k in 0..4 {
            size[permutations[k]] = self.size[k];
            stride[permutations[k]] = self.stride[k];
        }

        Self {
            size,
            stride,
            rank: self.rank,
            offset: self.offset,
        }
    }

    /// Checks if a tensor with this shape is contiguous in memory (in row-major order).
    pub fn is_contiguous(&self) -> bool {
        let mut stride = 1;
        for i in (0..self.rank as usize).rev() {
            if self.stride[i] != 0 && self.stride[i] != stride {
                return false;
            }

            stride *= self.size[i];
        }

        true
    }

    /// Broadcast `self` and `other` to make their dimensions compatible, if possible.
    pub fn broadcast(mut self, mut other: Self) -> Option<(Self, Self)> {
        // To simplify the code, ensure the smallest rank is in `self`.
        let flip = self.rank > other.rank;

        if flip {
            std::mem::swap(&mut other, &mut self);
        }

        // Equalize ranks.
        let rank_diff = (other.rank - self.rank) as usize;
        for k in (0..self.rank as usize).rev() {
            self.size[k + rank_diff] = self.size[k];
            self.stride[k + rank_diff] = self.stride[k];
        }
        for k in 0..rank_diff {
            self.size[k] = 1;
            // NOTE: strides will be adjusted after.
        }
        self.rank = other.rank;

        // Adjust strides.
        for k in 0..self.rank as usize {
            if self.size[k] == 1 {
                self.stride[k] = 0;
            }
            if other.size[k] == 1 {
                other.stride[k] = 0;
            }
        }

        // Check compatibility.
        for k in 0..self.rank as usize {
            if self.size[k] != 1 && other.size[k] != 1 && self.size[k] != other.size[k] {
                return None;
            }
        }

        // Return results (flip is needed).
        if flip {
            Some((other, self))
        } else {
            Some((self, other))
        }
    }

    /// Same as [`Self::broadcast`] except that it also return `None` if the result of an assignment
    /// from `source` to `self` would grow the dimension of `self`.
    pub fn broadcast_assign(self, source: Self) -> Option<(Self, Self)> {
        let (target, source) = self.broadcast(source)?;
        for k in 0..target.rank as usize {
            if target.size[k] < source.size[k] {
                // TODO: return an Error instead of an option so that the user can differentiate
                //       between incorrect broadcast and the fact the operation would grow `self`.
                return None;
            }
        }

        Some((target, source))
    }

    /// Creates a view with the specified shape and strides within this shape.
    pub fn view(&self, offset: u32, shape: &[u32], stride: &[Option<u32>]) -> Self {
        assert!(shape.len() <= 4);
        assert!(stride.len() <= shape.len());

        if !self.is_contiguous() {
            panic!("Cannot take a view of a non-contiguous tensor: {:?}.", self);
        };

        let available_elts = self.size.iter().product::<u32>();
        let needed_elts = shape.iter().product::<u32>() + offset;
        assert!(
            needed_elts <= available_elts,
            "Source tensor is too small for reshaping. Expected at least {needed_elts} elements (shape: {shape:?}, offset: {offset}), found {available_elts} (shape: {:?})",
            self
        );

        let new_rank = shape.len();

        let mut size = [1; 4];
        size[..new_rank].copy_from_slice(shape);

        let mut new_stride = [0; 4];
        let mut curr_stride = 1;
        for k in (0..new_rank).rev() {
            new_stride[k] = stride.get(k).copied().flatten().unwrap_or(curr_stride);
            curr_stride = new_stride[k] * size[k];
        }

        Self {
            size,
            stride: new_stride,
            rank: new_rank as u32,
            offset: self.offset + offset,
        }
    }

    pub fn narrow(&self, axis: u32, first_elt: u32, new_nelts: u32) -> Self {
        assert!(axis < self.rank, "Axis index out of bounds.");

        let nelts = self.size[axis as usize];
        let mut new_size = self.size;
        new_size[axis as usize] = new_nelts;

        assert!(
            first_elt + new_nelts <= nelts,
            "{} + {} <= {} (shape: {:?})",
            first_elt,
            new_nelts,
            nelts,
            self
        );
        TensorLayout {
            size: new_size,
            stride: self.stride,
            rank: self.rank,
            offset: self.offset + self.stride[axis as usize] * first_elt,
        }
    }

    /// Returns a view of the `matrix_id`-th matrix in this tensor.
    pub fn matrix(&self, matrix_id: u32) -> Self {
        self.narrow(2, matrix_id, 1)
    }

    /// Returns a view containing `new_ncols` columns starting from `first_col`.
    pub fn columns(&self, first_col: u32, new_ncols: u32) -> Self {
        self.narrow(1, first_col, new_ncols)
    }

    /// Returns a view of the specified column.
    pub fn column(&self, col: u32) -> Self {
        self.columns(col, 1)
    }

    /// Returns a view containing `new_nrows` rows starting from `first_row`.
    pub fn rows(&self, first_row: u32, new_nrows: u32) -> Self {
        self.narrow(0, first_row, new_nrows)
    }

    /// Returns a view of the specified row.
    pub fn row(&self, row: u32) -> Self {
        self.rows(row, 1)
    }

    /// Converts the shape `self` for a buffer `&[f32]` to a buffer `&[vec4f]`.
    pub fn f32_to_vec4(self) -> Self {
        let dim = (self.rank.max(1) - 1) as usize;

        assert_eq!(
            self.stride[dim], 1,
            "Cannot convert from f32 to vec4 with a stride[{dim}] of {} != 1",
            self.stride[dim]
        );
        assert_eq!(
            self.size[dim] % 4,
            0,
            "Matrix row count no properly aligned."
        );

        let new_stride = self.stride.map(|s| {
            assert!(s == 1 || s % 4 == 0);
            s.div_ceil(4)
        });
        let mut new_size = self.size;
        new_size[dim] /= 4;

        Self {
            size: new_size,
            stride: new_stride,
            rank: self.rank,
            offset: self.offset, // div_ceil 4 ?
        }
    }

    /// Checks if this shape contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the total number of elements in this shape.
    pub fn len(&self) -> u64 {
        self.size[..self.rank as usize]
            .iter()
            .map(|e| *e as u64)
            .product::<u64>()
            .max(1)
    }
}

/// A map between a `TensorLayout` and a uniform storage `Buffer` containing its value on the gpu.
///
/// When the `push_constants` feature is enabled, shapes can be passed directly to shaders via
/// push constants instead of uniform buffers, which is more efficient for small, frequently
/// changing data like tensor shapes.
#[derive(Default)]
pub struct TensorLayoutBuffers {
    buffers: HashMap<TensorLayout, GpuBuffer<GpuTensorLayout>>,
    tmp_buffers: HashMap<TensorLayout, GpuBuffer<GpuTensorLayout>>,
    // TODO: is this still needed?
    recycled: Mutex<Vec<GpuBuffer<GpuTensorLayout>>>,
}

impl TensorLayoutBuffers {
    /// Creates an empty map.
    pub fn new(_backend: &GpuBackend) -> Self {
        Self {
            buffers: HashMap::new(),
            tmp_buffers: HashMap::new(),
            recycled: Mutex::new(vec![]),
        }
    }

    /// Clears temporary shape buffers and recycles them for reuse.
    pub fn clear_tmp(&mut self) {
        // TODO PERF: apparently, not recycling the buffer is actually faster.
        //            (in other words, re-creating the shape buffer is faster than
        //             write_buffer).
        self.tmp_buffers.clear();
        // let mut recycled = self.recycled.lock().unwrap();
        // recycled.extend(self.tmp_buffers.drain().map(|(_, buf)| buf));
    }

    /// Stores a temporary shape buffer for the given shape, creating one if needed.
    pub fn put_tmp(
        &mut self,
        backend: &GpuBackend,
        shape: TensorLayout,
    ) -> Result<(), GpuBackendError> {
        if self.contains(shape) {
            return Ok(());
        }

        let mut recycled = self.recycled.lock().unwrap();
        let buffer = if let Some(mut buffer) = recycled.pop() {
            backend.write_buffer(&mut buffer, 0, &[shape.into_gpu()])?;
            buffer
        } else {
            // println!("Couldn't find recycling for {:?}", shape);
            drop(recycled);
            Self::make_buffer(
                backend,
                shape,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::STORAGE,
            )?
        };

        self.tmp_buffers.insert(shape, buffer);
        Ok(())
    }

    fn make_buffer(
        backend: &GpuBackend,
        shape: TensorLayout,
        usage: BufferUsages,
    ) -> Result<GpuBuffer<GpuTensorLayout>, GpuBackendError> {
        // println!("Making buffer for shape: {:?}", shape);
        backend.init_buffer(&[shape.into_gpu()], usage | BufferUsages::STORAGE)
    }

    /// Checks if a buffer for the given shape exists (permanent or temporary).
    pub fn contains(&self, shape: TensorLayout) -> bool {
        self.buffers.contains_key(&shape) || self.tmp_buffers.contains_key(&shape)
    }

    /// Inserts or retrieves a mutable buffer for the given shape.
    pub fn insert(
        &mut self,
        backend: &GpuBackend,
        shape: TensorLayout,
    ) -> Result<&mut GpuBuffer<GpuTensorLayout>, GpuBackendError> {
        if let Some(buffer) = self.tmp_buffers.get_mut(&shape) {
            return Ok(buffer);
        }

        let buf = match self.buffers.entry(shape) {
            Entry::Vacant(e) => e.insert(Self::make_buffer(
                backend,
                shape,
                BufferUsages::UNIFORM | BufferUsages::STORAGE,
            )?),
            Entry::Occupied(e) => e.into_mut(),
        };
        Ok(buf)
    }

    /// Gets the gpu uniform storage `Buffer` containing the value of `shape`.
    ///
    /// Returns `None` if it doesn't exist.
    pub fn get(&self, shape: TensorLayout) -> Option<&GpuBuffer<GpuTensorLayout>> {
        self.tmp_buffers
            .get(&shape)
            .or_else(|| self.buffers.get(&shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_same_shape() {
        let a = TensorLayout::contiguous(&[3, 4]);
        let b = TensorLayout::contiguous(&[3, 4]);
        let (a_broadcast, b_broadcast) = a.broadcast(b).expect("broadcast should succeed");

        assert_eq!(a_broadcast, a);
        assert_eq!(b_broadcast, b);
    }

    #[test]
    fn broadcast_scalar_to_matrix() {
        let scalar = TensorLayout::contiguous(&[1]);
        let matrix = TensorLayout::contiguous(&[3, 4]);
        let (s_broadcast, m_broadcast) =
            scalar.broadcast(matrix).expect("broadcast should succeed");

        assert_eq!(s_broadcast.size, [1, 1, 1, 1]);
        assert_eq!(m_broadcast.size, [3, 4, 1, 1]);
        assert_eq!(s_broadcast.rank, 2);
        assert_eq!(m_broadcast.rank, 2);
        // Scalar should have zero stride for broadcasting.
        assert_eq!(s_broadcast.stride[0], 0);
        assert_eq!(s_broadcast.stride[1], 0);
    }

    #[test]
    fn broadcast_vector_to_matrix() {
        // Column vector [3, 1] broadcast with [3, 4].
        let col_vec = TensorLayout::contiguous(&[3, 1]);
        let matrix = TensorLayout::contiguous(&[3, 4]);
        let (v_broadcast, m_broadcast) =
            col_vec.broadcast(matrix).expect("broadcast should succeed");

        assert_eq!(v_broadcast.size, [3, 1, 1, 1]);
        assert_eq!(m_broadcast.size, [3, 4, 1, 1]);
        // Stride along dimension with size 1 should be 0.
        assert_eq!(v_broadcast.stride[1], 0);
    }

    #[test]
    fn broadcast_row_vector_to_matrix() {
        // Row vector [1, 4] broadcast with [3, 4].
        let row_vec = TensorLayout::contiguous(&[1, 4]);
        let matrix = TensorLayout::contiguous(&[3, 4]);
        let (v_broadcast, m_broadcast) =
            row_vec.broadcast(matrix).expect("broadcast should succeed");

        assert_eq!(v_broadcast.size, [1, 4, 1, 1]);
        assert_eq!(m_broadcast.size, [3, 4, 1, 1]);
        // Stride along dimension with size 1 should be 0.
        assert_eq!(v_broadcast.stride[0], 0);
    }

    #[test]
    fn broadcast_different_ranks() {
        // [4] broadcast with [3, 4, 2].
        let vec = TensorLayout::contiguous(&[2]);
        let tensor = TensorLayout::contiguous(&[3, 4, 2]);
        let (v_broadcast, t_broadcast) = vec.broadcast(tensor).expect("broadcast should succeed");

        // Vec should be promoted to rank 3.
        assert_eq!(v_broadcast.rank, 3);
        assert_eq!(t_broadcast.rank, 3);
        // Sizes should be [1, 1, 4] for the vector (prepended with 1s).
        assert_eq!(v_broadcast.size[0], 1);
        assert_eq!(v_broadcast.size[1], 1);
        assert_eq!(v_broadcast.size[2], 2);
        // Strides for prepended dimensions should be 0.
        assert_eq!(v_broadcast.stride[0], 0);
        assert_eq!(v_broadcast.stride[1], 0);
    }

    #[test]
    fn broadcast_incompatible_shapes() {
        let a = TensorLayout::contiguous(&[3, 4]);
        let b = TensorLayout::contiguous(&[8]);
        assert!(a.broadcast(b).is_none(), "incompatible shapes should fail");

        let a = TensorLayout::contiguous(&[3, 4]);
        let b = TensorLayout::contiguous(&[3]);
        assert!(a.broadcast(b).is_none(), "incompatible shapes should fail");
    }

    #[test]
    fn broadcast_incompatible_inner_dim() {
        let a = TensorLayout::contiguous(&[3, 4]);
        let b = TensorLayout::contiguous(&[3, 5]);
        assert!(a.broadcast(b).is_none(), "incompatible shapes should fail");
    }

    #[test]
    fn broadcast_3d_tensors() {
        // [2, 1, 4] broadcast with [2, 3, 4].
        let a = TensorLayout::contiguous(&[2, 1, 4]);
        let b = TensorLayout::contiguous(&[2, 3, 4]);
        let (a_broadcast, b_broadcast) = a.broadcast(b).expect("broadcast should succeed");

        assert_eq!(a_broadcast.size, a_broadcast.size);
        assert_eq!(b_broadcast.size, b_broadcast.size);
        // Stride for dimension with size 1 should be 0.
        assert_eq!(a_broadcast.stride[1], 0);
    }

    #[test]
    fn broadcast_preserves_order() {
        // Ensure the returned shapes are in the same order as the inputs.
        let a = TensorLayout::contiguous(&[3, 4, 5]);
        let b = TensorLayout::contiguous(&[5]);

        let (a_broadcast, b_broadcast) = a.broadcast(b).expect("broadcast should succeed");
        // `a` was the first argument and had larger rank, so a_broadcast should have same size as original a.
        assert_eq!(a_broadcast, a);
        // `b` was the second argument and had smaller rank.
        assert_eq!(b_broadcast.rank, a.rank);
        assert_eq!(b_broadcast.size[0], 1);
        assert_eq!(b_broadcast.size[1], 1);
        assert_eq!(b_broadcast.size[2], 5);
        assert_eq!(b_broadcast.stride[0], 0);
        assert_eq!(b_broadcast.stride[1], 0);
        assert_eq!(b_broadcast.stride[2], 1);
    }

    #[test]
    fn broadcast_4d_tensors() {
        let a = TensorLayout::contiguous(&[2, 3, 1, 5]);
        let b = TensorLayout::contiguous(&[2, 1, 4, 5]);
        let (a_broadcast, b_broadcast) = a.broadcast(b).expect("broadcast should succeed");

        assert_eq!(a_broadcast.size, a.size);
        assert_eq!(b_broadcast.size, b.size);
        assert_eq!(a_broadcast.rank, 4);
        assert_eq!(b_broadcast.rank, 4);
        // Strides for size-1 dimensions should be 0.
        assert_eq!(a_broadcast.stride[2], 0);
        assert_eq!(b_broadcast.stride[1], 0);
    }
}
