//! Shape representation for tensors.

use glamx::UVec4;

/// Shape descriptor for a 4D tensor.
///
/// The tensor dimension is specified in the NCHW format
/// (Samples, Channels, Height, Width), where height is the row count, and width the column count.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Shape {
    /// Number of rows in each matrix of the tensor.
    pub n: u32,
    /// Number of columns in each matrix of the tensor.
    pub c: u32,
    /// Number of matrices in the tensor.
    pub h: u32,
    /// Number of cubes (3-tensors) in the tensor.
    pub w: u32,
    /// Number of elements between two successive rows.
    pub n_stride: u32,
    /// Number of elements between two successive columns.
    pub c_stride: u32,
    /// Number of elements between two successive matrices.
    pub h_stride: u32,
    /// Number of elements between two successive cubes (3-tensors).
    pub w_stride: u32,
}

impl Shape {
    /// Whether the tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total number of elements in the tensor.
    #[inline]
    pub fn len(&self) -> u32 {
        self.n * self.c * self.h * self.w
    }

    /// Index of the element at row `i`, column `j` of the matrix `k` of the cube `l` in this tensor.
    #[inline]
    pub fn it(&self, n: u32, c: u32, h: u32, w: u32) -> u32 {
        n * self.n_stride + c * self.c_stride + h * self.h_stride + w * self.w_stride
    }

    /// Index of the element from a 4D index (in NCHW order).
    #[inline]
    pub fn it_vec(&self, id: UVec4) -> u32 {
        id.x * self.n_stride + id.y * self.c_stride + id.z * self.h_stride + id.w * self.w_stride
    }

    // /// Indexes the tensor, but overflowing indices wrap around the dimension.
    // ///
    // /// For example if the row index is 4 and the number of rows is 3, the row index
    // /// effectively used by this function is `4 % 3 = 1`.
    // #[inline]
    // pub fn it_wrapping(&self, n: u32, c: u32, h: u32, w: u32) -> u32 {
    //     self.it(
    //         n % self.n,
    //         c % self.c,
    //         h % self.h,
    //         w % self.w,
    //     )
    // }

    /// Indexes the tensor with wrapping from a 4D index.
    #[inline]
    pub fn it_repeating_vec(&self, id: UVec4) -> u32 {
        self.it_vec(id % UVec4::new(self.n, self.c, self.h, self.w))
    }

    /// Decomposes a linear index `i` into a 4D tensor index.
    #[inline]
    pub fn decompose(&self, i: u32) -> UVec4 {
        let i3 = i.checked_div(self.c * self.h * self.w).unwrap_or(0);
        let i3_offset = i3 * (self.c * self.h * self.w);
        let i2 = (i - i3_offset).checked_div(self.h * self.w).unwrap_or(0);
        let i2_offset = i2 * (self.h * self.w);
        let i1 = (i - i3_offset - i2_offset).checked_div(self.w).unwrap_or(0);
        let i0 = i - i3_offset - i2_offset - i1 * self.w;
        UVec4::new(i3, i2, i1, i0)
    }
}

/// Division rounding up, for u32 divided by 4.
#[inline]
#[allow(clippy::manual_div_ceil)]
pub fn div_ceil4(a: u32) -> u32 {
    (a + 3) / 4
}

// Push constant wrapper structs for when the `push_constants` feature is enabled.
// These provide efficient data transfer for small, frequently-changing data like shapes.

/// Push constants containing two shapes (for binary operations).
#[cfg(feature = "push_constants")]
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Shapes2 {
    /// First shape (typically output or left operand).
    pub shape_a: Shape,
    /// Second shape (typically input or right operand).
    pub shape_b: Shape,
}

/// Push constants containing three shapes (for ternary operations like GEMM).
#[cfg(feature = "push_constants")]
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Shapes3 {
    /// Output shape.
    pub shape_out: Shape,
    /// Left operand shape.
    pub shape_lhs: Shape,
    /// Right operand shape.
    pub shape_rhs: Shape,
}

/// Push constants containing a single shape.
#[cfg(feature = "push_constants")]
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Shapes1 {
    /// The shape.
    pub shape: Shape,
}
