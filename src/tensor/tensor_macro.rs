macro_rules! impl_layout_modifiers {
    () => {
        /// Get the i-th element along axis 0.
        /// The rank of the result is lowered by 1.
        pub fn index(self, i: u32) -> Self {
            let l = self.layout.index(i);
            self.with_layout(l)
        }

        /// Returns a transposed view of this tensor.
        pub fn transpose(self, axis_a: usize, axis_b: usize) -> Self {
            let l = self.layout.transpose(axis_a, axis_b);
            self.with_layout(l)
        }

        pub fn transpose_last_dims(self) -> Self {
            let l = self.layout.transpose_last_dims();
            self.with_layout(l)
        }

        /// Permutes the dimensions of this view according to the given permutation array.
        pub fn permute(self, permutations: [usize; 4]) -> Self {
            let l = self.layout.permute(permutations);
            self.with_layout(l)
        }

        /// Permutes the dimensions according to GGML's dimension ordering convention.
        pub fn permute_ggml(self, permutations: [usize; 4]) -> Self {
            let l = self.layout.permute_ggml(permutations);
            self.with_layout(l)
        }

        pub fn canonicalize(self) -> Self {
            let l = self.layout.canonicalize();
            self.with_layout(l)
        }

        /// Inserts a dimension of size 1 at the given `axis` of this tensor.
        pub fn unsqueeze(self, axis: u32) -> Self {
            if let Some(new_layout) = self.layout.unsqueeze(axis) {
                self.with_layout(new_layout)
            } else {
                panic!("Not enough rank available for unsqueezing.");
            }
        }

        /// Reshapes this view to the specified shape, preserving the matrix ordering.
        pub fn reshape(self, shape: &[u32]) -> Self {
            let l = self.layout.reshape(shape);
            self.with_layout(l)
        }

        /// Reshapes this view using GGML's dimension ordering convention.
        pub fn reshape_ggml(self, _shape: &[u32]) -> Self {
            todo!()
            // shape.swap(0, 1);
            // self.reshape(shape)
        }

        /// Creates a view of a sub-tensor with the specified offset, shape, and optional strides.
        pub fn view(self, offset: u32, shape: &[u32], stride: &[Option<u32>]) -> Self {
            let l = self.layout.view(offset, shape, stride);
            self.with_layout(l)
        }

        /// Creates a view using GGML's dimension ordering convention.
        pub fn view_ggml(self, _offset: u32, _shape: &[u32], _stride: &[Option<u32>]) -> Self {
            todo!()
            // shape.swap(0, 1);
            // stride.swap(0, 1);
            // self.view(offset, shape, stride)
        }

        /// Returns a view containing `new_nelts` elements starting from index `first_elt` along the given `axis`.
        ///
        /// This does not change the rank of the tensor.
        pub fn narrow(self, axis: u32, first_elt: u32, new_nelts: u32) -> Self {
            let l = self.layout.narrow(axis, first_elt, new_nelts);
            self.with_layout(l)
        }

        /// Returns a view of the `matrix_id`-th matrix in this tensor.
        pub fn matrix(self, matrix_id: u32) -> Self {
            let l = self.layout.matrix(matrix_id);
            self.with_layout(l)
        }

        /// Returns a view containing `new_ncols` columns starting from `first_col`.
        pub fn columns(self, first_col: u32, new_ncols: u32) -> Self {
            let l = self.layout.columns(first_col, new_ncols);
            self.with_layout(l)
        }

        /// Returns a view of the specified column.
        pub fn column(self, col: u32) -> Self {
            let l = self.layout.column(col);
            self.with_layout(l)
        }

        /// Returns a view containing `new_nrows` rows starting from `first_row`.
        pub fn rows(self, first_row: u32, new_nrows: u32) -> Self {
            let l = self.layout.rows(first_row, new_nrows);
            self.with_layout(l)
        }

        /// Returns a view of the specified row.
        pub fn row(self, row: u32) -> Self {
            let l = self.layout.row(row);
            self.with_layout(l)
        }

        /// Removes the given axis if its dimension is 1.
        ///
        /// Panics if the axis doesn’t have a dimension of 1.
        pub fn squeeze_axis(self, axis: u32) -> Self {
            let l = self.layout.squeeze_axis(axis);
            self.with_layout(l)
        }

        // TODO: rename this to `squeeze_all` and then rename `squeeze_axis` to `squeeze`?
        pub fn squeeze(self) -> Self {
            let l = self.layout.squeeze();
            self.with_layout(l)
        }
    };
}
pub(crate) use impl_layout_modifiers;
