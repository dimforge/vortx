# Changelog

## v0.5.0

### Added

- New `ml` feature enabling the `vortx::ml` module of machine-learning operators: activations (incl. `tanh`/`elu` backward), unary ops, softmax/log-softmax, RMS norm, layer norm, RoPE, SiLU, fused attention, axis reductions, concat, gather, select, conv2d/conv-transpose-2d, im2col, pooling, window partitioning, the Adam optimizer, PPO gradients, and quantized GEMV (Q4/Q5/Q8 and K-quants, with CPU-side `quantization` blocks). The optional `rand` feature adds `Distribution` impls for sampling random quantized blocks.
- `cuda-oxide` feature to compile the CUDA kernels with cuda-oxide instead of rust-cuda. It is commented out in the published crates: it needs khal from GitHub through `[patch]`.
- `vortx_shaders::linalg::{contiguous, op_assign}::MAX_NUM_THREADS` are now public.

### Changed

- Update to `khal`/`khal-std`/`khal-builder` 0.4.
- `gemm_tiled` accumulates with `Vec4` FMAs.

### Fixed

- `Contiguous` and `OpAssign` now clamp their dispatch size to `MAX_NUM_THREADS`, fixing tensors too large for the workgroup-count limit.

## v0.4.0

### Changed

- Update to `khal`/`khal-std`/`khal-builder` 0.3.
- Update to `wgpu` 30.0.


## v0.3.0

### Added

- Integer reductions: `sum`/`product`/`min`/`max` now have dedicated `u32` and `i32` kernels in addition to `f32`.
- `Tensor::with_capacity` to allocate an empty (len 0) rank-1 tensor with preallocated room for `capacity` elements.
- New `unsafe_remove_boundchecks` feature to compile shaders without bounds checks.

### Fixed

- Replaced `(a..b).step_by(c)` with a custom `StepRng` iterator in the reduce, `op_assign`, and `contiguous` kernels: the standard combinator introduces non-uniform control flow that breaks workgroup barriers when targeting WebGPU in the browser.

## v0.2.0

### Added

- New `metal` feature enabling the Metal GPU backend, with backend tests for `contiguous`, `gemm`, `op_assign`, and `reduce`. ([#2](https://github.com/dimforge/vortx/pull/2))

### Changed

- Update to `khal`/`khal-std`/`khal-builder` 0.2. ([#2](https://github.com/dimforge/vortx/pull/2))
- Update `nalgebra` to 0.35 and `glamx` to 0.3. ([#2](https://github.com/dimforge/vortx/pull/2))
- Replace the manual `any(target_arch = "spirv", target_arch = "nvptx64")` GPU-target guards with the `target_arch_is_gpu` cfg provided by `khal-std`, and delegate the shader crate's build script to `khal_std::setup_shader_crate_build()`. ([#2](https://github.com/dimforge/vortx/pull/2))
