# Changelog

## Unreleased

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
