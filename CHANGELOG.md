# Changelog

## v0.2.0

### Added

- New `metal` feature enabling the Metal GPU backend, with backend tests for `contiguous`, `gemm`, `op_assign`, and `reduce`. ([#2](https://github.com/dimforge/vortx/pull/2))

### Changed

- Update to `khal`/`khal-std`/`khal-builder` 0.2. ([#2](https://github.com/dimforge/vortx/pull/2))
- Update `nalgebra` to 0.35 and `glamx` to 0.3. ([#2](https://github.com/dimforge/vortx/pull/2))
- Replace the manual `any(target_arch = "spirv", target_arch = "nvptx64")` GPU-target guards with the `target_arch_is_gpu` cfg provided by `khal-std`, and delegate the shader crate's build script to `khal_std::setup_shader_crate_build()`. ([#2](https://github.com/dimforge/vortx/pull/2))
