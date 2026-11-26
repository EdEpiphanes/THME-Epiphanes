# 0001. Dual GPU Backend: Embedded-PTX CUDA for NVIDIA and Pure-Rust `wgpu` Cross-Vendor

Date: 2026-11-26

## Status

Accepted

## Context

THME targets any OS and any GPU (NVIDIA, AMD, Intel, Apple Silicon) while keeping external runtime dependencies minimal and maintaining Pure-Rust purity where feasible. Benchmarking on SmolLM2-1.7B (RTX 4060 Ti 16 GB) showed:
- Pure-Rust `wgpu` (`F16` + `EXPERIMENTAL_COOPERATIVE_MATRIX`) reaches **3,691–4,242 tok/s**, limited by `wgpu`'s `8×8×8` `f32` cooperative-matrix constraint.
- NVIDIA `cuda.rs` with precompiled embedded `decoder.ptx` (`include_str!`) and optional dynamic-loaded `cublas64_12.dll` reaches **7,038 tok/s** (exact `CUBLAS_COMPUTE_32F`) and **9,506 tok/s** (opt-in `THME_CUDA_F16_ACC=1`), with zero `nvrtc` runtime dependency and automatic fallback to pure-PTX or `wgpu` when CUDA/cuBLAS libraries are absent.

## Decision

Keep the unified `GpuBackend` trait dispatching at runtime:
1. **Windows / Linux with NVIDIA GPU:** Try `CudaContext` (embedded `decoder.ptx`, `is_culib_present()` no-panic probing, `CUBLAS_COMPUTE_32F` exact default, `THME_CUDA_F16_ACC=1` opt-in).
2. **AMD, Intel, Apple Silicon, or missing CUDA driver:** Fall back automatically to Pure-Rust `wgpu` (Vulkan / DX12 / Metal), with CPU fallback when no GPU is present.
3. **Future cross-vendor upgrade:** Add a Pure-Rust `ash` Vulkan backend using `VK_KHR_cooperative_matrix` (`16×16×16 f16`) to close the Vulkan speed gap on AMD/Intel/NVIDIA without external C++ translation layers (ZLUDA/HIP/SYCL).

## Consequences

- Zero external `nvrtc` DLL requirement; missing `nvcuda` or `cublas` never panics.
- NVIDIA users get 24×–34× speedups immediately while AMD/Intel/Apple Silicon users retain Pure-Rust GPU acceleration.
- Two GPU backend implementations (`cuda.rs` and `wgpu` in `backend.rs`/`decoder.rs`) must preserve numerical parity across `test_gpu_batched`.
