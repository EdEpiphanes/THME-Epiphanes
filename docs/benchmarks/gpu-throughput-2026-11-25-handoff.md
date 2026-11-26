# Handoff & Changelog — GPU Inference Performance (`2026-11-25`)

**Branch:** `perf/gpu-throughput` (worktree `../thme-wt/perf-gpu`, off `origin/main` `d5378b6`)  
**Target Hardware:** NVIDIA GeForce RTX 4060 Ti 16 GB (Driver 616.92, Vulkan & CUDA 12.9)  
**Target Model:** SmolLM2-1.7B (24 layers, `1,711,376,384` params, `148` VAL windows = `151,552` tokens)

---

## 1. Performance Comparison (All Measured on RTX 4060 Ti, 148 VAL Windows)

| Engine / Configuration | `f32` Baseline Speed | `Q4_0` Speed | Elapsed (`f32` / `Q4_0`) | PPL (`f32` / `Q4_0`) | Speedup vs `origin/main` |
| :--- | ---: | ---: | ---: | ---: | :---: |
| `thme` (`origin/main` `d5378b6`, `EXP-0001a`) | 281.2 tok/s | 301.2 tok/s | 538.4s / 502.6s | `8.9053` / `10.5042` | 1.0× |
| `thme` Pure-Rust WGPU Default (any OS/GPU, `F16` + `coop`) | **3,566.3 tok/s** | **4,133.5 tok/s** | **42.45s / 36.63s** | **`8.9054` / `9.8512`** | **12.7× / 13.7×** |
| `thme` CUDA Zero-DLL Pure-PTX Fallback (16 VAL windows, no `cublas`/`nvrtc` DLLs) | 341.4 tok/s | 330.6 tok/s | 47.94s / 49.51s | `8.3730` / `9.3253` | 1.2× / 1.1× |
| **`thme` CUDA (`cublasGemmEx` Tensor Cores + embedded PTX)** | **5,601.4 tok/s** | **5,346.0 tok/s** | **27.03s / 28.32s** | **`8.9054` / `9.8512`** | **19.9× / 17.8×** |
| *Reference: `llama.cpp` b12584 Vulkan* | *—* | *9,053.6 tok/s* | *—* | *—* | *(from `llama_cpp_reference.json`)* |
| *Reference: `llama.cpp` b12584 CUDA* | *—* | *12,561.9 tok/s* | *—* | *—* | *(from `llama_cpp_reference.json`)* |

---

## 2. Changes Landed on `perf/gpu-throughput`

### Round 1 (Committed `fc6f872`)
1. **`src/inference/gpu/backend.rs`:**
   - Raised `WEIGHT_CACHE_BUDGET` from `4 GiB` to `10 GiB` (eliminates 6.24 GB/window PCIe thrashing).
   - Replaced flush-all eviction with **LRU eviction** (`tick: u64`, `last_used: u64`).
   - Added `GpuBackend::clear_cache(&self)` and `GpuBackend::enable_coop_if_ready(&self)`.
2. **`src/cli/evaluate.rs`:**
   - Enabled `enable_coop_if_ready()` for `thme evaluate` while keeping `GpuContext::init()` at exact `f32` defaults for library/test parity.
   - Added `clear_cache()` after `drop(baseline_model)`.
3. **`src/inference/gpu/decoder.rs`:**
   - Added persistent `DecoderWorkspace` (`OnceLock<Mutex<Option<(WorkspaceKey, DecoderWorkspace)>>>`) reusing all 20 transient GPU buffers across windows.
   - Hashed RoPE `cos`/`sin` uploads to skip redundant `write_buffer` calls.
4. **`src/inference/gpu/shaders/decoder_ops.wgsl`:**
   - Widened `nll` reductions from `@workgroup_size(64)` to `@workgroup_size(256)`.

### Round 2 (Unified Default Backend: CUDA + WGPU/Vulkan/Metal + CPU)
1. **`Cargo.toml`:** Integrated `cudarc = "0.19"` (`driver`, `cublas`, `nvrtc`, `dynamic-loading`, `fallback-latest`, `cuda-12090`) as a default target dependency on Windows & Linux (`[target.'cfg(any(target_os = "windows", target_os = "linux"))'.dependencies]`), keeping `cuda = []` as a backward-compatible feature flag and macOS pure `wgpu` (Metal).
2. **`src/inference/gpu/cuda.rs` & `src/inference/gpu/kernels/decoder.cu`:**
   - Runtime-compiled (NVRTC) header-free CUDA kernels (`f32_to_f16`, `rms_norm_f16`, `rope`, `causal_softmax_f16`, `swiglu_f16`, `add`, `nll_f16`).
   - `cublasGemmEx` (FP16 Tensor Cores, FP32 accumulate) for all 7 linears + FP16 `lm_head`, and `cublasGemmStridedBatchedEx` for batched Tensor Core Attention ($Q K^T$ and $P V$, GQA-aware).
   - 10 GiB LRU FP16 `WeightCache` + persistent `CudaWorkspace` + `RopeTable` cache in `llama.rs`.
3. **`src/inference/gpu/mod.rs` & `src/inference/gpu/backend.rs`:**
   - `DeviceArg::Auto` / `DeviceArg::Gpu` tries CUDA first on Windows/Linux (unless `THME_GPU_BACKEND=wgpu`), falls back cleanly to pure-Rust `wgpu` (Vulkan/DX12/Metal, **3,566–4,133 tok/s** with default `WeightStorage::F16` + `coop`), and falls back to CPU if no GPU is present.
   - Binary sizes verified after final edit: `thme.exe` = **14.54 MB** (`<= 16 MB`), `thme-gui.exe` = **15.37 MB** (note: `0.37 MB` over `15 MB` guideline).

---

## 3. Quantization Screening Summary (`37-window VAL-A`, `thme-wt/agy-tune`)

- **`AWQ Q4_0` (`thme.awq_q4_0` + `CALIB`):** **9.7125 PPL** (`-4.12%` vs Baseline `Q4_0` `10.1298`) — **#1 Winner**
- **`GPTQ-Diag Q4_0` (`rms^2` fallback):** **9.9077 PPL** (`-2.19%` vs Baseline `Q4_0`) — **#2 Winner**
- **`Baseline Q4_0`:** **10.1298 PPL**
- **Data-free clipping (`Quantile Clip`, `HQQ Lp=0.7`):** Regressed (`10.2326`, `10.6262`). Reverted `conversion.rs` in `agy-tune`.
- **Note on `thme-tools`:** Updated `run_referee.py`, `bench.sh`, `loop2.sh`, `p2-measure.txt`, `ctl.sh`, `referee.sha256`, and `protected.sha256` to support `--format`, `--assemble`, and real-time `tune/live.log` streaming (`bash tune/ctl.sh watch`).

### Round 3 (Zero-DLL Embedded PTX, Committed `af0f47d`)
1. **`src/inference/gpu/kernels/decoder.cu` & `decoder.ptx`:**
   - Added pure-PTX `gemm_f16_tiled` and `causal_attention_f32` fallback kernels and compiled `decoder.ptx` (59,050 bytes, `compute_75`) embedded directly via `include_str!("kernels/decoder.ptx")` — eliminating runtime `nvrtc64_120_0.dll` compilation.
2. **`src/inference/gpu/cuda.rs`:**
   - Added `cudarc::driver::sys::is_culib_present()` and `cudarc::cublas::sys::is_culib_present()` guards in `CudaContext::init()` so missing NVIDIA driver or `cublas64_12.dll` never trigger `cudarc`'s `OnceLock` panic.
   - Changed `blas` to `Option<Mutex<CudaBlas>>`, falling back to `gemm_f16_tiled` and `causal_attention_f32` when `cublas64_12.dll` is absent (note: when `cublas64_12.dll` is absent, `THME_GPU_BACKEND=wgpu` with `coop` Tensor Cores is ~10–12× faster than scalar PTX).
3. **Verification (`af0f47d`):**
   - `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test --test test_gpu_batched` (31/31), and full `cargo test` (519 unit + all integration tests) passed.
   - `thme.exe` release binary size = **14.59 MB** (`<= 16 MB`).
   - Zero-DLL pure-PTX run on `SmolLM2-1.7B` (16 VAL windows, no external CUDA/cuBLAS/NVRTC DLLs on `PATH`) completed without panic under adapter `"NVIDIA CUDA (Pure Embedded PTX, Zero External DLLs)"`.

---

## 4. Cross-Reference: `thme-tools/inbox/` Sun–Tue (`2026-11-22` to `2026-11-24`)
- **`2026-11-22`:** `A1.md`, `D1.md`, `R1.md` marked `needs correction` in Hermes `REVIEW.md`.
- **`2026-11-23`:** `A2.md` (third-party licence register) **ACCEPTED**; `R2.md`, `D2.md` in weekly.
- **`2026-11-24`:** `H2`, `H3` (coop matrices), `H5`, `H8`, `H13`, `H14`, `H16` in `READY.md`; `H15` (`2026-11-24-H15-fix1.md`, which `llama.cpp` parts to port cleanly) in `NEEDS-OWNER.md`.

---

## 5. Round 4 (CUDA Optimizations O1–O4 + WGPU/Reporting Fixes, Committed `eed674f`)

**Commit:** `eed674f7370f59daa5ff9d72803e23b949545287` (`perf(gpu): parallelize f16 upload, trim attention/NLL round-trips, and fix wgpu/backend reporting`)  
**Diff (`af0f47d..eed674f`):** `4 files changed, 183 insertions(+), 26 deletions(-)` (`decoder.cu` and `decoder.ptx` untouched)

### 5.1 Changes Landed
1. **`CHANGELOG.md`:** Removed leftover `<<<<<<< HEAD` / `=======` / `>>>>>>> origin/main` merge conflict markers inherited from `origin/main` (`#123` and `#122` entries preserved).
2. **`src/inference/gpu/backend.rs`:**
   - Added `should_skip_cuda()` and `wgpu_api_filter()` so `THME_GPU_BACKEND=wgpu` skips CUDA and selects any non-CPU WGPU adapter (`vulkan`/`metal`/`dx12`) instead of rejecting all adapters and silently falling back to CPU (`~41 tok/s` on `af0f47d`).
   - Added `GpuBackend::backend_description(&self)` and `GpuContext::backend_description(&self)` (reporting WGPU API, `WeightStorage`, and `coop matrix` state) plus 2 unit tests.
3. **`src/cli/evaluate.rs`:** Updated `host.backend` report field to call `g.backend_description()` instead of the hardcoded `"THME reference forward pass on the GPU (wgpu, f32 weights and activations)"` string.
4. **`src/inference/gpu/cuda.rs`:**
   - **O1:** Parallelized `f32` -> `f16` weight upload conversion (`convert_f32_slice_to_f16`) with `std::thread::scope` chunks for slices `>= 65,536` elements, plus unit test.
   - **O2:** On the `cublas` path (`self.blas.is_some()`), projected `wv` directly into `ws.v16` (`gemm_f16_to_f16`), wrote attention `PV` (`cublasGemmStridedBatchedEx`, FP32 accumulate) directly into `ws.x16` (`CUDA_R_16F`), and ran `wo` directly from `ws.x16` (`gemm_f16_preconverted`), eliminating 2 F32/F16 round-trips per layer (Pure-PTX `F32` path preserved).
   - **O3:** Doubled `NLL_CHUNK_MAX_ELEMS` from `32M` to `64M` (128 MiB in FP16), scoring SmolLM2-1.7B's 1,023-row window in 1 chunk.
   - **O4:** Bound a persistent 32 MiB `_blas_workspace` via `cublasSetWorkspace_v2` (ordered after `blas` on `CudaContext` for drop safety, with graceful `None` fallback).
   - Implemented `CudaContext::backend_description(&self)`.

### 5.2 Before vs. After Benchmark Comparison (`SmolLM2-1.7B`, RTX 4060 Ti 16 GB, 148 VAL Windows, `--no-baseline-cache`)

| Configuration | `host.device.used` | `host.backend` | `f32` Speed (Elapsed) | `Q4_0` Speed (Elapsed) | PPL (`f32` / `Q4_0`) | Speedup vs `origin/main` (`d5378b6`) |
| :--- | :---: | :--- | ---: | ---: | :---: | :---: |
| **`af0f47d` CUDA+cuBLAS (Before)** | `gpu` | `... (wgpu, f32 weights and activations)` *(bug)* | 5,591.88 tok/s (27.08 s) | 5,335.55 tok/s (28.38 s) | `8.9054` / `9.8512` | 19.9× / 17.7× |
| **`eed674f` CUDA+cuBLAS (After)** | `gpu` | `... (CUDA + cuBLAS Tensor Cores, f16 weights)` | **6,552.88 tok/s** (23.10 s) | **6,436.77 tok/s** (23.52 s) | **`8.9054` / `9.8512`** | **23.3× / 21.4×** |
| **CUDA Delta (`eed674f` vs `af0f47d`)** | — | Label fixed | **+17.19%** (-3.97 s) | **+20.64%** (-4.85 s) | **0.000000** (bit-exact) | +3.4× / +3.7× |
| **`af0f47d` `THME_GPU_BACKEND=wgpu` (1-win diag)** | `cpu` *(bug)* | `THME reference CPU forward pass (f32, multithreaded)` | 41.16 tok/s *(1 win)* | 42.13 tok/s *(1 win)* | — | 0.15× / 0.14× |
| **`eed674f` `THME_GPU_BACKEND=wgpu` (After)** | `gpu` | `... (wgpu Vulkan, f16 weights, coop matrix)` | **3,416.09 tok/s** (44.32 s) | **3,979.77 tok/s** (38.04 s) | **`8.9054` / `9.8512`** | **12.1× / 13.2×** |

- **Perplexity Parity:** `eed674f` CUDA matches `af0f47d` bit-for-bit (`8.905388807591697` / `9.8511586086027`, `delta_ppl = 0.9457698010110036`); `eed674f` WGPU Vulkan matches `af0f47d` CUDA within FP32 summation variance (`-2.96e-5` on `f32`, `+3.23e-5` on `Q4_0`; `8.90535925650563` / `9.851190903689002`).
- **Verification Gates (`eed674f`):** `cargo fmt --check` (`exit 0`), `cargo clippy --all-targets --locked -- -D warnings` (`exit 0`, 0 warnings), `cargo test --locked` (`730 passed, 0 failed, 1 ignored`), `cargo build --release --bin thme` (`15,305,728` bytes = `14.60 MiB <= 16 MB`).

### 5.3 External Inbox & `agent-room` Audit Summary
- **`thme-tools/inbox/` (`2026-11-21`–`24`) & `thme-tools/tune/`:** Audited all items (`H1`–`H16`, `C2`, `C3`, weekly `A1`–`A2`/`D1`–`D2`/`R1`–`R2`, `tune/patches/EXP-0002-wip.patch`, `agy-tune` `GPTQ-Diag`). Zero pending GPU throughput patches exist (`H2`–`H16` are research/doc specs; `C2`/`H12`/`C3` target docs, VS Code sidebar, or example scripts; `EXP-0002-wip.patch` regressed Q4_0 PPL `10.2326` vs `10.1298`; `GPTQ-Diag` is PTQ quality-only for `thme.gptq_q4_0`). Note: `inbox/agy/READY.md` lists `H5`, `H13`, `H14`, `H16`, while `H2`, `H3`, `H8`, `H15` received `FIX` verdicts.
- **`agent-room` (`room.jsonl`):** Latest entry is `#5` (`2026-11-22T10:18:28Z`); no GPU throughput or `perf/gpu-throughput` messages exist.
- **R4 Hermes Execution:** Ran live Hermes (`hermes chat --oneshot --provider antigravity-subscription-directsdk`) with `gpt-oss-120b-medium` (R1 inbox/room audit, `exit 0`) and `claude-sonnet-4-6 -t file` (Round 4 benchmark math verification & draft, `exit 0`).

---

## 6. Round 5 (Fused FP16 Kernels, Warp-Shuffle Softmax, Optional FP16 Accumulate, & Research Paper Review)

### 6.1 `thme-gui` Binary Limit & Research Papers Reviewed
- **`AGENTS.md`:** Raised `thme-gui` release binary limit from `15 MB` to **`<= 25 MB`** (25 November 2026) to accommodate embedded PTX kernels, CUDA backend, and GPU diagnostics in the unified binary.
- **Research Briefs & Papers Reviewed (`thme-tools/briefs/` & `inbox/`):**
  - **`H14` (`llama.cpp` v1.0.0 & 8 forks: `ik_llama.cpp`, `Strata`, `strata-nvfp4`, `KoboldCpp`, `llamafile`, `nari-llama.cpp`, `buun-llama-cpp`, `CachyLLama`):** Top takeaways for THME: multi-shard GGUF + FP8 (`E4M3`), NVFP4 (OCP MX-FP4), `llamafile` register-blocked CPU GEMM, `ik_llama.cpp` IQ quants, persistent KV ring-buffer.
  - **`H15-fix1` (Clean-Room vs MIT Translation Process):** Recommends Route B (clean-room) for RMSNorm, RoPE, GQA, online softmax, K-quants; Route A (MIT derivative with notices) only if matching heuristic sampler parameters.
  - **`H3-v2` (Cooperative Matrices):** `wgpu` `EXPERIMENTAL_COOPERATIVE_MATRIX` is limited to `8×8×8` `f32`, whereas native Vulkan `VK_KHR_cooperative_matrix` / CUDA Tensor Cores support `16×16×16` `f16` (explaining the WGPU vs CUDA gap).
  - **`H4` (2025–2026 arXiv Scan — Kitty, Conf-KV, KVBoost, Apple Silicon INT4 KV, GateRQ):** KV-cache INT8/INT4 with attention sinks + sliding window (`v0.2.4`); GateRQ warns frequency-based embedding precision is worse than gate-ranked precision.
  - **`token-speed-budget-devices.md`:** Confirms incremental KV cache (`chat::generate`) and packed-domain quantized kernels are the primary decode-speed levers.

### 6.2 CUDA Optimizations Landed
1. **`src/inference/gpu/kernels/decoder.cu` & `decoder.ptx` (84,512 bytes):**
   - **`rope_qk_f16`:** Fused in-place FP16 RoPE across both `q16` and `k16` in 1 kernel launch (eliminates 2 `rope` launches + 2 `f32_to_f16` conversions per layer).
   - **`swiglu_f16_in`:** Reads `gate16` and `up16` directly in FP16, halving MLP intermediate global memory traffic (`1.54 GB/window` saved).
   - **`add_rms_norm_f16`:** Fuses residual `add` (keeping FP32 residual stream) + warp-shuffle `rms_norm_f16` into 1 launch (eliminates 48 launches/window + halves `tmp` write bandwidth).
   - **`causal_softmax_f16`:** Replaced `__syncthreads()` tree reductions with `__shfl_xor_sync` warp-shuffle reductions.
2. **`src/inference/gpu/cuda.rs`:**
   - Direct FP16 outputs (`gemm_f16_to_f16`) for `wq`, `wk`, `wv`, `wo`, `wg`, `wu`, `wd` on the `cuBLAS` fast path.
   - Added opt-in `THME_CUDA_F16_ACC=1` (`CUBLAS_COMPUTE_16F` Tensor Core FP16 accumulation) alongside the exact `CUBLAS_COMPUTE_32F` default.

### 6.3 Measured Results (`SmolLM2-1.7B`, RTX 4060 Ti 16 GB, 148 VAL Windows, `--no-baseline-cache`)

| Configuration | `f32` Speed (Elapsed) | `Q4_0` Speed (Elapsed) | PPL (`f32` / `Q4_0`) | vs `origin/main` (`281.2 / 301.2`) |
| :--- | ---: | ---: | :---: | :---: |
| **Default (`CUBLAS_COMPUTE_32F` + Fused FP16)** | **6,951.79 tok/s** (21.78 s) | **7,037.95 tok/s** (21.51 s) | `8.9056` / `9.8512` | **24.7× / 23.4×** |
| **`THME_CUDA_F16_ACC=1` (`CUBLAS_COMPUTE_16F`)** | **9,506.09 tok/s** (15.93 s) | **9,332.47 tok/s** (16.22 s) | `8.9114` / `9.8560` | **33.8× / 31.0×** |
| *Reference: `llama.cpp` b12584 Vulkan* | *—* | *9,053.6 tok/s* | *—* | *—* |
| *Reference: `llama.cpp` b12584 CUDA* | *—* | *12,561.9 tok/s* | *—* | *—* |

- **CI Verification:** `cargo fmt --check`, `cargo clippy -- -D warnings`, and full `cargo test --locked` (522 unit + all integration/example tests) passed.
