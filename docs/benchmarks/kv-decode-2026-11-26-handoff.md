# THME Engineering Status & Handoff Report for Claude (`2026-11-26`)

## 1. Repository, Worktree & Branch State

- **Primary Worktree:** `..\thme-wt\perf-gpu`
- **Active Branch:** `feat/kv-cache-gpu-decode` (branched off `perf/gpu-throughput` `8e54a99`, which is off `origin/main` `d5378b6`)
- **Latest Commits on `feat/kv-cache-gpu-decode`:**
  - `a97b07e` — `feat(gpu): add device-resident CUDA KV cache for incremental decode (forward_extend)`
  - `4399f61` — `perf(cuda): add aligned Q4_0 GEMV kernel and resident_q4_0 for fast 1-token KV decode`
  - `0708184` — `feat(cli): add thme chat KV-decode benchmark command, bits_per_token metric, and --f16-acc flag`
  - `aa8e949` — `perf(cuda): add CUDA Graph capture/replay, parallel Q4_0 packing, and chat decode metrics`
  - *(plus release CI binary-limit sync in `.github/workflows/release.yml`)*

---

## 2. Binary Footprint Limits (Checked & Synced)

- **`AGENTS.md` (both `LLM-Tweaker` and `thme-wt/perf-gpu`):**
  - `thme` (CLI): **`<= 16 MB`** (currently measured at **`14.75 MB`** for release `thme.exe`).
  - `thme-gui` (GUI): **`<= 25 MB`** (raised from 15 MB on 25 Nov 2026 for embedded PTX kernels, CUDA backend, and GPU diagnostics).
- **CI Fix Applied:** `.github/workflows/release.yml` line 77 in `thme-wt/perf-gpu` still had the outdated `15 * 1024 * 1024` check for `thme-gui`; updated to `25 * 1024 * 1024` to match `AGENTS.md`.

---

## 3. Implemented Architecture & Optimizations

1. **CUDA + cuBLAS Backend (`src/inference/gpu/cuda.rs`, `src/inference/gpu/kernels/decoder.cu` / `decoder.ptx`):**
   - Embedded precompiled `decoder.ptx` (`compute_75`) with optional runtime `cuBLAS` (`CUBLAS_COMPUTE_32F` default; `--f16-acc` / `THME_CUDA_F16_ACC=1` opt-in).
   - Fused kernels: `rope_qk_f16`, `add_rms_norm_f16`, `swiglu_f16_in`, warp-cooperative `causal_softmax_f16`.
   - **CUDA Graph Capture & Replay:** `SendCudaGraph(CudaGraph)` in `CudaWorkspace` captures and replays the GPU compute body in `CudaContext::forward_decoder` (`begin_capture` / `end_capture` / `launch`), keyed by workspace shape, `Arc` weight pointers, `use_flash`, `THME_CUDA_F16_ACC`, `scored`, `chunk_rows`, and `plan.eps`. Disable via `THME_CUDA_GRAPH=0`.
2. **Device-Resident Incremental KV Cache (`src/inference/kv.rs`, `src/inference/llama.rs`, `cuda.rs`):**
   - `DeviceKvHandle` + `kv_append_f16` + `kv_attention_f16` in `forward_decoder_extend` and `Llama::forward_extend` with `ROPE_CACHE` — zero per-token host KV copies during decode.
3. **Packed `Q4_0` GPU Decode (`cuda.rs`, `decoder.cu`):**
   - `resident_q4_0` parallelized across CPU threads with `std::thread::scope` (`0.5625` B/weight, `3.56x` smaller than `F16`).
   - Warp-cooperative `gemv_q4_0_aligned` (128-bit `uint4` coalesced loads + shared-memory `x16`) for `t == 1` decode (`THME_CUDA_Q4_DECODE=1`).
4. **CLI & Evaluation Metrics (`src/chat.rs`, `src/cli/chat.rs`, `src/cli/evaluate.rs`):**
   - `thme chat` command with `--device`, `--max-new-tokens`, `--json`, reporting `prefill_secs`, `decode_tokens_per_sec`, `wall_secs`, and overall `tokens_per_sec`.
   - `thme evaluate` with `--f16-acc` and `bits_per_token` in JSON output.

---

## 4. Verified Checks & Measured Benchmarks (`SmolLM2-1.7B`, RTX 4060 Ti 16 GB)

- **Code Quality & Tests (all passing on `aa8e949`):**
  - `cargo fmt --check` — **PASS**
  - `cargo clippy -- -D warnings` — **PASS** (zero `unwrap`/`expect`/`panic` outside tests)
  - `cargo test --locked` — **PASS** (all unit, integration, and example test suites)
  - `cargo build --release --bin thme` — **PASS** (`14.75 MB`)
- **`thme chat --device gpu --max-new-tokens 64 --json`:**
  - **`F16`:** `decode_tokens_per_sec` = **62.87 tok/s**, `prefill_secs` = **1.78 s**, `wall_secs` = **2.78 s** (`23.00 tok/s` overall).
  - **`Q4_0` (`THME_CUDA_Q4_DECODE=1`):** `decode_tokens_per_sec` = **16.88 tok/s**, `prefill_secs` = **1.84 s**, `wall_secs` = **5.57 s** (`11.48 tok/s` overall — >3× faster than `17.68 s` / `3.62 tok/s` prior to parallelizing `resident_q4_0`).
- **`thme evaluate --device gpu --f16-acc --no-baseline-cache --eval-max-tokens 151552` (148 windows, WikiText-2 test):**
  - **Baseline (`F16`):** **9,152.77 tok/s** (`16.54 s`, PPL `8.9114`, `bits_per_token` `3.1557`).
  - **Candidate (`MlpDown` L0 `Q8_0`):** **9,267.78 tok/s** (`16.34 s`, PPL `8.9126`, Δ `+0.0011`, `PASSED`).

---

## 5. Known Pitfalls (Do Not Repeat)

1. Do **not** use `CUBLAS_COMPUTE_16F` on `PV` in `launch_attention_qkv16` (regresses throughput vs `COMPUTE_32F` on `PV`).
2. Do **not** fuse `Gate+Up` via `resident_f16_pair` (doubles VRAM pressure and host copy).
3. Do **not** use raw `wmma` register fragments without shared-memory store (produces non-finite output at `t=1024`).
4. On Windows, runtime tests/eval require `cublas64_12.dll` on `PATH` (e.g. `cuda12-rt\nvidia\cublas\bin`). If `link.exe` hits transient `LNK1201`, clear `target\debug\build\thme`.

---

## 6. Next Priorities for Claude

1. **Direct `.epiph` Packed GPU Upload:** Skip CPU `Vec<f32>` expansion for `.epiph` `Q4_0` / `F16` tensors and upload packed weights straight to CUDA/wgpu (`resident_q4_0` currently quantizes from `f32` on CPU).
2. **Decode Speed & Cross-Vendor Parity:**
   - Profile `gemv_q4_0_aligned` (`16.88 tok/s` vs `F16` `62.87 tok/s`) and add CUDA Graph capture to `forward_decoder_extend` (`t == 1` decode path) so each decode token avoids host launch overhead.
   - Implement `forward_decoder_extend` and `gemv_q4_0` on `wgpu` (for AMD / Intel / Apple Silicon), plus INT8/INT4 KV cache quantization + StreamingLLM sink/window eviction.
3. **Evaluation Suite (`v0.2.8.3`):** Add `bits_per_byte` (BPB) to `thme evaluate` and wire `src/bench/` for MMLU / ARC-Challenge / HellaSwag / GSM8K.
4. **Release & Merge:** Build `thme-gui` to verify it stays `<= 25 MB`, then merge `feat/kv-cache-gpu-decode` -> `perf/gpu-throughput` -> `main`.
