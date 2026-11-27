# MLC LLM survey (for THME)

Read-only survey of the shallow clone at `<engines>\mlc-llm`. Paths are relative to that folder. Every claim names a file that was opened; anything else is marked unverified. Effort sizes (S/M/L) are estimates, not measurements.

Important limit: the submodule folders `3rdparty/tvm` and `3rdparty/xgrammar` are empty in this clone (`ls` shows nothing; `.gitmodules` points TVM at `mlc-ai/relax`). TVM internals therefore cannot be read here; everything about TVM below comes from how `python/mlc_llm/` calls it and is marked where inferred.

THME today (opened: `README.md`, `src/inference/gpu/`): CUDA through embedded PTX built from `src/inference/gpu/kernels/decoder.cu`, and wgpu through hand-written WGSL shaders in `src/inference/gpu/shaders/` (gemv and gemm per format, attention, activation conversion), a feature audit of the adapter (`src/inference/gpu/features.rs`: `audit`, `coop_f16_usable`), and a backend enum with weight caches (`src/inference/gpu/backend/`). Vulkan, DX12 and Metal are reached through wgpu, not through separate THME backends (inferred from the module layout; the wgpu backend choice itself was not opened).

## 1. Licence
- `LICENSE` first lines: "Apache License / Version 2.0, January 2004". Only the unfilled appendix template is present, so `LICENSE` names no holder.
- `NOTICE` exists. Full content: "MLC LLM" and "Copyright (c) 2023-2025 by MLC LLM Contributors".
- Different licence in any directory: the find for LICENSE/NOTICE/COPYING found only the root `LICENSE` and `NOTICE`. The submodules (`3rdparty/tvm`, `xgrammar`, `tokenizers-cpp`, `argparse`, `googletest`, `stb`, listed in `.gitmodules`) are empty here, so their licences are unverified.
- Obligations (Apache-2.0 section 4(d) in `LICENSE`): redistributions of the work or derivatives that include a NOTICE file must keep its attribution text. If any source is copied, THME must also ship the Apache licence text, keep notices and mark changes. Clean-room Rust code that copies nothing needs only credit in docs. AGPL-3.0-only can include Apache-2.0 code.

## 2. What it is
MLC LLM is a compiler-based LLM deployment stack: models are written once as Python `nn.Module` descriptions, quantized by Python passes, and compiled with Apache TVM (Relax graph IR, TIR kernels) into per-target libraries. Its runtime is a C++ serving engine with paged KV cache, radix-tree prefix cache and speculative decoding. Targets named in `README.md` include Vulkan, CUDA, ROCm, Metal, OpenCL (Adreno, Mali), WebGPU and WASM, plus iOS and Android apps.

## 3. Components
| Component | What it does | Main paths opened |
| :--- | :--- | :--- |
| Model descriptions | One `nn.Module` per architecture with `prefill`, `decode`, `batch_prefill`, `batch_decode`, `batch_verify` entry points taking a `PagedKVCache` | `python/mlc_llm/model/llama/llama_model.py`, `model/model.py` (names), `model/` folder listing (about 50 families) |
| Compile driver | Builds the module, picks a target, runs a named pipeline, exports a library and metadata | `python/mlc_llm/interface/compile.py` (`_compile`, `CompileArgs`, `relax.get_pipeline`), `interface/compiler_flags.py` (`OptimizationFlags`) |
| Pass pipeline | Four phases: attach info and dispatch; graph-level fusion; lowering to TIR; TIR fusion and Dlight scheduling | `python/mlc_llm/compiler_pass/pipeline.py`, `fuse_dequantize_matmul_ewise.py`, `fuse_add_norm.py`, `low_batch_specialization.py`, `dispatch_kv_cache_creation.py` |
| Target selection | Maps device and hints to TVM targets (cuda, rocm, metal, vulkan, opencl, webgpu, llvm) and registers CUDA hooks | `python/mlc_llm/support/auto_target.py`, `support/auto_device.py` (`AUTO_DETECT_DEVICES`) |
| Quantization | `GroupQuantize` presets (`q3f16_0`, `q4f16_1`, `q4bf16_0`, `q4f32_1`, ...) that rewrite linear layers; quantize and dequantize are written as TVM compute expressions | `python/mlc_llm/quantization/quantization.py`, `group_quantization.py`, `awq_quantization.py`, `fp8_quantization.py` |
| Custom ops | Attention, MoE, speculative verify, top-p pivot, Triton, cutlass, FasterTransformer GEMM | `python/mlc_llm/op/` listing |
| Serving runtime | Engine loop and actions (prefill, decode, draft, verify, eagle, jump-forward), radix-tree prefix cache, logit processor, samplers | `cpp/serve/engine.cc`, `radix_tree.h`, `prefix_cache.h`, `engine_actions/` listing |
| WebGPU metadata | Records required features (`shader-f16`, `subgroups`) in the compiled artifact | `interface/compile.py` (webgpu branch) |

## 4. What THME lacks that MLC LLM has, ranked
Frame: THME is not a compiler and should not become one. Value is in design lessons for its CUDA and wgpu kernel set and in model-description reuse.

### 4.1 Capability-recorded kernel variants (rank 1)
- What MLC does: `compile.py` writes the features a compiled WebGPU program requires (`shader-f16`, `subgroups`) into artifact metadata and `auto_target.py` toggles subgroup support per target, so a runtime can refuse or fall back by capability rather than by device name.
- THME angle: `src/inference/gpu/features.rs` already audits the adapter and has `coop_f16_usable`. The gap is a declared requirement list per shader variant (for example `gemm_coop.wgsl` needs cooperative matrices, `gemm_coop_f16w.wgsl` needs f16) and a test that every variant either runs or is skipped with a stated reason. Algorithm only, no TVM. Effort S. Risk low.
- Test oracle: CPU reference forward for each variant that the adapter supports; skipped variants are listed in the report.

### 4.2 Dequantize-matmul fusion as a named pattern list (rank 2)
- Paths: `compiler_pass/fuse_dequantize_matmul_ewise.py`, `fuse_dequantize_take.py`, `fuse_dequantize_transpose.py`, `fuse_add_norm.py`, `fuse_transpose_matmul.py`. These show which fusions MLC treats as essential: dequantize into matmul (and the embedding lookup), add plus RMSNorm, transpose into matmul.
- THME angle: THME already has fused quantized GEMV shaders (`gemv_q4_k.wgsl`, `gemv_q5_k.wgsl`, `gemv_iq4_xs.wgsl`, `gemm_q8dot.wgsl`). Use the list as a checklist of fusions not yet done (add+norm, dequantizing embedding lookup; whether THME has these is unverified). Translate: hand-written kernels, not the TVM passes. Effort M per fusion. Risk low-medium. Oracle: the unfused CPU path, tolerance per format; measured speed on named hardware.

### 4.3 Low-batch GEMV specialization and rule-based scheduling (rank 3)
- Paths: `compiler_pass/low_batch_specialization.py`, `pipeline.py` Phase 4 (Dlight rules `Matmul`, `GEMV`, `Reduction`, `GeneralReduction`, `Fallback`; a CPU `GEMV` rule for llvm). Dlight picks a schedule by pattern of the kernel, with no autotuning, which is a cheap alternative to tuning (inferred from the pipeline; Dlight source not available).
- THME angle: choose between `gemv_*` and `gemm_*` shaders by batch size and shape with a small table; THME has both families. Effort S. Risk low. Oracle: same logits; measured throughput per batch size.

### 4.4 One model description, many targets: what to copy (rank 4)
- MLC writes the model once (`llama_model.py`) and the compiler produces CUDA, Vulkan, Metal and others; THME writes WGSL (run by wgpu on Vulkan, DX12, Metal) plus a separate CUDA path. The lesson: keep one description of the decoder step (op list with shapes and formats) and generate or dispatch per backend, so tests compare backends against one CPU oracle. THME's `DecoderPlan` (`src/inference/gpu/decoder.rs`, exported in `mod.rs`; body not opened here) is the closest existing piece; whether it is already backend-neutral is unverified.
- TVM concepts that matter, as seen through the MLC Python (TVM source unavailable): Relax is a graph IR with symbolic shapes (`_get_variable_bounds`, `symbolic_sizes` in `compile.py`); TIR is the loop-level kernel IR; passes are sequenced into phases (`pipeline.py`); targets are objects with kind and attributes (`Target`, `auto_target.py`). For THME only the target-attribute idea and the phase order matter.
- Effort: M for a backend-neutral op list (estimate). Risk: medium (over-abstraction; keep it a plain enum plus plan). Oracle: CPU reference forward versus each backend.

### 4.5 Serving features that duplicate vLLM and SGLang (rank 5)
- `cpp/serve/radix_tree.h`, `prefix_cache.h`, `engine_actions/` (`batch_verify.cc`, `batch_jumpforward.cc`, `eagle_batch_verify.cc`). Same ideas as vllm.md and sglang.md; the C++ radix tree is another reference to read, and `batch_jumpforward` implies jump-forward decoding for grammars (file listing only; behavior unverified). Prefer the SGLang and vLLM notes for these.

### 4.6 Group quantization presets (low)
- `quantization/quantization.py` presets such as `q3f16_0` (int3, group size 40, uint32 storage) and the `*_0` versus `*_1` linear layout switch (`linear_weight_layout="KN"` shown for `q3f16_0`). THME has far more formats (28 GGML types, README), so little to gain. Note the 3-bit group size 40 as an unusual design point only.

## 5. What NOT to port
- TVM itself (Relax, TIR, Dlight, VM, runtime): a very large C++ dependency, the submodule is not even present here, and it would replace THME's hand-checked kernels with a compiler whose output THME cannot audit.
- The compile-to-library workflow and per-device library artifacts: conflicts with THME's `.epiph` one-file model.
- Mobile and web apps (`android/`, `ios/`, `web/`): out of scope.
- Triton, cutlass, FasterTransformer and FlashInfer integrations (`python/mlc_llm/op/`, `compile` flags): vendor library bindings.
- The C++ serving engine as code (`cpp/serve/`): read for ideas only; rewrite in Rust if a serving mode is built.
- xgrammar and tokenizers-cpp submodules: external native dependencies (unverified contents here).
