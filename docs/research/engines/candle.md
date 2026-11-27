# candle (Hugging Face) survey for THME

Read-only survey of the shallow clone at `C:\Users\TechTroniX\engines\candle` (paths below are relative to it).
Effort sizes (S, M, L) are estimates, not measurements. Nothing here was run.

## 1. Licence

- Files at the root: `LICENSE-MIT` and `LICENSE-APACHE` (dual). No `NOTICE` file (root listing shows none). `Cargo.toml` line 30: `license = "MIT OR Apache-2.0"`.
- `LICENSE-MIT` opens with "Permission is hereby granted, free of charge, to any person obtaining a copy of this software ..." and has NO copyright line, so the holder is not named in the licence text. Who holds copyright: unverified.
- `LICENSE-APACHE` is the stock Apache License 2.0 text ("Apache License, Version 2.0, January 2004"); its appendix is unfilled boilerplate, so no named holder there either.
- Embedded third-party code: `candle-kernels/README.md` says some kernels come from the dfdx crate; `candle-kernels/src/quantized.cu` header says kernels are adapted from llama.cpp `ggml-cuda.cu`; `candle-kernels/src/mmq_gguf/mmq_common.cuh` says it replaces llama.cpp headers. llama.cpp is MIT upstream (not in this clone, unverified), so ported CUDA quant kernels carry a second notice.
- Obligations if THME copies code: MIT requires the copyright and permission notice in copies; Apache-2.0 requires the licence text, kept notices, and a statement of changes on modified files. Both are permissive and compatible with AGPL-3.0-only, but notices must stay in file headers and in THME's third-party register. Safest rule: port ideas and re-derive from published algorithms (as THME already does for llm.c and PEFT), copy only with notices.

## 2. What it is

A minimalist Rust ML framework from Hugging Face with a PyTorch-like `Tensor`, CPU, CUDA and Metal backends, and a large zoo of model ports in `candle-transformers`. Backward is a tape built from forward tensor ops, so training runs on any backend that runs the forward. Quantized inference follows llama.cpp: GGML block types, a GGUF reader and writer, and llama.cpp-derived CUDA kernels.

## 3. Components

| Component | What it does | Paths opened |
| :--- | :--- | :--- |
| candle-core | Tensor, devices, dtypes, autograd, safetensors, npy, pickle | `candle-core/src/backprop.rs`, `custom_op.rs` |
| quantized | GGML types, QTensor, QMatMul, GGUF and GGML readers, imatrix reader, SIMD kernels, CUDA path | `candle-core/src/quantized/mod.rs`, `k_quants.rs`, `gguf_file.rs`, `cuda.rs`, `imatrix_file.rs` (file sizes listed: `repack.rs`, `fast_mmq.rs`, `fast_mmvq.rs`, not read) |
| candle-kernels | CUDA `.cu` compiled to PTX by `build.rs`; static kernels for GGUF mmq, mmvq, MoE | `candle-kernels/src/lib.rs`, `build.rs`, `quantized.cu`, `mmq_gguf/mmq_common.cuh`, `README.md` |
| candle-nn | layers, optimisers, KV cache, rotary, sampling, CPU flash attention, MoE | `candle-nn/src/kv_cache.rs`, `sampling.rs`, `ops.rs` |
| candle-flash-attn | CUDA flash attention crates | `candle-flash-attn/README.md` |
| candle-transformers | model ports (list below) | `candle-transformers/src/models/` directory listing, `quantized_llama.rs` head |
| tests | gradient and quantization tests | `candle-core/tests/grad_tests.rs`, `quantized_tests.rs` |

Architectures present (file and folder names in `candle-transformers/src/models`; only names were checked, not behaviour):
- LLM: llama, llama2_c, mistral, mixtral, gemma, gemma2, gemma3, gemma4, phi, phi3, qwen2, qwen2_moe, qwen3, qwen3_moe, qwen3_vl, deepseek2, glm4, glm4_new, chatglm, granite, granitemoehybrid, helium, lfm2, olmo, olmo2, falcon, mpt, stable_lm, starcoder2, persimmon, yi, bigcode, codegeex4_9b, smol.
- State space and recurrent: mamba, mamba2, rwkv_v5, rwkv_v6, rwkv_v7, recurrent_gemma.
- Encoders: bert, modernbert, nomic_bert, jina_bert, distilbert, debertav2, xlm_roberta, t5, marian, stella_en_v5, nvembed_v2.
- Vision and multimodal: clip, siglip, pixtral, llava, paligemma, moondream, blip, vit, dinov2, resnet, convnext and more.
- Audio and image generation: whisper, encodec, mimi, dac, parler_tts, metavoice, voxtral, csm; stable_diffusion, flux, wuerstchen, mmdit.
- Quantized variants (`quantized_*.rs`): llama, mistral, phi, phi3, qwen2, qwen3, qwen3_moe, gemma3, glm4, lfm2, rwkv_v5, rwkv_v6, t5, stable_lm, mixformer, mpt and more.

How backward is handled: `backprop.rs` walks the op graph in topological order (`sorted_nodes`, `backward`) and each op's gradient is written with ordinary tensor ops (matmul backward is two matmuls and a transpose, lines 459 to 470). Custom ops add a `bwd` hook (`custom_op.rs` lines 39, 85, 143) or opt out with `apply_op*_no_bwd` (lines 157 to 171). Unsupported: ceil and ConvTranspose1D (`backprop.rs` lines 314, 609). A grep for `bwd` or `backward` found hits in `candle-nn` (`ops.rs`, `rotary_emb.rs`, `optim.rs`, `init.rs`, `lib.rs`) but none in `candle-flash-attn/src` or `candle-kernels/src`, so the CUDA kernels are forward-only and backward reuses forward kernels.

## 4. What THME lacks that candle has (ranked)

THME today (from `README.md` and `src/`): LLaMA, GPT-NeoX and Gemma forward (`src/inference/arch.rs`, `llama.rs`, `neox.rs`, `gemma.rs`), 28 GGML types read, GGUF writer and K-quants (`src/exporter/`), CUDA via embedded PTX and wgpu WGSL kernels (`src/inference/gpu/`), a CPU-only hand-written LLaMA backward (`src/trainer/backward.rs`, 1189 lines), LoRA and AdamW (`src/trainer/lora.rs`), speculative decoding as an analytic model (`src/speculative/`).

1. **imatrix-weighted quantization.**
   - Why: importance-matrix rounding is a direct accuracy lever for K-quants and pairs with THME calibration (`src/inference/calibration.rs`, `src/exporter/calibration.rs`).
   - Paths: `candle-core/src/quantized/mod.rs` (`quantize_imatrix` at 183, `quantize_imatrix_onto` at 216, `from_float_imatrix` at 441), `imatrix_file.rs` (85 lines), `k_quants.rs`.
   - Translate: add weighted error terms to THME `src/exporter/kquant.rs` and an imatrix reader.
   - Drags in: nothing. Effort: S to M. Risk: low.
   - Test: oracle is llama.cpp (THME's README says a THME export loaded in llama.cpp b12584), plus error bounds like `quantized_tests.rs` lines 12 to 17 (total error 0.002, dot product error 0.02).
2. **Tape autograd reusing forward ops (GPU backward without new kernels).**
   - Why: THME backward is CPU, one sequence, LLaMA only (stated in the header of `src/trainer/backward.rs`). A tape runs wherever the forward runs.
   - Paths: `candle-core/src/backprop.rs`, `custom_op.rs`.
   - Translate: THME has no generic tensor with an op graph, so this means a new tensor layer over `GpuBackend`, not a file port.
   - Drags in: no crate, but a large new abstraction. Effort: L. Risk: high (two backward implementations to keep in agreement).
   - Test: keep `src/trainer/backward.rs` and Hugging Face gradients as the oracle. Candle's own oracles are hand values and PyTorch-derived numbers in comments (`grad_tests.rs` lines 260 to 290).
3. **CUDA quantized matmul kernels (mmvq and mmq, Q4_0 to Q6_K).**
   - Why: quantized decode speed. THME has WGSL GEMV for q4_k, q5_k, iq4_xs, q8dot and one CUDA `decoder.cu`; candle has mmq tiles for q2_k, q3_k, q4_0, q4_1, q4_k, q5_0, q5_1, q5_k, q6_k, q8_0 (`candle-kernels/src/mmq_gguf/`).
   - Translate: kernels are llama.cpp-derived (licence note above). Candle compiles with nvcc in `build.rs` (lines 14 to 23); THME checks in PTX, so the build path differs.
   - Drags in: nvcc build step. Effort: L. Risk: medium (numerics, GPU arch coverage).
   - Test: THME CPU dequantized forward (`src/inference/llama.rs`) as oracle, with dot-product error bounds.
4. **CPU repacked quantized matmul and ISA kernels.**
   - Paths: `quantized/repack.rs`, `repack_x86.rs`, `avx.rs`, `neon.rs`, `simd128.rs` (sizes only; not read). THME has `src/inference/llama/simd.rs`.
   - Effort: M to L. Risk: medium, per ISA. Test: THME scalar path, bit-exact on integer dots.
5. **More architectures** (Qwen2 and Qwen3, Mistral, Phi-3, Mixtral and Qwen MoE, Mamba, RWKV).
   - Paths: `candle-transformers/src/models/qwen2.rs`, `qwen3.rs`, `mistral.rs`, `phi3.rs`, `mixtral.rs`, `mamba.rs`, `rwkv_v7.rs` (not opened; value unverified). Map onto THME's `CausalLm` trait (`src/inference/arch.rs`).
   - Effort: S for Qwen2 and Mistral (LLaMA variants), M to L for MoE and state-space. Risk: low. Test: Hugging Face logits.
6. **GGUF reader and writer** (`quantized/gguf_file.rs`: read 469, tensor 569, write 590). THME already reads and writes GGUF with tests; use candle only as a second opinion in tests. Effort: S. Not a port.
7. **MoE kernels** (`candle-kernels/src/moe/`). Needed only once THME runs a MoE model. Effort: L. Defer.

## 5. What NOT to port

- Metal backend (`candle-core/src/metal_backend`, `candle-metal-kernels`): THME targets CUDA and wgpu, and wgpu already reaches Metal.
- candle-pyo3, wasm examples, candle-onnx, and the vision, audio and diffusion models: outside a quantization workbench.
- The full `Tensor` and `Op` stack just to get autograd (item 2): take the design lesson, not the code.
- candle's dense `Cache` and `KvCache` (`candle-nn/src/kv_cache.rs`) and sampler: THME already has quantized KV (`src/inference/kv.rs`, `polar_kv.rs`).
- Candle has no ISQ, paged attention, speculative decoding or LoRA layer that I found (only `deepseek2.rs` and `rwkv_v7.rs` matched a grep for gptq, awq or lora; not examined). Use mistral.rs for those.
