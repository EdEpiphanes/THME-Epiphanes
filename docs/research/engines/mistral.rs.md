# mistral.rs survey for THME

Read-only survey of the shallow clone at `C:\Users\TechTroniX\engines\mistral.rs` (paths relative to it).
S, M, L are estimates, not measurements. Nothing was run. Many files were sized and skimmed by headings only; those are marked.

## 1. Licence

- Root file: `LICENSE` only (single licence, no `-APACHE` twin). No `NOTICE*` file (`ls NOTICE*` failed). `Cargo.toml` line 47: `license = "MIT"`.
- `LICENSE` begins: "MIT License / Copyright (c) 2024 Eric Buehler / Permission is hereby granted, free of charge, to any person obtaining a copy ...". Holder: Eric Buehler.
- Sub-crate caveat: `mistralrs-quant/Cargo.toml` line 12 says `license = "MIT AND Apache-2.0"`, and `mistralrs-quant/third_party/` holds `deepgemm_sm90` and `ggml_mmvq` (names only; their licences were not read, unverified).
- Embedded Apache-2.0 code: `mistralrs-paged-attn/src/cuda/pagedattention.cuh` header says "Adapted from NVIDIA FasterTransformer ... Copyright (c) 2023, The vLLM team. Copyright (c) 2020-2023, NVIDIA CORPORATION ... Licensed under the Apache License, Version 2.0". The block pool and prefix-hash code follows the vLLM v1 design (`paged_attention/block_pool.rs` lines 1 to 11), an idea not a copy as far as the comments say.
- Obligations if THME copies code: MIT needs the Buehler copyright and permission notice kept; any file derived from the vLLM or NVIDIA kernels also needs the Apache-2.0 text, kept notices and a change statement. All fit inside AGPL-3.0-only if recorded in THME's third-party register. Prefer re-deriving from the algorithms.
- It builds on candle: `Cargo.toml` lines 51 to 54 depend on `candle-core`, `candle-nn`, `candle-flash-attn-v3`, `candle-metal-kernels` 0.11.1. So the model code is candle-tensor code; porting means translating off candle's `Tensor` (see `candle.md`).

## 2. What it is

A Rust LLM inference server and library (OpenAI-compatible HTTP, Python, Rust API) built on candle, with in-situ quantization, paged attention with prefix caching, speculative decoding, LoRA and X-LoRA, multimodal and MoE models. The architecture is three layers (server, engine, pipeline) with one engine thread and one scheduler per loaded model (`docs/src/content/docs/developer/architecture.md`). It is an application stack for serving; THME is a measurement workbench, so the transferable parts are the quantization and memory-planning ideas, not the server.

## 3. Components

| Component | What it does | Paths opened |
| :--- | :--- | :--- |
| mistralrs-quant | ISQ types, imatrix, UQFF format, GPTQ and Marlin, HQQ, AFQ, FP8, MXFP4, GGUF kernels | `mistralrs-quant/src/isq_type.rs`, `imatrix.rs`, `uqff/mod.rs`, `isq_executor.rs` (comments), directory listings of `gptq/`, `hqq/`, `gguf/` |
| ISQ pipeline | quantize at load, topology overrides, online calibration | `mistralrs-core/src/pipeline/isq_tests.rs`, `topology/mod.rs`, `docs/.../guides/quantization/online-calibration.mdx`, `guides/perf/topology.mdx` |
| paged attention | block pool, prefix hashing, scheduler, cache engine, CUDA kernels | `mistralrs-core/src/paged_attention/mod.rs`, `block_pool.rs`, `scheduler.rs`, `mistralrs-paged-attn/src/cuda/pagedattention.cuh` |
| speculative | MTP heads and drafters, DFlash block drafter, verifier, autotuner | `mistralrs-core/src/speculative/mod.rs`, `config.rs`, `dflash/mod.rs`, `proposer.rs`, `verifier.rs`, `docs/.../guides/perf/speculative-decoding.mdx` |
| LoRA, X-LoRA | LoRA linear layers, runtime adapter registry, X-LoRA classifier | `mistralrs-core/src/lora/mod.rs`, `xlora_models/classifier.rs`, `adapter/` listing, `docs/.../guides/customize/lora-adapters.mdx` |
| engine, scheduler | per-model engine thread, continuous batching, multi-model registry | `mistralrs-core/src/engine/mod.rs` (sizes), `scheduler/mod.rs`, `lib.rs` (`add_model`, line 2279), `developer/architecture.md`, `developer/inference-runtime.md` |
| models | llama, mistral, mixtral, gemma, gemma2, glm4, glm4_moe, gpt_oss, granite, lfm2, lfm3, phi2, phi3, phi3_5_moe, qwen2, qwen3, qwen3_moe, qwen3_next, smollm3, starcoder2, deepseek2, deepseek3, hunyuan, quantized_llama (listing of `mistralrs-core/src/models`) | listing only |

Notes from the files opened:
- ISQ: `IsqType` (`isq_type.rs` lines 4 to 30) lists Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K to Q6K, Q8K, HQQ8, HQQ4, F8E4M3, AFQ2 to AFQ8, F8Q8, MXFP4. `IsqBits` maps a bit target to a type per platform. imatrix statistics are collected during forward passes, including per-expert for MoE (`imatrix.rs` lines 20 to 135), saved as `.cimatrix` (line 173). Online calibration requantizes a running server from live traffic; only K-quants use it (`online-calibration.mdx`).
- Topology: a YAML maps layer ranges to device and ISQ type (`topology.mdx`), the closest thing to a THME recipe, but with no measured gate.
- Paged attention: flat block pool with a doubly linked LRU free list, content-hash prefix caching (`block_pool.rs` 1 to 11, `block_hash.rs` 558 lines, not read), scheduler with chunked prefill and starvation counter (`scheduler.rs` lines 66 to 77), plus a memory plan that sizes the cache at load (`inference-runtime.md` headings "Sizing the KV cache at load", "Admitting each forward pass").
- Speculative: `SpeculativeConfig` is `Off` or `Mtp(MtpConfig)` (`config.rs` 19 to 43). Drafters are a separate assistant model, a built-in multi-token-prediction head, or a DFlash block-diffusion drafter (`dflash/mod.rs` lines 1 to 8). A verifier does accept or reject with uniforms and a sparse CUDA rejection path (`verifier.rs` around 343 to 395). Works only with paged attention (`speculative-decoding.mdx` last line read). Classic small-draft-model speculation is not the focus.
- LoRA: per-request adapter choice from several loaded adapters, hot load and remove while serving; X-LoRA blends adapters per token through a trained classifier (`lora-adapters.mdx`, `classifier.rs` struct at line 20). Training of LoRA is not here (not found; unverified).
- Multi-model: one engine thread per model, requests routed by model id, models added at runtime via `add_model` (`architecture.md`, `lib.rs` line 2279).

## 4. What THME lacks that mistral.rs has (ranked)

THME today: see `candle.md` section 4 and `README.md`. Relevant: THME has a measured quality gate and per-tensor recipes but no server, no paged KV, no real speculative runner (`src/speculative/engine.rs` defines verification strategies; README says "Model", no timing measured).

1. **imatrix collection from activations, including MoE and online recalibration.**
   - Why: ties straight into THME's measure-then-quantize loop; calibration data drives K-quant rounding (`src/inference/calibration.rs`, `src/exporter/calibration.rs`).
   - Paths: `mistralrs-quant/src/imatrix.rs` (453 lines), `online-calibration.mdx`.
   - Translate: a per-linear accumulator of squared activations hooked into THME's forward; the file format `.cimatrix` is project-specific, use llama.cpp's imatrix for interchange.
   - Drags in: nothing if re-derived. Effort: S to M. Risk: low.
   - Test: oracle is the plain definition (mean squared activation per input column, hand-computed on a tiny matrix) and llama.cpp's imatrix on the same prompts.
2. **Topology files: per-layer-range format map.** `topology/mod.rs`, `topology.mdx`. THME recipes already do this per tensor; porting is not needed, but the layer-range syntax is a cheap UX win. Effort: S. Risk: low. Test: parser unit tests.
3. **Memory planning and admission** (size KV cache at load, admit each forward). `docs/.../developer/inference-runtime.md`, `mistralrs-core/src/memory_plan.rs`, `memory_budget.rs`, `resource_plan.rs` (names only, unread). THME has `thme pathfinder` predictions and `src/inference/gpu/budget.rs`; a feedback loop from real free memory could close the gap. Effort: M. Risk: medium (driver query differences). Test: compare predicted against measured bytes, which THME already labels.
4. **Paged KV with prefix caching.** `paged_attention/block_pool.rs`, `kv_cache_manager.rs`, `block_hash.rs`, kernels `mistralrs-paged-attn/src/cuda/pagedattention_v1_f16.cu`, `reshape_and_cache_kernel.cu`.
   - Why: needed for batching and long contexts; THME's KV is single-sequence (`src/inference/kv.rs`) and its README lists "GPU cache; 4,096-token context" as in progress.
   - Translate: block pool and hash logic are pure Rust (portable, M); the CUDA kernels are Apache-2.0 vLLM-derived (L, needs PTX build). A wgpu paged attention would be new work.
   - Drags in: nothing beyond kernels. Effort: L. Risk: high. Test: oracle is THME's contiguous-cache forward; logits must match within fp tolerance; plus the invariants in `scheduler_tests.rs` (e.g. block hash behaviour tests at lines 270 to 440).
5. **Quantization formats: HQQ, AFQ, FP8, MXFP4, GPTQ with Marlin.** `mistralrs-quant/src/hqq/` (quantize.rs, optimize.rs), `gptq/marlin_backend.rs`, `mxfp4/`, `fp8/` (listings only). HQQ is calibration-free and cheap to port conceptually (optimizer over scale and zero). Effort: M for HQQ, L for Marlin. Risk: medium. Test: round-trip error bounds plus perplexity through THME's gate.
6. **Speculative decoding with a real verifier.** `speculative/verifier.rs`, `proposer.rs`. The accept or reject math is standard; THME's `src/speculative/engine.rs` has the strategies. A measured runner would turn "Model" into "Measured". Effort: M (greedy plus rejection sampling with a small draft model; MTP and DFlash are not needed). Risk: medium. Test: oracle is distribution equality, verified by sampling statistics, and greedy equality with the target-only run.
7. **LoRA serving** (hot-swap adapters, merged vs unmerged path). `lora/loralinear.rs` (251 lines), `adapter/`. THME already has LoRA training and merge. Only the runtime registry is new. Effort: S to M. X-LoRA classifier: skip (see 5).
8. **Engine and scheduler design, multi-model.** One thread per model, continuous batching (`scheduler/default_scheduler.rs`, 790 lines, unread). Needed only if THME becomes a server. Effort: L. Defer.

## 5. What NOT to port

- The HTTP server, MCP, agentic loop, sessions, web search, code execution, sandbox crates (`mistralrs-server-core`, `mistralrs-mcp`, `mistralrs-sandbox`, `mistralrs-code-exec` directories): off mission and a large attack surface for a security-minded tool.
- Vision, audio, speech, diffusion and video pipelines, Metal paths: outside scope.
- X-LoRA classifier and AnyMoE (`xlora_models/`, `amoe/`): research-grade, per-architecture copies, and no measured value for a quantization gate.
- DFlash and built-in MTP drafters: tied to specific model families (`docs/.../inference-runtime.md` names Gemma 4, Qwen3.5, Qwen4-Exp).
- CUDA graph machinery (`pipeline/cuda_graph/`): worth revisiting only after THME's decode loop is measured.
- Anything that needs candle's `Tensor`: the code is written against it, so copying files drags the candle dependency.
