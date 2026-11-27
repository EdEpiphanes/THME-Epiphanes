# THME Engineering Backlog

> Source of truth for planned work. Tickets between the `backlog:start` and `backlog:end` markers
> are mirrored to GitHub Issues by `scripts/sync_github_backlog.py`. Process: [docs/WORKFLOW.md](docs/WORKFLOW.md).
> Milestones and exit criteria: [ROADMAP.md](ROADMAP.md).

Each ticket lists: milestone, roadmap item, priority (p0 stop the line · p1 this milestone ·
p2 next), area, type and status (triage · ready · blocked), plus the observable result that
closes it.

<!-- backlog:start -->

## Active: v0.2.0 close-out

### [THME-2005] Release v0.2.0

- **Milestone**: v0.2.0 · **Roadmap**: v0.2.0.9 · **Priority**: p1 · **Area**: ci · **Type**: chore · **Status**: deferred (owner decision)
- **Problem**: The measured core is complete but unreleased.
- **Done means**: CHANGELOG section `[0.2.0]`, Cargo version 0.2.0, tag `v0.2.0`; the release workflow publishes four signed archives with checksums.
- **Progress** (27 November 2026): The owner can release v0.2.0 but the version number lags the work done (engine, GPU path, formats, gate, sampler). Decide the release version and name first, then do the CHANGELOG section, Cargo version and tag; no code is blocking.

### [THME-2205] Second-order rounding (GPTQ-style)

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.5 · **Priority**: p2 · **Area**: byoa · **Type**: feature · **Status**: done
- **Problem**: Error compensation across columns is the strongest known post-training method.
- **Progress** (19 November 2026): `thme.gptq_q4_0` and `thme.gptq_q8_0` follow the reference algorithm (scales fitted per 32-column group, Cholesky of the inverse Hessian, damping with retry). Tested on synthetic matrices: output error below plain rounding, result exactly on the exporter grid. Not measured on a real model; K-quants not supported yet.
- **Progress** (27 November 2026): measured on both models against round-to-nearest, AWQ and the ceiling (`docs/benchmarks/*/data/gptq_comparison.json`); issue closed 27 November 2026.
- **Done means**: Implemented as a BYOA rule using calibration activations; measured against THME-2204.

### [THME-2206] Measure Hadamard rotation on real models

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.6 · **Priority**: p2 · **Area**: research · **Type**: experiment · **Status**: done
- **Problem**: `RotatedQuantize` lowers rounding error on outlier-heavy weights in tests; its effect on perplexity is unmeasured.
- **Progress** (27 November 2026): measured with and without rotation on both models; the 4-bit and ternary arms were unusable, no benefit claimed (`docs/benchmarks/*/data/hadamard_comparison.json`); issue closed 27 November 2026.
- **Done means**: `thme.hadamard_int4` and `thme.hadamard_ternary` compared with their unrotated rules via `thme evaluate` on both models; reports committed.
- **Progress** (27 November 2026): measured on both models under a pre-registered 1 percent rule (`docs/benchmarks/*/data/hadamard_comparison.json`). Pythia-14M (CPU): int4 rotation worse (311,383 to 1,714,157), ternary better (37,099 to 28,238). SmolLM-135M (CUDA): both better (int4 7.3M to 466k, ternary 75M to 28.6M). Every arm is unusable against the ceilings (114.3 and 19.80): the whole-tensor absolute-maximum scale destroys these models with or without rotation, so rotation is not a consistent improvement and no quality claim follows. Status was blocked; the measurement part of the done-means is met.

### [THME-2208] llama.cpp compatibility checks for exported files

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.8 · **Priority**: p2 · **Area**: exporter · **Type**: feature · **Status**: ready
- **Progress** (19 November 2026): `src/exporter/gguf_check.rs` checks a finished GGUF v3 file: header, counts, alignment, tensor names, dimensions, types, truncation, and now the data section (offsets, sizes, bounds, overlap) and the metadata keys (architecture, the five llama keys). About 30 tests on synthetic files, plus a test that a file written by THME's own writer passes. Not done: the optional run through llama.cpp. Three spec facts were written from memory and are marked `shortcut:` in the code: the alignment rule (multiple of 8, power of two), the data-section start rule, and the llama key names. Confirm them against the GGUF specification before relying on the check.
- **Progress** (20 November 2026, llama.cpp b12584, SmolLM2-1.7B; report `docs/benchmarks/smollm2-1.7b/data/llama_cpp_reference.json`): the optional run through llama.cpp was done and it does not pass. `examples/check_gguf.rs` runs the checker (F16, Q8_0 and Q4_0 files: 0 errors, 0 warnings), yet llama.cpp (CUDA and Vulkan builds) refuses all of them: first `cannot find tokenizer vocab in model file` (the GGUF carries no vocabulary), then `tensor 'token_embd.weight' not found` (THME writes Hugging Face tensor names, not GGML names). Also missing or different for llama.cpp: BF16 norm weights (needs F32), q/k rows not permuted for its rope layout, no `llama.rope.dimension_count`. With an outside-the-repo adapter that fixes exactly those points (weight blocks copied unchanged apart from the q/k row permutation) all three files load and generate the same sensible text on both builds. Also found: `thme export --quant fp16|q8_0` fails on this BF16 checkpoint (recipe path works). Not done: the exporter writing GGML names, tokenizer and the permutation itself (follow-up ticket needed; the check above is the acceptance test).
- **Problem**: Exported GGUF files are only re-read by THME itself.
- **Done means**: The existing optional CI step runs a heterogeneous export through llama.cpp when the runtime is configured.

### [THME-2101] Live 3D: attention maps, drill-down and side-by-side

- **Milestone**: v0.2.3 · **Roadmap**: v0.2.1.9 extensions · **Priority**: p2 · **Area**: gui · **Type**: feature · **Status**: ready
- **Problem**: The Live 3D view shows per-tensor activity but not per-head attention, single weights, or source-vs-candidate differences.
- **Done means**: The four extensions listed in ROADMAP.md under "Studio Live 3D view" are implemented and recaptured.

### [THME-2300] Epic: Recovery and distillation

- **Milestone**: v0.2.3 · **Roadmap**: v0.2.3.1-v0.2.3.9 · **Priority**: p2 · **Area**: research · **Type**: feature · **Status**: triage
- **Problem**: Aggressive formats such as ternary fail without retraining.
- **Done means**: Broken into tickets per ROADMAP.md v0.2.3 when v0.2.2 closes.

### [THME-2400] Epic: Runtime, speed and KV cache

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.1-v0.2.4.9 · **Priority**: p2 · **Area**: inference · **Type**: feature · **Status**: triage
- **Problem**: No speed, memory or KV-cache figure is measured yet.
- **Done means**: Broken into tickets per ROADMAP.md v0.2.4 when v0.2.3 closes.

### [THME-3079] Token-frequency count for Pythia-14M (weights-only)

- **Milestone**: v0.2.4 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: experiment · **Status**: in review
- **Problem**: Pythia-14M stores about 92 percent of its parameters in the two embedding matrices, and the output head is about 85 percent of the bytes read per decoded token. Which rows matter depends on how often tokens occur, and nothing counts that yet.
- **Progress** (20 November 2026): `examples/token_frequency.rs` counts the 2,419,837 WikiText-2 train tokens (Pythia tokenizer, no model pass) and writes `docs/benchmarks/pythia-14m/data/token_frequency.json`, pinned by the token-file, corpus and tokenizer hashes. Measured: the top 1,024 / 4,096 / 8,192 rows cover 66.2% / 82.6% / 90.7% of train tokens; the same train rows cover 65.4% / 81.2% / 89.2% of the 287,159 test tokens (rows ranked on the test split itself would cover 69.3% / 87.0% / 94.5%). 17,145 of 50,304 rows never occur in train; 848 test tokens (0.3%) fall in them. Unit test on a tiny synthetic array passes.
- **Not done**: citation from THME-3081 and the banded-row experiments of THME-3080.
- **Done means**: A command or example counts token frequencies over the WikiText-2 train split with the model's own tokenizer (no model forward pass), writes a hash-pinned frequency table, and reports the share of tokens covered by the top 1,024, 4,096 and 8,192 rows; the result is committed as a report and cited by THME-3080 and THME-3081.

### [THME-3080] Per-row precision bands for the embedding and output matrices (Pythia-14M)

- **Milestone**: v0.2.4 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: experiment · **Status**: in progress
- **Problem**: The linear layers are about 8 percent of Pythia-14M, so uniform quantisation of them barely changes its size; the embedding rows are the lever. Rare tokens may tolerate fewer bits than frequent ones.
- **Progress** (20 November 2026): first question measured (rule written in the report before the first run; `docs/benchmarks/pythia-14m/data/embedding_vs_linear.json`, GPU, RTX 4060 Ti, full test split, default gate). At Q8_0 the perplexity rise over the 114.286 baseline is +0.083 for embeddings and head alone (passes), +0.666 for the linear layers alone (fails the 0.45 margin), +0.744 for both (fails); by the rule the linear layers are the cause. At Q4_0 the rises are +12.4, +78.7 and +102.1, all failing. The written GGUF is 16.07 MB (embeddings Q8_0), 27.04 MB (linears Q8_0), 14.96 MB (both) at Q8_0, and 9.63, 26.45 and 7.93 MB at Q4_0. The reproduction check missed: both-at-Q8_0 gives +0.744, the committed uniform Q8_0 report says +0.834 (limit 0.02); CPU gives the same +0.744, so the device is not the cause. Untested guess: the committed path also quantises norms and biases.
- **Not done**: frequency-banded rows with shuffled-band controls, the uniform-lower-bit comparison at equal bytes, the `.epiph` byte counts, the 0.83 gap explained, SmolLM-135M.
- **Done means**: Measured on Pythia-14M with the existing quality gate and a pre-registered rule written in the report before the run: bytes of the written `.epiph` and perplexity on the WikiText-2 test split for uniform Q8_0, uniform lower-bit, and frequency-banded rows (bands from THME-3079), each with a shuffled-band control; also reports whether the Q8_0 gate failure on Pythia comes from the embeddings or the linear layers. A positive result is the condition for the per-row-band codec of THME-3078.

### [THME-3072] GPU path faster than CPU on the small models

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.3 · **Priority**: p1 · **Area**: gpu · **Type**: perf · **Status**: in review
- **Related**: THME-3016 (placement plan, no runtime reads it yet), THME-3038 (Studio device selector, GPU labelled experimental), THME-3061 (GPU probe and CPU-wins rule), THME-3045/3046 (GPU unlock, unrelated to compute).
- **Problem**: The GPU path ran one matrix-vector product per prompt row with a submit and a wait each (26,881 submits per 128-token pass on SmolLM-135M), re-uploaded every weight tensor on every pass (211 uploads, 774 ms), and ran only the linear layers on the device. Measured before: 7,680 ms (GPU) vs 372 ms (CPU) for 128 tokens (`docs/benchmarks/smollm-135m/data/gpu_profile_before.json`).
- **Progress** (20 November 2026): batched tiled matrix-matrix kernel (one dispatch for all rows), linears sharing one command buffer with one read-back, slice weights resident on the device, explicit adapter choice over Vulkan, Metal and DX12, and a device-resident LLaMA-style decoder (norms, RoPE, causal attention, SwiGLU, residuals, head) in a single submit. Measured after (RTX 4060 Ti, Vulkan, release build, 11 interleaved pairs, CPU job loading the machine at 60 to 90 percent, so CPU timings are biased against the CPU): SmolLM-135M 128-token pass GPU 102 ms vs CPU 552 ms, 2-token pass 7.7 vs 120.8 ms; Pythia-14M 128-token pass 17.7 vs 36.6 ms, 2-token pass 9.6 vs 14.7 ms (GPU faster in 9 of 11 pairs). Report: `docs/benchmarks/*/data/gpu_profile.json`; GPU vs CPU logits within 5e-3 (worst observed 3.7e-3 on Pythia-14M, 1.6e-5 on SmolLM-135M).
- **Not done**: Pythia-14M (GPT-NeoX) still uses the per-operation GPU path (25 submits per pass) because the device-resident decoder covers LLaMA-style layers only; `forward_nll`, scheduled, traced and exit-readout passes use the per-operation path; the "decode" figure is a 2-token pass, not an incremental decode step (no KV cache on the device, owned by the incremental-decoding work); contexts above 2048 tokens fall back; f16 and subgroup kernels not tried; attention kernel is simple (about 97 ms of the 102 ms SmolLM prefill is device time, GEMM about 1 to 2 TFLOPS); the CPU comparison is not clean while the other job runs; Metal and DX12 were not run (no such hardware or run here).
- **Done means**: GPU beats CPU on paired, interleaved runs for both small models at 128-token prefill and decode steps on a quiet machine; parity tests green; a committed report with device, build and method.

### [THME-3084] GPU speed round 2: profile, matmul, attention, resident weights, f16 and quantized weights

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.3 · **Priority**: p1 · **Area**: gpu · **Type**: perf · **Status**: in review
- **Related**: THME-3072 (round 1), THME-3073 (GPU perplexity), THME-3070 (KV cache), `docs/research/gpu-backends.md` (Compute Bridge).
- **Problem**: A full WikiText-2 test pass of SmolLM2-1.7B on the GPU took about 23 minutes (4.46 ms per token, 224 tokens per second, about 0.75 TFLOPS against a card peak of about 22 TFLOPS for f32). Nobody had profiled it per kernel.
- **Profile** (RTX 4060 Ti, Vulkan, driver 616.92, one 1,024-token window, per-dispatch timestamp queries, `docs/benchmarks/smollm2-1.7b/data/gpu_speed_round2.json`): of 4,641 ms wall, 2,737 ms was weights re-uploaded because the 4 GiB resident budget flushed a 6.9 GB model every window (218 uploads), and of 1,665 ms device time attention took 818 ms (49 percent) and the matmuls 830 ms (50 percent, about 4 TFLOPS each). Norms, RoPE, residual adds, SwiGLU and the loss together were under 1 percent.
- **Progress** (20 November 2026), 16,384-token slice, GPU only, same binary with flags, one change at a time: baseline 73.0 s (224 tokens/s); register-blocked 128 x 128 matmul (8 x 8 outputs per thread, vec4 loads, padded shared stride) 65.1 s (the re-upload dominates); weights kept resident (`set_weight_budget`) 19.3 s; tiled causal attention with online softmax 7.7 s (2,113 tokens/s, 9.4 times the baseline). Matmul alone, per shape: 4.0 to 4.6 TFLOPS before, 7.3 to 9.1 after (8.8 to 9.5 on the large shapes in the in-model profile); target of 5 reached. Full test split (303,118 scored tokens, 297 windows): **145.5 s** (2,084 tokens/s) f32 and 139.9 s f16, against an extrapolated 1,352 s (22.5 minutes) for the baseline; the machine was loaded by other sessions (CPU mean 32 percent, so not the quiet run the plan asked for; treat as an upper bound). SmolLM-135M full split 97.8 s to 30.0 s (3.3 times); Pythia-14M unchanged (34.4 s to 31.5 s, within noise) because it runs the per-operation path with CPU attention.
- **Weight storage** (`GpuContext::set_weight_storage`, f32 accumulation everywhere): f16 (packed two per word, no shader-f16 feature needed), Q8_0 and Q4_0 (scales plus codes, dequantized in the kernel, GGML block semantics), for batched matmul and few-row matrix-vector kernels, and for quantized `QuantizedTensor` (Q8_0, Q4_0, F16) so a quantized model stays quantized on the card. Resident bytes per weight: 4, 2, 1.125, 0.625. Parity on the 16,384-token slice: f32 paths within 4.1e-8 nats of the baseline (tolerance 1e-4); f16 7.1e-8 (lossless for bf16 and fp16 checkpoints); Q8_0 +5.2e-4 nats (perplexity 8.377 against 8.373); Q4_0 +0.165 nats (9.870), the real quality cost. Q8_0 and Q4_0 kernels match the CPU block decoders in tests. Speed on this card is the same as f32 (7.4 s against 7.7 s): prefill is compute-bound, so the gain is memory (SmolLM2-1.7B 6.9 GB to 3.4, 1.9 or 1.1 GB) and decode bytes per token, not prefill time.
- **Not done / did not help**: Pythia-14M (GPT-NeoX) has no device-resident decoder (null result above); a 32-query attention block was tried and was not faster (reverted); f16 arithmetic was not tried because this GeForce runs fp16 ALU at the f32 rate and wgpu exposes no tensor-core path, so f16 saves bytes, not time; the matmul is at about 9.5 of about 22 TFLOPS (double buffering, 8 x 8 bank swizzle and subgroup operations untried); attention is at about 1.2 TFLOPS (further tiling untried); no incremental decode on the device (no device KV cache); a model larger than the weight budget still re-uploads every pass (flush-all, no LRU); Metal and DX12 not run.
- **Compute Bridge notes**: the per-dispatch profile (`set_op_profile`) and the paired old/new flags (`--old-gemm`, `--old-attention`) are the tools to compare any new backend against; all kernels are original WGSL from the public WebGPU specification and the tiling ideas in `docs/research/gpu-backends.md`, no code from other engines. The weight budget is a setting because wgpu does not report VRAM: a future run planner (`thme pathfinder`) should set it from the adapter and from `nvidia-smi` style probes, and pick the storage format from the free memory.
- **Done means**: Paired before and after per step on SmolLM2-1.7B, SmolLM-135M and Pythia-14M with device, driver, build, method and load note in a committed report; parity tests green; a full-split time on a quiet machine (the committed one was taken under load, repeat when quiet).

### [THME-3085] Reference comparison against llama.cpp on this machine

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.3 · **Priority**: p1 · **Area**: gpu · **Type**: perf · **Status**: ready
- **Problem**: The only outside numbers we have (community scoreboards: about 3,400 prompt tokens per second for a 7B model at Q4_0 on an RTX 4060 Ti with CUDA; reported, mixed builds) are not measured on our hardware, so the gap to mature engines is unknown.
- **Progress** (20 November 2026): measured on this machine with llama.cpp b12584 (CUDA 12.4 and Vulkan builds, official release, digests verified; reference report and compatibility findings on branch `feat/llama-cpp-reference`, pull request pending): prompt processing at 1,024 tokens 11,348 to 14,348 tokens per second and generation 74 (F16), 131 (Q8_0), 221 to 225 (Q4_0) for SmolLM2-1.7B; THME at that time 2,090 to 2,170 tokens per second for the full pass, after THME-3087 about 7,000 on a 16,384-token slice with f16 weights (different measurement shapes: see the report).
- **Done means**: llama.cpp (MIT, prebuilt binaries downloaded into a separate folder, never bundled) run through its own benchmark tool on the same SmolLM2-1.7B file in its CUDA and Vulkan builds: prompt processing and token generation tokens per second, build number, flags, driver and load recorded in a committed report; our passes compared to those rows with the same token counts; the result sets the real target for THME-3086 to 3089.

### [THME-3086] GPU feature audit: what the adapter really exposes, with checked kernels

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.3 · **Priority**: p1 · **Area**: gpu · **Type**: feature · **Status**: in review
- **Related**: THME-3084 (speed round 2), THME-3087, THME-3088, `docs/benchmarks/_adapters/rtx-4060-ti_features.json`.
- **Problem**: Round 2 assumed "wgpu exposes no tensor-core path" and "fp16 ALU runs at the f32 rate". Nobody had asked the adapter.
- **Progress** (20 November 2026): `thme::inference::gpu::features::audit` and `examples/gpu_features.rs` request every optional feature the adapter lists and run one tiny kernel per feature whose output is compared with a CPU result (a listed feature with a failing kernel is reported as failed). RTX 4060 Ti, Vulkan, driver NVIDIA 616.92, wgpu 30.0.2: SHADER_F16 present and checked; SUBGROUP present, subgroup size 32 (min and max), checked; EXPERIMENTAL_COOPERATIVE_MATRIX present, shapes 16x16x16, 16x8x16 and 16x8x8 with f16 inputs and f16 or f32 accumulate (no f32 inputs, no integer shapes), the 16x16x16 f16/f32 and f16/f16 kernels checked; packed 4x8 integer dot product (`dot4I8Packed`) compiles, runs and gives the right numbers (the report does not say whether the driver runs it as a single instruction). The DX12 adapter of the same card exposes none of f16, subgroups or cooperative matrices in wgpu 30. naga 30 accepts only square 8x8 and 16x16 cooperative matrices, so the 16x8 shapes the card lists are not reachable from WGSL.
- **Not done**: `thme doctor` does not call the audit (not cheap: it needs the device with experimental features); Metal and Windows-on-ARM not run.
- **Done means**: A committed per-adapter JSON with device, driver, backend, each feature present or absent, the matrix shapes, and a passed or failed kernel check per feature. Met for the RTX 4060 Ti.

### [THME-3087] Cooperative-matrix (tensor-core) matmul and attention

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.3 · **Priority**: p1 · **Area**: gpu · **Type**: perf · **Status**: in review
- **Related**: THME-3086, THME-3084, `docs/benchmarks/smollm2-1.7b/gpu_tensor_core_plan.md` (rules written before the runs), `docs/benchmarks/smollm2-1.7b/data/gpu_coop_*.json`.
- **Rule (pre-registered)**: adopt if at least 25 TFLOPS isolated on large shapes, at least 6,000 tokens per second on the 16,384-token slice, and perplexity within 0.01 nats of the f32 path.
- **Progress** (20 November 2026, RTX 4060 Ti, Vulkan, driver 616.92, release build, GPU only, SmolLM2-1.7B, 16,384-token slice, two runs per arm; shared machine, no other GPU job from this session): original WGSL from the Vulkan cooperative-matrix and wgpu documentation (clean room, no other engine read). Kernels: 128 x 128 tile, four subgroups each owning 64 x 64 (sixteen 16 x 16 f32 accumulators), f16 weights and f16 activations read in place (activations converted once per matmul by a small pass), row tiles scheduled fastest so the workgroups in flight share each weight tile in L2 (the head matmul went from 12 to 40 TFLOPS with that one change); a staged variant converts f32, Q8_0 and Q4_0 weights in workgroup memory (slower, 10 to 20 TFLOPS); a flash-style attention for head size 64 (two passes over key tiles instead of an online rescale). Isolated matmul (1,024 rows; shapes 2048x2048, 2048x8192, 8192x2048, 2048x49152, 576x1536): f32 kernel 9.0, 8.2, 9.1, 9.3, 7.3 TFLOPS; cooperative matrices 28.0, 35.7, 26.1, 40.0, 27.8. Slice, f16 weights: 2,215 tokens/s before, 5,350 with the matmul, **7,000 to 7,020** with matmul and attention (3.2 times); perplexity 8.37288 before, 8.37286 after (3e-6 nats). One 1,024-token window: matmul 368 ms to 99 ms, attention 58 ms to 13 ms. f32 weights (8 GiB budget so they stay resident): 2,099 to 3,951 tokens/s with the staged kernel on every storage. **Go on all three criteria**: cooperative matrices are the default for f16 weight storage (and attention of head size 64) when the adapter passes a start-up probe of the kernels; every other storage keeps the exact f32 kernels unless `set_coop_matrix(true)`; `THME_GPU_COOP=0` turns it off.
- **Parity**: f16 inputs, f32 accumulate; matmul tests against the CPU within 5e-3 relative on tile-edge shapes, attention against the tiled kernel within 5e-3 on four shapes (ragged blocks, GQA), full SmolLM2-1.7B slice within 3e-6 nats.
- **Not done**: attention only for head size 64 (others fall back to the tiled kernel); Q8_0 and Q4_0 still use the slow staged kernel or the int8 route of THME-3088 (a pre-pass that dequantizes to f16 for the in-place kernel is the next try); activations are converted again for each of q, k, v and gate, up (about 2.5 percent); the matmul is at 26 to 40 of a measured 37 to 40 TFLOPS ceiling for this structure without staging; no full-split pass (slice only, the card reached 50 C); Metal and DX12 have no cooperative matrices in wgpu 30.
- **Done means**: Paired before and after with device, driver, build, method and load note in committed reports; parity tests green; a quiet-machine full-split run.

### [THME-3088] Packed 8-bit integer dot product for Q8_0 and Q4_0 weights

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.3 · **Priority**: p2 · **Area**: gpu · **Type**: perf · **Status**: in review (no-go at this tuning)
- **Related**: THME-3086, THME-3087, `docs/benchmarks/smollm2-1.7b/gpu_tensor_core_plan.md`.
- **Rule (pre-registered)**: adopt if at least 20 TOPS-equivalent isolated and at least 6,000 tokens per second on the slice, with the perplexity change reported.
- **Progress** (20 November 2026, same machine and method as THME-3087): activations quantized per block of 32 to int8 on the fly, Q8_0 codes as stored and Q4_0 nibbles unpacked to int8 in the kernel, `dot4I8Packed` in integer, scaled to f32; 128 x 128 tile, 16 x 16 threads, all 64 accumulators in named registers (a first version with indexed arrays ran at 1 TOPS from spilling). Isolated: 10.5, 11.4, 10.3, 12.9, 10.1 TOPS-equivalent on the five shapes (Q8_0; Q4_0 the same), against 9 to 10 for the exact float kernel. Slice: Q8_0 2,205 to 3,028 tokens/s (with the cooperative attention), perplexity 8.3773 to 8.3914 (+0.0017 nats); Q4_0 2,219 to 3,045, 9.8701 to 9.8900 (+0.0020 nats). **No-go**: below 20 TOPS and below 6,000 tokens per second. Kept as an opt-in (`set_int8_dot(true)`, `--int8-dot`) with a parity test against the float kernel.
- **Not done**: tuning (two blocks per staged tile, scale folding, a wider tile); checking whether the driver issues one dp4a per `dot4I8Packed`; the dequantize-to-f16 route for the in-place cooperative kernel; Q4_K and the other K-quants.
- **Done means**: Isolated TOPS and slice tokens per second against the rule above, with the perplexity change, in committed reports. Measured; rule not met.

### [THME-3092] Gate code reaches the declared-range specification

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p0 · **Area**: quality · **Type**: fix · **Status**: triage
- **Problem**: the quality gate used an invalid upper e-process (see docs/research/THME_Paper_v5_Code_Review_2026-11-27.md item 1).
- **Done means**: `hedged_epiphanes_bounds` takes a declared range, fails a block above it, reproduces paper Table 4, and has a coverage test; remaining: surface the range as a `thme validate` option and in the gate report.

### [THME-3093] Run the Step 1 gate on real per-block losses

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p0 · **Area**: quality · **Type**: feature · **Status**: triage
- **Problem**: the gate has never run on a candidate; paper appendix E needs per-window loss files for a source and candidate (148 VAL windows, float32 logits, identity-form loss).
- **Done means**: a committed gate report with decision, bounds, screens and the permutation and source-versus-source checks, labelled Measured.

### [THME-3094] Measure BITCOS and BITCOS-2:4 accuracy

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p0 · **Area**: formats · **Type**: research · **Status**: triage
- **Problem**: no perplexity exists for either format and the GPU kernel on main only handles pre-expanded 2-bit signs.
- **Done means**: a heterogeneous export with WikiText-2 perplexity against the source, and a GPU kernel decoding the real super-block (block_min, group scales, nz mask) with a CPU-oracle test.

### [THME-3095] Q4_0 decode reaches the bandwidth ceiling

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.3 · **Priority**: p0 · **Area**: gpu · **Type**: perf · **Status**: triage
- **Problem**: reported Q4_0 decode is 16.88 tokens per second against 62.87 for F16, 5.6 percent of the 299.2 tokens per second ceiling (RTX 4060 Ti, 288 GB/s).
- **Done means**: Q4_0 decode faster than F16 on the same prompt with the fraction of the ceiling reported, in a committed report.

### [THME-3096] Commit JSON reports for the SmolLM2-1.7B GPU, decode and screening runs

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p1 · **Area**: benchmarks · **Type**: chore · **Status**: triage
- **Problem**: the 9,267.8 tokens per second prefill, 62.87 decode and AWQ 9.7125 figures exist only in handoff documents and were not re-run on main.
- **Done means**: reports with the git commit recorded, so paper sections 11.1 to 11.3 can carry the Measured label.

### [THME-3098] Commit the paper and its check scripts

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p1 · **Area**: docs · **Type**: chore · **Status**: triage
- **Problem**: paper v5 cites checks/verify_all.py, checks/make_figures.py and figures/ that are not in the repository.
- **Done means**: the paper source and scripts committed with licence headers, CI runs the verification script, paper labels updated for the merged code.

### [THME-3099] Loop breaker behaviour matches its specification

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p2 · **Area**: inference · **Type**: fix · **Status**: triage
- **Problem**: paper section 7.9 says a detected loop raises temperature and resets DoLa; the code stops generation.
- **Done means**: either the behaviour is implemented behind a setting or the paper text states stopping, with a test.
- **Progress** (27 November 2026): owner path chosen: the paper text states stopping. The rule (trailing 4-gram seen three times in the last 256) is pinned by tests in `src/inference/grounded_decode.rs`, and `docs/research/paper_v5_1_notes.md` gives the replacement sentence. Not done: the paper itself is not on disk, so its section 7.9 is not edited yet.

### [THME-3101] Repository-wide status overhaul: roadmap, charts, status files, labels

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: docs · **Type**: chore · **Status**: in progress
- **Problem**: Several top-level files describe an older state: `TASKS.md` (last updated 10 November 2026), `TEST_READY.md` (milestone v0.1.7), roadmap state colours, README figures, status counts, and GitHub labels (only seven of 26 BACKLOG areas existed as labels).
- **Done means**: ROADMAP "Where things stand" and each milestone state re-derived from BACKLOG statuses with no unsupported "done"; README numbers (tests, binary sizes, measured reports) taken from a command run on the merged tree and dated; `scripts/generate_status.py` and `scripts/generate_charts.py` rerun and their outputs committed; `TASKS.md`, `TEST_READY.md`, `PROJECT.md` refreshed or archived under `docs/archive/` with a pointer; GitHub labels cover every BACKLOG area and status; no release tag created (v0.2.0 is deferred by the owner).
- **Progress** (27 November 2026): labels `area:*` for all BACKLOG areas and `status:in-progress|in-review|done|shelved` created on GitHub; status board regenerated (846 tests counted by the generator; 856 passed in a full `cargo test` run on 27 November 2026, release `thme.exe` 14.90 MiB); the paper itself is now in the repository as `docs/research/paper/THME_Epiphanes_Paper_v5.1.md` (version 5 kept unchanged beside it): new Section 3.8 (US EPA PS-16 bands transposed to the Epiphanes gate), Section 7.9 item 4 corrected to the code, Section 8.7 binary size, new Section 11.6 (measurements since version 5); its figures and `checks/verify_all.py` were not on this machine and no PDF of 5.1 was built; `docs/research/paper_v5_1_notes.md` lists measured code state (856 tests, 14.90 MiB), gate declared range and v2, DC-CCCP sign split, the llama.cpp match, KV precision and the softmax note, each with a suggested sentence (the paper itself is still not on disk); README "Now building" table, measurement paragraph, Studio chat note and roadmap diagram brought up to date (llama.cpp match, KV precision, pathfinder, the v0.3.0 node, published reports; two statements that were no longer true were corrected); getting-started and the manual describe the llama.cpp layout export, BF16 `--quant`, `--kv-precision`, `thme pathfinder`, `--allow-root` and gate v2 with their evidence levels (checked: `validate` has no `--gate`, only `evaluate` does); `docs/status/roadmap_vs_backlog.md` compares roadmap states with ticket statuses (finding: v0.2.8 has every ticket done or done in code while the roadmap calls it next; v0.2.1 has no BACKLOG tickets); chart regeneration changed only line endings, so nothing was committed; the 85 legacy tickets without a Status bullet now have one copied from their heading (31 done as v0.1 deliverables, 9 completed, 45 marked triage because no status was written; nothing is marked measured); five stale top-level agent notes untracked (see THME-3050); crate version set to 0.2.0.

### [THME-3102] GGUF export: write the pre-tokenizer type llama.cpp expects

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.8 · **Priority**: p2 · **Area**: exporter · **Type**: bug · **Status**: triage
- **Problem**: A `--llama-cpp-layout` export has no `tokenizer.ggml.pre`, and llama.cpp b12584 warns "missing pre-tokenizer type, using: 'default'" and "generation quality will be degraded". The prompt used in THME-3091 tokenised identically, but other text may split differently.
- **Done means**: The key is written from the public documentation of the value set and a rule that follows from `tokenizer.json` (its pre-tokenizer and split pattern), or the export says it could not tell and what to set; a test with a tokenizer.json of each supported kind; a comparison of token ids between THME's tokenizer and llama.cpp on a few hundred varied strings.
- **Progress** (27 November 2026): `thme export --llama-cpp-layout` now prints a warning that no `tokenizer.ggml.pre` was written (`ExportReport.warnings`, tested). The key itself is not written because choosing the right value from `tokenizer.json` needs the public value set and a tested rule; the ticket stays triage.

### [THME-3103] THME's GGUF reader does not undo the q/k row permutation of llama.cpp layout files

- **Milestone**: v0.2.2 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: bug · **Status**: in review
- **Problem**: The forward pass uses the Hugging Face rotary layout. A GGUF written for llama.cpp (including THME's own `--llama-cpp-layout` export) has permuted q and k rows, so `thme forward` on it gives different logits (SmolLM-135M: top logit 7.53 against 13.06 for the checkpoint; the top token matched for one prompt).
- **Done means**: The GGUF import undoes the permutation for the LLaMA family (using the inverse in `src/exporter/ggml_names.rs`), with a round-trip test (export with the layout, read back, logits equal to the checkpoint within the format tolerance) and a check that a file already in Hugging Face layout is not changed.
- **Progress** (27 November 2026): the loader (`src/inference/llama.rs`) undoes the q/k permutation for tensors named `attn_q`/`attn_k` in a GGUF whose `general.architecture` is `llama`; HF-named tensors and other architectures are left alone. Test `tests/test_gguf_layout_roundtrip.rs`: tiny LLaMA exported with `--llama-cpp-layout` in f32 and read back gives the checkpoint's logits within 1e-4 (largest gap 1.29e-3 without the fix, so the test discriminates). Also tested: an HF-named GGUF (flag off) is not touched, five token sequences, and the architecture key the fix depends on. Not done: a check against a GGUF made by llama.cpp's own converter, and other architectures that use the interleaved layout (none are handled; the check is for `llama` only). Side finding, fixed: when a source recorded no norm epsilon the engine assumed 1e-6 while the exporter wrote 1e-5; both now use `FALLBACK_RMS_NORM_EPS` (1e-6), with a test.

### [THME-3105] Teacher-to-student distillation pipeline (`thme distill`)

- **Milestone**: v0.2.3 · **Roadmap**: v0.2.3.2 · **Priority**: p1 · **Area**: trainer · **Type**: feature · **Status**: triage
- **Problem**: The headline use is a frontier model teaching a smaller open-weight model quickly, with THME measuring the result. THME has the teacher bridge, the gate, `.epiph` provenance and LoRA merge, but no pipeline joins them, and the trainer is synthetic until THME-3036 lands.
- **Done means**: The design in `docs/research/distillation-pipeline-design.md` is built in its first slice: quantization-aware distillation on SmolLM-135M with the source as teacher, a held-out evaluation set whose hash is recorded and never used in training, a decision rule written before the run, cost accounting, and a committed report comparing the result with round-to-nearest and AWQ; then sequence-level distillation from a configured teacher command, with the teacher's identity and terms note recorded in the dataset.
- **Progress** (27 November 2026): first slice measured under a pre-registered protocol (`docs/benchmarks/smollm-135m/data/qad_lora.json`): quantization-aware distillation with LoRA on the Q4_0 SmolLM-135M student, the source model as teacher, no hosted teacher. Test perplexity 20.413 (three seeds, SD 0.018) against round-to-nearest 23.70, AWQ 22.04, GPTQ 22.23 and ceiling 19.80: better than all three by more than 2 sigma, still 0.61 above the ceiling so it fails the 0.45 gate. Cost: 19.6 MB float32 adapter, 4.88M trainable parameters, about 10 minutes of training per run on the RTX 4060 Ti, 2.1M tokens. The replication gate passed (PyTorch reproduces THME's committed numbers to 6 digits) and THME's own evaluate on the merged model agrees to 0.002 percent. Limits: learning rate at the grid edge, one model, adapter not folded into 4-bit weights. Not built yet: sequence-level distillation from a hosted teacher, the teacher registry and the curriculum. Design written. The PyTorch and PEFT bridge is used now for real-model training (owner decision, 27 November 2026, replacing 'later'), with llama.cpp (MIT) as reference engine and tool; a CUDA backward pass is shelved (THME-3107). Owner decision: distillation with a hosted teacher is for functionality testing only, never released or used commercially (files stay local, annotated `use=test-only`, `distill/` is git-ignored). Claude is the first teacher, with a registry for several teachers and a curriculum of subjects (one teacher, prompt set, weight and held-out set per subject). Whether a hosted teacher's usage terms allow this is for the owner and counsel; the pipeline needs a filled-in terms note before it runs one. Release or commercial use of a model taught by a hosted provider's model needs that provider's permission, held by the owner; planned guard (deferred by the owner on 27 November 2026, to be installed later): files annotated `use=test-only` cannot be signed, exported for sharing or released without `--permission-note` (recorded, not checked). Not built.

### [THME-3106] Measurement drift protocol: bias, drift and a tolerance band for every measured result

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.3 · **Priority**: p1 · **Area**: bench · **Type**: feature · **Status**: in progress
- **Problem**: A single run cannot tell a real difference from noise. Contention, thermal state, warm-up and GPU floating-point order move readings, and a reference check before and after a batch is the usual way to expose a bias that every run shares.
- **Done means**: `docs/research/measurement-drift-protocol.md` is applied by the report scripts: each batch has a warm-up run, n repeats in interleaved order, a reference check before and after, a recorded drift, an accept or flag decision against a stated tolerance (5 percent for timings by default, tighter for perplexities), a guard band that turns a verdict into Inconclusive when the drift can flip it, and, when the tolerance is exceeded, an attribution table (run index, other processes, GPU temperature, clock, power, CPU load); committed reports carry the numbers; the compatibility report repeats its GPU runtimes under it.
- **Progress** (27 November 2026): protocol written; `scripts/drift.py` with 14 tests worked by hand from the 100 ppm sensor example. Not yet applied to any committed report. Owner decision: 5 percent for every measured quantity until the benchmark suites (mathematics, literature and others) can assess the models; the tolerance only decides stability, the guard band uses the observed drift.

### [THME-3107] CUDA backward pass inside THME

- **Milestone**: v0.5 · **Roadmap**: unassigned · **Priority**: p3 · **Area**: gpu · **Type**: feature · **Status**: shelved (owner decision, 27 November 2026: a far-future milestone, v0.4 or v0.5)
- **Problem**: THME's backward pass (THME-3036) is CPU only and checked on a tiny model. Training a real model quickly needs GPU kernels for the backward pass; the shortcut for now is an external PyTorch and PEFT trainer.
- **Done means**: Backward kernels for the LLaMA block on the embedded-PTX and cuBLAS path (matmul gradients through cuBLAS; RMSNorm, rotary, attention and SwiGLU kernels), each tested against the CPU backward pass and the Hugging Face oracle (`tests/fixtures/torch_gradients_tiny.json`), then a measured speed against PyTorch on the same model.
- **Progress** (27 November 2026): none. Shelved. No evidence found that NVIDIA has opened CUDA to other GPUs; third-party layers (ZLUDA, SCALE) are untested with THME's PTX.

### [THME-3108] Unified engine port: add what THME is missing from the top permissive engines

- **Milestone**: v0.3.0 · **Roadmap**: unassigned · **Priority**: p1 · **Area**: engine · **Type**: feature · **Status**: in progress
- **Problem**: The best features of the top open-source inference engines (llama.cpp, candle, mistral.rs, burn, vLLM, SGLang, exllamav2, MLC-LLM, ollama) overlap with and exceed THME's engine in places. The owner wants the missing good parts in one Rust codebase to cut coding, testing and research time.
- **Done means**: Each ported feature meets the process in `docs/research/unified-engine-port-plan.md`: a gap THME actually has, a permissive licence read from the LICENSE file, an attribution row, a test whose expected values come from the source engine, the full gate, and a measured effect under the drift protocol; the ranked candidate list exists and its first slice is done; the binary stays under its size caps.
- **Progress** (27 November 2026): nine engines cloned shallowly outside the repository (`<engines>`); licences read from their LICENSE files (llama.cpp, mistral.rs, ollama, exllamav2 MIT; candle, burn, vLLM, SGLang, MLC-LLM Apache-2.0). The nine surveys are written (`docs/research/engines/*.md`) and merged into a ranked, gap-driven list in the plan: six small items first (first slice: rotate before quantizing the KV cache, tested on the Pythia-14M keys case; second: imatrix and the missing encoders, tested against the local `llama-quantize`), then medium and large ones. Licence cautions recorded (two MIT files with no copyright line, one NOTICE file to carry). Nothing is ported yet. Scheduled after the v0.2 merge.

### [THME-3109] Band-capable quality gate (RATA-style tolerances)

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.3 · **Priority**: p2 · **Area**: quality · **Type**: feature · **Status**: triage
- **Problem**: Gate v2 has two regimes (absolute below a switch point, relative above). The owner describes a relative-accuracy-test shape with more bands: 10 percent above 100, an absolute difference below about 20, something in between. `docs/research/rata-style-tolerance-check.md` shows the bands give the same verdicts as v2 on every committed result at an absolute limit of 0.45, so the change is about future results and clarity.
- **Done means**: Criteria accept an ordered list of bands (from, relative and absolute limits) validated and hashed like the rest of the gate file; v1 and v2 files and hashes unchanged; a gate v3 file with the owner's confirmed limits; tests on both sides of every band edge; a documented table of the committed results re-judged.
- **Progress** (27 November 2026): the owner named US EPA Performance Specification 16; its text was read and its numbers are in `docs/research/ps16-transposition.md` (RATA 10 percent above 100, 20 percent from 100 to 10, absolute 2 below 10; audit absolute at 20 or less; bias, F-test and correlation 0.8; nine runs at three levels) with a mapping to the Epiphanes gate. The committed results were re-judged under the literal bands. Not built. Waiting for the owner on: absolute threshold 10 or 20, the absolute limit in perplexity units, and whether to adopt the bias, variance and correlation tests and the three levels. Owner decision, 27 November 2026: the absolute threshold stays 20 for now; the full transposition (bias, variance, correlation, levels, gate v3) is done later.

### [THME-3100] Calibrate the allocator link constant C

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p2 · **Area**: research · **Type**: research · **Status**: triage
- **Problem**: the link between the allocator objective and logit divergence is a modelling assumption (paper section 12.2) and ternary effects are not additive (sum +1,509, joint +15,877,276 perplexity).
- **Done means**: C estimated against joint measurements, or the surrogate replaced by an empirically bounded constant.

### [THME-3069] Visual check of every Studio tab, then recapture screenshots

- **Milestone**: v0.4 · **Roadmap**: none · **Priority**: p2 · **Area**: gui · **Type**: chore · **Status**: ready
- **Problem**: Nobody has looked at the Studio window since the eframe 0.37 upgrade: tabs, the Chat tab with its device selector, the Welcome panel and labels are untested by eye, and the Live 3D capture crop no longer fits the new layout (THME-2002).
- **Done means**: The owner opens each tab and records what is wrong; layout fixes land; all Studio screenshots and recordings are recaptured from the real window.

### [THME-3071] Provenance ledger: record where every part of the code and data came from

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: legal · **Type**: chore · **Status**: ready
- **Problem**: Dependencies and licence headers are checked, but nothing records who or what wrote each part of the code (the owner, an AI tool and when, a third-party source) or which datasets and models a report used. The owner wants every part of what is written and released to have clear rights.
- **Done means**: A ledger file in the repository lists each area (directory or file group) with its origin, licence and date; CI fails when a new top-level area has no entry; agents are told in AGENTS.md to add code only from the repository itself or from sources the licence audit approved; the contributor agreement and counsel review are tracked privately.
>
### [THME-3074] Benchmark runner in Rust: one process, model loaded once, progress

- **Milestone**: v0.2.4 · **Roadmap**: none · **Priority**: p2 · **Area**: bench · **Type**: perf · **Status**: ready
- **Problem**: `scripts/run_awq_calibration.py` starts a new `thme` process for each of the 34 or so arms per model, so every arm re-reads and re-decodes the weights and repeats the unquantized baseline pass. The Python driver itself is not the slow part; the repeated work is. The driver also hard-codes a CPU hardware string in the provenance.
- **Done means**: A Rust `thme` subcommand (or example) runs all arms of a pre-registered protocol in one process: model loaded once, baseline and calibration statistics shared, device chosen by `--device` and recorded with adapter and driver, resumable, prints `[done/total pct, elapsed, ETA]`; produces the same JSON as the Python driver (compare on Pythia-14M); the dataset download and tokenisation may stay in Python until a Rust equivalent exists.

### [THME-3075] Run metrics and progress for every long command, in Rust

- **Milestone**: v0.2.4 · **Roadmap**: none · **Priority**: p1 · **Area**: bench · **Type**: feature · **Status**: ready
- **Problem**: Long runs (`evaluate`, `calibrate`, `search`, the examples, the benchmark runner of THME-3074) print little while they work and record few facts about the run, so nobody can see how far along a run is, how fast it goes or what hardware produced a number. Each script invents its own counters.
- **Done means**: One small Rust module used by every long command: (1) progress events `done/total`, percent, elapsed, ETA from measured pace, items per second (tokens per second for model passes); (2) a human line on stderr and, with `--progress json`, one JSON line per event that Studio and the live window can read; (3) a `run` block in every report: start and end time, wall time and time per phase, device, adapter and driver, threads, sampled CPU and GPU load, peak memory and VRAM, cache hits (for example the cached baseline pass), seed, thme version and git commit, exit status; (4) the block has a schema under `docs/schemas/` and the dossier generator reads it instead of hard-coded strings; (5) unit tests with an injected clock; (6) the Python collector's progress line becomes a thin reader of the JSON events until THME-3074 replaces it.

### [THME-3078] `.epiph` flexible encodings: any tensor layout, safely extensible

- **Milestone**: v0.2.6 · **Roadmap**: none · **Priority**: p2 · **Area**: format · **Type**: feature · **Status**: triage
- **Progress** (20 November 2026): chain `.epiph` to GGUF measured on SmolLM2-1.7B: `thme epiph pack` (218 tensors, 7 attachments, 3,426,560,087 bytes), `thme epiph verify` OK (no signature), `thme export` of the `.epiph` with a Q8_0 recipe gives a GGUF byte-identical to the one exported from the checkpoint directory (same sha256), so tensor payloads survive losslessly; the GGUF carries no tokenizer or attachments and no signature. Details in `docs/benchmarks/smollm2-1.7b/data/llama_cpp_reference.json` (`epiph_chain`).
- **Problem**: Each tensor today has one storage format. The memory work needs layouts inside a tensor (for example embedding rows split into precision bands by token frequency, factorised or codebook tensors, row-sparse blocks). The format should accept new layouts without breaking readers, and stay readable by tools that do not know them.
- **Done means**: A design and a spec section (`docs/formats/epiph-v1.md`, a minor version): a tensor entry carries an `encoding` description (codec id, parameters, optional row bands with a band map) from an open registry; each codec is marked critical (a reader that does not know it refuses with an exact message and the way to convert) or optional (a reader may skip it when a canonical copy is present); every non-canonical encoding can be written together with, or converted to, a canonical block-format copy; the manifest lists the features used so a reader can tell before reading tensors; unknown optional manifest fields and attachment kinds are ignored; tests for old reader and new file, new reader and old file, unknown critical codec, band map round trip, and the golden-hash portability test extended to the new encodings. The first codec (per-row-band precision) is added only if experiment A1 of the unified concept note shows a measured gain in bytes at an acceptable perplexity change on Pythia-14M.

### [THME-3081] Frequent-row draft pass for the output head (Pythia-14M)

- **Milestone**: v0.2.4 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: experiment · **Status**: ready
- **Problem**: Decode reads the whole output head for every token. If the greedy token is almost always among the most frequent rows, a draft over those rows plus an exact verification step could cut the bytes read without changing the output.
- **Done means**: One pass measures how often the greedy token lies in the top 1,024, 4,096 and 8,192 frequent rows (rows from THME-3079) over the test split; a decision rule fixed in the report before the run says whether to build the verified draft; if built, the output must be identical to normal greedy decoding and tokens per second are measured paired against it.

### [THME-3082] KV cache in int8 and int4 per group, counted in the placement budget

- **Milestone**: v0.2.4 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: experiment · **Status**: ready
- **Problem**: The KV cache is now real (THME-3070): 12.6 MB at 2,048 context for Pythia-14M, more for larger models. Lower-precision KV could cut it, mainly for models at SmolLM size and above.
- **Done means**: KV stored in int8 and int4 per group behind the existing cache type; cache bytes, perplexity at 1k and 4k context and decode tokens per second measured against the f32 cache on Pythia-14M, with SmolLM-135M as the late confirmation; the placement budget (THME-3016) counts the real KV bytes.
- **Progress** (27 November 2026): part 1 in code: `KvPrecision` (F32, Int8, Int4 per group of 32) behind `KvCache::with_precision`, real packed bytes reported, round-trip tests. Part 2: `thme chat --kv-precision f32|int8|int4` (default f32) builds the cache at that precision on the CPU path and reports `kv_precision` and `kv_cache_bytes`; the CUDA path declines non-f32 caches. Spot check on the CPU (27 November 2026, release build, one prompt, greedy, 64-token cap, not a perplexity measurement): SmolLM-135M with 32 generated tokens held 2,073,600 bytes in f32, 583,200 in int8 and 324,000 in int4 (3.6 times and 6.4 times smaller), the generated text was identical at all three, and decode speed was 36.1, 35.2 and 35.1 tokens per second. Pythia-14M diverged from the f32 text at int8 and at int4 (it also stopped on the loop breaker after 12, 15 and 8 tokens, so its byte counts are for different lengths and are not compared). Part 3 (27 November 2026): `quality::kv_perplexity::kv_windowed_perplexity` and `examples/kv_precision_ppl.rs` score windows token by token through a cache at the chosen precision (f32 case checked against an independent prefix-by-prefix oracle at every position). Measured on SmolLM-135M, WikiText-2, CPU: at 4 windows of 256 tokens perplexity is 30.392 (f32), 30.397 (int8), 32.153 (int4) with cache bytes 11.75 MB, 3.30 MB, 1.84 MB; at 2 windows of 1,024 tokens 16.094, 16.087, 17.185 with 47.1 MB, 13.3 MB, 7.4 MB. Reading (SmolLM-135M only): int8 is within noise of f32 at both lengths; int4 costs about 6 to 7 percent perplexity. On Pythia-14M the same method gives a very different answer (report `docs/benchmarks/pythia-14m/data/kv_precision_ppl.json`): perplexity 190.8 (f32), 245.2 (int8), 5,805 (int4) at 4 windows of 256 tokens, and 75.5, 101.5, 10,350 at one 2,048-token window. So int8 is not safe for every model and int4 can break one; the cause is not isolated: a group of 8 instead of 32 did not fix it (int8 234.3, int4 4,983 at 4x256), so coarse scales are not the whole story, and no cache bug showed up on a tiny synthetic GPT-NeoX model (tests `neox_int8_cache_stays_close_to_the_f32_cache` and `neox_int4_cache_is_finite_and_not_catastrophic` pass, LLaMA control too), so the sensitivity looks specific to Pythia-14M's real weights (not proven). A local, reverted experiment showed that the keys cause all of the loss on Pythia-14M: with only the keys quantized, perplexity is 236.1 (int8) and 7,349 (int4); with only the values quantized it is 190.9 and 190.7, equal to f32 (190.8).  Report: `docs/benchmarks/smollm-135m/data/kv_precision_ppl.json`. Not yet: 4,096 context (both small models are trained for 2,048), a per-channel key scale to see whether it fixes Pythia-14M, a separate key and value precision (the values alone can be int4 on Pythia-14M at no cost), decode speed, GPU cache, placement budget; the ticket stays open and the default stays f32.
- **Progress** (27 November 2026, part 4): keys and values can have different precisions: `KvCache::with_precisions`, `thme chat --kv-key-precision/--kv-value-precision`, `kv_windowed_perplexity_mixed`, and `examples/kv_precision_ppl --key-precision --value-precision`; JSON reports `kv_key_precision`, `kv_value_precision` and `kv_precision` ("mixed" when they differ). Measured on Pythia-14M (4x256 tokens): f32 keys with int4 values gives perplexity 190.66 against 190.83 at f32 with a cache 1.7 times smaller; int8 keys with int4 values gives 237.0. Not measured on SmolLM-135M or at longer context, and the CUDA path still declines any non-f32 cache.

### [THME-3083] Layer-removal check on the small models

- **Milestone**: v0.2.4 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: experiment · **Status**: ready
- **Problem**: Published work reports that some layers can be removed from larger models with little loss; small models may be more fragile.
- **Done means**: Block influence per layer computed for Pythia-14M (and SmolLM-135M late), perplexity measured against the number of removed blocks on the same token hash with a random-removal control; the rule and the stop threshold are written before the run.

### [THME-3089] Device decoder with KV cache and fused quantized matrix-vector for decode

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.3 · **Priority**: p1 · **Area**: gpu · **Type**: perf · **Status**: ready
- **Problem**: Decode (one token at a time) is limited by memory bandwidth, not arithmetic: about 288 GB/s on this card, so a 1.7B model at Q4_0 (1.1 GB) has a ceiling of about 260 tokens per second. We have no device KV cache and a per-operation GPT-NeoX path, so decode on the GPU is unmeasured and likely far below that ceiling.
- **Done means**: Device-resident decode for LLaMA-family and GPT-NeoX models with a device KV cache and fused quantized matrix-vector kernels, one submit per token; tokens per second measured against the bandwidth ceiling (target at least 80 percent of it for Q4_0 and Q8_0) and against llama.cpp from THME-3085; Pythia-14M first.

### [THME-3090] Automatic weight budget for the GPU path (default is 7x slower than it should be)

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.3 · **Priority**: p1 · **Area**: gpu · **Type**: bug · **Status**: ready
- **Problem**: The GPU path keeps weights resident only up to a budget that defaults to 4 GiB. A model larger than the budget is flushed and re-uploaded for every window. Measured on SmolLM2-1.7B (6.9 GB in f32), RTX 4060 Ti, full test split, same build: 1,016 s with the default budget and 145 s with a 12 GiB budget (same perplexity, 9.1091), so a user who runs the engine with defaults loses a factor of seven. wgpu does not report VRAM, so the right default is unknown.
- **Done means**: The budget is chosen automatically (adapter limits, a probe allocation with an out-of-memory fallback, or `thme pathfinder`) and never smaller than needed when memory allows; a model that does not fit falls back to f16 or a quantized device format before it falls back to re-uploading; the report states the budget used; a test with a fake backend covers the fallback order.
- **Progress** (27 November 2026): the decision logic is in `src/inference/gpu/budget.rs` (format order F32, F16, Q8_0, Q4_0, out-of-memory fallback, six tests on a fake probe). Not yet wired into `GpuContext`, and the report does not yet state the budget used.
- **Follow-up (owner, 27 November 2026)**: the fake-backend test only proves the decision logic. After the code lands, repeat it against the real backends (CUDA and wgpu) on SmolLM2-1.7B on the RTX 4060 Ti and record the measured budget and timings here. Until then the status stays "done in code; not run on a real GPU".

### [THME-3091] GGUF export that llama.cpp loads (names, tokenizer, rope layout, norms, flags)

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.8 · **Priority**: p1 · **Area**: exporter · **Type**: bug · **Status**: ready
- **Problem**: GGUF files from `thme export` pass THME's own checker but do not load in llama.cpp (measured 20 November 2026 with b12584): no tokenizer vocabulary in the file, Hugging Face tensor names instead of GGML names, BF16 norm weights where F32 is required, q/k rows not permuted for llama.cpp's rope layout, and no rope dimension key. `thme export --quant fp16` and `--quant q8_0` fail on checkpoints with a BF16 embedding (only per-tensor recipes work). A small adapter outside the repository made all three files load and generate identical text, so the gaps are known and fixable.
- **Done means**: `thme export` writes, from the public GGUF specification (ggml docs/gguf.md) and the documented tensor-name conventions and not from llama.cpp source: the tokenizer (vocabulary, merges, special tokens) from the model's tokenizer files, GGML tensor names, F32 norms, the rope key and layout needed for the architecture, and the `--quant` flags work for BF16 sources; THME's checker gains the consumer's requirements (vocabulary present, names recognised, norm types); a test loads the exported file in llama.cpp when a llama.cpp build is configured (CI step off by default) and compares 20 greedy tokens against the adapter-free export; the `.epiph` to GGUF chain keeps its byte-identical payloads.
- **Progress** (27 November 2026): slice 1 in `src/exporter/ggml_names.rs`: Hugging Face to GGML tensor names for the LLaMA family, norm detection, and the q/k row permutation (and its inverse) derived from the rotary pair layout, with hand-worked tests. Not wired into `thme export`; still missing: tokenizer vocabulary, F32 norms in the writer, rope dimension key, BF16 `--quant` fix, checker rules, and the llama.cpp load test (needs a configured llama.cpp build).
- **Progress** (27 November 2026, slice 2): `thme export --llama-cpp-layout` (opt-in, default off and byte-identical) writes GGML tensor names, F32 norms, q/k rows permuted before quantising, and `rope.dimension_count`; tests read the file back with literal expectations. Slice 3: `--llama-cpp-layout` also writes `tokenizer.ggml.*` (tokens, token types, merges, special ids, gpt2 or llama model) from a `tokenizer.json` next to the model. Slice 4: `--quant fp16|q8_0` accepts BF16 sources (the test failed before the fix); `check_llama_cpp_requirements` checks tokenizer, tensor names, F32 norms and rope dimension, and `--llama-cpp-layout` refuses a file that fails it, so it needs a `tokenizer.json`. Llama.cpp check (27 November 2026, build b12584 from this machine, CPU only, one model): SmolLM-135M exported with `--quant fp16 --llama-cpp-layout` loads in `llama-completion`, and 20 of 20 greedy tokens after "The capital of France is" equal THME's own forward pass; the same export without the q/k permutation repeats itself and does not match. Evidence: `docs/benchmarks/smollm-135m/data/llama_cpp_load.json`. Not yet: `tokenizer.ggml.pre` (llama.cpp warns about a degraded default; see THME-3102), other models and quant types, and the opt-in CI step; the ticket stays open.

## Engine, format, security and quality tracks (v0.2.5 to v0.2.9)

> From the research dossier v3.1 (`docs/research/`) and the owner's direction of 13 November 2026.
> Each ticket names its milestone and sub-milestone in ROADMAP.md.

### [THME-3002] Exact dynamic-programming allocator behind `thme search --solver` (dossier WP2)

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.2 · **Priority**: p1 · **Area**: engine · **Type**: feature · **Status**: blocked
- **Problem**: Docs promised a knapsack solver; `ParetoAllocator` uses sensitivity thresholds and `thme search` is greedy.
- **Progress**: Implemented: `engine::alloc` (exhaustive-search oracle on 300 instances) and `thme search --solver dp --dp-grid N`; never worse than greedy on predicted cost (200 random tables). Default stays greedy. Blocked on the comparison on real cost tables, which needs model runs (paused).
- **Done means**: `engine::alloc` matches exhaustive search on small instances and is never worse than greedy at equal budget; default solver unchanged until compared on committed cost tables.

### [THME-3008] Permission and ownership record for reverse-engineering features (dossier WP19)

- **Milestone**: v0.2.7 · **Roadmap**: v0.2.7.4 · **Priority**: p1 · **Area**: core · **Type**: feature · **Status**: triage
- **Problem**: Architecture inference and closed-format analysis must only run on models the user owns or may analyze.
- **Done means**: Those features refuse to run without a stored basis and purpose statement next to the model hash; wording reviewed by a lawyer.

### [THME-3011] Measured bits per weight from the written artifact (placeholder)

- **Milestone**: v0.2.6 · **Roadmap**: none · **Priority**: p3 · **Area**: exporter · **Type**: feature · **Status**: triage
- **Problem**: Reports show nominal bits per weight (format bit-widths averaged over all parameters, unchanged tensors at stored width). `TensorFormat::bits_per_weight` also says 8.0 for Q8_0 and 4.5 for Q4_0, while the stored blocks are 8.5 and 4.5. llama.cpp reports file-level bits per weight (artifact bytes x 8 / parameters), which includes block scales.
- **Done means**: Every report that writes an artifact also states measured bits per weight from its byte size, next to the nominal figure, each labelled; the nominal definition is kept for unexported candidates. No committed number changes without a new report.

### [THME-3017] Trademark search and filing for the format name

- **Milestone**: v0.2.6 · **Roadmap**: none · **Priority**: p1 · **Area**: legal · **Type**: chore · **Status**: blocked (owner: funding and registration)
- **Problem**: The owner wants the format name protected under Epiphanes. Extensions are not usually registrable; the name is.
- **Done means**: A lawyer has searched and advised on the name; filing decided by the owner.
- **Progress** (27 November 2026): The Epiphanes site exists but the domain and the business are not registered yet; funding for registration is being sought. A lawyer search and a filing decision follow registration.

### [THME-3020] Developer sandbox for untrusted code

- **Milestone**: v0.2.7 · **Roadmap**: v0.2.7.5, v0.2.7.6 · **Priority**: p2 · **Area**: security · **Type**: feature · **Status**: shelved (owner: opportunities and the commercial sector)
- **Problem**: Remote-code models, plugins (dossier L.4) and benchmark code (R.4) need to run isolated.
- **Done means**: Design doc first: separate process, no network, file access limited to a work folder, time and memory caps, per-OS mechanism; then an implementation with escape tests.
- **Progress**: design done ([docs/sandbox.md](docs/sandbox.md)). Implementation shelved by the owner for the commercial version (unmerged branches `feature/sandbox-windows` and `feature/sandbox-linux`, PRs #65 and #66). Until then the open-source version does not run untrusted code.
- **Progress** (27 November 2026): Design stays in docs/sandbox.md; the implementation branches (feature/sandbox-windows, feature/sandbox-linux, PRs 65 and 66) are kept for the commercial version. Not on the open-source roadmap.

### [THME-3021] Guardrail detection and evaluation (read-only)

- **Milestone**: v0.2.7 · **Roadmap**: v0.2.7.7 · **Priority**: p2 · **Area**: security · **Type**: feature · **Status**: triage
- **Problem**: Owner asks to find guardrails. Scope: detect safety components (safety classifiers, refusal instructions in chat templates and system prompts) and measure refusal behaviour on a fixed prompt set. Out of scope: any feature that removes or disables a model's safety behaviour.
- **Done means**: Scan reports safety-related components it finds; an evaluation reports refusal rates with intervals; optional safety layers that THME adds on top of a model can be switched on or off.

### [THME-3036] LoRA training with a real backward pass

- **Milestone**: v0.2.3 · **Roadmap**: v0.2.3.1 · **Priority**: p2 · **Area**: trainer · **Type**: feature · **Status**: triage
- **Problem**: THME has no backward pass, so it cannot fine-tune; the current trainer is synthetic.
- **Done means**: Hand-written CPU backward for the LLaMA and Gemma blocks, checked against finite differences, then LoRA adapters with AdamW on a tiny model. Written from the published formulas (permissive references only); a heavy training framework is not added.
- **Design note** (27 November 2026, from a summary the owner pasted of arXiv 2610.31642 "A Fix for Softmax and Cross-Entropy"; the paper itself was not read or verified here): softmax can lose the gradient of the dominant class once logit margins pass a precision-dependent size (reported about 16.6 in fp32), and distillation between near-equal student and teacher logits can leave a gradient noise floor. When the real backward is written, compute the cross-entropy and distillation gradients as direct differences (for example `p - onehot` and `p_student - p_teacher` from log-probabilities in f64) and add a test with margins above 17 in f32 that keeps the gradients summing to zero. Today's evaluation path is not affected: `src/inference/llama.rs` computes the loss from the maximum-subtracted form with the sum in f64, and `kv_perplexity` does the same. The current trainer (`src/trainer/mod.rs`) is synthetic and uses f32 softmax gradients.

### [THME-3038] Studio chat tab

- **Milestone**: v0.3.0 · **Roadmap**: v0.3.0.4 · **Priority**: p3 · **Area**: gui · **Type**: feature · **Status**: ready
- **Progress** (19 November 2026): the Studio has a Chat tab. `src/chat.rs` renders the model's own chat template, encodes, and decodes greedily on a background thread with Cancel, a max-new-tokens slider and a status line (tokens, seconds, tokens per second, labelled Measured on this machine). A banner says replies come from the slow CPU reference forward pass, and that the mock model's text is meaningless. 9 tests on the mock model with a fixture tokenizer. Not done: the live window was never opened, so layout and look are unchecked by eye; there is no KV cache, so long replies are slow; chat needs a model loaded from a file (the synthetic demo model has no folder). No real model was used.
- **Progress** (20 November 2026): device selector added (CPU default, GPU and CPU + GPU marked experimental with a tooltip). The loaded-model cache is keyed by device, so a change reloads. GPU was slower when this was written: 4.3 s vs 1.1 s for a 128-token pass on SmolLM-135M (RTX 4060 Ti); see THME-3072 for the fix and the new paired numbers. Not done: resolved GPU device name in the reply stats; CPU + GPU not measured; no live window check.
- **Problem**: Users cannot talk to or test a loaded model inside the Studio.
- **Done means**: A tab that sends a prompt through the reference forward pass and shows the reply, labelled with the measured speed. Depends on THME-3037.

### [THME-3039] Animated live-layers figure

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.8 · **Priority**: p3 · **Area**: figures · **Type**: feature · **Status**: ready
- **Progress** (19 November 2026): `thme figure MODEL --gif` also writes a per-layer GIF (one frame per layer, from the real layer structure; 5 tests on the mock model). Not done: a person must look at the rendered figure, because the lit layer is subtle in the light theme.
- **Progress** (19 November 2026, later): the lit layer is now ink glass with the rest faded back (checked by eye in light and dark). Still to do: our own Epiphanes card and layout, dark-theme GIFs, and the ink splash on the layer frames.
- **Problem**: README GIFs of the other Studio tabs came out static (identical frames collapse to one), so only Live 3D is animated. The committed `thme figure MODEL --gif` (`src/report/animate.rs`, `docs/figures/mock_llama-forward.gif`) walks the forward pass tensor by tensor; a per-layer view of the whole transformer stack is not built yet.
- **Done means**: `thme figure MODEL --gif` shows the layers lighting in turn from the real structure of the model; frames checked by eye; no invented figures.

### [THME-3040] `thme discover`: find the models already on this machine

- **Milestone**: v0.3.0 · **Roadmap**: v0.3.0.1 · **Priority**: p1 · **Area**: onboarding · **Type**: feature · **Status**: ready
- **Problem**: A new user should not have to move files to start. Models already sit in the Hugging Face cache, Ollama, LM Studio and llama.cpp folders.
- **Done means**: A read-only command lists the models found in those places (name, tool, format, size, path) and accepts extra folders; tested on fake folder layouts built in a temporary directory; no network and no writes.

### [THME-3042] Bridges to the tools people already use

- **Milestone**: v0.3.0 · **Roadmap**: v0.3.0.5 · **Priority**: p2 · **Area**: onboarding · **Type**: feature · **Status**: triage
- **Progress** (19 November 2026): part 1 is built: `thme download ORG/NAME [--revision] [--out] [--file] [--dry-run]` picks the config, tokenizer and weight files from the hub listing, prints the licence first, skips pickle files unless named, verifies size and sha256 where the hub gives one, renames atomically, never prints the `HF_TOKEN`, and runs the import scan on the result. 19 tests with a fake fetcher; no real hub request was made. New crates: `ureq` 3.5.2 (MIT OR Apache-2.0) and `rustls` with `ring` (build script contains C and assembly; no OpenSSL, no aws-lc); `hf-hub` was not used. Not done: bridges to Ollama, LM Studio and llama.cpp (part 2), resume, a real download test, Windows trust-store check.
- **Problem**: Analysts, engineers and scientists start from notebooks, Ollama, LM Studio, llama.cpp or Hugging Face. THME should read what they have and write what they can use.
- **Done means**: Download from the Hugging Face hub through a permissively licensed client crate (licence and binary-size cost proposed first); export GGUF that llama.cpp and Ollama load (already tested for structure by THME-2208); a documented MCP setup for editors and agents. Each bridge lists what was tested and what was not.

### [THME-3043] Evaluate the `neutron-iso` Rust port of the anatomy kit

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.8 · **Priority**: p3 · **Area**: figures · **Type**: chore · **Status**: triage
- **Problem**: The owner ported the anatomy kit into `neutron-iso` (sibling folder `neutron-lab`). It builds and `neutron iso` renders valid SVG, but `shadow` writes HTML entities into an `.svg`, `gallery` writes HTML into a file named `.svg`, the crate has no tests, and the files carry no MIT notice for the anatomy kit. THME already has `engine::iso`, byte-compared with the kit.
- **Done means**: Either the useful renderers move into THME with tests and the notice, or the crate stays outside with its output fixed; the decision is written down.

### [THME-3047] Clean public repository from a clean export

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: legal · **Type**: chore · **Status**: triage
- **Problem**: Old public history holds the Windows username and an absolute path (`.claude_handover.md`, commits `a0befc7`, `b31dd81`), an old personal account name, and the GPL-2.0 derived GPU-unlock port (PR #82). A fork would carry all of it. Local refs (`refs/agents/*`, stashes) also hold the removed `agent-hub/` and `neutron-*` crates.
- **Done means**: A new repository is built from `git archive` of a clean tree with fresh history (not a clone, not a mirror push), after a scripted pre-publish scan (secrets, usernames and absolute paths, `hermes`, `medgemma`, binary extensions, SPDX, third-party register, `cargo deny`) and the full CI on the new repository. `LICENSING.md` no longer cites a commit that will not exist. The owner decides whether to archive or privatise the old repository. First-pass audit, 19 November 2026: H2, M9.

### [THME-3048] VS Code extension: no shell injection, honest commands

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: security · **Type**: bug · **Status**: ready
- **Progress** (20 November 2026): done on branch `fix/vscode-and-mcp-docs`. Commands run with `shell: false` and raw argument arrays (`editors/vscode/src/thme.ts`); numbers and paths validated; `validate` asks for measured perplexities and reports the CLI's exit status; surgery and the other commands pass the arguments the CLI really takes (they were wrong before: `inspect --model`, `plan --tolerance`, `tweak` without `--model`); the MCP terminal starts the binary without a shell string; manifest has `capabilities.untrustedWorkspaces.supported: false`. `npm test` in `editors/vscode` runs 5 tests (quote-and-ampersand path, validation, no-shell). Not done: the extension was not run inside VS Code; the sidebar trees still show static example text.
- **Problem**: `editors/vscode/src/extension.ts` runs commands through `spawn(..., { shell: true })` and pastes Copilot tool input into quoted arguments, so a prompt-injected tool call can run shell commands (audit H3). `validateQuality` runs the gate on hard-coded perplexities and reports PASSED, and the surgery message claims a rollback checkpoint it does not check (M5). The binary is resolved from the workspace before PATH, and the manifest does not declare untrusted-workspace support (L5).
- **Done means**: `shell: false` with raw argument arrays, numeric inputs validated, `modelPath` checked to be an existing file, a unit test with a quote-and-ampersand path; validate asks for real inputs or is labelled a demo and reports the CLI's own verdict; surgery states only what it checked; the manifest declares `capabilities.untrustedWorkspaces: { supported: false }`.

### [THME-3049] Release pipeline hardening

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: ci · **Type**: chore · **Status**: ready
- **Progress** (20 November 2026): archives now ship `LICENSES/` and `LICENSING.md`; every action in `ci.yml`, `release.yml` and `clean_machine_test.yml` is pinned by full commit SHA with a version comment (SHAs read from the action repositories through the GitHub API); the three workflows default to `contents: read`, only `publish` gets `contents: write`; `MINISIGN_KEY` is visible to the signing step only; publish fails when the key is missing and `scripts/sign_artifacts.sh` no longer skips; a verify step runs `minisign -V` against `minisign.pub` for each archive and `SHA256SUMS.txt`. Release builds pass `--remap-path-prefix` through `RUSTFLAGS`. Not done: no release was triggered, so the workflow changes are unrun; `Swatinem/rust-cache`, `softprops/action-gh-release` and `EmbarkStudios/cargo-deny-action` are pinned to the commit behind their moving `v2`/`v3` tag (the API showed no exact release tag on that commit), so they need a deliberate bump to move.
- **Problem**: Release archives omit the anatomy-kit MIT notice for the compiled `iso.rs` port (M1). Actions are pinned by moving tags (M6). `contents: write` covers every job and `ci.yml` has no `permissions:` block, and `MINISIGN_KEY` is readable by every step of the publish job (M7). Signing is skipped silently when the key is missing although SECURITY.md promises signed archives (M8).
- **Done means**: Each archive ships `LICENSES/` and `LICENSING.md`; every action is pinned by full commit SHA with a version comment; both workflows default to `contents: read` and only `publish` gets write access, with the key passed only to the signing step; publish fails when the key is missing and a verify step checks signatures against `minisign.pub`.

### [THME-3050] Remove agent notes, personal settings and withdrawn claims from the tree

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: legal · **Type**: chore · **Status**: ready
- **Problem**: Tracked agent working notes and personal editor settings: `.claude_handover.md`, `TASKS.md`, `TEST_READY.md`, `TEST_INFRA.md`, `PROJECT.md`, `.vscode/settings.json` (M2). `docs/research/THME_measured_findings_2026-11.md` publishes perplexity figures whose report was removed (M4). `docs/research/THME_AUTHENTICITY_AUDIT_AND_ROADMAP.md` credits a third-party organisation (L8). README says recipes are measured with no committed report and has a broken sentence (L9).
- **Progress** (27 November 2026): `.claude_handover.md`, `TASKS.md`, `TEST_READY.md`, `TEST_INFRA.md` and `PROJECT.md` are no longer tracked (they stay in git history). Not done: `.vscode/settings.json`, the withdrawn findings note, the audit note's AI-author notice, README statuses.
- **Done means**: The notes and personal settings are no longer tracked; the findings note is marked withdrawn (pre-THME-3023, to be re-measured) at the top and per table, or removed; the audit note says it was written by an AI agent and is unverified; README statuses match the evidence.

### [THME-3051] Drop the hard-coded model fallback from MCP and tests

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: inference · **Type**: bug · **Status**: ready
- **Progress** (20 November 2026): the MCP server no longer reads a MedGemma path from `%LOCALAPPDATA%` (six places in `src/cli/mcp.rs`; `THME_MODEL` or an explicit argument only). The real-model tests use Pythia-14M and SmolLM-135M from `THME_TEST_MODELS_DIR` behind `THME_REAL_MODEL_TESTS=1` and print a SKIP reason when they do not run; no test reads a model from a user folder. Not done: the gated tests were not run on the real models (no model runs in this change), so they are unverified until someone runs them with the variables set.
- **Problem**: `src/cli/mcp.rs` (six places), `tests/test_mcp.rs`, `tests/test_gpu_acceleration.rs` and `tests/test_medgemma_inference.rs` fall back to `%LOCALAPPDATA%/hermes/models/medgemma-...gguf`, which exposes the local agent setup and silently loads a file from a user-writable place (M3).
- **Done means**: The MCP server loads a model only from `THME_MODEL` or an explicit argument; tests that need a real model are gated behind `THME_REAL_MODEL_TESTS=1` and never search a fixed location.

### [THME-3052] Repository hygiene sweep

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p2 · **Area**: ci · **Type**: chore · **Status**: ready
- **Progress** (20 November 2026): `.gitignore` has `.claude/` and `.mcp.json`; the CI path guard also rejects Windows user-folder paths with either slash, macOS `/Users/` and `AppData` paths, and a new CI step rejects tracked `.claude/`, `.agents/`, `.mcp.json` and `*.local.md`; `deny.toml` lost the five licences the lockfile never meets (`cargo deny check advisories bans licenses sources` passes with no warnings); AGENTS.md sizes and MSRV were already correct, and the stale 10 MiB comment in `Cargo.toml` and MSRV 1.89 in ROADMAP are fixed. The committed `.cargo/config.toml` holds only `strip=symbols` (machine-local job limits live in the global config). Path remapping cannot be written in a committed Cargo config (it needs machine-specific prefixes and `trim-paths` is unstable on 1.100), so the release workflow passes it through `RUSTFLAGS`; AGENTS.md rule 1 says so. Not done: the guard does not yet name the five tracked agent notes (`.claude_handover.md`, `TASKS.md`, `TEST_READY.md`, `TEST_INFRA.md`, `PROJECT.md`) because THME-3050 still has to remove them, so add them to the guard in that change; the `models/SmolLM-135M/config.json` register row and the mascot and logo art origins (L7) need the owner's facts and are open.
- **Problem**: `.gitignore` lacks `.claude/` and `.mcp.json` (M9). `.cargo/config.toml` has no path remapping although AGENTS.md rule 1 says it stays active (L1). The CI path guard misses `C:/Users/`, macOS `/Users/` and `AppData` (L10). AGENTS.md still says CLI <= 10 MB and MSRV 1.89 (L6). `deny.toml` allows licences it never meets (L12). `models/SmolLM-135M/config.json` has no register row and the mascot and logo art have no recorded origin (L7).
- **Done means**: Each item fixed or recorded; the path guard also rejects agent-note filenames.

### [THME-3053] Enforce the teacher subprocess timeout; delete dead code

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: bug · **Status**: ready
- **Progress** (20 November 2026): `AgentBridge::request_recipe` now runs the subprocess through `run_with_deadline` (pipes serviced on threads, poll, then kill at `timeout_secs`); test `subprocess_is_killed_at_the_timeout` uses the test binary itself as the sleeping child, so it needs no shell or OS-specific command. `evaluate_external` is removed (it had no timeout field and no callers). Open: `thme co-surgery` still builds an `AgentBridgeConfig` (30 s) that nothing invokes, so no agent runs there yet; the timeout applies as soon as the bridge is wired in.
- **Problem**: `timeout_secs` is ignored in `src/surgery/agent_bridge.rs`, `src/cli/co_surgery.rs` and `src/quality/evaluator.rs`, so a teacher subprocess can hang forever; `evaluate_external` has no callers (L4).
- **Done means**: The timeout is enforced (wait with a deadline, then kill) with a test using a sleeping child; `evaluate_external` is removed.

### [THME-3054] Document the MCP trust model and the memory-map limit

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p3 · **Area**: security · **Type**: chore · **Status**: ready
- **Progress** (20 November 2026): trust model documented in `tutorials/mcp_copilot_setup.md`, memory-map limit in `SECURITY.md`, example configs point at `target/release/thme`. Allowed roots are implemented (27 November 2026): `thme mcp --allow-root DIR` (repeatable) refuses a `model_path` that resolves outside every root, or cannot be resolved, before any tool runs (`process_json_rpc_with_roots`, five tests; the one place the argument is read is the tool-call dispatch). The VS Code extension's MCP terminal passes one `--allow-root` per workspace folder (`mcpArgs`, test added, 12 extension tests pass), and the example configs and docs show the flag. Not done: the model given with `--model` is not checked, the extension was not run inside VS Code, and the flag is optional, so the default is still unrestricted. Status stays ready for that reason.
- **Problem**: Any connected MCP client can make the server open and parse any readable path (read-only and scan-gated, no writes or process spawning found), and the configs start it through `cargo run`, which runs build scripts (L3). The parsers use `unsafe Mmap::map` on user files, so truncation by another process while in use can crash the process (L11).
- **Done means**: The MCP trust model is documented with optional allowed roots; configs point at a built binary; SECURITY.md states the memory-map limit.

### [THME-3067] precision_sweep: random_matched seeds 2 and 3 gave identical perplexity on Pythia-14M

- **Milestone**: v0.2.0 · **Roadmap**: v0.2.0.9 · **Priority**: p2 · **Area**: research · **Type**: bug · **Status**: triage
- **Problem**: On Pythia-14M, `examples/precision_sweep` `random_matched_seed2` and `random_matched_seed3` printed the same perplexity. The likely cause is that the pool of same-shape tensors is too small, so shuffling by seed picks the same placement (unverified; not reproduced in this ticket).
- **Done means**: Either the seeds are shown to yield different placements (log the chosen tensor indices per seed in the report) and the perplexity differs, or the sweep refuses to report random controls that repeat a placement. Checked on Pythia-14M with a short token file.

### [THME-3068] llama.cpp compatibility test needs a configured runtime

- **Milestone**: v0.2.9 · **Roadmap**: v0.2.9.6 · **Priority**: p2 · **Area**: ci · **Type**: chore · **Status**: triage
- **Problem**: `tests/test_exporter_compatibility.rs` (`exported_gguf_runs_in_configured_llama_cpp_and_ollama`) returns early unless `THME_LLAMA_CPP`, `THME_OLLAMA` and a model source are set. The `test` job in `.github/workflows/ci.yml` runs it with `THME_REQUIRE_RUNTIME_COMPAT=1` only when the repository variables `THME_COMPAT_MODEL`, `THME_LLAMA_CPP` and `THME_OLLAMA` are all set; otherwise the step is skipped and the run passes without loading the file in any runtime. Whether those variables are set on the repository was not checked here (unverified).
- **Done means**: A CI job or a documented local run sets `THME_REQUIRE_RUNTIME_COMPAT=1` with a pinned llama.cpp build, and the run output records which runtime loaded the exported GGUF. Until then the README and BACKLOG say the runtime check was not run.

### [THME-3046] Original Rust GPU-unlock engine (placeholder)

- **Milestone**: v0.2.7 · **Roadmap**: none · **Priority**: p2 · **Area**: security · **Type**: feature · **Status**: triage
- **Problem**: Placeholder only. The owner wants THME to write its own GPU-unlock engine in Rust as original work, after the GPL-2.0 derived code is removed (THME-3045); it must be written clean-room: from public vendor documentation, by someone who has not read the GPL source.
- **Done means**: Not scoped yet; a later ticket defines the design, tests and review.

<!-- backlog:end -->

---

## Archive

- Finished tickets: [docs/archive/BACKLOG_done.md](docs/archive/BACKLOG_done.md) (49 tickets).
- Old sprints (v0.1, superseded and deferred plans): [docs/archive/BACKLOG_sprints_v0.1.md](docs/archive/BACKLOG_sprints_v0.1.md).
