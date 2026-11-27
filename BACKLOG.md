# THME Engineering Backlog

> Source of truth for planned work. Tickets between the `backlog:start` and `backlog:end` markers
> are mirrored to GitHub Issues by `scripts/sync_github_backlog.py`. Process: [docs/WORKFLOW.md](docs/WORKFLOW.md).
> Milestones and exit criteria: [ROADMAP.md](ROADMAP.md).

Each ticket lists: milestone, roadmap item, priority (p0 stop the line · p1 this milestone ·
p2 next), area, type and status (triage · ready · blocked), plus the observable result that
closes it.

<!-- backlog:start -->

## Active: v0.2.0 close-out

### [THME-2001] Cross-check the forward pass against the Hugging Face reference

- **Milestone**: v0.2.0 · **Roadmap**: v0.2.0.6 · **Priority**: p1 · **Area**: inference · **Type**: feature · **Status**: done
- **Problem**: Measured perplexities rely on THME's own forward pass, which is unit-tested but has never been compared with the reference implementation.
- **Progress** (20 November 2026): Pythia-14M passes (Hugging Face reference run on CPU and on CUDA; THME on its default device), max absolute logit difference 1.9e-4 against a 0.005 tolerance (report: docs/research/hf-cross-check-pythia-14m.json). The script now loads the reference in fp32; newer transformers default to the stored dtype (fp16 for Pythia), which had shown a false 0.36 gap. SmolLM-135M passes too, 2.1e-5 (docs/research/hf-cross-check-smollm-135m.json).
- **Done means**: A script compares THME and Hugging Face logits on fixed prompts for SmolLM-135M and Pythia-14M; maximum absolute logit difference is reported and below an agreed tolerance. Blocked while new model runs are paused.

### [THME-2002] Upgrade egui/eframe to clear RUSTSEC-2026-0194/0195

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: dependencies · **Type**: chore · **Status**: done
- **Problem**: eframe 0.31 pulls quick-xml 0.30 (two advisories, ignored in deny.toml with this ticket) and several unmaintained transitive crates.
- **Progress** (19 November 2026): eframe and egui upgraded to 0.37; the two `ignore` entries are gone from deny.toml and `cargo deny check` passes; fmt, clippy and tests are green. Still open: all Studio tabs recaptured and visually checked (a recording on 19 November showed the Live 3D capture crop no longer fits the new layout).
- **Done means**: eframe upgraded; the two `ignore` entries removed from deny.toml; `cargo deny check` passes. The visual check of every Studio tab moved to THME-3069 (v0.4) on 20 November 2026 by the owner.

### [THME-2003] Split src/gui/app.rs into tab modules

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p2 · **Area**: gui · **Type**: chore · **Status**: done
- **Problem**: app.rs is about 6,000 lines with ten tab renderers sharing mutable state (supersedes THME-910).
- **Done means**: Each `render_tab_*` lives in `src/gui/tabs/<name>.rs`; no behaviour change; screenshots identical apart from timestamps.

### [THME-2004] Publish the first measured recipe search and Pareto chart

- **Milestone**: v0.2.0 · **Roadmap**: v0.2.1.7 · **Priority**: p2 · **Area**: research · **Type**: experiment · **Status**: done
- **Problem**: `thme search` and the Pareto chart exist but no measured search report is committed.
- **Progress** (20 November 2026): `docs/benchmarks/{pythia-14m,smollm-135m}/data/search_report.json` committed (Q8_0, Q4_0, grouped by layer, 8,192 calibration tokens, 16,384 validation tokens, 5 random controls each); pareto charts regenerated. Pythia-14M: the 50% budget was **not met** (58.3% achieved, validation Δ +0.303, passes the 0.45 gate); the 30% Pareto point could not be reached and repeats the 58.3% plan. SmolLM-135M: 50% met (validation Δ −0.018); all points are within about ±0.02 PPL of the baseline, which is within noise (no interval computed). Searched plans beat all five controls on Pythia and are not separated from them on SmolLM.
- **Done means**: `docs/benchmarks/<model>/data/search_report.json` committed for Pythia-14M and SmolLM-135M with five or more random controls; chart and dossier section regenerated. Blocked while new model runs are paused.

### [THME-2005] Release v0.2.0

- **Milestone**: v0.2.0 · **Roadmap**: v0.2.0.9 · **Priority**: p1 · **Area**: ci · **Type**: chore · **Status**: deferred (owner decision)
- **Problem**: The measured core is complete but unreleased.
- **Done means**: CHANGELOG section `[0.2.0]`, Cargo version 0.2.0, tag `v0.2.0`; the release workflow publishes four signed archives with checksums.
- **Progress** (27 November 2026): The owner can release v0.2.0 but the version number lags the work done (engine, GPU path, formats, gate, sampler). Decide the release version and name first, then do the CHANGELOG section, Cargo version and tag; no code is blocking.

### [THME-2006] Configure repository protection and secrets

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: ci · **Type**: chore · **Status**: done
- **Problem**: Branch protection, required checks and the signing key live in GitHub settings.
- **Progress**: Ruleset on `main` (pull requests, squash only, up-to-date branches, 11 required checks), `MINISIGN_KEY` secret and published public key are in place.
- **Done means**: Settings match docs/WORKFLOW.md section 7; a test pull request cannot merge with a failing required check.

## Next: v0.2.2 Formats and calibration

### [THME-2201] Per-tensor formats in the GGUF writer

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.1 · **Priority**: p1 · **Area**: exporter · **Type**: feature · **Status**: done
- **Problem**: The exporter writes every tensor as one format, so a recipe and its artifact differ.
- **Done means**: Exporting a recipe writes each tensor in its assigned format; re-importing and evaluating the file reproduces `thme evaluate`'s perplexity for the same recipe.

### [THME-2202] Q4_K, Q5_K and Q6_K encoders

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.2 · **Priority**: p1 · **Area**: exporter · **Type**: feature · **Status**: done
- **Problem**: Only Q8_0, Q4_0 and TQ1_0 can be encoded; K-quants are the common deployment formats.
- **Done means**: Round-trip tests against the existing decoders; the formats are usable in recipes and `thme search --formats`.

### [THME-2203] Capture calibration activations from the forward pass

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.3 · **Priority**: p1 · **Area**: inference · **Type**: feature · **Status**: done
- **Problem**: Activation-aware methods need per-layer input statistics.
- **Done means**: A probe records per-linear-tensor input channel statistics on calibration tokens and saves them; unit-tested on the synthetic model.

### [THME-2204] Activation-aware scaling (AWQ-style)

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.4 · **Priority**: p1 · **Area**: byoa · **Type**: feature · **Status**: done for Pythia-14M and SmolLM-135M (C4 arm and size check pending)
- **Problem**: Round-to-nearest 4-bit fails the gate on Pythia-14M (Δ +31 to +74).
- **Progress**: Implemented and tested (`thme.awq_q4_0`, `thme.awq_q4_k`, `--calibration`). 20 November 2026: control A (`ScaleSource::PermutedScales`; rule JSON without the field keeps real scales) and train-split calibration windows (`scripts/prepare_eval_tokens.py --split train --calibration-windows 16 --window-len 2048 --seed S`, S in 1234, 1, 2, 3, 4; control B1 adds `--shuffle`, control B2 `--random-ids --vocab-size V`) are implemented and tested offline. Calibration, control and test token files are prepared and hashed; provenance is in `docs/benchmarks/*/data/awq_calibration.json`. Scoring corpus unchanged: the test split reproduces the existing verification reports' hashes.
- **Pre-registered** (20 November 2026, before any measured run): AWQ beats round-to-nearest only if RTN perplexity minus the mean AWQ perplexity over the five seeds exceeds 2 x pooled SD (pooled SD from the sample variances of AWQ, control A and control B1, n = 5 each) and every seed's AWQ perplexity is below RTN. Gate: mean dPPL vs unquantized at most 0.45. A null or negative result is reported as such.
- **Measured, Pythia-14M** (20 November 2026, `scripts/run_awq_calibration.py`, WikiText-2 test, reference CPU forward pass, pre-registered rule applied unchanged; numbers in `docs/benchmarks/pythia-14m/data/awq_calibration.json` and the dossier). Q4_0: ceiling 114.2862, RTN 193.0028, AWQ mean 170.7025 (SD 2.75), control A 194.6773, control B1 172.8148, B2 193.3853. AWQ beats RTN (gap 22.30 > 2 x pooled SD 10.26, all seeds below RTN) and beats control A, but does NOT beat control B1 (gap 2.11). Q4_K: RTN 168.2951, AWQ 161.2814 (SD 3.70), control A 167.4629, control B1 158.0235, B2 173.4992. AWQ beats RTN (gap 7.01 > 6.05, all seeds) and control A, does NOT beat control B1 (B1 is lower by 3.26). Neither format passes the 0.45 gate (Delta +56.4 and +47.0 PPL vs the unquantized model). Reading: the scale vector carries real information (permuting it removes the gain), but a shuffled-token calibration does as well as ordered train text, so per-channel statistics, not text order, drive the gain. Caveats: one tiny model; Q4_K blocks on Pythia-14M cross rows (128-wide rows, 256-value blocks); per-tensor alpha is not reported by the CLI.
- **Fixed on the way**: `thme calibrate` keyed GPT-NeoX statistics to the bias tensors, so AWQ could not run on Pythia; fixed in `src/inference/calibration.rs` with a test.
- **Measured, SmolLM-135M** (20 November 2026, on the GPU: RTX 4060 Ti, Vulkan; `scripts/run_awq_calibration.py --device gpu`, WikiText-2 test, same pre-registered rule, report `docs/benchmarks/smollm-135m/data/awq_calibration.json`): Q4_0 round-to-nearest 23.70, AWQ mean 22.04 (SD 0.06), control A 23.61, shuffled-calibration control 22.00; Q4_K round-to-nearest 24.62, AWQ mean 21.64 (SD 0.13), control A 24.38, shuffled 21.62; unquantized 19.80. AWQ beats round-to-nearest in both formats (every seed, gap far above 2 x pooled SD) and beats the permuted-scale control, but not the shuffled-calibration control, and the 0.45 gate still fails (+2.24 and +1.84). Finding: the gain comes from generic per-channel statistics, not from the particular text or its order; the random-token stress arm reaches 22.36 (Q4_0).
- **PENDING**: SmolLM-135M (not run: one full-test-split forward pass took about 10 minutes on this CPU and `thme evaluate` repeats the unquantized baseline pass in every run, so about 20 minutes per run and 34 runs, roughly 11 hours; caching the baseline would roughly halve it; needs an owner decision on running time), the C4 out-of-domain arm (not downloaded), the 8k / 32k / 131k size check, and a per-run alpha report.
- **Done means**: A BYOA rule using captured activations; measured against plain Q4_0 on the same split with random controls.

### [THME-2205] Second-order rounding (GPTQ-style)

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.5 · **Priority**: p2 · **Area**: byoa · **Type**: feature · **Status**: ready
- **Problem**: Error compensation across columns is the strongest known post-training method.
- **Progress** (19 November 2026): `thme.gptq_q4_0` and `thme.gptq_q8_0` follow the reference algorithm (scales fitted per 32-column group, Cholesky of the inverse Hessian, damping with retry). Tested on synthetic matrices: output error below plain rounding, result exactly on the exporter grid. Not measured on a real model; K-quants not supported yet.
- **Done means**: Implemented as a BYOA rule using calibration activations; measured against THME-2204.

### [THME-2206] Measure Hadamard rotation on real models

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.6 · **Priority**: p2 · **Area**: research · **Type**: experiment · **Status**: blocked
- **Problem**: `RotatedQuantize` lowers rounding error on outlier-heavy weights in tests; its effect on perplexity is unmeasured.
- **Done means**: `thme.hadamard_int4` and `thme.hadamard_ternary` compared with their unrotated rules via `thme evaluate` on both models; reports committed.

### [THME-2207] SmoothQuant-style scale migration

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.7 · **Priority**: p2 · **Area**: byoa · **Type**: feature · **Status**: done
- **Problem**: The existing SmoothQuant rule clips weights only; true migration rescales weights and activations together.
- **Done means**: Migration folded into the preceding norm so the model's function is preserved before quantization; a test confirms unchanged outputs without quantization.

### [THME-2208] llama.cpp compatibility checks for exported files

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.8 · **Priority**: p2 · **Area**: exporter · **Type**: feature · **Status**: ready
- **Progress** (19 November 2026): `src/exporter/gguf_check.rs` checks a finished GGUF v3 file: header, counts, alignment, tensor names, dimensions, types, truncation, and now the data section (offsets, sizes, bounds, overlap) and the metadata keys (architecture, the five llama keys). About 30 tests on synthetic files, plus a test that a file written by THME's own writer passes. Not done: the optional run through llama.cpp. Three spec facts were written from memory and are marked `shortcut:` in the code: the alignment rule (multiple of 8, power of two), the data-section start rule, and the llama key names. Confirm them against the GGUF specification before relying on the check.
- **Progress** (20 November 2026, llama.cpp b12584, SmolLM2-1.7B; report `docs/benchmarks/smollm2-1.7b/data/llama_cpp_reference.json`): the optional run through llama.cpp was done and it does not pass. `examples/check_gguf.rs` runs the checker (F16, Q8_0 and Q4_0 files: 0 errors, 0 warnings), yet llama.cpp (CUDA and Vulkan builds) refuses all of them: first `cannot find tokenizer vocab in model file` (the GGUF carries no vocabulary), then `tensor 'token_embd.weight' not found` (THME writes Hugging Face tensor names, not GGML names). Also missing or different for llama.cpp: BF16 norm weights (needs F32), q/k rows not permuted for its rope layout, no `llama.rope.dimension_count`. With an outside-the-repo adapter that fixes exactly those points (weight blocks copied unchanged apart from the q/k row permutation) all three files load and generate the same sensible text on both builds. Also found: `thme export --quant fp16|q8_0` fails on this BF16 checkpoint (recipe path works). Not done: the exporter writing GGML names, tokenizer and the permutation itself (follow-up ticket needed; the check above is the acceptance test).
- **Problem**: Exported GGUF files are only re-read by THME itself.
- **Done means**: The existing optional CI step runs a heterogeneous export through llama.cpp when the runtime is configured.

### [THME-2209] Measured artifact size in every report

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.9 · **Priority**: p1 · **Area**: exporter · **Type**: feature · **Status**: done
- **Problem**: Recipe and search sizes are nominal bits per weight.
- **Done means**: Reports include the written file size including scales and metadata alongside the nominal figure.

## Later

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

### [THME-3097] DC-CCCP sign split

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p1 · **Area**: allocator · **Type**: fix · **Status**: done in code; not run on a real model
- **Problem**: `src/allocator/dc_cccp.rs` majorizes every coupling entry; paper section 4.2 keeps negative-coefficient terms convex and majorizes only positive ones.
- **Done means**: sign-split majorizer with a test that the surrogate is at least the objective at random points and equal at the expansion point, and monotone descent.
- **Progress** (27 November 2026): `sign_split_surrogate` and `mm_descend` in `src/allocator/dc_cccp.rs`. Tests: the surrogate is at least the objective at 200 random instances whose gamma_plus has negative off-diagonal entries and equals it at the expansion point; majorize-minimize steps never raise the objective and stay in budget; the solver is never worse than its diagonal-DP start. Not done: a measurement on a real model.

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
- **Progress** (27 November 2026): labels `area:*` for all BACKLOG areas and `status:in-progress|in-review|done|shelved` created on GitHub; status board regenerated (846 tests counted by the generator; 856 passed in a full `cargo test` run on 27 November 2026, release `thme.exe` 14.90 MiB); `docs/research/paper_v5_1_notes.md` lists measured code state (856 tests, 14.90 MiB), gate declared range and v2, DC-CCCP sign split, the llama.cpp match, KV precision and the softmax note, each with a suggested sentence (the paper itself is still not on disk); README "Now building" table, measurement paragraph, Studio chat note and roadmap diagram brought up to date (llama.cpp match, KV precision, pathfinder, the v0.3.0 node, published reports; two statements that were no longer true were corrected); getting-started and the manual describe the llama.cpp layout export, BF16 `--quant`, `--kv-precision`, `thme pathfinder`, `--allow-root` and gate v2 with their evidence levels (checked: `validate` has no `--gate`, only `evaluate` does); `docs/status/roadmap_vs_backlog.md` compares roadmap states with ticket statuses (finding: v0.2.8 has every ticket done or done in code while the roadmap calls it next; v0.2.1 has no BACKLOG tickets); chart regeneration changed only line endings, so nothing was committed; the 85 legacy tickets without a Status bullet now have one copied from their heading (31 done as v0.1 deliverables, 9 completed, 45 marked triage because no status was written; nothing is marked measured); five stale top-level agent notes untracked (see THME-3050); crate version set to 0.2.0.

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

### [THME-3104] Finish every v0.2 sub-milestone

- **Milestone**: v0.2.9 · **Roadmap**: v0.2.9.9 · **Priority**: p1 · **Area**: docs · **Type**: chore · **Status**: in progress
- **Problem**: 44 of the 90 sub-milestones of v0.2.0 to v0.2.9 are not marked done. Some are small measurements, some need hardware or an owner decision, and v0.2.3 is a training program.
- **Done means**: Each sub-milestone is either done (its own done-means text met, evidence committed) or closed with a dated reason (owner decision, missing hardware); `docs/status/v0.2_completion_plan.md` is current; v0.2.9.9 is closed by re-reading it.
- **Progress** (27 November 2026): the plan groups the 44 items (CPU work, hardware, owner decisions, the training program) and fixes the order. No sub-milestone is changed yet.

### [THME-3105] Teacher-to-student distillation pipeline (`thme distill`)

- **Milestone**: v0.2.3 · **Roadmap**: v0.2.3.2 · **Priority**: p1 · **Area**: trainer · **Type**: feature · **Status**: triage
- **Problem**: The headline use is a frontier model teaching a smaller open-weight model quickly, with THME measuring the result. THME has the teacher bridge, the gate, `.epiph` provenance and LoRA merge, but no pipeline joins them, and the trainer is synthetic until THME-3036 lands.
- **Done means**: The design in `docs/research/distillation-pipeline-design.md` is built in its first slice: quantization-aware distillation on SmolLM-135M with the source as teacher, a held-out evaluation set whose hash is recorded and never used in training, a decision rule written before the run, cost accounting, and a committed report comparing the result with round-to-nearest and AWQ; then sequence-level distillation from a configured teacher command, with the teacher's identity and terms note recorded in the dataset.
- **Progress** (27 November 2026): design written. Not built. Owner decision: distillation with a hosted teacher is for functionality testing only, never released or used commercially (files stay local, annotated `use=test-only`, `distill/` is git-ignored). Claude is the first teacher, with a registry for several teachers and a curriculum of subjects (one teacher, prompt set, weight and held-out set per subject). Whether a hosted teacher's usage terms allow this is for the owner and counsel; the pipeline needs a filled-in terms note before it runs one. Release or commercial use of a model taught by a hosted provider's model needs that provider's permission, held by the owner; planned guard (deferred by the owner on 27 November 2026, to be installed later): files annotated `use=test-only` cannot be signed, exported for sharing or released without `--permission-note` (recorded, not checked). Not built.

### [THME-3106] Measurement drift protocol: bias, drift and a tolerance band for every measured result

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.3 · **Priority**: p1 · **Area**: bench · **Type**: feature · **Status**: in progress
- **Problem**: A single run cannot tell a real difference from noise. Contention, thermal state, warm-up and GPU floating-point order move readings, and a reference check before and after a batch is the usual way to expose a bias that every run shares.
- **Done means**: `docs/research/measurement-drift-protocol.md` is applied by the report scripts: each batch has a warm-up run, n repeats in interleaved order, a reference check before and after, a recorded drift, an accept or flag decision against a stated tolerance (5 percent for timings by default, tighter for perplexities), a guard band that turns a verdict into Inconclusive when the drift can flip it, and, when the tolerance is exceeded, an attribution table (run index, other processes, GPU temperature, clock, power, CPU load); committed reports carry the numbers; the compatibility report repeats its GPU runtimes under it.
- **Progress** (27 November 2026): protocol written; `scripts/drift.py` with 14 tests worked by hand from the 100 ppm sensor example. Not yet applied to any committed report. The tolerances are proposals for the owner to adjust.

### [THME-3100] Calibrate the allocator link constant C

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p2 · **Area**: research · **Type**: research · **Status**: triage
- **Problem**: the link between the allocator objective and logit divergence is a modelling assumption (paper section 12.2) and ternary effects are not additive (sum +1,509, joint +15,877,276 perplexity).
- **Done means**: C estimated against joint measurements, or the surrogate replaced by an empirically bounded constant.

### [THME-3070] Incremental decoding with a KV cache

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.1 · **Priority**: p1 · **Area**: inference · **Type**: feature · **Status**: done
- **Progress** (20 November 2026): `KvCache` and `forward_extend` for LLaMA and GPT-NeoX behind `CausalLm`; the Studio chat reply loop prefills once and steps the cache (Gemma keeps the whole-sequence loop). Incremental logits equal full-sequence logits at every position: 0 on Pythia-14M and SmolLM-135M (24 tokens, real models, `THME_REAL_MODEL_TESTS=1`) and at most 7.6e-6 on a 96-token synthetic LLaMA (tolerance 1e-4). KV bytes equal `KvCacheEstimator` exactly. Measured paired old-vs-cached decode speed is in `docs/benchmarks/<model>/data/decode_speed.json` (shared machine; see the folder README). Not done: GPU path not measured; threads and matmul kernels unchanged (separate tickets).
- **Problem**: `chat::generate` calls `forward_logits` on the whole growing sequence for every new token, so each token costs a full pass over all tokens so far. Research note `docs/research/token-speed-budget-devices.md` (section 4) ranks this as the largest single loss for decode.
- **Done means**: A KV cache for the LLaMA and GPT-NeoX reference models behind `CausalLm` (`forward_logits` unchanged for the perplexity path). Incremental logits equal full-sequence logits at every position within a stated tolerance on a synthetic model, and on Pythia-14M and SmolLM-135M behind `THME_REAL_MODEL_TESTS=1`. Studio chat and the reply loop use prefill then one-token steps. `examples/decode_speed.rs` gives paired, interleaved old-vs-new decode tokens/s with spread, committed under `docs/benchmarks/<model>/data/decode_speed.json` with machine facts; KV bytes vs context checked against `src/profiling/kv_cache.rs`. Threads and matmul kernels are not changed here.

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

### [THME-3073] Perplexity on the GPU, device in every report, baseline cache

- **Milestone**: v0.2.4 · **Roadmap**: none · **Priority**: p1 · **Area**: gpu · **Type**: perf · **Status**: done
- **Problem**: Benchmarks (full WikiText-2 test split perplexity) ran only on the slow CPU path, and `thme evaluate` repeated the unquantized baseline pass in every run.
- **Progress** (20 November 2026): `forward_nll` runs on the device-resident decoder for LLaMA-family models (one submit per window, only the per-token losses read back); GPT-NeoX keeps per-operation layers with a fused GPU head and loss. `thme evaluate` and `thme calibrate` take `--device`; reports record requested and used device, adapter, backend and driver; the baseline pass is cached (keyed on model hash, token hash, context, max tokens, device and version). Every GPU wait has a timeout that becomes an error. Measured, full test split, RTX 4060 Ti (Vulkan), 3 paired runs each, report `docs/benchmarks/<model>/data/gpu_perplexity.json`: SmolLM-135M CPU 504.7 s, GPU 98.1 s (5.14x, range 4.91 to 5.20, perplexity 19.79846 vs 19.79842, quiet machine in all pairs); Pythia-14M CPU 51.1 s, GPU 31.7 s (1.53x, range 1.52 to 1.70, perplexity 114.28621 vs 114.28600; first two pairs not quiet).
- **Not done**: a device-resident GPT-NeoX decoder (Pythia gain is modest), device KV cache, f16 and subgroup kernels, Metal and DX12 timings.
- **Done means**: Perplexity over the full test split on GPU matches CPU within 1e-4 nats per window and is faster on both small models; the report records the device.

### [THME-3076] VS Code sidebar shows fixed example text as if it were live data

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p2 · **Area**: security · **Type**: bug · **Status**: done
- **Problem**: The extension's sidebar trees show static example text (a model name such as "thme-hybrid-12l" and a throughput such as "620 tok/sec") that THME never measured, which breaks the rule that no unmeasured number is presented as a result (found by the THME-3048 fix).
- **Progress** (21 November 2026): the four trees are data-driven. `thme-models` reads `thme inspect --json`; `thme-decisions` reads the file `thme plan` writes; `thme-telemetry` reads the file `thme profile` writes (layer sensitivity and the KV-cache projection — the old REA-LLM/EWHA figures had no CLI command, so they are gone rather than left as examples); `thme-copilot` is labelled as an example and carries no numbers. With no model selected, each tree says "no model selected". New commands `THME: Select Model` and `THME: Refresh Sidebar from the CLI` drive it, and `THME: Inspect Model` selects as well. `editors/vscode/src/treeData.ts` builds every row and `src/treeData.test.ts` lists every string the trees can show and fails if a row carries a number the CLI output does not contain; `npm test` passes 11 tests. The fixtures under `src/fixtures/` are real CLI output captured on `examples/mock_llama` (inspect trimmed to two tensors). Not done: the view is still named "REA-LLM & EWHA Telemetry" while showing `thme profile` numbers; renaming it is a product decision.
- **Done means**: Each sidebar value is read from the CLI's own output for the selected model, or the tree says "no model selected" or is labelled as an example; a test lists every string the tree can show and checks none is a made-up number.

### [THME-3077] Quality gate v2: a margin that fits the model's perplexity (two regimes)

- **Milestone**: v0.2.8 · **Roadmap**: none · **Priority**: p1 · **Area**: quality · **Type**: feature · **Status**: done in code (27 November 2026); the default gate stays v1, v2 is chosen with `--gate`
- **Problem**: The default gate (`thme-perplexity-v1`) is one absolute limit of 0.45 perplexity. That fits models whose perplexity is in the single digits, but Pythia-14M has a baseline of 114, where even a uniform Q8_0 export (+0.83, 0.73 percent) fails. An accuracy rule from emissions monitoring (a relative limit when the level is high, an absolute limit when it is low, because a relative limit misbehaves near zero) is the model: the owner's recollection, not yet checked against the standard's text.
- **Done means**: A versioned gate `thme-perplexity-v2` whose criterion applies a relative margin when the baseline is above a stated threshold and an absolute margin when it is below, with the threshold, margins and the rule written in the gate file and covered by its hash; both regimes documented in `docs/quality-gate.md` with the reasoning and a table of what each committed report would have been under v1 and v2; old reports keep their v1 verdicts; the decision rule is stated before it is applied to any new run; unit tests on both sides of the threshold.

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

### [THME-3000] Consolidate the dossier v3.1 claim register under docs/research/

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.9 · **Priority**: p1 · **Area**: research · **Type**: docs · **Status**: done
- **Problem**: Claims V-01 to V-135 are spread over the appendices of the PDF; citations were written from memory.
- **Done means**: One register file with unique IDs, status letters and the section each claim lives in; every citation checked or marked unverified.

### [THME-3001] Engine error module (dossier WP1)

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.1 · **Priority**: p1 · **Area**: engine · **Type**: feature · **Status**: done
- **Problem**: Engine primitives need one error type that never panics and never falls back silently.
- **Done means**: `src/engine/error.rs`: invalid-input, infeasible, no-convergence and non-finite variants plus a finite-value guard, each reached by a test; no behaviour change elsewhere.

### [THME-3002] Exact dynamic-programming allocator behind `thme search --solver` (dossier WP2)

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.2 · **Priority**: p1 · **Area**: engine · **Type**: feature · **Status**: blocked
- **Problem**: Docs promised a knapsack solver; `ParetoAllocator` uses sensitivity thresholds and `thme search` is greedy.
- **Progress**: Implemented: `engine::alloc` (exhaustive-search oracle on 300 instances) and `thme search --solver dp --dp-grid N`; never worse than greedy on predicted cost (200 random tables). Default stays greedy. Blocked on the comparison on real cost tables, which needs model runs (paused).
- **Done means**: `engine::alloc` matches exhaustive search on small instances and is never worse than greedy at equal budget; default solver unchanged until compared on committed cost tables.

### [THME-3003] One versioned quality gate; retire the stale 0.25 tolerance (dossier WP20)

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.1 · **Priority**: p1 · **Area**: quality · **Type**: fix · **Status**: done
- **Problem**: The threshold lives in three places: CLI defaults (0.45), `configs/quality_gates.yaml` (0.25, unread) and evaluator tests (0.25).
- **Owner decision (2026-11-13)**: thresholds are customizable; a quality gate may hold one criterion or many.
- **Done means**: A gate is versioned data with a content hash, absolute and relative margins; every report records its gate version; MMLU/GSM8K marked not measured.

### [THME-3004] Shared basis module: one Walsh-Hadamard transform (dossier WP5)

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.5 · **Priority**: p2 · **Area**: engine · **Type**: refactor · **Status**: done
- **Problem**: Three implementations existed: `MatrixMath::fast_walsh_hadamard_transform`, `byoa::wht_blocks` and an inline copy in the `OrthogonalRotation` rule. Two copies can drift and confound a comparison.
- **Done means**: `engine::basis` owns the transform; both callers use it with bit-identical results before and after; orthonormality and involution tests.

### [THME-3005] SPDX headers on every source file

- **Milestone**: v0.2.7 · **Roadmap**: v0.2.7.9 · **Priority**: p2 · **Area**: ci · **Type**: chore · **Status**: done
- **Problem**: The dossier's rules assume `SPDX-License-Identifier: AGPL-3.0-only` headers; no file carries one.
- **Done means**: Every `.rs` and script file has the header; the hygiene CI job fails on a file without it.

### [THME-3006] Number-format conversions with stated error bounds (dossier WP3)

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.3 · **Priority**: p2 · **Area**: engine · **Type**: feature · **Status**: done
- **Problem**: Codecs live in the exporter without declared error bounds.
- **Also**: absmean ternary is written three times (`MatrixMath::discretize_bitnet_ternary`, `BitNetPacker::pack_matrix`, the BYOA `AbsmeanTernary` rule) with different all-zero handling; one shared quantizer.
- **Progress**: `engine::numfmt` started with the shared absmean-ternary quantizer; the three copies call it. Other formats and error bounds open.
- **Done means**: `engine::numfmt` covers integer, ternary, binary, codebook, FP16/BF16/FP8 and phase, each with a bound its tests confirm.

### [THME-3007] Dependency-free SVG visual engine (dossier WP9)

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.8 · **Priority**: p2 · **Area**: engine · **Type**: feature · **Status**: done
- **Problem**: Charts come from Python scripts; the Studio and dossier cannot share a headless renderer.
- **Done means**: Deterministic SVG for log-axis traces, heatmaps, scatter with retained-variance label, dataflow graphs and Pareto fronts; NaN or non-positive log values return errors.

### [THME-3008] Permission and ownership record for reverse-engineering features (dossier WP19)

- **Milestone**: v0.2.7 · **Roadmap**: v0.2.7.4 · **Priority**: p1 · **Area**: core · **Type**: feature · **Status**: triage
- **Problem**: Architecture inference and closed-format analysis must only run on models the user owns or may analyze.
- **Done means**: Those features refuse to run without a stored basis and purpose statement next to the model hash; wording reviewed by a lawyer.

### [THME-3009] Versioned schemas under docs/schemas/ (dossier B.6.4)

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.9 · **Priority**: p2 · **Area**: core · **Type**: docs · **Status**: done
- **Problem**: Benchmark task, benchmark result, quality-gate version, plugin protocol and conversion-registry entry have no schema.
- **Done means**: Each schema carries a version field and validates a fixture in CI.
- **Progress**: benchmark task, result record and quality gate schemas done (`docs/schemas/*-v1.json`, `tests/schemas.rs`). Plugin protocol and conversion-registry schemas wait until those exist (plugins are commercial-only for now).

### [THME-3010] BYOA rules get calibration through their interface, not tensor metadata

- **Milestone**: v0.2.2 · **Roadmap**: none · **Priority**: p2 · **Area**: byoa · **Type**: refactor · **Status**: done
- **Problem**: `ActivationAware` reads statistics from a JSON string in tensor metadata and needs a special case in `apply_selected_format`; each new data-needing rule would add another.
- **Done means**: A rule receives an apply context (calibration statistics, row width) at one seam; the metadata side channel and special case are gone; AWQ tests unchanged.

### [THME-3011] Measured bits per weight from the written artifact (placeholder)

- **Milestone**: v0.2.6 · **Roadmap**: none · **Priority**: p3 · **Area**: exporter · **Type**: feature · **Status**: triage
- **Problem**: Reports show nominal bits per weight (format bit-widths averaged over all parameters, unchanged tensors at stored width). `TensorFormat::bits_per_weight` also says 8.0 for Q8_0 and 4.5 for Q4_0, while the stored blocks are 8.5 and 4.5. llama.cpp reports file-level bits per weight (artifact bytes x 8 / parameters), which includes block scales.
- **Done means**: Every report that writes an artifact also states measured bits per weight from its byte size, next to the nominal figure, each labelled; the nominal definition is kept for unexported candidates. No committed number changes without a new report.

### [THME-3012] `.epiph` v1.0: open container with full model metadata

- **Milestone**: v0.2.6 · **Roadmap**: v0.2.6.1 · **Priority**: p1 · **Area**: exporter · **Type**: feature · **Status**: done
- **Problem**: THME had no native format that carries a model with its recipe, provenance, quality results and calibration.
- **Done means**: Spec `docs/formats/epiph-v1.md`; `src/epiph` reader/writer; `thme epiph pack|verify`; the importer reads `.epiph`; round trip, corruption, truncation and GGUF-conversion tests pass.

### [THME-3013] `.epiph`: pack tensors in their selected formats

- **Milestone**: v0.2.6 · **Roadmap**: v0.2.6.3 · **Priority**: p1 · **Area**: exporter · **Type**: feature · **Status**: done
- **Problem**: v1.0 stores tensors as stored; a candidate's quantized formats are not packed yet.
- **Done means**: Packing a recipe candidate stores each tensor in its selected format (reusing the exporter codecs); the forward pass on the packed file equals the in-memory candidate.

### [THME-3014] `.epiph`: tokenizer and GGUF metadata attachments

- **Milestone**: v0.2.6 · **Roadmap**: v0.2.6.4 · **Priority**: p2 · **Area**: exporter · **Type**: feature · **Status**: done
- **Problem**: GGUF metadata (including the tokenizer) is not serializable yet, so GGUF to .epiph to GGUF loses it.
- **Done means**: GGUF -> .epiph -> GGUF keeps every metadata key; SafeTensors tokenizer files are attached.

### [THME-3015] `.epiph`: lossless payload compression

- **Milestone**: v0.2.6 · **Roadmap**: v0.2.6.6 · **Priority**: p2 · **Area**: exporter · **Type**: feature · **Status**: done
- **Progress** (19 November 2026): `.epiph` format 1.3 adds a per-payload encoding (`identity` or `deflate`, flate2's pure-Rust backend). `thme epiph pack --compress deflate` only compresses a payload when it actually shrinks; `verify` checks both the stored and decoded digests, and decompression is bounded by the declared length so a corrupt or hostile file cannot expand past it (a decompression-bomb test is included). 1.2 files load unchanged. 13 tests on synthetic tensors. Not measured on a real model; readers older than 1.2 refusing deflate is stated in the spec but not tested here.
- **Problem**: Owner asks for a compressed variant. Quantization already shrinks weights; lossless coding on top needs a pure-Rust codec dependency.
- **Done means**: Dependency proposed and approved (licence, binary size); per-payload `encoding` field; compression ratio and decode speed reported as measured on the synthetic checkpoint only.

### [THME-3016] `.epiph`: GPU/CPU placement plan

- **Milestone**: v0.2.6 · **Roadmap**: v0.2.6.7 · **Priority**: p2 · **Area**: engine · **Type**: feature · **Status**: done
- **Progress** (19 November 2026): `src/epiph/placement.rs` plans GPU, CPU and streamed tensors exactly from a VRAM and host budget; the plan is stored in the manifest (`.epiph` format 1.2, older files still verify), `thme epiph pack --placement VRAM_MIB:HOST_MIB` writes it and `verify` checks it against the tensors and prints it. Tested on synthetic tensor sets only. No runtime reads the plan yet, because the GPU runtime does not use `.epiph` files.
- **Problem**: Owner asks for files that place their own parameters across GPU and CPU. THME has no GPU backend yet (v0.3).
- **Done means**: A `placement` section from the exact allocator for a stated memory budget per tier; tested on synthetic budgets; used by the GPU runtime when it exists.

### [THME-3017] Trademark search and filing for the format name

- **Milestone**: v0.2.6 · **Roadmap**: none · **Priority**: p1 · **Area**: legal · **Type**: chore · **Status**: blocked (owner: funding and registration)
- **Problem**: The owner wants the format name protected under Epiphanes. Extensions are not usually registrable; the name is.
- **Done means**: A lawyer has searched and advised on the name; filing decided by the owner.
- **Progress** (27 November 2026): The Epiphanes site exists but the domain and the business are not registered yet; funding for registration is being sought. A lawyer search and a filing decision follow registration.

### [THME-3018] Static model security scanner (`thme scan`)

- **Milestone**: v0.2.7 · **Roadmap**: v0.2.7.1 · **Priority**: p1 · **Area**: security · **Type**: feature · **Status**: done
- **Problem**: Model files can execute code on load (pickle imports, template injection, trust_remote_code) or hide payloads in unused bytes.
- **Done means**: `src/scan`: pickle opcode walker (plain and PyTorch zip), GGUF/SafeTensors header and coverage checks, metadata and chat-template rules, code and executable files, `.epiph` verification; `--fail-on` exit code; tests on built fixtures, no file loaded or run.

### [THME-3019] Signed `.epiph` files

- **Milestone**: v0.2.6 · **Roadmap**: v0.2.6.2 · **Priority**: p1 · **Area**: security · **Type**: feature · **Status**: done
- **Problem**: Owner wants files signed by Epiphanes. Third-party converters can read the open format but get plain weights only; signature, provenance, quality evidence and placement are THME features.
- **Done means**: Optional signature section over the manifest and payload digests; `thme epiph sign` / `verify` check it; key handling documented; a pure-Rust signature dependency proposed and approved first.

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

### [THME-3022] Scanner: more formats and compressed pickles

- **Milestone**: v0.2.7 · **Roadmap**: v0.2.7.2 · **Priority**: p2 · **Area**: security · **Type**: feature · **Status**: done
- **Progress** (19 November 2026): deflate pickles, NumPy object arrays, oversize pickles, Keras and ONNX are done and tested. Legacy HDF5 `.h5` files are now inspected in a bounded way: the Keras `model_config` JSON is searched for in the readable bytes (64 MiB cap) and `Lambda` layers are flagged High, unknown custom layers Medium. 6 tests on fixtures built in the test. Limits: attributes stored compressed, chunked or in fragmented heaps can be missed, so a clean result means no code-executing layer was found in the readable part, never that the file is safe; marshalled-code fields are not flagged.
- **Problem**: Legacy HDF5 `.h5` Keras files were not inspected.
- **Done means**: Each format added with fixtures; compressed pickles inspected with a pure-Rust inflate dependency proposed first.

### [THME-3024] Benchmark registry: task files and result records

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.2 · **Priority**: p1 · **Area**: quality · **Type**: feature · **Status**: done
- **Problem**: Dossier R.4 and WP31: benchmarks must be declarative data with one result format, so runs are reproducible and intervals and paired tests can be computed without re-running.
- **Done means**: Task file (dataset path and hash, split, licence, prompt template, extraction, metric) and result record (every Table R2 field, per-item results) with strict validation; a folder registry; `thme bench list/validate/verify-dataset`; tests show malformed files are refused and a record round-trips.

### [THME-3027] Benchmark metrics library

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.4 · **Priority**: p1 · **Area**: quality · **Type**: feature · **Status**: done
- **Problem**: Dossier WP33: the runner needs one tested implementation of each extraction rule and metric a task file can name.
- **Done means**: Extraction (exact, first line, last number, choice letter, after marker), exact and normalized match, numeric tolerance, likelihood choice, unbiased pass@k; Table R3 pass@k values reproduced; pass@k equal to enumeration.

### [THME-3028] Benchmark statistics library

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.5 · **Priority**: p1 · **Area**: quality · **Type**: feature · **Status**: done
- **Problem**: Dossier WP34 and R.5: every score needs an interval, comparisons on the same items need paired tests, and small sets must be recognised as too small.
- **Done means**: Wilson, percentile bootstrap, paired bootstrap, exact McNemar, required-items calculator, Holm correction; Table R3 values reproduced; simulated coverage near nominal.

### [THME-3029] Three-state quality gate

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.6 · **Priority**: p1 · **Area**: quality · **Type**: feature · **Status**: done
- **Problem**: Dossier WP35 and R.5: a point-estimate gate passes candidates whose interval says the drop could be several times the margin.
- **Done means**: The gate decides pass, fail or inconclusive from a degradation interval through the existing decision path; point measures decide as before; two result records are compared with a paired bootstrap only when comparable; synthetic cases with known decisions.

### [THME-3030] Benchmark contamination check

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.7 · **Priority**: p1 · **Area**: quality · **Type**: feature · **Status**: done
- **Problem**: Dossier WP36 and Q.6: a score on items that appear in training or teacher-written data is not clean, and the overlap must be listed before a score is shown.
- **Done means**: Normalized n-gram overlap between a task's items and any corpus, streamed; a report of overlapping items with fraction and source; planted overlaps found in tests; `thme bench contamination` fails when any item overlaps.

### [THME-3031] Benchmark runner: imported mode

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.3 · **Priority**: p1 · **Area**: quality · **Type**: feature · **Status**: done
- **Problem**: Dossier R.4: numbers produced by other tools must enter THME through the same record format, labelled as supplied and never as measured.
- **Done means**: Outputs (answers, log-likelihoods, pass counts) scored with the metric library into a validated result record with run mode and evidence `imported`; every dataset item scored exactly once; one dataset reader shared with the contamination check.

### [THME-3032] Benchmark charts

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.8 · **Priority**: p2 · **Area**: quality · **Type**: feature · **Status**: done
- **Problem**: Dossier WP37 and R.7: benchmark results need charts drawn from stored records, with intervals and evidence status.
- **Done means**: Scores with intervals, paired changes against a baseline, a model-by-benchmark heatmap, from result records through the visual engine; light and dark; evidence label on each.

### [THME-3033] Isometric figure kit in Rust (anatomy port)

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.8 · **Priority**: p2 · **Area**: core · **Type**: feature · **Status**: done
- **Problem**: Owner wants engine, 3D visuals and speed in Rust; the isometric figure kit existed only in TypeScript.
- **Done means**: Camera, extruded solids with facing-based shading, painter's order, lines, dots, wires and the card in `engine::iso`, byte-identical to the kit on a committed set of cases (`scripts/iso_golden.mjs` regenerates them from the kit), and timed against it (`examples/iso_bench.rs`).

### [THME-3023] Re-measure reports affected by the FP16 rounding fix

- **Milestone**: v0.2.2 · **Roadmap**: none · **Priority**: p1 · **Area**: research · **Type**: chore · **Status**: done
- **Progress** (19 November 2026): the stale committed reports, charts and dossier were removed instead of re-run for now; results will be re-measured and published with their method.
- **Progress** (20 November 2026): verification, precision-sweep, depth-probe and search reports re-run with the fixed code for both models and committed; the dossier caveat is replaced by a note that the reports come from the fixed code. Gate results as measured: Pythia-14M exported Q8_0 Δ +0.834 (FAILED, gate 0.45); recipe Δ +504.2 (FAILED). SmolLM-135M exported Q8_0 Δ +0.034 (PASSED); recipe Δ +28.9 (FAILED).
- **Problem**: Until 13 November 2026, f32-to-f16 conversion lost the rounding carry for values just below a power of two, so about 1 in 2,000 Q8_0, Q4_0 and TQ1_0 block scales (and FP16 values) were stored at half their size. Committed reports were measured with that code.
- **Done means**: The verification, precision-sweep and depth-probe reports for both models are re-run with the fixed code and committed; the dossier caveat is removed. Blocked while model runs are paused.

### [THME-3034] Third-party code register

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: governance · **Type**: chore · **Status**: done
- **Progress** (19 November 2026): `docs/third-party.md` lists the outside sources the repository already uses (crate licences checked against `cargo metadata`); `scripts/third_party.py --check` runs in CI beside the SPDX check and fails when a file whose header names a non-AGPL licence has no row. Code copied without any header is not detected.
- **Problem**: Reused algorithms and crates need a record of source, licence and notice. Code under GPL, non-commercial or "do not copy" terms is never copied; the idea is rewritten from the paper.
- **Done means**: A table in the repository lists every outside source (name, URL, licence read from the LICENSE file, what was reused, notice kept) and CI fails when a copied file lacks its row.

### [THME-3035] Replace mock and simulated data in the Studio and engine

- **Milestone**: v0.2.0 · **Roadmap**: none · **Priority**: p1 · **Area**: gui · **Type**: bug · **Status**: done
- **Progress** (19 November 2026): all five audit verdicts are handled. Studio values are labelled Estimate, Simulated or Demo (a guard test fails when one loses its label), including the stats ribbon, the speculative acceptance rate (Simulated) and the hard-coded demo constants (Demo), in the Studio, `thme speculate` and the MCP tool text. No number or algorithm was changed. Known smell: the speculative engine imports `gui::provenance`; moving the module to a neutral place is a small follow-up.
- **Problem**: The Studio telemetry and baseline figures are heuristic formulas and the micro-trainer, speculative acceptance and token flow are synthetic (see the audit in `docs/research/THME_AUTHENTICITY_AUDIT_AND_ROADMAP.md`, which still needs to be checked line by line). The Studio screenshots show them next to measured numbers.
- **Done means**: Each such value is either computed from the reference forward pass or carries a visible "simulated" label; a test fails on an unlabelled constant.

### [THME-3036] LoRA training with a real backward pass

- **Milestone**: v0.2.3 · **Roadmap**: v0.2.3.1 · **Priority**: p2 · **Area**: trainer · **Type**: feature · **Status**: triage
- **Problem**: THME has no backward pass, so it cannot fine-tune; the current trainer is synthetic.
- **Done means**: Hand-written CPU backward for the LLaMA and Gemma blocks, checked against finite differences, then LoRA adapters with AdamW on a tiny model. Written from the published formulas (permissive references only); a heavy training framework is not added.
- **Design note** (27 November 2026, from a summary the owner pasted of arXiv 2610.31642 "A Fix for Softmax and Cross-Entropy"; the paper itself was not read or verified here): softmax can lose the gradient of the dominant class once logit margins pass a precision-dependent size (reported about 16.6 in fp32), and distillation between near-equal student and teacher logits can leave a gradient noise floor. When the real backward is written, compute the cross-entropy and distillation gradients as direct differences (for example `p - onehot` and `p_student - p_teacher` from log-probabilities in f64) and add a test with margins above 17 in f32 that keeps the gradients summing to zero. Today's evaluation path is not affected: `src/inference/llama.rs` computes the loss from the maximum-subtracted form with the sum in f64, and `kv_perplexity` does the same. The current trainer (`src/trainer/mod.rs`) is synthetic and uses f32 softmax gradients.

### [THME-3037] Tokenizer and chat templates

- **Milestone**: v0.3.0 · **Roadmap**: v0.3.0.3 · **Priority**: p2 · **Area**: inference · **Type**: feature · **Status**: done
- **Progress** (19 November 2026): `thme tokenize MODEL_DIR TEXT` and `thme chat-template MODEL_DIR --messages FILE` encode, decode and render a model's own chat template (string or named-list form, bos/eos tokens, `raise_exception` errors). New direct crates, all Apache-2.0 and pure Rust: `tokenizers` 0.22.2 (no onig, no C++), `minijinja` 3.1.0 and `minijinja-contrib` 3.1.0. 15 tests on fixtures built in the test; no real Hugging Face model or real Llama or Mistral template was used. The release CLI grew to about 12.5 MiB, so the CLI limit was raised from 10 MiB to 16 MiB (owner decision, 19 November 2026).
- **Problem**: THME cannot encode text or apply a chat template, so it cannot chat with a model or run text benchmarks itself.
- **Done means**: Encode and decode through a permissively licensed tokenizer crate and render `chat_template` with a template engine, each proposed with its licence and binary-size cost first; round trips tested on committed fixtures.

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

### [THME-3041] `thme doctor` and a guided first run

- **Milestone**: v0.3.0 · **Roadmap**: v0.3.0.2 · **Priority**: p2 · **Area**: onboarding · **Type**: feature · **Status**: done
- **Progress** (19 November 2026): `thme doctor [--json]` reports operating system, memory, disk, CPU threads, GPU adapters found through wgpu and the models `thme discover` finds, each marked checked or assumed, then prints a short Next list. 9 tests on a `HostFacts` struct, none depending on this machine. Not done: the Studio first-start summary.
- **Progress** (19 November 2026, later): the Studio shows the doctor summary and the discover list in a Welcome panel when no model is loaded (`src/gui/app/welcome.rs`); checks run on a background thread, Open runs the safety scan first, at most 10 rows with the total. 12 state-machine tests. The live window was not opened.
- **Problem**: People come from different backgrounds and tools. They need one command that says what works on their machine and what to do next.
- **Done means**: `thme doctor` reports operating system, memory, disk, GPU adapter found through wgpu, and the models `thme discover` finds, then prints the next command to try. The Studio shows the same summary on first start. Every line states what was checked and what was only assumed.

### [THME-3042] Bridges to the tools people already use

- **Milestone**: v0.3.0 · **Roadmap**: v0.3.0.5 · **Priority**: p2 · **Area**: onboarding · **Type**: feature · **Status**: triage
- **Progress** (19 November 2026): part 1 is built: `thme download ORG/NAME [--revision] [--out] [--file] [--dry-run]` picks the config, tokenizer and weight files from the hub listing, prints the licence first, skips pickle files unless named, verifies size and sha256 where the hub gives one, renames atomically, never prints the `HF_TOKEN`, and runs the import scan on the result. 19 tests with a fake fetcher; no real hub request was made. New crates: `ureq` 3.5.2 (MIT OR Apache-2.0) and `rustls` with `ring` (build script contains C and assembly; no OpenSSL, no aws-lc); `hf-hub` was not used. Not done: bridges to Ollama, LM Studio and llama.cpp (part 2), resume, a real download test, Windows trust-store check.
- **Problem**: Analysts, engineers and scientists start from notebooks, Ollama, LM Studio, llama.cpp or Hugging Face. THME should read what they have and write what they can use.
- **Done means**: Download from the Hugging Face hub through a permissively licensed client crate (licence and binary-size cost proposed first); export GGUF that llama.cpp and Ollama load (already tested for structure by THME-2208); a documented MCP setup for editors and agents. Each bridge lists what was tested and what was not.

### [THME-3043] Evaluate the `neutron-iso` Rust port of the anatomy kit

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.8 · **Priority**: p3 · **Area**: figures · **Type**: chore · **Status**: triage
- **Problem**: The owner ported the anatomy kit into `neutron-iso` (sibling folder `neutron-lab`). It builds and `neutron iso` renders valid SVG, but `shadow` writes HTML entities into an `.svg`, `gallery` writes HTML into a file named `.svg`, the crate has no tests, and the files carry no MIT notice for the anatomy kit. THME already has `engine::iso`, byte-compared with the kit.
- **Done means**: Either the useful renderers move into THME with tests and the notice, or the crate stays outside with its output fixed; the decision is written down.

### [THME-3044] Re-measure on Pythia-14M with the corrected code

- **Milestone**: v0.2.0 · **Roadmap**: v0.2.0.9 · **Priority**: p1 · **Area**: research · **Type**: chore · **Status**: done
- **Problem**: The earlier reports were measured before the FP16 rounding fix and have been removed. Pythia-14M (Apache-2.0, 14,067,712 parameters, 6 layers) is downloaded for local testing and loads in `thme inspect`.
- **Progress** (20 November 2026): `docs/benchmarks/pythia-14m/data/verification_report.json` (source perplexity 114.2862 on the full WikiText-2 test split, 286,878 scored tokens), `precision_sweep.json` (`--layer-scan`), `depth_probe.json`, and `search_report.json` committed; charts and dossier regenerated with `scripts/generate_dossier.py` and `scripts/generate_charts.py`. The tokenizer was read from a local `tokenizer.json`, not downloaded for this run; its hash is in the report but was not compared with the hub copy (unverified).
- **Done means**: Perplexity of the source model, the Q8_0 round trip and a precision sweep are re-run with the fixed code, committed as reports under `docs/benchmarks/` with their method, and charts regenerate from them. Results are published as they come, with the method beside them.

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

### [THME-3045] Remove GPL-2.0 derived GPU-unlock code; plan an original Rust unlock engine

- **Milestone**: v0.2.7 · **Roadmap**: none · **Priority**: p1 · **Area**: legal · **Type**: chore · **Status**: done
- **Problem**: `crates/neutron-gpu` was ported from an upstream GPU-unlock project whose kernel module and packaging are GPL-2.0-only (its README says so; the firmware part has its own notice). GPL-2.0-only is incompatible with this repository's AGPL-3.0-only and with the planned commercial licence, and a port is a derivative work that would itself have to be GPL-2.0 with attribution. THME shipped it under AGPL with no attribution, so it is removed. The unlock engine, ROP payload synthesis and register-offset data are gone.
- **Done means**: `crates/neutron-gpu/src/` contains only PCI/vGPU discovery and a plain hardware recipe table; the unlock engine, ROP synthesis and register-offset data are gone. A future ticket designs an original, tested, Rust-native unlock engine from scratch, not derived from any GPL or otherwise incompatibly licensed source.

### [THME-3046] Original Rust GPU-unlock engine (placeholder)

- **Milestone**: v0.2.7 · **Roadmap**: none · **Priority**: p2 · **Area**: security · **Type**: feature · **Status**: triage
- **Problem**: Placeholder only. The owner wants THME to write its own GPU-unlock engine in Rust as original work, after the GPL-2.0 derived code is removed (THME-3045); it must be written clean-room: from public vendor documentation, by someone who has not read the GPL source.
- **Done means**: Not scoped yet; a later ticket defines the design, tests and review.

<!-- backlog:end -->

---

## Archive

Tickets below are the original sprint ledger. Sprints 1 to 9 are completed or superseded by the
active backlog above; Sprints 10 to 13 describe work deferred to v0.3 and later (see ROADMAP.md).

### [THME-3055] VRAM and adapter facts in `HostFacts`

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p1 · **Area**: doctor · **Type**: feature · **Status**: triage
- **Problem**: only an adapter name is reported.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: `Fact<u64>` dedicated VRAM total/free per adapter, device type (discrete, integrated, CPU), shared-budget field, physical core count; unit tests with fake data; JSON schema updated; unknown stays `Unknown`.

### [THME-3056] Memory model for a run

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p1 · **Area**: inference · **Type**: feature · **Status**: triage
- **Problem**: placement counts weights only.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: pure function from header + context + KV type to weights, KV and scratch bytes; table test against hand-computed LLaMA and GPT-NeoX shapes; no model read.
- **Progress** (27 November 2026): `src/profiling/run_memory.rs` implements and tests the function (hand-computed literals checked independently). Not yet read from a real header, and the outside-engine reference audit above was not needed because the arithmetic is clean-room; status stays triage until `RunPlan` (THME-3057) uses it.

### [THME-3057] `RunPlan` type and decision order, CPU-only

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p1 · **Area**: cli · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: `thme pathfinder --model X` prints the plan for a header, with basis labels, using only `predicted`; golden-file test for a mock model (`examples/generate_mock_model.rs`).
- **Progress** (27 November 2026): `thme pathfinder <MODEL_PATH> [--budget-gb] [--context] [--kv] [--json]` and `profiling::run_plan` exist, every decision labelled `predicted`; golden test `tests/test_pathfinder.rs` with `tests/golden/pathfinder_mock.json` (numbers checked by hand: KV 8,388,608, scratch 143,360, weights 24,252,928 for the mock LLaMA). The man page and manual now list every subcommand, generated from `thme --help` (27 November 2026). Not done: the plan does not drive a real run.

### [THME-3058] Shrink loop

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p1 · **Area**: inference · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: property test: for any budget the result fits or errors; context never below the floor except by explicit user value; fixed flags unchanged.
- **Progress** (27 November 2026): `src/profiling/shrink.rs` implements the order KV type, then context, then weights, with hand-worked literal tests and a 500-case deterministic property test (fits or errors, context bounds, fixed flags unchanged). Pure function over `run_memory`; not yet used by a CLI command (needs THME-3057).

### [THME-3059] Roofline decode estimate per device

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p2 · **Area**: profiling · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: predicted decode tok/s from bandwidth and bytes per token for CPU and GPU, GPU dispatch-overhead term; unit tests; output labelled predicted; no claim in docs.

### [THME-3060] CPU probe

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p1 · **Area**: inference · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: threads sweep on synthetic matvec under 1 s; deterministic test using an injected clock; result struct with raw samples.

### [THME-3061] GPU probe and CPU-wins rule

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p1 · **Area**: gpu · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: probe uses the real dispatch pattern, 2 s timeout, returns an error instead of hanging; with a fake backend the plan picks CPU when the GPU is slower; on a real machine, a report under `docs/benchmarks/` records probe vs real tokens/s (pending hardware run; mark pending, no LLM run by agents).

### [THME-3062] Hybrid placement through `epiph::placement`

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: plan feeds budget after KV, scratch and margin; partial offload accepted only when the probe says it beats CPU.

### [THME-3063] Plan cache

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: key and TTL tests, corrupt file ignored, `--no-cache`, `--reprobe`.

### [THME-3064] Wire `DeviceArg::Auto` and Studio selector to the plan

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p1 · **Area**: inference · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: Auto no longer means "GPU if init works"; Studio shows the recommended device and reason; test with a stubbed plan.

### [THME-3065] `--verify` predicted vs measured report

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: JSON report with error percentages per field; schema in `docs/schemas/`; becomes the source for any published number.

### [THME-3066] Windows shared-memory guard

- **Milestone**: v0.3.0 · **Roadmap**: none · **Priority**: p2 · **Area**: inference · **Type**: feature · **Status**: triage
- **Problem**: Models are loaded with fixed settings; the run planner (docs/research/engine-autotuning.md, section 5) needs this piece.
- **Before building**: reference audit of the outside engines named in docs/research/engine-autotuning.md (licence, clean-room or integrate).
- **Done means**: plan counts dedicated VRAM only; warns when the driver reports shared memory spill; documented in the manual.

## Sprint 1: Format Correctness & Fuzzing (Milestone v0.1.2) [COMPLETED]

### [THME-101] Full GGUF v2/v3 Quantization Type Decoder (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::parser::gguf`
- **Problem**: Current GGUF reader detects quant type IDs but lacks full block header decoders for `Q4_K_M`, `Q5_K_S`, `Q8_0`, and `IQ4_XS` super-blocks.
- **Acceptance Criteria**:
  - Implement zero-allocation block structures for all standard llama.cpp quant types.
  - Compute exact per-block scale and min values during weight extraction.
  - Validate against standard test fixtures without runtime allocations.
- **Done Means**: `cargo test test_gguf_all_quant_types` passes for synthetic headers representing each GGUF type.

### [THME-102] Truncated File & Malformed Header Recovery (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::parser`
- **Problem**: Corrupted downloads or partial transfers can cause premature parser abortion without actionable diagnostic contexts.
- **Acceptance Criteria**:
  - Parser returns structured `THMEError::CorruptedHeader` with byte offset and expected vs actual size.
  - Implement partial header recovery to salvage intact layer metadata from damaged files.
  - Never panic under any slice bounds.
- **Done Means**: Property test feeding 100,000 random byte mutations to SafeTensors and GGUF parsers yields 0 panics.

### [THME-103] Workspace Crate Modularization (`thme-core`, `thme-cli`, `thme-gui`) (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: Cargo Workspace
- **Problem**: Monolithic crate structure couples lightweight headless algorithms with heavy `egui`/`eframe` dependencies.
- **Acceptance Criteria**:
  - Extract pure IR, parsers, and math algorithms into `crates/thme-core`.
  - Extract terminal commands into `crates/thme-cli`.
  - Retain desktop studio in `crates/thme-gui`.
- **Done Means**: `cargo check -p thme-core --no-default-features` builds in < 5 seconds without GUI dependencies.

---

## Sprint 2: Real Telemetry & SIMD Vector Kernels (Milestone v0.1.3) [COMPLETED]

### [THME-201] AVX2 / AVX-512 / ARM NEON Walsh-Hadamard Transform (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::transform::hadamard`
- **Problem**: Scalar FWHT (Fast Walsh-Hadamard Transform) is single-threaded and CPU-bound for large hidden dimensions ($D=4096$).
- **Acceptance Criteria**:
  - Implement vector butterfly stages using portable SIMD or explicit AVX2 / NEON intrinsics.
  - Fall back gracefully to scalar implementation on non-AVX systems.
  - Ensure zero numerical deviation from scalar baseline ($|y_{\text{simd}} - y_{\text{scalar}}| < 10^{-6}$).
- **Done Means**: Benchmark shows $\ge 4.5\times$ speedup on 4096-dim vectors with AVX2 enabled.

### [THME-202] Criterion Micro-Benchmark Suite in CI (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `benches/`
- **Problem**: Telemetry figures currently rely on simulator estimates rather than empirical hardware benchmarks.
- **Acceptance Criteria**:
  - Add `benches/bench_matrix_ops.rs` measuring FWHT, SVD power iteration, and BitNet discretization.
  - Add `benches/bench_token_throughput.rs` measuring single-token forward simulation latency.
  - Integrate criterion reports into CI artifacts.
- **Done Means**: `cargo bench` executes in CI and generates automated regression comparison markdown.

---

## Sprint 3: Empirical Quality Verification (Milestone v0.1.4) [COMPLETED]

### [THME-301] External Perplexity Evaluation Bridge (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::quality::evaluator`
- **Problem**: Quality gates require ground-truth perplexity ($PPL$) evaluation on benchmark corpora (WikiText-2, C4).
- **Acceptance Criteria**:
  - Implement IPC/subprocess bridge to `llama.cpp/perplexity` or standalone inference runner.
  - Stream progress and token loss in real-time to GUI Quality tab.
  - Calculate empirical degradation $\Delta PPL = PPL_{\text{opt}} - PPL_{\text{base}}$.
- **Done Means**: `thme validate --model <path> --test-set wikitext2` runs end-to-end and outputs structured pass/fail gate verdict.

### [THME-302] Automated Anti-Degradation Quality Gate (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::quality::gate`
- **Problem**: Models must automatically reject optimizations that exceed allowable perplexity degradation ($\epsilon = 2.0\%$).
- **Acceptance Criteria**:
  - Enforce rollback if candidate exceeds $\epsilon$.
  - Output Pareto frontier curve comparing storage reduction vs $PPL$ impact.
- **Done Means**: Quality gate test triggers automatic rollback when degradation threshold is violated.

---

## Sprint 4: Surgery Daemon, IDE Extensions & PyTorch Bridge (Milestone v0.1.5) [COMPLETED]

### [THME-401] Model Context Protocol (MCP) Server (`thme mcp`) (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::cli::mcp`
- **Problem**: Autonomous agents, IDEs, and Teacher LLMs need standard JSON-RPC tools to inspect model topology and apply surgery.
- **Acceptance Criteria**:
  - Standard JSON-RPC 2024-11-05 protocol implementation over stdin/stdout.
  - Expose 8 specialized tools: inspect model, autodecide, apply surgery, validate quality, simulate token flow, run experiment, micro-train, convert paradigm.
- **Done Means**: `cargo test --test test_mcp` verifies tool discovery, schema generation, and execution.

### [THME-402] VS Code Extension & GitHub Copilot Integration (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `editors/vscode`
- **Problem**: Developers need native IDE access to model topologies, surgery recipes, and Copilot tools.
- **Acceptance Criteria**:
  - Packaged extension (`thme-vscode-0.1.5.vsix`) with 4 explorer sidebars.
  - GitHub Copilot Language Model Tools integration (`thme_copilot_inspect`, `thme_copilot_surgery`, `thme_copilot_validate`).
- **Done Means**: Extension packages cleanly with zero warnings and installs in VS Code.

### [THME-403] PyTorch & CUDA Ecosystem Interoperability Bridge (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::interop`
- **Problem**: Practitioners requiring traditional fine-tuning need seamless migration to and from PyTorch/CUDA environments.
- **Acceptance Criteria**:
  - Pure Rust generator for `train_cuda.py` supporting LoRA, QLoRA (BitsAndBytes 4-bit), Unsloth Fast, and DeepSpeed ZeRO-2/3.
  - Hugging Face `config.json` generator for major architectures (LLaMA, Mistral, Gemma, Qwen, DeepSeek).
  - PEFT LoRA adapter zero-copy weight merger ($W = W_0 + \frac{\alpha}{r} B \times A$).
- **Done Means**: `cargo test --test test_interop` passes for script generation, configs, and LoRA math.

### [THME-404] Universal 8 Specialized AI Model Paradigm Converter (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::ir::paradigm`, `thme::cli::convert`
- **Problem**: Ingesting and cross-converting diverse model architectures (LLM, LCM, LAM, MoE, VLM, SLM, MLM, SAM) requires universal tensor manifold transformations.
- **Acceptance Criteria**:
  - Implement heuristic and structural detection across all 8 specialized AI paradigms.
  - Universal cross-conversion matrix with sparse upcycling (MoE), BitNet b1.58 ternary discretization (SLM), vision tower augmentation (VLM/SAM), neuro-symbolic intent routing (LAM), latent concept space projection (LCM), bidirectional masking (MLM), and dense consolidation (LLM).
  - CLI `thme convert` command with full parameter customization.
  - Desktop Studio GUI converter integrated in Tab 1 (Models Manager).
  - Declarative surgery recipe support (`TweakOperation::ConvertParadigm`) and MCP tool `thme_convert_paradigm`.
- **Done Means**: `cargo test --test test_model_paradigm_conversion` passes for detection, recipe surgery, and all 8 model round-trip conversions.

---

## Sprint 5: Speculative Decoding & Draft Synthesizer (Milestone v0.1.6) [COMPLETED]

### [THME-501] Draft Model Manifold Synthesizer (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::speculative::synthesizer`
- **Problem**: Generating tokens with multi-billion parameter models autoregressively is bottlenecked by memory bandwidth; smaller draft models are needed to propose candidate tokens.
- **Acceptance Criteria**:
  - Extract lightweight prefix draft sub-manifolds (e.g. 2 to 4 layers) from parent target models.
  - Support dimension downscaling and BitNet b1.58 ternary discretization for low-latency CPU/edge execution.
  - Calculate empirical parameter reductions, memory savings, and latency cost ratios $r = \frac{T_{\text{draft}}}{T_{\text{target}}}$.
- **Done Means**: `DraftSynthesizer::extract_draft_model` generates valid `ModelIR` sub-networks with matching vocabularies and heads.

### [THME-502] Multi-Token Tree Speculation Engine (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::speculative::engine`, `thme::speculative::tree`
- **Problem**: Validating speculative candidate sequences requires mathematically sound rejection sampling and multi-branch tree attention.
- **Acceptance Criteria**:
  - Implement speculative rejection sampling with temperature scaling.
  - Compute acceptance rate $\alpha_{\text{spec}}$, emitted tokens per step $\mathbb{E}[k] + 1$, and Leviathan effective wall-clock speedup $S = \frac{\mathbb{E}[k]+1}{1 + \gamma \cdot r}$.
  - Multi-candidate tree topology synthesis with causal 2D tree attention mask generation.
  - CLI `thme speculate` command and MCP tool `thme_simulate_speculative_decoding`.
  - Desktop Studio GUI Tab 4 Speculative Deck with live stat gauges and simulation runner.
- **Done Means**: `cargo test --test test_speculative_decoding` passes 5/5 tests and verifies effective speedups $> 1.5\times$.

---

## Sprint 6: Hardware-Calibrated Roofline & Micro-Kernel Profiler (Milestone v0.1.7) [COMPLETED]

### [THME-601] Hardware Target Specifications & Machine Balance Database (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::profiling::hardware` (`src/profiling/hardware.rs`)
- **Problem**: Calculating Roofline boundaries requires calibrated peak compute (TFLOP/s) and peak memory bandwidth (GB/s) parameters across heterogeneous hardware platforms (NVIDIA, Apple Silicon, AMD, Intel).
- **Acceptance Criteria**:
  - Implement predefined hardware specifications: NVIDIA H100 SXM5 (989 TFLOP/s FP16, 3.35 TB/s HBM3), NVIDIA RTX 4090 (82.6 TFLOP/s FP16, 1.01 TB/s GDDR6X), Apple M3 Max (31.2 TFLOP/s FP16, 400 GB/s Unified), Apple M2 Ultra (54.4 TFLOP/s FP16, 800 GB/s Unified), AMD Instinct MI300X (1307 TFLOP/s FP16, 5.3 TB/s HBM3), and Intel Xeon Platinum 8490H (5.0 TFLOP/s AVX-512, 307 GB/s DDR5).
  - Compute machine balance knee $I^* = \frac{\text{Peak TFLOP/s}}{\text{Peak Bandwidth GB/s}}$ [FLOPs/byte] for each device.
  - Support external JSON hardware specification loading via CLI flag `--hardware-config <path>`.
- **Production Pipeline / CI/CD Impact**: Establishes authoritative reference ceilings for automated benchmark regression gates and multi-hardware CI target profiling.
- **Done Means**: `cargo test test_hardware_database` passes and validates specs, units, and knee values across all 6 predefined hardware profiles.

### [THME-602] Linear Projection & Attention Operational Arithmetic Intensity Profiler (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::profiling::intensity` (`src/profiling/intensity.rs`)
- **Problem**: Determining whether attention projections and matrix multiplications are memory-bandwidth-bound or compute-bound requires tracking exact FLOPs and byte transfers across decode and prefill phases.
- **Acceptance Criteria**:
  - Compute FLOP counts and memory traffic for linear projections ($W_q, W_k, W_v, W_o$) based on tensor shapes, sequence lengths, and precision types.
  - Compute Scaled Dot-Product Attention FLOPs ($2 B H S^2 D_{\text{head}}$) and quadratic activation transfer volumes.
  - Distinguish $B=1$ autoregressive decode traffic (memory-bound, $I \approx 1.0$) from $B \ge 64$ prompt prefill traffic (compute-bound, $I \gg I^*$).
  - Model KV cache memory traffic across context window lengths $S \in [1, 32768]$.
- **Production Pipeline / CI/CD Impact**: Feeds layer classification into the Pareto optimizer to prevent over-optimizing compute-bound layers while neglecting memory-bound bottlenecks.
- **Done Means**: `cargo test test_linear_attention_intensity` passes, confirming exact FLOP and byte calculations for attention layers of LLaMA and Mistral models.

### [THME-603] SwiGLU FFN & FlashAttention-2 Arithmetic Intensity Mathematical Models (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::profiling::intensity` (`src/profiling/intensity.rs`)
- **Problem**: Modern LLMs rely heavily on SwiGLU feed-forward networks (accounting for ~65% of total parameters) and tiled FlashAttention-2, which require specialized non-linear intensity models.
- **Acceptance Criteria**:
  - Model SwiGLU FFN FLOP count ($6 B D D_{\text{ffn}}$) and memory traffic ($3 D D_{\text{ffn}} b_w + B(2D + 2D_{\text{ffn}}) b_{\text{act}}$).
  - Model elementwise SiLU activation and Hadamard product arithmetic intensity.
  - Model FlashAttention-2 SRAM tiling memory reduction factor ($\mathcal{O}(B H S^2 d^2 / M_{\text{SRAM}})$) compared to standard attention.
- **Production Pipeline / CI/CD Impact**: Informs precision pruning suggestions for feed-forward blocks in automated surgery recipes and benchmark reports.
- **Done Means**: `cargo test test_swiglu_flashattn_intensity` passes, verifying correct intensity curves for SwiGLU FFN across various hidden dimensions.

### [THME-604] 4-Tier Memory Hierarchy Cache Simulator (L1/L2/L3/DRAM) (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::profiling::cache_sim` (`src/profiling/cache_sim.rs`)
- **Problem**: Transformer layers exhibit distinct tensor working set sizes that fit into or spill out of on-chip caches, introducing multi-cycle latency penalties not captured by simple DRAM bandwidth equations.
- **Acceptance Criteria**:
  - Model 4 memory hierarchy tiers: L1 Data (128 KB, 1 cycle / 0.5 ns), L2 Cache (4-16 MB, 15 cycles / 4 ns), L3 / System-Level Cache (32-96 MB, 50 cycles / 15 ns), and Main Memory / VRAM (HBM3/GDDR6/DDR5, 200 cycles / 60 ns).
  - Simulate hit/miss latency penalties: $T = \sum_{k} (M_k \cdot \text{Latency}_k + \frac{\text{Volume}_k}{\text{Bandwidth}_k})$.
  - Track active tensor footprint across layer transitions to determine residency.
- **Production Pipeline / CI/CD Impact**: Provides synthetic cache hit/miss latency projections in headless CI environments without requiring physical hardware performance counters.
- **Done Means**: `cargo test test_cache_simulator_hierarchy` passes and demonstrates stepped latency degradation when tensor footprints exceed L1, L2, and L3 boundaries.

### [THME-605] Working Set Size & Cache Eviction Latency Degradation Model (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::profiling::cache_sim` (`src/profiling/cache_sim.rs`)
- **Problem**: Transformer layers with large intermediate activations cause cache thrashing, evicting KV cache pages and model weights, leading to unpredictable latency spikes.
- **Acceptance Criteria**:
  - Calculate working set size $W_{\text{active}} = \text{size}(W_{\text{layer}}) + \text{size}(\text{Activations}) + \text{size}(\text{KV}_{\text{page}})$.
  - Estimate cache line eviction probability using LRU / pseudo-LRU working set simulation.
  - Compute effective memory latency penalty scaling factor based on predicted eviction rates.
- **Production Pipeline / CI/CD Impact**: Enables surgery recipes to identify layers that can fit completely into L2/L3 cache upon BitNet b1.58 ternary quantization.
- **Done Means**: `cargo test test_working_set_eviction_model` validates that quantizing a layer below L2 cache capacity reduces simulated memory stalls by $\ge 60\%$.

### [THME-606] Empirical Roofline Boundary Curve & Memory/Compute Classifier (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::profiling::roofline` (`src/profiling/roofline.rs`)
- **Problem**: Practitioners need automated, mathematical classification of each model layer to determine whether precision reduction or kernel fusion is the appropriate optimization.
- **Acceptance Criteria**:
  - Compute attainable performance ceiling: $P(I) = \min(P_{\text{peak}}, I \times B_{\text{peak}})$.
  - Classify layers as `MemoryBound` ($I < I^*$) or `ComputeBound` ($I \ge I^*$).
  - Output efficiency metric $\eta = \frac{P_{\text{observed}}}{P(I)}$ and compute distance to knee $|I - I^*|$.
  - Emit structured optimization recommendations: suggest low-bit quantization for `MemoryBound` and kernel fusion/tensor core acceleration for `ComputeBound`.
- **Production Pipeline / CI/CD Impact**: Enforces automated profiling quality gates, preventing optimizations that target the wrong hardware bottleneck.
- **Done Means**: `cargo test test_roofline_classifier` passes, correctly labeling decode vs prefill across all 32 layers of LLaMA-8B.

### [THME-607] Dynamic TDP Power & Thermal Throttling Projection Engine (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::profiling::thermal` (`src/profiling/thermal.rs`)
- **Problem**: Sustained compute-bound operations trigger thermal throttling and TDP power clamping, degrading sustained token throughput compared to burst performance.
- **Acceptance Criteria**:
  - Model dynamic power dissipation $P_{\text{dyn}} = C V^2 f \alpha + P_{\text{static}}$.
  - Project thermal accumulation over time: $T(t) = T_{\text{ambient}} + (T_{\max} - T_{\text{ambient}})(1 - e^{-t/\tau_{\text{thermal}}})$.
  - Predict thermal throttle frequency scaling factor $\beta_{\text{throttle}} \in [0.6, 1.0]$ when $T(t) > T_{\text{crit}}$.
  - Compute sustained vs peak token throughput projections.
- **Production Pipeline / CI/CD Impact**: Integrates thermal and energy constraints into mobile and edge deployment verification benchmarks.
- **Done Means**: `cargo test test_thermal_throttling_model` passes, verifying simulated frequency back-off under sustained high-TDP workloads.

### [THME-608] CLI Subcommand `thme roofline` & MCP Tool `thme_profile_roofline` (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::cli::roofline` (`src/cli/roofline.rs`), `thme::cli::mcp` (`src/cli/mcp.rs`)
- **Problem**: Terminal users and autonomous MCP agents need seamless programmatic access to Roofline profiling and JSON reports.
- **Acceptance Criteria**:
  - CLI command `thme roofline --model <path> --hardware <device> --batch-size <B> --seq-len <S>`.
  - Generate ASCII log-log Roofline plot with plotted layer markers and hardware ceiling lines.
  - Support `--json` flag outputting complete telemetry schema (layer coordinates, classifications, recommendations).
  - Implement MCP tool `thme_profile_roofline` for IDE and agent interaction.
- **Production Pipeline / CI/CD Impact**: Enables CI pipelines to capture Roofline telemetry artifacts and agents to inspect model bottlenecks via JSON-RPC.
- **Done Means**: `thme roofline --help` executes cleanly, and unit tests in `tests/test_roofline_profiler.rs` verify CLI and MCP outputs.

### [THME-609] Desktop Studio Tab 4 Interactive SVG/egui Roofline Deck (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::gui::app` (`src/gui/app.rs`)
- **Problem**: Visual studio users require an interactive, real-time graphical representation of the Roofline model with dynamic parameter manipulation.
- **Acceptance Criteria**:
  - Integrate interactive Roofline deck in Tab 4 (Benchmarks) using `egui` native rendering.
  - Provide interactive dropdowns for predefined hardware profiles and custom JSON configs.
  - Live sliders for Batch Size ($B = 1 \dots 128$) and Context Length ($S = 128 \dots 32768$) dynamically shifting layer markers.
  - Hover tooltips displaying exact FLOPs, memory transfer bytes, arithmetic intensity, and optimization advice per layer.
- **Production Pipeline / CI/CD Impact**: Provides graphical validation deck for interactive release demos and user research workflows.
- **Done Means**: GUI builds with zero clippy warnings, runs at 60 FPS, and renders the Roofline plot with responsive hover tooltips.

---

## Sprint 7: Binary Quantization Exporter (Milestone v0.1.8) [COMPLETED]

### [THME-701] Vectorized BitNet b1.58 Ternary Bitstream Packer (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::exporter::bitnet_pack` (`src/exporter/bitnet_pack.rs`)
- **Problem**: Storing 1.58-bit ternary weights ($\{-1, 0, +1\}$) requires dense 2-bit packing (4 weights per byte) with SIMD acceleration to minimize export latency.
- **Acceptance Criteria**:
  - Implement 2-bit encoding: $-1 \mapsto 00_2$, $0 \mapsto 01_2$, $+1 \mapsto 10_2$ ($11_2$ reserved).
  - Vectorized packing kernel compressing slices of `i8` or `f32` ternary states into contiguous `u8` buffers.
  - Support AVX2 / NEON / portable SIMD intrinsics with scalar fallback.
  - Achieve $\ge 8.0\times$ storage reduction compared to FP16 baselines.
- **Production Pipeline / CI/CD Impact**: Core serialization kernel ensuring fast export times in CI packaging jobs.
- **Done Means**: `cargo test test_bitnet_vectorized_packer` passes, verifying exact bitwise round-trip packing fidelity.

### [THME-702] Vectorized Ternary Bitstream Unpacker with FP16 Scale Product (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::exporter::bitnet_pack` (`src/exporter/bitnet_pack.rs`)
- **Problem**: Export verification and downstream decoding require high-speed unpacking of 2-bit representations back into floating-point tensors with scale restoration.
- **Acceptance Criteria**:
  - Zero-allocation unpacking routine converting packed `u8` bytes into `f32`/`f16` activation buffers.
  - Vectorized fused unpack-and-multiply applying per-channel scale factors $\alpha_i$ on-the-fly.
  - Guarantee zero loss of numerical accuracy compared to pre-packed ternary values.
- **Production Pipeline / CI/CD Impact**: Used by CI verification harnesses to validate exported artifacts before deployment.
- **Done Means**: `cargo test test_ternary_unpacker_fidelity` passes across 100,000 randomized weight matrices.

### [THME-703] Direct Pure Rust GGUF v3 Binary Writer & 32-Byte Alignment (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::exporter::gguf_writer` (`src/exporter/gguf_writer.rs`)
- **Problem**: Deploying models to `llama.cpp` and Ollama requires writing compliant GGUF v3 binary files from pure Rust without external runtime dependencies.
- **Acceptance Criteria**:
  - Serialize GGUF header: magic `GGUF` (`0x46554747`), version 3, tensor count, and key-value metadata count.
  - Write string, array, and numeric metadata records following GGUF v3 specification.
  - Serialize tensor info table (name, dimensions, GGML type, data offset).
  - Enforce strict 32-byte alignment padding before tensor data payloads.
- **Production Pipeline / CI/CD Impact**: Enables THME to produce first-class deployable GGUF files in CI release workflows.
- **Done Means**: `cargo test test_gguf_v3_writer` passes, and output files pass GGUF header parser validation.

### [THME-704] GGUF Metadata Key-Value Reconstitution & Synchronization Engine (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::exporter::metadata` (`src/exporter/metadata.rs`)
- **Problem**: Restructuring layers or modifying precision alters model architecture metadata, which must be synchronized into the GGUF header to prevent downstream runtime crashes.
- **Acceptance Criteria**:
  - Automatically update `general.architecture`, `llama.block_count`, `llama.embedding_length`, `llama.feed_forward_length`, and `llama.attention.head_count`.
  - Inject tokenizer vocabulary metadata, BOS/EOS token IDs, and RoPE frequency scaling parameters.
  - Reconcile parameter changes resulting from layer pruning or MoE sparse upcycling.
- **Production Pipeline / CI/CD Impact**: Prevents metadata drift and corrupt model configurations in published model checkpoints.
- **Done Means**: `cargo test test_gguf_metadata_sync` passes, confirming metadata integrity across all 8 paradigm architectures.

### [THME-705] Direct Pure Rust SafeTensors Binary Serializer & 8-Byte Prefix (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::exporter::safetensors_writer` (`src/exporter/safetensors_writer.rs`)
- **Problem**: Exporting to PyTorch, Hugging Face `transformers`, and vLLM requires writing compliant SafeTensors binary archives.
- **Acceptance Criteria**:
  - Write 8-byte little-endian unsigned integer indicating JSON header length.
  - Construct formatted, sorted JSON metadata map with tensor data offsets, shapes, and dtypes (`F32`, `F16`, `BF16`, `I8`).
  - Serialize binary tensor payloads aligned to 8-byte boundaries.
  - Optional `__metadata__` block embedding THME surgery recipe provenance and git commit hash.
- **Production Pipeline / CI/CD Impact**: Supplies standard model weights directly to Hugging Face Hub upload pipelines.
- **Done Means**: `cargo test test_safetensors_serializer` passes, and exported files load successfully via the standard `safetensors` crate.

### [THME-706] Low-Memory Streaming Chunked File Writer Engine (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::exporter::streaming` (`src/exporter/streaming.rs`)
- **Problem**: Exporting multi-billion parameter models (e.g. 8B to 70B) in one in-memory buffer causes Out-Of-Memory (OOM) crashes on developer workstations.
- **Acceptance Criteria**:
  - Implement buffered chunked streaming writer operating with bounded $\le 64 \text{ MB}$ RAM footprint.
  - Quantize, pack, and stream tensors sequentially layer-by-layer directly to disk.
  - Maintain backpressure and asynchronous chunk flushing.
  - Compute running SHA-256 checksum during streaming.
- **Production Pipeline / CI/CD Impact**: Allows resource-constrained CI runners (7 GB RAM) to export large model weights reliably.
- **Done Means**: `cargo test test_streaming_chunked_writer` exports an 8B synthetic model without exceeding 64 MB of resident memory.

### [THME-707] Calibrated Per-Channel & Per-Group MSE-Optimal Scale Search (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::exporter::calibration` (`src/exporter/calibration.rs`)
- **Problem**: Standard absmean scale factors ($\alpha = \frac{1}{N}\|W\|_1$) can cause significant quantization error on outlier distributions; MSE optimization is needed.
- **Acceptance Criteria**:
  - Implement grid search / Golden Section search finding $\alpha^* = \arg\min_\alpha \|W - \alpha \tilde{W}\|_F^2$.
  - Support row-wise, column-wise, and group-wise ($G = 128$) scale factor calibration.
  - Fall back to analytical absmean when matrix variance is uniform.
- **Production Pipeline / CI/CD Impact**: Improves representation fidelity of exported 1.58-bit models, reducing perplexity penalty by up to 0.35 points.
- **Done Means**: `cargo test test_mse_scale_calibration` passes, demonstrating lower Frobenius reconstruction error than baseline absmean.

### [THME-708] CLI Subcommand `thme export` & Desktop Studio Tab 1 Exporter Wizard (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `thme::cli::export` (`src/cli/export.rs`), `thme::gui::app` (`src/gui/app.rs`)
- **Problem**: Users need intuitive CLI and graphical controls to export models with chosen formats, bit-widths, and destination paths.
- **Acceptance Criteria**:
  - CLI command `thme export --input <model> --output <path> --format <gguf|safetensors> --quant <bitnet-b1.58|q8_0|fp16>`.
  - CLI progress bar showing per-layer quantization and disk write throughput (MB/s).
  - Desktop Studio Tab 1 Export Wizard modal with target format selectors, scale options, and SHA-256 digest display.
- **Production Pipeline / CI/CD Impact**: Standard entry point for model deployment automation in production CI/CD workflows.
- **Done Means**: `thme export --help` displays clean options, and unit tests verify CLI argument parsing and GUI modal state.

### [THME-709] Downstream Runtime Compatibility Testbed (Done)

- **Status**: done (v0.1 deliverable; code landed, see the evidence-level table in ROADMAP.md for what it can be relied on for)
- **Component**: `tests/test_exporter_compatibility.rs`
- **Problem**: Exported binary artifacts must be validated against downstream runtimes (`llama.cpp`, Ollama, vLLM) to prevent shipping unparseable models.
- **Acceptance Criteria**:
  - Integration test fixture loading exported GGUF files with external or mock `llama.cpp` parser.
  - Verify tensor names, dimensions, and type tags match standard architecture conventions.
  - Verify forward pass numerical stability (zero NaN or Inf values).
- **Production Pipeline / CI/CD Impact**: Serves as the mandatory release quality gate for all quantization exporter updates.
- **Done Means**: `cargo test --test test_exporter_compatibility` passes, verifying 100% format compliance across test fixtures.

---

## Sprint 8: Closed-Loop Teacher-Student Co-Surgery (Milestone v0.1.9) [COMPLETED]

### [THME-801] Co-Surgery Closed-Loop State Machine & Controller [COMPLETED]

- **Status**: completed
- **Component**: `thme::surgery::closed_loop` (`src/surgery/closed_loop.rs`)
- **Problem**: Manually tuning heterogeneous layer precision is labor-intensive; an autonomous closed-loop state machine is needed to orchestrate iterative surgery.
- **Acceptance Criteria**:
  - State machine tracking states: `Idle` $\to$ `ProfileSensitivity` $\to$ `SynthesizeRecipe` $\to$ `ApplySurgery` $\to$ `ValidateQuality` $\to$ `EvaluatePareto` $\to$ `CommitOrRollback`.
  - Support configurable loop budgets: max iterations ($N_{\max}$), target parameter reduction (%), or target perplexity delta ($\Delta PPL_{\max}$).
  - Emit structured telemetry events at each state transition.
- **Production Pipeline / CI/CD Impact**: Enables unattended, continuous model compression runs in nightly optimization pipelines.
- **Done Means**: `cargo test test_closed_loop_state_machine` completes 10 simulated iterations and transitions to `Completed`.

### [THME-802] Teacher LLM Topology Analysis Prompt Synthesizer [COMPLETED]

- **Status**: completed
- **Component**: `thme::surgery::prompt_synth` (`src/surgery/prompt_synth.rs`)
- **Problem**: External Teacher LLMs require concise, mathematically rich context summarizing layer sensitivities, spectral energy decay, and memory bottlenecks to formulate effective surgery recipes.
- **Acceptance Criteria**:
  - Formulate structured markdown/JSON prompt payloads containing model architecture summary, top-K sensitive layers, condition numbers, and EWHA reach profiles.
  - Include declarative recipe schema specification and strict generation constraints in the prompt.
  - Keep prompt token footprint compact ($\le 2,048$ tokens).
- **Production Pipeline / CI/CD Impact**: Standardizes prompt engineering across LLM backends (Anthropic Claude, Google Gemini, OpenAI, local models).
- **Done Means**: `cargo test test_prompt_synthesizer` passes, verifying correct markdown and JSON prompt structure.

### [THME-803] Headless JSON-RPC & Subprocess Teacher Agent Communication Bridge [COMPLETED]

- **Status**: completed
- **Component**: `thme::surgery::agent_bridge` (`src/surgery/agent_bridge.rs`)
- **Problem**: The co-surgery engine needs a reliable communication channel to invoke external AI agents (via CLI subprocess, HTTP REST, or MCP JSON-RPC).
- **Acceptance Criteria**:
  - Subprocess bridge executing external CLI agents (`claude`, `gemini`, `curl`).
  - Stdin/stdout pipe handler with timeout protection and stderr logging.
  - Parse generated JSON surgery recipes with automated schema validation.
- **Production Pipeline / CI/CD Impact**: Decouples the Rust execution engine from external AI agent providers.
- **Done Means**: `cargo test test_agent_bridge` passes with mock agent returning valid surgery recipes.

### [THME-804] Multi-Objective 3D Pareto Optimizer [COMPLETED]

- **Status**: completed
- **Component**: `thme::optimization::pareto_solver` (`src/optimization/pareto_solver.rs`)
- **Problem**: Compression decisions involve trade-offs across 3 competing objectives: perplexity degradation ($\Delta PPL$), memory traffic ($W_{\text{bytes/token}}$), and wall-clock speedup ($S$).
- **Acceptance Criteria**:
  - Track 3D objective vectors $\mathbf{y} = (\Delta PPL, W_{\text{token}}, -S)$.
  - Implement Pareto dominance relation: $\mathbf{a} \prec \mathbf{b}$ iff $a_i \le b_i \ \forall i$ and $\exists j: a_j < b_j$.
  - Maintain dynamic archive of non-dominated Pareto optimal checkpoints.
- **Production Pipeline / CI/CD Impact**: Generates optimal candidate frontiers for deployment engineers to pick based on hardware constraints.
- **Done Means**: `cargo test test_3d_pareto_optimizer` discovers and maintains the non-dominated set across 50 simulated candidates.

### [THME-805] Pareto Hypervolume Indicator Convergence Metric [COMPLETED]

- **Status**: completed
- **Component**: `thme::optimization::hypervolume` (`src/optimization/hypervolume.rs`)
- **Problem**: Autonomous co-surgery needs an objective mathematical stopping criterion to determine when further iterations yield diminishing returns.
- **Acceptance Criteria**:
  - Calculate hypervolume indicator $\mathcal{S}(\mathcal{P}, \mathbf{r})$ bounded by reference point $\mathbf{r}$.
  - Compute relative hypervolume gain $\Delta \mathcal{S}_k = \frac{\mathcal{S}_k - \mathcal{S}_{k-1}}{\mathcal{S}_{k-1}}$.
  - Trigger termination when $\Delta \mathcal{S}_k < \epsilon_{\text{conv}}$ for 3 consecutive iterations.
- **Production Pipeline / CI/CD Impact**: Eliminates wasteful compute by stopping automated optimization runs once convergence is reached.
- **Done Means**: `cargo test test_hypervolume_convergence` correctly halts simulated optimization upon frontier stabilization.

### [THME-806] Automated Self-Healing Parameter Relaxation Heuristic [COMPLETED]

- **Status**: completed
- **Component**: `thme::surgery::self_healing` (`src/surgery/self_healing.rs`)
- **Problem**: Aggressive quantization proposals can breach quality tolerances; rather than discarding the entire recipe, fine-grained relaxation should recover quality.
- **Acceptance Criteria**:
  - Identify the single layer contributing highest perplexity degradation.
  - Relax that layer to higher precision (e.g. Ternary $\to$ INT8 or INT8 $\to$ FP16) while keeping other layers compressed.
  - Re-evaluate candidate quality before triggering full rollback.
- **Production Pipeline / CI/CD Impact**: Increases recipe success rate by $> 40\%$ in automated overnight optimization pipelines.
- **Done Means**: `cargo test test_self_healing_relaxation` demonstrates recovery of breached quality gate without discarding valid optimizations.

### [THME-807] Anti-Degradation Quality Gate Watchdog & Rollback Engine [COMPLETED]

- **Status**: completed
- **Component**: `thme::surgery::watchdog` (`src/surgery/watchdog.rs`)
- **Problem**: Models must never enter production in a degraded or broken state; instant rollback is mandatory when validation fails.
- **Acceptance Criteria**:
  - Watchdog monitoring validation output after every surgery step.
  - If $\Delta PPL > \epsilon_{\max}$ or NaN/Inf is detected, trigger immediate zero-cost pointer restoration to prior snapshot.
  - Record structured audit event (`SurgeryRollback`) with diagnostic failure reasons.
- **Production Pipeline / CI/CD Impact**: Critical safety gate preventing corrupt checkpoints from advancing in the release pipeline.
- **Done Means**: `cargo test test_co_surgery_watchdog_rollback` verifies immediate state restoration upon injected catastrophic degradation.

### [THME-808] CLI Subcommand `thme co-surgery` & Progress Telemetry Stream [COMPLETED]

- **Status**: completed
- **Component**: `thme::cli::co_surgery` (`src/cli/co_surgery.rs`)
- **Problem**: DevOps engineers need a terminal command to launch autonomous co-surgery runs with real-time logging and progress reporting.
- **Acceptance Criteria**:
  - CLI `thme co-surgery --model <path> --agent <cmd> --max-iterations <N> --target-reduction <%> --max-ppl-delta <float>`.
  - Live terminal progress stream: iteration counter, proposed recipe diff, validation $\Delta PPL$, Pareto status.
  - Support `--non-interactive` flag emitting JSONL events for CI log ingestion.
- **Production Pipeline / CI/CD Impact**: Primary integration command for continuous model surgery cron jobs and GitHub Actions workflows.
- **Done Means**: `thme co-surgery --help` displays correct flags, and unit tests verify command dispatch.

### [THME-809] Desktop Studio Tab 2 Closed-Loop Co-Surgery Dashboard [COMPLETED]

- **Status**: completed
- **Component**: `thme::gui::app` (`src/gui/app.rs`)
- **Problem**: Interactive users need visual oversight of autonomous co-surgery sessions with live graphs and manual override buttons.
- **Acceptance Criteria**:
  - Interactive co-surgery panel in Tab 2 (Layer Tweaker & Surgery).
  - Real-time iteration history table, loss curve plot, and 3D Pareto scatter projection.
  - Start / Pause / Stop buttons and manual intervention controls.
- **Production Pipeline / CI/CD Impact**: Visual monitoring interface for researchers and internal demonstrations.
- **Done Means**: Desktop Studio compiles cleanly and renders the Co-Surgery dashboard with responsive controls.

---

## Sprint 9: Production Core Release (Milestone v0.2.0) [SUPERSEDED by the active backlog]

### [THME-901] SmolLM-135M Open-Weights End-to-End Verification Pipeline

- **Status**: triage (legacy record, no status was written)
- **Component**: `examples/production_verification.rs`
- **Problem**: Release validation requires end-to-end proof-of-correctness on genuine, publicly available open weights.
- **Acceptance Criteria**:
  - Download and ingest genuine SmolLM-135M checkpoint (SafeTensors).
  - Execute sensitivity profiling, Pareto BitNet b1.58 surgery, and GGUF export.
  - Validate perplexity on WikiText-2, ensuring $\Delta PPL \le 0.45$.
  - Complete execution within 3 minutes on standard CI runner.
- **Production Pipeline / CI/CD Impact**: Mandatory integration test gate required before tagging v0.2.0.
- **Done Means**: `cargo test --example production_verification -- smollm-135m` passes all quality assertions.

### [THME-902] LLaMA-3.2-1B Open-Weights End-to-End Verification Pipeline

- **Status**: triage (legacy record, no status was written)
- **Component**: `examples/production_verification.rs`
- **Problem**: Verifying scalability to billion-parameter architectures requires a full pipeline run on LLaMA-3.2-1B.
- **Acceptance Criteria**:
  - Ingest LLaMA-3.2-1B checkpoint under bounded memory constraints ($\le 4 \text{ GB}$ RAM).
  - Execute layer surgery and export valid GGUF and SafeTensors weights.
  - Verify generation output coherence on sample prompts.
- **Production Pipeline / CI/CD Impact**: Validates that THME algorithms scale correctly beyond toy models to modern frontier architectures.
- **Done Means**: `cargo test --example production_verification -- llama-3.2-1b` runs to completion and verifies tensor integrity.

### [THME-903] Automated Reproducible Benchmark Script Suite

- **Status**: triage (legacy record, no status was written)
- **Component**: `scripts/benchmark_suite.py`, `scripts/benchmark_suite.sh`
- **Problem**: External researchers and users need one-command scripts to reproduce all published release benchmarks on their own hardware.
- **Acceptance Criteria**:
  - Multi-platform benchmark script (Python 3 and Bash) with zero external library dependencies.
  - Benchmarks: ingest throughput (MB/s), SVD profiling latency, Pareto allocation time, export speed, and token simulation.
  - Collect hardware telemetry (CPU model, RAM, OS, cache sizes).
- **Production Pipeline / CI/CD Impact**: Generates benchmark artifacts uploaded to GitHub Actions release assets.
- **Done Means**: Running `./scripts/benchmark_suite.sh` on clean environment completes in $< 5$ minutes and outputs `benchmark_results.json`.

### [THME-904] Published Benchmark Dossier Generator

- **Status**: triage (legacy record, no status was written)
- **Component**: `docs/benchmark_dossier.md`, `scripts/generate_dossier.py`
- **Problem**: Benchmark results must be formatted into transparent, publication-grade documentation with clear hardware attribution.
- **Acceptance Criteria**:
  - Synthesize benchmark data into Markdown and CSV tables comparing baseline vs THME-optimized models.
  - Include parameter counts, memory footprint, throughput (tok/s), WikiText-2 PPL, and EWHA energy (mJ/tok).
  - Include strict hardware attribution (NVIDIA H100, RTX 4090, Apple M3 Max, AMD MI300X, Intel Xeon).
- **Production Pipeline / CI/CD Impact**: Publishes empirical performance documentation alongside release announcements.
- **Done Means**: Generated `docs/benchmark_dossier.md` passes format check and contains complete empirical tables.

### [THME-905] Multi-Platform LTO Stripped Binary Release Packaging

- **Status**: triage (legacy record, no status was written)
- **Component**: `.github/workflows/release.yml`, `Cargo.toml`
- **Problem**: Production binaries must be optimized for size and speed across all supported operating systems and architectures.
- **Acceptance Criteria**:
  - Configure Release profile with Fat LTO (`lto = "fat"`), single codegen unit (`codegen-units = 1`), and panic abort (`panic = "abort"`).
  - Strip debug symbols: binary sizes $\le 10 \text{ MB}$ for CLI and $\le 15 \text{ MB}$ for GUI.
  - Build matrices for Linux (x86_64, aarch64), Windows (x64), and macOS (Apple Silicon, Intel).
- **Production Pipeline / CI/CD Impact**: Automates multi-platform asset compilation and packaging upon release tag push.
- **Done Means**: Release workflow builds all 5 target platform archives and validates binary footprint limits.

### [THME-906] Minisign & GPG Cryptographic Artifact Signing Pipeline

- **Status**: triage (legacy record, no status was written)
- **Component**: `.github/workflows/release.yml`, `scripts/sign_artifacts.sh`
- **Problem**: Binary releases must have cryptographic provenance to guarantee authenticity and prevent supply-chain tampering.
- **Acceptance Criteria**:
  - Sign release tarballs and executables using Minisign (`.minisig`) and GPG (`.asc`).
  - Publish public signing key in repository root (`SECURITY.md`).
  - Automated signature verification check in release workflow.
- **Production Pipeline / CI/CD Impact**: Establishes a tamper-evident software supply chain for enterprise adopters.
- **Done Means**: `minisign -Vm <archive> -p epiphanes.pub` verifies signatures with exit code 0.

### [THME-907] SHA-256 Checksum Manifest Generation

- **Status**: triage (legacy record, no status was written)
- **Component**: `.github/workflows/release.yml`
- **Problem**: Users and package managers require standard SHA-256 checksum manifests to verify download integrity.
- **Acceptance Criteria**:
  - Generate `SHA256SUMS.txt` manifest covering all release archives and standalone binaries.
  - Automated verification step verifying `sha256sum -c SHA256SUMS.txt`.
- **Production Pipeline / CI/CD Impact**: Standard release asset referenced by Homebrew, Winget, and Arch AUR package formulas.
- **Done Means**: Generated `SHA256SUMS.txt` verifies successfully across all built artifacts.

### [THME-908] Comprehensive Architecture Playbook, User Manual & Man Pages

- **Status**: triage (legacy record, no status was written)
- **Component**: `docs/manual.md`, `docs/man/thme.1`
- **Problem**: Production users and system administrators need exhaustive documentation and standard Unix man pages.
- **Acceptance Criteria**:
  - Authoritative User Manual (`docs/manual.md`) covering all CLI commands, GUI tabs, and JSON recipe schemas.
  - Standard troff Unix man page (`docs/man/thme.1`) installable to `/usr/share/man/man1`.
  - Architecture Playbook documenting internal data structures and extension points.
- **Production Pipeline / CI/CD Impact**: Verified by documentation linters in CI to prevent broken links or outdated command flags.
- **Done Means**: `man ./docs/man/thme.1` renders cleanly without groff warnings.

### [THME-909] End-to-End Clean-Machine Verification Test Harness

- **Status**: triage (legacy record, no status was written)
- **Component**: `.github/workflows/clean_machine_test.yml`
- **Problem**: Binaries must execute on clean operating system installations without hidden dynamic library dependencies or environment assumptions.
- **Acceptance Criteria**:
  - Ephemeral CI runner with clean Ubuntu, Windows, and macOS images.
  - Download release binary, execute `thme --version`, run synthetic profiling, and export model.
  - Verify zero missing shared library errors (`ldd` / Dependency Walker check).
- **Production Pipeline / CI/CD Impact**: Final quality gate certifying production readiness before public release announcement.
- **Done Means**: Clean machine CI workflow passes 100% across all 3 OS platforms.

### [THME-910] Post-Refactoring Risk Register & Technical Debt Amortization (Pre-Release Gate)

- **Status**: triage (legacy record, no status was written)
- **Component**: `src/gui/app.rs`, `examples/generate_showcase_gifs.rs`, `src/exporter/calibration.rs`
- **Problem**: Large architectural monolithic files and un-sandboxed visual generators deferred during fast prototyping create maintainability and borrow-checker fragility prior to public release.
- **Risk Assessment & Mitigation Matrix**:
  1. **Monolithic `src/gui/app.rs` (~5,376 lines)**:
     - *Risk*: Heavy state coupling across 10 tab renderers; subtle mutable borrow collisions (`&mut self`) during asynchronous telemetry updates.
     - *Mitigation*: Modularize into discrete tab components under `src/gui/tabs/{dashboard, models, tweaker, export, cosurgery}.rs` with isolated UI state sub-structs before tagged binary freeze.
  2. **Unchecked Expectations in Exporter Calibration & Test Utilities**:
     - *Risk*: `calibration.rs` contains `.expect("calibrate exact ternary")` and writer tests use panicking assertions that must never bleed into runtime parser boundaries.
     - *Mitigation*: Enforce strict `Result<T, QuantError>` propagation across all public exporter entry points; verify against `AGENTS.md` parser infallibility rules.
  3. **Orphaned Example Generator `generate_showcase_gifs.rs` (~2,085 lines)**:
     - *Risk*: Standalone GIF showcase generator is unreferenced in `Cargo.toml` and carries manual font bitmaps and non-standard image encoding loops.
     - *Mitigation*: Promote into `Cargo.toml` under `[[example]]` or transition into a documentation asset utility in `scripts/`.
     - *Resolved (v0.2.0)*: Removed with the drawn GIFs, which did not depict the application. Real screenshots now come from the Studio's capture mode (`scripts/capture_studio.ps1`).
- **Done Means**: Tab modularization verified with zero GUI borrow conflicts; zero unhandled errors in release codepaths.

---

## Sprint 10: Direct GPU Driver Execution Engine (former v0.2.1) [DEFERRED to v0.3+]

### [THME-1001] Hardware Abstraction Layer (HAL) Core Traits

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::hardware::hal` (`src/hardware/hal.rs`)
- **Problem**: Supporting multiple GPU backends (CUDA, Vulkan, Metal) requires a unified trait architecture for buffer allocation, memory copies, and kernel launches.
- **Acceptance Criteria**:
  - Define core traits: `GpuDevice`, `DeviceBuffer`, `StreamQueue`, `ComputeKernel`, and `KernelLaunchConfig`.
  - Unified host-to-device, device-to-host, and device-to-device memory copy abstractions.
  - Device capability querying (compute units, shared memory per block, warp size).
- **Production Pipeline / CI/CD Impact**: Foundational trait contract decoupling backend implementations from high-level model execution.
- **Done Means**: `cargo test test_hal_traits` passes with mock CPU backend implementing full HAL interface.

### [THME-1002] NVIDIA CUDA Driver API Backend via Pure Rust `cudarc`

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::hardware::cuda` (`src/hardware/cuda.rs`)
- **Problem**: External PyTorch runtimes introduce gigabytes of dependencies; THME requires direct driver communication via `libcuda.so` / `nvcuda.dll`.
- **Acceptance Criteria**:
  - Integrate `cudarc` for driver API initialization and context management.
  - Support JIT PTX compilation and loading pre-compiled CUBIN modules.
  - Asynchronous stream execution and CUDA event synchronization.
- **Production Pipeline / CI/CD Impact**: Enables high-performance NVIDIA GPU execution without requiring Python or PyTorch installed.
- **Done Means**: `cargo test test_cuda_driver_backend` initializes CUDA context and executes vector-add kernel on available NVIDIA GPU.

### [THME-1003] Vulkan 1.3 SPIR-V / WGSL Compute Pipeline Backend

- **Component**: `thme::inference::gpu` (`src/inference/gpu/`)
- **Problem**: Cross-platform GPU execution across Linux, Windows, and AMD/Intel/NVIDIA GPUs requires standard Vulkan compute shaders with fused quantized GEMV kernels.
- **Acceptance Criteria**:
  - Initialize Vulkan compute device via WGPU backend with Ada Lovelace / NVIDIA hardware recipe resolution via `neutron-gpu`.
  - Compile and dispatch WGSL compute shaders for GEMV and fused dequantization (`gemv_f32`, `gemv_q4_k`, `gemv_q5_k`, `gemv_iq4_xs`).
  - Manage dynamic staging buffers, parameter uniforms, and graceful CPU fallback for oversized vocab projection matrices.
- **Production Pipeline / CI/CD Impact**: Delivers zero-dependency pure-Rust GPU acceleration on host hardware with automated CPU SIMD parity verification.
- **Done Means**: `cargo test --test test_gpu_acceleration` passes 100% on host GPU with numerical parity matching CPU reference outputs within $10^{-3}$ tolerance.
- **Status**: implemented; `cargo test --test test_gpu_acceleration` (8 tests) passes on one machine with an NVIDIA GeForce RTX 4060 Ti over Vulkan, 19 November 2026. No committed report yet, and no speed figure is claimed.

### [THME-1004] Apple Metal 3 Performance Shaders Backend

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::hardware::metal` (`src/hardware/metal.rs`)
- **Problem**: Apple Silicon Macs feature high-bandwidth unified memory that requires native Metal 3 Performance Shaders to achieve zero-copy execution.
- **Acceptance Criteria**:
  - Metal device creation and command queue management via `metal-rs`.
  - Allocate shared CPU-GPU storage buffers (`MTLStorageModeShared`) with zero memory copies.
  - Dispatch custom MSL compute shaders and integrate Metal Performance Shaders (MPS) matrix kernels.
- **Production Pipeline / CI/CD Impact**: Enables native, highly efficient inference and surgery on macOS developer laptops.
- **Done Means**: `cargo test test_metal_backend` passes on macOS runner, verifying zero-copy execution.

### [THME-1005] Fused BitNet b1.58 GEMV / GEMM Kernel

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::hardware::kernels::bitnet_gemm` (`src/hardware/kernels/bitnet_gemm.rs`)
- **Problem**: Unpacking 2-bit ternary weights to global memory before GEMM introduces severe memory bandwidth overhead; unpacking must happen inside GPU registers.
- **Acceptance Criteria**:
  - CUDA, Vulkan, and Metal compute kernels unpacking 2-bit weights directly within register files.
  - Replace multiplication with conditional addition/subtraction: $y += (w == 1) ? x : ((w == -1) ? -x : 0)$.
  - Achieve $\ge 2.5\times$ speedup over un-quantized FP16 GEMV during batch size $B=1$ decode.
- **Production Pipeline / CI/CD Impact**: Core inference speedup engine for deployable BitNet models.
- **Done Means**: `cargo bench --bench bench_bitnet_gemm` demonstrates $\ge 2.5\times$ throughput improvement.

### [THME-1006] Fused RMSNorm Micro-Kernel

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::hardware::kernels::rmsnorm` (`src/hardware/kernels/rmsnorm.rs`)
- **Problem**: Separate reduction and scaling passes for RMSNorm incur unnecessary global memory round-trips.
- **Acceptance Criteria**:
  - Single-pass thread block cooperative reduction computing root mean square: $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$.
  - Fused weight scaling $y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$ writing directly to activation output.
- **Production Pipeline / CI/CD Impact**: Reduces layer transition latency across all Transformer blocks.
- **Done Means**: `cargo test test_fused_rmsnorm` passes and verifies bitwise alignment with reference implementation.

### [THME-1007] Fused SwiGLU Activation Micro-Kernel

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::hardware::kernels::swiglu` (`src/hardware/kernels/swiglu.rs`)
- **Problem**: SwiGLU requires computing $\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)$, which wastes memory bandwidth if intermediate activations are written to DRAM.
- **Acceptance Criteria**:
  - Fused elementwise kernel computing $\text{SiLU}(a) = \frac{a}{1 + e^{-a}}$ and multiplying by $b$ in local registers.
  - Zero global memory allocation for intermediate activation buffers.
- **Production Pipeline / CI/CD Impact**: Accelerates MLP execution by $\ge 35\%$ across modern LLaMA-style architectures.
- **Done Means**: `cargo test test_fused_swiglu` verifies numerical correctness within $10^{-5}$ tolerance.

### [THME-1008] Fused Rotary Position Embedding (RoPE) Kernel

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::hardware::kernels::rope` (`src/hardware/kernels/rope.rs`)
- **Problem**: Applying RoPE to $Q$ and $K$ matrices requires complex sinusoidal rotation that can bottleneck attention pre-processing.
- **Acceptance Criteria**:
  - Fused in-place 2D rotation kernel: $\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix} \mapsto \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$.
  - Support precomputed frequency tables and dynamic RoPE scaling (YaRN, Linear).
- **Production Pipeline / CI/CD Impact**: Streamlines attention projection pipeline in pure Rust driver execution.
- **Done Means**: `cargo test test_fused_rope` verifies exact rotation matching Hugging Face LLaMA reference.

### [THME-1009] CLI Subcommand `thme run` Direct Inference

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cli::run` (`src/cli/run.rs`)
- **Problem**: Users need a direct CLI command to run LLM inference using native GPU drivers without Python.
- **Acceptance Criteria**:
  - CLI `thme run --model <path> --backend <cuda|vulkan|metal|cpu> --prompt <text> --max-tokens <N>`.
  - Token generation loop with live tok/s throughput and latency metrics.
  - Automatic hardware backend detection and graceful fallback to CPU SIMD.
- **Production Pipeline / CI/CD Impact**: Primary command for standalone inference benchmarks and edge deployment.
- **Done Means**: `thme run --help` executes cleanly, and integration test runs end-to-end token generation on test fixture.

---

## Sprint 11: Tiered Out-of-Core NVMe SSD Streaming (former v0.2.2) [DEFERRED to v0.3+]

### [THME-1101] Linux Direct Asynchronous I/O Engine (`io_uring`)

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::storage::direct_io` (`src/storage/direct_io.rs`)
- **Problem**: Standard OS file reading incurs page cache buffering and context switch overhead, capping disk throughput below PCIe 4.0/5.0 line rates.
- **Acceptance Criteria**:
  - Implement `io_uring` submission and completion queue polling via `io-uring` crate.
  - Use `O_DIRECT` flag to bypass OS kernel buffer cache and read directly into user-space buffers.
  - Achieve $> 6,000 \text{ MB/s}$ sustained sequential read throughput on NVMe SSDs.
- **Production Pipeline / CI/CD Impact**: Establishes high-speed storage streaming backbone for Linux servers.
- **Done Means**: `cargo test test_io_uring_direct_io` passes on Linux runner, reading 1 GB test file with zero page cache caching.

### [THME-1102] Windows Direct Asynchronous I/O Engine (Overlapped I/O)

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::storage::direct_io` (`src/storage/direct_io.rs`)
- **Problem**: Windows developers require native unbuffered disk streaming matching Linux `io_uring` throughput.
- **Acceptance Criteria**:
  - Implement Win32 Overlapped I/O using `FILE_FLAG_NO_BUFFERING` and `FILE_FLAG_OVERLAPPED`.
  - Asynchronous completion handling via I/O Completion Ports (`IOCP`).
  - Enforce sector-aligned memory buffer allocation (`VirtualAlloc` on 4096-byte boundaries).
- **Production Pipeline / CI/CD Impact**: Enables high-speed SSD model streaming on Windows developer workstations.
- **Done Means**: `cargo test test_windows_overlapped_io` passes on Windows runner with verified direct I/O throughput.

### [THME-1103] Sequential Layer Streaming & Eviction Engine

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::storage::layer_streamer` (`src/storage/layer_streamer.rs`)
- **Problem**: Executing a 70B parameter model requires 140 GB of weights, exceeding consumer GPU VRAM; layers must be loaded and evicted sequentially.
- **Acceptance Criteria**:
  - Layer-by-layer forward execution: stream layer $k$ from disk $\to$ transfer to GPU $\to$ execute forward step $\to$ release buffer for layer $k+1$.
  - Maintain single-layer active memory ceiling in VRAM.
- **Production Pipeline / CI/CD Impact**: Enables massive model execution in memory-constrained CI environments.
- **Done Means**: `cargo test test_layer_streaming_execution` completes forward pass across 32 layers using single-layer memory budget.

### [THME-1104] Double-Buffered Prefetch Pipeline

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::storage::prefetch` (`src/storage/prefetch.rs`)
- **Problem**: Sequential loading stalls GPU compute while waiting for disk reads; I/O and computation must be overlapped.
- **Acceptance Criteria**:
  - Double-buffered queue: asynchronous disk thread prefetches layer $k+1$ while GPU compute stream executes layer $k$.
  - Synchronize streams using non-blocking barriers.
  - Eliminate $\ge 80\%$ of disk I/O latency stalls during continuous autoregressive generation.
- **Production Pipeline / CI/CD Impact**: Maximizes throughput during out-of-core model surgery and profiling.
- **Done Means**: `cargo test test_double_buffered_prefetch` demonstrates concurrent disk reading and mock layer compute.

### [THME-1105] Pinned Memory Staging Buffers (`mlock` / `VirtualLock`)

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::memory::pinned` (`src/memory/pinned.rs`)
- **Problem**: GPU DMA transfers from pageable host RAM require intermediate copies into pinned memory, reducing transfer bandwidth.
- **Acceptance Criteria**:
  - Allocate page-locked host memory buffers using `mlock` on Unix and `VirtualLock` on Windows.
  - Enable direct DMA transfers from host buffer to GPU VRAM at full PCIe bandwidth ($\sim 31.5 \text{ GB/s}$ on PCIe 4.0 x16).
- **Production Pipeline / CI/CD Impact**: Eliminates host memory copy bottlenecks during layer streaming.
- **Done Means**: `cargo test test_pinned_memory_allocation` verifies lock status and zero-copy DMA transfer capabilities.

### [THME-1106] REA-LLM Tiered Heterogeneous Memory Placement Manager

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::memory::tiered_allocator` (`src/memory/tiered_allocator.rs`)
- **Problem**: Different model layers exhibit different reach probabilities $q_l$; memory placement should be tiered strategically across VRAM, Host RAM, and NVMe SSD.
- **Acceptance Criteria**:
  - Placement policy: hot layers ($q_l > 0.8$) permanently pinned in VRAM, warm layers ($0.4 \le q_l \le 0.8$) staged in Host RAM, deep layers ($q_l < 0.4$) streamed from NVMe SSD.
  - Dynamically adapt layer placement based on runtime reach telemetry.
- **Production Pipeline / CI/CD Impact**: Optimizes execution speed under strict hardware memory constraints.
- **Done Means**: `cargo test test_tiered_memory_placement` verifies correct layer distribution across simulated 3-tier hierarchy.

### [THME-1107] Dynamic VRAM Memory Accountant

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::memory::accountant` (`src/memory/accountant.rs`)
- **Problem**: Out-of-core execution without strict VRAM accounting risks hitting hard CUDA OOM errors.
- **Acceptance Criteria**:
  - Track active VRAM allocations in real-time (model weights, KV cache, activation buffers).
  - Enforce user-defined hard ceiling (e.g. `--max-vram 8GB`), throttling prefetch queue if VRAM headroom is $< 512 \text{ MB}$.
- **Production Pipeline / CI/CD Impact**: Prevents catastrophic OOM aborts in CI and production deployments.
- **Done Means**: `cargo test test_vram_accountant_ceiling` strictly caps allocated memory under stress testing.

### [THME-1108] 70B Model in 8 GB VRAM End-to-End Testbed

- **Status**: triage (legacy record, no status was written)
- **Component**: `tests/test_out_of_core_streaming.rs`
- **Problem**: Verifying out-of-core streaming requires full multi-layer autoregressive execution on a genuine 70B parameter topology under an 8 GB memory limit.
- **Acceptance Criteria**:
  - Generate synthetic 70B parameter model topology (80 layers, $D=8192$, $D_{\text{ffn}}=28672$).
  - Execute end-to-end forward token step strictly bounded within 8 GB VRAM footprint.
  - Verify numerical consistency with standard un-streamed reference output.
- **Production Pipeline / CI/CD Impact**: Primary proof that THME enables large-model inference on consumer GPUs.
- **Done Means**: `cargo test --test test_out_of_core_streaming` passes and verifies $< 8 \text{ GB}$ peak VRAM usage.

### [THME-1109] PCIe Bandwidth Saturation & Disk I/O Telemetry Monitor

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::profiling::io_monitor` (`src/profiling/io_monitor.rs`)
- **Problem**: Operators need visibility into PCIe bus saturation, disk read queue depths, and streaming bottlenecks.
- **Acceptance Criteria**:
  - Real-time telemetry: PCIe transfer rate (GB/s), NVMe read throughput (MB/s), and I/O wait latency fraction ($f_{\text{wait}} \in [0, 1]$).
  - Expose metrics to CLI status bars and Desktop Studio Tab 4 benchmark deck.
- **Production Pipeline / CI/CD Impact**: Identifies storage and bus bottlenecks during hardware profiling.
- **Done Means**: `cargo test test_io_telemetry_monitor` validates accurate metric computation and reporting.

---

## Sprint 12: Distributed Heterogeneous Multi-Node Clustering (former v0.2.3) [DEFERRED to v0.3+]

### [THME-1201] Intra-Node Tensor Parallelism (TP) Projection Splitter

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cluster::tensor_parallel` (`src/cluster/tensor_parallel.rs`)
- **Problem**: Multi-GPU workstations require splitting Attention and MLP projection matrices across local accelerators to reduce per-device memory footprint and latency.
- **Acceptance Criteria**:
  - Column-parallel linear projections for $W_q, W_k, W_v$ and MLP $W_{\text{gate}}, W_{\text{up}}$.
  - Row-parallel linear projections for $W_o$ and MLP $W_{\text{down}}$ with local partial sum reduction.
  - Zero numerical divergence from single-device forward pass.
- **Production Pipeline / CI/CD Impact**: Enables multi-GPU scaling on single compute nodes.
- **Done Means**: `cargo test test_tensor_parallel_projections` passes, verifying exact output match across 2 and 4 simulated devices.

### [THME-1202] Inter-Node Pipeline Parallelism (PP) & Layer Partitioner

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cluster::pipeline_parallel` (`src/cluster/pipeline_parallel.rs`)
- **Problem**: Distributing multi-layer models across networked nodes requires partitioning consecutive layer blocks across heterogeneous machines.
- **Acceptance Criteria**:
  - Dynamic layer partitioning algorithm assigning layers to nodes based on compute capacity and network link speed.
  - Activation communication pipeline forwarding intermediate activation tensors between nodes.
- **Production Pipeline / CI/CD Impact**: Allows clustering mixed machines (e.g. desktop + server) to run massive models.
- **Done Means**: `cargo test test_pipeline_parallel_partitioning` splits 32 layers across 3 nodes with balanced execution.

### [THME-1203] 1F1B (One-Forward-One-Backward) Pipeline Bubble Minimizer

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cluster::schedule_1f1b` (`src/cluster/schedule_1f1b.rs`)
- **Problem**: Naive pipeline parallelism leaves nodes idle while waiting for activations (pipeline bubbles); a 1F1B schedule is needed.
- **Acceptance Criteria**:
  - Implement 1F1B micro-batch scheduling interleaving forward and backward passes.
  - Reduce pipeline bubble fraction to $\frac{p - 1}{m}$ where $p$ is pipeline depth and $m$ is micro-batch count.
- **Production Pipeline / CI/CD Impact**: Maximizes cluster accelerator utilization during distributed micro-training.
- **Done Means**: `cargo test test_1f1b_scheduler` verifies micro-batch ordering and idle time minimization.

### [THME-1204] Pure Rust Asynchronous TCP Ring All-Reduce

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cluster::ring_allreduce` (`src/cluster/ring_allreduce.rs`)
- **Problem**: Distributed communication must avoid heavy MPI or NCCL runtime dependencies, providing pure Rust network communication.
- **Acceptance Criteria**:
  - Asynchronous Ring All-Reduce over non-blocking TCP socket streams using `tokio`.
  - Scatter-Reduce and All-Gather phases transferring $2 \frac{N-1}{N} S$ bytes per node.
  - Achieve $> 85\%$ network bandwidth utilization on 10 GbE / 100 GbE connections.
- **Production Pipeline / CI/CD Impact**: Pure Rust distributed primitives running anywhere without complex cluster runtime configuration.
- **Done Means**: `cargo test test_ring_allreduce_tcp` passes across 4 simulated local nodes.

### [THME-1205] Hardware RDMA / RoCE Network Fallback Engine

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cluster::rdma` (`src/cluster/rdma.rs`)
- **Problem**: High-end HPC clusters feature InfiniBand / RoCE networking; THME should leverage kernel-bypass RDMA when available.
- **Acceptance Criteria**:
  - Detect InfiniBand / RoCE hardware interfaces via `libibverbs` FFI bindings.
  - Establish RDMA Queue Pairs (QP) for zero-copy memory transfers.
  - Fall back transparently to TCP ring all-reduce if RDMA hardware is not present.
- **Production Pipeline / CI/CD Impact**: Delivers enterprise-grade multi-node scaling on HPC clusters.
- **Done Means**: `cargo test test_rdma_detection_and_fallback` verifies fallback behavior and queue pair setup.

### [THME-1206] Multicast / mDNS Cluster Node Discovery & Capability Handshake

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cluster::discovery` (`src/cluster/discovery.rs`)
- **Problem**: Ad-hoc clustering requires machines on a local subnet to automatically discover each other without manual IP configuration.
- **Acceptance Criteria**:
  - mDNS / UDP multicast beacon broadcasting THME node ID, compute capability (TFLOP/s), and available VRAM.
  - Dynamic capability handshake establishing cluster topology and leader election.
- **Production Pipeline / CI/CD Impact**: Zero-configuration cluster setup for local network workstations.
- **Done Means**: `cargo test test_cluster_mdns_discovery` discovers peers and exchanges hardware capability payloads.

### [THME-1207] Continuous 100 ms Heartbeat & Dead-Node Watchdog

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cluster::watchdog` (`src/cluster/watchdog.rs`)
- **Problem**: Nodes in heterogeneous or ad-hoc clusters may disconnect, reboot, or experience network drops during long runs.
- **Acceptance Criteria**:
  - Periodic heartbeat ping every 100 ms between cluster nodes.
  - Detect dead or unresponsive nodes after 3 missed heartbeats (300 ms).
  - Emit cluster health alert and trigger failover protocol.
- **Production Pipeline / CI/CD Impact**: Protects distributed training and inference jobs from silent network hangs.
- **Done Means**: `cargo test test_cluster_heartbeat_watchdog` detects simulated node failure within 300 ms.

### [THME-1208] Automated Dynamic Layer Re-Sharding Failover

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cluster::reshard` (`src/cluster/reshard.rs`)
- **Problem**: When a node disconnects, its assigned model layers become orphaned; the cluster must redistribute them without terminating the job.
- **Acceptance Criteria**:
  - Automated re-sharding algorithm redistributing orphaned layers to remaining healthy nodes.
  - Re-establish communication pipelines and resume forward pass from current checkpoint.
- **Production Pipeline / CI/CD Impact**: Guarantees fault-tolerant execution in volatile distributed environments.
- **Done Means**: `cargo test test_dynamic_resharding` demonstrates forward pass recovery after dropping 1 of 3 nodes.

### [THME-1209] CLI Subcommand `thme cluster` & Cluster Formation Wizard

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cli::cluster` (`src/cli/cluster.rs`)
- **Problem**: Operators need a command-line interface to initialize, join, and inspect distributed clusters.
- **Acceptance Criteria**:
  - CLI `thme cluster init --port <port>`, `thme cluster join --leader <ip:port>`, and `thme cluster status`.
  - Terminal dashboard showing active nodes, hardware specifications, assigned layers, and network latency matrix.
- **Production Pipeline / CI/CD Impact**: Primary CLI for managing distributed THME deployments in data centers and local labs.
- **Done Means**: `thme cluster --help` executes cleanly, unit tests verify cluster command dispatch.
- **Progress** (26 November 2026): `src/cli/cluster.rs` and the planner in `src/cluster/mod.rs` exist (interconnect classification, stage partition, Q4_1 wire codec); init, join and status networking not done.

---

## Sprint 13: Universal Agentic AI Connectivity Gateway (former v0.2.4) [DEFERRED to v0.3+]

### [THME-1301] High-Throughput Asynchronous WebSocket Event Streaming Daemon

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::daemon::server` (`src/daemon/server.rs`)
- **Problem**: Remote agents, IDEs, and dashboards require real-time bidirectional WebSocket streaming of per-token progress, layer execution, and telemetry.
- **Acceptance Criteria**:
  - Headless daemon (`thme daemon --port 9090`) based on `tokio-tungstenite`.
  - Broadcast structured JSON event streams: `TokenGenerated`, `LayerExecuted`, `SurgeryApplied`, `ThermalWarning`.
  - Support $> 100$ concurrent client WebSocket connections with $< 1 \text{ ms}$ broadcast latency.
- **Production Pipeline / CI/CD Impact**: Core event hub connecting all external clients to running THME instances.
- **Done Means**: `cargo test test_websocket_streaming_daemon` verifies concurrent client connections and message broadcasting.

### [THME-1302] JSON-RPC 2.0 Method Router & Remote Protocol Dispatcher

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::daemon::router` (`src/daemon/router.rs`)
- **Problem**: External tools need a standardized remote procedure call API to invoke THME operations (inspect, surgery, validate, export).
- **Acceptance Criteria**:
  - JSON-RPC 2.0 specification compliant router supporting batch requests and error codes.
  - Dispatch methods: `thme_inspectModel`, `thme_applySurgery`, `thme_validateQuality`, `thme_exportModel`, `thme_profileRoofline`.
- **Production Pipeline / CI/CD Impact**: Provides stable, language-agnostic API for third-party developer integrations.
- **Done Means**: `cargo test test_jsonrpc_method_router` validates JSON-RPC request and response payloads.

### [THME-1303] Cryptographic HMAC-SHA256 Surgery Recipe Attestation

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::daemon::attestation` (`src/daemon/attestation.rs`)
- **Problem**: Exposing model surgery over a network creates security vulnerabilities; recipes must be authenticated and tamper-proof.
- **Acceptance Criteria**:
  - HMAC-SHA256 signature verification on incoming JSON surgery recipe payloads using pre-shared keys or bearer tokens.
  - Reject unsigned or altered payloads with `401 Unauthorized` / `AttestationFailed` error.
  - Rate limiting and request throttling per authenticated identity.
- **Production Pipeline / CI/CD Impact**: Enforces software supply-chain security and prevents unauthorized model modifications in shared environments.
- **Done Means**: `cargo test test_hmac_recipe_attestation` rejects tampered recipes and accepts validly signed payloads.

### [THME-1304] Autonomous CLI Agent Adapter

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::daemon::adapters::cli_agent` (`src/daemon/adapters/cli_agent.rs`)
- **Problem**: Coding agents (Claude Code, Gemini CLI, Aider, OpenHands) need formatted tool schemas and direct execution wrappers.
- **Acceptance Criteria**:
  - Generate agent tool configuration blocks compatible with popular coding agent frameworks.
  - Structured error messages suggesting specific remediation steps when surgery operations fail.
- **Production Pipeline / CI/CD Impact**: Enables autonomous coding agents to perform automated model surgery directly from CLI prompts.
- **Done Means**: `cargo test test_cli_agent_adapter` verifies agent schema generation and tool calling.

### [THME-1305] CI/CD Orchestration Harness Adapter

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::daemon::adapters::cicd` (`src/daemon/adapters/cicd.rs`)
- **Problem**: Enterprise CI/CD systems (Harness CI/CD, GitHub Actions, GitLab CI) need structured step outputs, JUnit reports, and exit code mappings.
- **Acceptance Criteria**:
  - Emit GitHub Actions workflow commands (`::set-output`, `::error`, `::notice`).
  - Generate standard JUnit XML test reports from model validation and perplexity benchmarks.
- **Production Pipeline / CI/CD Impact**: Seamlessly integrates THME quality gates into enterprise CI/CD deployment pipelines.
- **Done Means**: `cargo test test_cicd_adapter_reports` verifies JUnit XML generation and workflow command formatting.

### [THME-1306] Desktop Studio Remote Daemon Connection Bridge

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::gui::remote_client` (`src/gui/remote_client.rs`)
- **Problem**: Users running `thme-gui` on lightweight laptops want to connect to remote GPU servers hosting THME daemon instances.
- **Acceptance Criteria**:
  - Remote connection modal in Desktop Studio: enter host IP, port, and authentication token.
  - Remote state synchronization mirroring remote model inventory, telemetry, and surgery state in the local GUI.
- **Production Pipeline / CI/CD Impact**: Expands Desktop Studio usability to remote headless GPU clouds.
- **Done Means**: `cargo test test_gui_remote_bridge` establishes WebSocket session and synchronizes model state.

### [THME-1307] IDE & GitHub Copilot Extension Bridge

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::daemon::adapters::ide_bridge` (`src/daemon/adapters/ide_bridge.rs`)
- **Problem**: VS Code, Cursor, and Windsurf extensions require live daemon connectivity to stream model topology changes into editor sidebars.
- **Acceptance Criteria**:
  - Bi-directional bridge connecting editor language servers to THME daemon.
  - Live workspace telemetry updating VS Code tree views upon layer surgery.
- **Production Pipeline / CI/CD Impact**: Closes the loop between developer code editors and active model surgery daemons.
- **Done Means**: `cargo test test_ide_bridge_synchronization` verifies event passing between editor client and daemon.

### [THME-1308] Real-Time System Telemetry & Resource Pressure Broadcasting

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::daemon::telemetry_broadcaster` (`src/daemon/telemetry_broadcaster.rs`)
- **Problem**: Remote monitoring systems (Grafana, Datadog) need real-time streams of VRAM pressure, GPU utilization, and token throughput.
- **Acceptance Criteria**:
  - Stream metrics in OpenTelemetry / Prometheus format: `thme_tokens_per_second`, `thme_vram_used_bytes`, `thme_layer_latency_seconds`.
  - Publish metrics over HTTP endpoint `/metrics`.
- **Production Pipeline / CI/CD Impact**: Standard observability integration for enterprise model serving infrastructure.
- **Done Means**: `cargo test test_telemetry_prometheus_exporter` verifies `/metrics` response format and values.

### [THME-1309] CLI Subcommand `thme daemon` & Universal Daemon Controller

- **Status**: triage (legacy record, no status was written)
- **Component**: `thme::cli::daemon` (`src/cli/daemon.rs`)
- **Problem**: System administrators require a dedicated CLI command to launch, configure, and daemonize the gateway server.
- **Acceptance Criteria**:
  - CLI `thme daemon --port <port> --bind <ip> --token <secret> --cors <origins>`.
  - Support graceful shutdown on SIGINT/SIGTERM, draining active connections.
  - Built-in logging with `--verbose` and structured JSON logs.
- **Production Pipeline / CI/CD Impact**: Primary system service entry point for systemd / docker container deployments.
- **Done Means**: `thme daemon --help` displays clean options, and integration tests verify server start and graceful shutdown.
