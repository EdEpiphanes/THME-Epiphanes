<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Backlog: finished tickets

Tickets with status done, moved out of `BACKLOG.md` on 27 November 2026 to keep the active backlog readable. Text is unchanged.

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

### [THME-2207] SmoothQuant-style scale migration

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.7 · **Priority**: p2 · **Area**: byoa · **Type**: feature · **Status**: done
- **Problem**: The existing SmoothQuant rule clips weights only; true migration rescales weights and activations together.
- **Done means**: Migration folded into the preceding norm so the model's function is preserved before quantization; a test confirms unchanged outputs without quantization.

### [THME-2209] Measured artifact size in every report

- **Milestone**: v0.2.2 · **Roadmap**: v0.2.2.9 · **Priority**: p1 · **Area**: exporter · **Type**: feature · **Status**: done
- **Problem**: Recipe and search sizes are nominal bits per weight.
- **Done means**: Reports include the written file size including scales and metadata alongside the nominal figure.

## Later

### [THME-3097] DC-CCCP sign split

- **Milestone**: v0.2.4 · **Roadmap**: unassigned · **Priority**: p1 · **Area**: allocator · **Type**: fix · **Status**: done in code; not run on a real model
- **Problem**: `src/allocator/dc_cccp.rs` majorizes every coupling entry; paper section 4.2 keeps negative-coefficient terms convex and majorizes only positive ones.
- **Done means**: sign-split majorizer with a test that the surrogate is at least the objective at random points and equal at the expansion point, and monotone descent.
- **Progress** (27 November 2026): `sign_split_surrogate` and `mm_descend` in `src/allocator/dc_cccp.rs`. Tests: the surrogate is at least the objective at 200 random instances whose gamma_plus has negative off-diagonal entries and equals it at the expansion point; majorize-minimize steps never raise the objective and stay in budget; the solver is never worse than its diagonal-DP start. Not done: a measurement on a real model.

### [THME-3104] Finish every v0.2 sub-milestone

- **Milestone**: v0.2.9 · **Roadmap**: v0.2.9.9 · **Priority**: p1 · **Area**: docs · **Type**: chore · **Status**: done
- **Problem**: 44 of the 90 sub-milestones of v0.2.0 to v0.2.9 are not marked done. Some are small measurements, some need hardware or an owner decision, and v0.2.3 is a training program.
- **Done means**: Each sub-milestone is either done (its own done-means text met, evidence committed) or closed with a dated reason (owner decision, missing hardware); `docs/status/v0.2_completion_plan.md` is current; v0.2.9.9 is closed by re-reading it.
- **Progress** (27 November 2026): the plan groups the 44 items (CPU work, hardware, owner decisions, the training program) and fixes the order. Closed the same day: 61 of 90 Done, 29 Deferred with named gaps (`docs/status/v0.2_closure.md`).

### [THME-3070] Incremental decoding with a KV cache

- **Milestone**: v0.2.4 · **Roadmap**: v0.2.4.1 · **Priority**: p1 · **Area**: inference · **Type**: feature · **Status**: done
- **Progress** (20 November 2026): `KvCache` and `forward_extend` for LLaMA and GPT-NeoX behind `CausalLm`; the Studio chat reply loop prefills once and steps the cache (Gemma keeps the whole-sequence loop). Incremental logits equal full-sequence logits at every position: 0 on Pythia-14M and SmolLM-135M (24 tokens, real models, `THME_REAL_MODEL_TESTS=1`) and at most 7.6e-6 on a 96-token synthetic LLaMA (tolerance 1e-4). KV bytes equal `KvCacheEstimator` exactly. Measured paired old-vs-cached decode speed is in `docs/benchmarks/<model>/data/decode_speed.json` (shared machine; see the folder README). Not done: GPU path not measured; threads and matmul kernels unchanged (separate tickets).
- **Problem**: `chat::generate` calls `forward_logits` on the whole growing sequence for every new token, so each token costs a full pass over all tokens so far. Research note `docs/research/token-speed-budget-devices.md` (section 4) ranks this as the largest single loss for decode.
- **Done means**: A KV cache for the LLaMA and GPT-NeoX reference models behind `CausalLm` (`forward_logits` unchanged for the perplexity path). Incremental logits equal full-sequence logits at every position within a stated tolerance on a synthetic model, and on Pythia-14M and SmolLM-135M behind `THME_REAL_MODEL_TESTS=1`. Studio chat and the reply loop use prefill then one-token steps. `examples/decode_speed.rs` gives paired, interleaved old-vs-new decode tokens/s with spread, committed under `docs/benchmarks/<model>/data/decode_speed.json` with machine facts; KV bytes vs context checked against `src/profiling/kv_cache.rs`. Threads and matmul kernels are not changed here.

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

### [THME-3000] Consolidate the dossier v3.1 claim register under docs/research/

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.9 · **Priority**: p1 · **Area**: research · **Type**: docs · **Status**: done
- **Problem**: Claims V-01 to V-135 are spread over the appendices of the PDF; citations were written from memory.
- **Done means**: One register file with unique IDs, status letters and the section each claim lives in; every citation checked or marked unverified.

### [THME-3001] Engine error module (dossier WP1)

- **Milestone**: v0.2.5 · **Roadmap**: v0.2.5.1 · **Priority**: p1 · **Area**: engine · **Type**: feature · **Status**: done
- **Problem**: Engine primitives need one error type that never panics and never falls back silently.
- **Done means**: `src/engine/error.rs`: invalid-input, infeasible, no-convergence and non-finite variants plus a finite-value guard, each reached by a test; no behaviour change elsewhere.

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

### [THME-3009] Versioned schemas under docs/schemas/ (dossier B.6.4)

- **Milestone**: v0.2.8 · **Roadmap**: v0.2.8.9 · **Priority**: p2 · **Area**: core · **Type**: docs · **Status**: done
- **Problem**: Benchmark task, benchmark result, quality-gate version, plugin protocol and conversion-registry entry have no schema.
- **Done means**: Each schema carries a version field and validates a fixture in CI.
- **Progress**: benchmark task, result record and quality gate schemas done (`docs/schemas/*-v1.json`, `tests/schemas.rs`). Plugin protocol and conversion-registry schemas wait until those exist (plugins are commercial-only for now).

### [THME-3010] BYOA rules get calibration through their interface, not tensor metadata

- **Milestone**: v0.2.2 · **Roadmap**: none · **Priority**: p2 · **Area**: byoa · **Type**: refactor · **Status**: done
- **Problem**: `ActivationAware` reads statistics from a JSON string in tensor metadata and needs a special case in `apply_selected_format`; each new data-needing rule would add another.
- **Done means**: A rule receives an apply context (calibration statistics, row width) at one seam; the metadata side channel and special case are gone; AWQ tests unchanged.

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

### [THME-3018] Static model security scanner (`thme scan`)

- **Milestone**: v0.2.7 · **Roadmap**: v0.2.7.1 · **Priority**: p1 · **Area**: security · **Type**: feature · **Status**: done
- **Problem**: Model files can execute code on load (pickle imports, template injection, trust_remote_code) or hide payloads in unused bytes.
- **Done means**: `src/scan`: pickle opcode walker (plain and PyTorch zip), GGUF/SafeTensors header and coverage checks, metadata and chat-template rules, code and executable files, `.epiph` verification; `--fail-on` exit code; tests on built fixtures, no file loaded or run.

### [THME-3019] Signed `.epiph` files

- **Milestone**: v0.2.6 · **Roadmap**: v0.2.6.2 · **Priority**: p1 · **Area**: security · **Type**: feature · **Status**: done
- **Problem**: Owner wants files signed by Epiphanes. Third-party converters can read the open format but get plain weights only; signature, provenance, quality evidence and placement are THME features.
- **Done means**: Optional signature section over the manifest and payload digests; `thme epiph sign` / `verify` check it; key handling documented; a pure-Rust signature dependency proposed and approved first.

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

### [THME-3037] Tokenizer and chat templates

- **Milestone**: v0.3.0 · **Roadmap**: v0.3.0.3 · **Priority**: p2 · **Area**: inference · **Type**: feature · **Status**: done
- **Progress** (19 November 2026): `thme tokenize MODEL_DIR TEXT` and `thme chat-template MODEL_DIR --messages FILE` encode, decode and render a model's own chat template (string or named-list form, bos/eos tokens, `raise_exception` errors). New direct crates, all Apache-2.0 and pure Rust: `tokenizers` 0.22.2 (no onig, no C++), `minijinja` 3.1.0 and `minijinja-contrib` 3.1.0. 15 tests on fixtures built in the test; no real Hugging Face model or real Llama or Mistral template was used. The release CLI grew to about 12.5 MiB, so the CLI limit was raised from 10 MiB to 16 MiB (owner decision, 19 November 2026).
- **Problem**: THME cannot encode text or apply a chat template, so it cannot chat with a model or run text benchmarks itself.
- **Done means**: Encode and decode through a permissively licensed tokenizer crate and render `chat_template` with a template engine, each proposed with its licence and binary-size cost first; round trips tested on committed fixtures.

### [THME-3041] `thme doctor` and a guided first run

- **Milestone**: v0.3.0 · **Roadmap**: v0.3.0.2 · **Priority**: p2 · **Area**: onboarding · **Type**: feature · **Status**: done
- **Progress** (19 November 2026): `thme doctor [--json]` reports operating system, memory, disk, CPU threads, GPU adapters found through wgpu and the models `thme discover` finds, each marked checked or assumed, then prints a short Next list. 9 tests on a `HostFacts` struct, none depending on this machine. Not done: the Studio first-start summary.
- **Progress** (19 November 2026, later): the Studio shows the doctor summary and the discover list in a Welcome panel when no model is loaded (`src/gui/app/welcome.rs`); checks run on a background thread, Open runs the safety scan first, at most 10 rows with the total. 12 state-machine tests. The live window was not opened.
- **Problem**: People come from different backgrounds and tools. They need one command that says what works on their machine and what to do next.
- **Done means**: `thme doctor` reports operating system, memory, disk, GPU adapter found through wgpu, and the models `thme discover` finds, then prints the next command to try. The Studio shows the same summary on first start. Every line states what was checked and what was only assumed.

### [THME-3044] Re-measure on Pythia-14M with the corrected code

- **Milestone**: v0.2.0 · **Roadmap**: v0.2.0.9 · **Priority**: p1 · **Area**: research · **Type**: chore · **Status**: done
- **Problem**: The earlier reports were measured before the FP16 rounding fix and have been removed. Pythia-14M (Apache-2.0, 14,067,712 parameters, 6 layers) is downloaded for local testing and loads in `thme inspect`.
- **Progress** (20 November 2026): `docs/benchmarks/pythia-14m/data/verification_report.json` (source perplexity 114.2862 on the full WikiText-2 test split, 286,878 scored tokens), `precision_sweep.json` (`--layer-scan`), `depth_probe.json`, and `search_report.json` committed; charts and dossier regenerated with `scripts/generate_dossier.py` and `scripts/generate_charts.py`. The tokenizer was read from a local `tokenizer.json`, not downloaded for this run; its hash is in the report but was not compared with the hub copy (unverified).
- **Done means**: Perplexity of the source model, the Q8_0 round trip and a precision sweep are re-run with the fixed code, committed as reports under `docs/benchmarks/` with their method, and charts regenerate from them. Results are published as they come, with the method beside them.

### [THME-3045] Remove GPL-2.0 derived GPU-unlock code; plan an original Rust unlock engine

- **Milestone**: v0.2.7 · **Roadmap**: none · **Priority**: p1 · **Area**: legal · **Type**: chore · **Status**: done
- **Problem**: `crates/neutron-gpu` was ported from an upstream GPU-unlock project whose kernel module and packaging are GPL-2.0-only (its README says so; the firmware part has its own notice). GPL-2.0-only is incompatible with this repository's AGPL-3.0-only and with the planned commercial licence, and a port is a derivative work that would itself have to be GPL-2.0 with attribution. THME shipped it under AGPL with no attribution, so it is removed. The unlock engine, ROP payload synthesis and register-offset data are gone.
- **Done means**: `crates/neutron-gpu/src/` contains only PCI/vGPU discovery and a plain hardware recipe table; the unlock engine, ROP synthesis and register-offset data are gone. A future ticket designs an original, tested, Rust-native unlock engine from scratch, not derived from any GPL or otherwise incompatibly licensed source.

