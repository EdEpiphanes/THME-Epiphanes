# Roadmap vs. Backlog Comparison (v0.2.0 through v0.3.0)

Source files compared: [ROADMAP.md](../../ROADMAP.md) (sections `Where things stand` and `Milestones in detail`) and [BACKLOG.md](../../BACKLOG.md). Ticket `Status` values are quoted verbatim from `BACKLOG.md`; no status is inferred where none is written.

## Summary by Milestone

| Milestone | ROADMAP state (Mermaid / Heading) | Ticket count | Counts per Status value | Verdict |
| :--- | :--- | ---: | :--- | :--- |
| `v0.2.0` | `active` / `(ACTIVE)` | 21 | `done`: 9; `ready`: 8; `triage`: 2; `in progress`: 1; `deferred (owner decision)`: 1 | `consistent` |
| `v0.2.1` | `done` / none in heading | 0 | none | `no tickets` |
| `v0.2.2` | `active` / none in heading | 12 | `done`: 7; `ready`: 3; `blocked`: 1; `done for Pythia-14M and SmolLM-135M (C4 arm and size check pending)`: 1 | `consistent` |
| `v0.2.3` | `next` / none in heading | 3 | `triage`: 2; `ready`: 1 | `consistent` |
| `v0.2.4` | `next` / none in heading | 27 | `triage`: 9; `ready`: 8; `in review`: 5; `done`: 2; `in progress`: 1; `in review (no-go at this tuning)`: 1; `done in code; not run on a real model`: 1 | `consistent` |
| `v0.2.5` | `active` / none in heading | 8 | `done`: 5; `blocked`: 1; `ready`: 1; `triage`: 1 | `consistent` |
| `v0.2.6` | `active` / none in heading | 9 | `done`: 6; `triage`: 2; `blocked (owner: funding and registration)`: 1 | `consistent` |
| `v0.2.7` | `active` / none in heading | 8 | `done`: 4; `triage`: 3; `shelved (owner: opportunities and the commercial sector)`: 1 | `consistent` |
| `v0.2.8` | `next` / none in heading | 11 | `done`: 10; ``done in code (27 November 2026); the default gate stays v1, v2 is chosen with `--gate` ``: 1 | `consistent` |
| `v0.2.9` | `next` / none in heading | 1 | `triage`: 1 | `consistent` |
| `v0.3.0` | `active` / none in heading | 17 | `triage`: 13 (1 in active backlog, 12 in Archive); `done`: 2; `ready`: 2 | `consistent` |

Scope notes:
- `THME-3069` ("Visual check of every Studio tab, then recapture screenshots") has `Milestone: v0.4` and `Status: ready`, so it falls outside `v0.2.0` through `v0.3.0`.
- `THME-3055` through `THME-3066` carry `Milestone: v0.3.0` and `Status: triage`, but sit below `<!-- backlog:end -->` under `## Archive` in `BACKLOG.md`.

---

## Milestone Details

### v0.2.0

- **ROADMAP state**: Mermaid class `active` (`B["v0.2.0<br/>Measured core"]:::active`); section heading ``## v0.2.0 — Measured core (ACTIVE)``.
- **BACKLOG tickets (`Milestone: v0.2.0`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-2001 | Cross-check the forward pass against the Hugging Face reference | `done` |
| THME-2002 | Upgrade egui/eframe to clear RUSTSEC-2026-0194/0195 | `done` |
| THME-2003 | Split src/gui/app.rs into tab modules | `done` |
| THME-2004 | Publish the first measured recipe search and Pareto chart | `done` |
| THME-2005 | Release v0.2.0 | `deferred (owner decision)` |
| THME-2006 | Configure repository protection and secrets | `done` |
| THME-3101 | Repository-wide status overhaul: roadmap, charts, status files, labels | `in progress` |
| THME-3071 | Provenance ledger: record where every part of the code and data came from | `ready` |
| THME-3076 | VS Code sidebar shows fixed example text as if it were live data | `done` |
| THME-3034 | Third-party code register | `done` |
| THME-3035 | Replace mock and simulated data in the Studio and engine | `done` |
| THME-3044 | Re-measure on Pythia-14M with the corrected code | `done` |
| THME-3047 | Clean public repository from a clean export | `triage` |
| THME-3048 | VS Code extension: no shell injection, honest commands | `ready` |
| THME-3049 | Release pipeline hardening | `ready` |
| THME-3050 | Remove agent notes, personal settings and withdrawn claims from the tree | `ready` |
| THME-3051 | Drop the hard-coded model fallback from MCP and tests | `ready` |
| THME-3052 | Repository hygiene sweep | `ready` |
| THME-3053 | Enforce the teacher subprocess timeout; delete dead code | `ready` |
| THME-3054 | Document the MCP trust model and the memory-map limit | `ready` |
| THME-3067 | precision_sweep: random_matched seeds 2 and 3 gave identical perplexity on Pythia-14M | `triage` |

- **Counts per Status value**: `done`: 9, `ready`: 8, `triage`: 2, `in progress`: 1, `deferred (owner decision)`: 1 (total: 21).
- **Verdict**: `consistent` (ROADMAP says `active` / `(ACTIVE)`, and 12 of 21 tickets are not `done`).

### v0.2.1

- **ROADMAP state**: Mermaid class `done` (`C["v0.2.1<br/>Recipe and BYOA lab<br/>done"]:::done`); section heading ``## v0.2.1 — Recipe and BYOA lab`` (no status label in heading; all nine sub-items `v0.2.1.1` through `v0.2.1.9` are marked `**Done**` in the table).
- **BACKLOG tickets (`Milestone: v0.2.1`)**: none. (Related `Roadmap` references exist on `THME-2004` under `Milestone: v0.2.0` and `THME-2101` under `Milestone: v0.2.3`; Sprint 10 in `Archive` is labelled `former v0.2.1` with no `Milestone` field.)
- **Counts per Status value**: none (0 tickets).
- **Verdict**: `no tickets`.

### v0.2.2

- **ROADMAP state**: Mermaid class `active` (`D["v0.2.2<br/>Formats and calibration"]:::active`); section heading ``## v0.2.2 — Formats and calibration`` (no status label in heading).
- **BACKLOG tickets (`Milestone: v0.2.2`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-2201 | Per-tensor formats in the GGUF writer | `done` |
| THME-2202 | Q4_K, Q5_K and Q6_K encoders | `done` |
| THME-2203 | Capture calibration activations from the forward pass | `done` |
| THME-2204 | Activation-aware scaling (AWQ-style) | `done for Pythia-14M and SmolLM-135M (C4 arm and size check pending)` |
| THME-2205 | Second-order rounding (GPTQ-style) | `ready` |
| THME-2206 | Measure Hadamard rotation on real models | `blocked` |
| THME-2207 | SmoothQuant-style scale migration | `done` |
| THME-2208 | llama.cpp compatibility checks for exported files | `ready` |
| THME-2209 | Measured artifact size in every report | `done` |
| THME-3091 | GGUF export that llama.cpp loads (names, tokenizer, rope layout, norms, flags) | `ready` |
| THME-3010 | BYOA rules get calibration through their interface, not tensor metadata | `done` |
| THME-3023 | Re-measure reports affected by the FP16 rounding fix | `done` |

- **Counts per Status value**: `done`: 7, `ready`: 3, `blocked`: 1, `done for Pythia-14M and SmolLM-135M (C4 arm and size check pending)`: 1 (total: 12).
- **Verdict**: `consistent` (ROADMAP says `active`, and 5 of 12 tickets are not `done`).

### v0.2.3

- **ROADMAP state**: Mermaid class `next` (`E["v0.2.3<br/>Recovery and distillation"]:::next`); section heading ``## v0.2.3 — Recovery and distillation`` (no status label in heading).
- **BACKLOG tickets (`Milestone: v0.2.3`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-2101 | Live 3D: attention maps, drill-down and side-by-side | `ready` |
| THME-2300 | Epic: Recovery and distillation | `triage` |
| THME-3036 | LoRA training with a real backward pass | `triage` |

- **Counts per Status value**: `triage`: 2, `ready`: 1 (total: 3).
- **Verdict**: `consistent` (ROADMAP says `next`, and none of the 3 tickets are `done`).

### v0.2.4

- **ROADMAP state**: Mermaid class `next` (`F["v0.2.4<br/>Runtime, speed, KV cache"]:::next`); section heading ``## v0.2.4 — Runtime, speed and KV cache`` (no status label in heading; sub-item `v0.2.4.1` is `**Done**` and `v0.2.4.2`, `v0.2.4.3` are `**In progress**`).
- **BACKLOG tickets (`Milestone: v0.2.4`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-2400 | Epic: Runtime, speed and KV cache | `triage` |
| THME-3079 | Token-frequency count for Pythia-14M (weights-only) | `in review` |
| THME-3080 | Per-row precision bands for the embedding and output matrices (Pythia-14M) | `in progress` |
| THME-3072 | GPU path faster than CPU on the small models | `in review` |
| THME-3084 | GPU speed round 2: profile, matmul, attention, resident weights, f16 and quantized weights | `in review` |
| THME-3085 | Reference comparison against llama.cpp on this machine | `ready` |
| THME-3086 | GPU feature audit: what the adapter really exposes, with checked kernels | `in review` |
| THME-3087 | Cooperative-matrix (tensor-core) matmul and attention | `in review` |
| THME-3088 | Packed 8-bit integer dot product for Q8_0 and Q4_0 weights | `in review (no-go at this tuning)` |
| THME-3092 | Gate code reaches the declared-range specification | `triage` |
| THME-3093 | Run the Step 1 gate on real per-block losses | `triage` |
| THME-3094 | Measure BITCOS and BITCOS-2:4 accuracy | `triage` |
| THME-3095 | Q4_0 decode reaches the bandwidth ceiling | `triage` |
| THME-3096 | Commit JSON reports for the SmolLM2-1.7B GPU, decode and screening runs | `triage` |
| THME-3097 | DC-CCCP sign split | `done in code; not run on a real model` |
| THME-3098 | Commit the paper and its check scripts | `triage` |
| THME-3099 | Loop breaker behaviour matches its specification | `triage` |
| THME-3100 | Calibrate the allocator link constant C | `triage` |
| THME-3070 | Incremental decoding with a KV cache | `done` |
| THME-3074 | Benchmark runner in Rust: one process, model loaded once, progress | `ready` |
| THME-3075 | Run metrics and progress for every long command, in Rust | `ready` |
| THME-3073 | Perplexity on the GPU, device in every report, baseline cache | `done` |
| THME-3081 | Frequent-row draft pass for the output head (Pythia-14M) | `ready` |
| THME-3082 | KV cache in int8 and int4 per group, counted in the placement budget | `ready` |
| THME-3083 | Layer-removal check on the small models | `ready` |
| THME-3089 | Device decoder with KV cache and fused quantized matrix-vector for decode | `ready` |
| THME-3090 | Automatic weight budget for the GPU path (default is 7x slower than it should be) | `ready` |

- **Counts per Status value**: `triage`: 9, `ready`: 8, `in review`: 5, `done`: 2, `in progress`: 1, `in review (no-go at this tuning)`: 1, `done in code; not run on a real model`: 1 (total: 27).
- **Verdict**: `consistent` (ROADMAP says `next`, and 25 of 27 tickets are not `done`; note that active work is already underway across `in progress`, `in review`, and `done` tickets, and `THME-3090` also carries a follow-up note stating its status stays `"done in code; not run on a real GPU"` after landing while its `Status` line reads `ready`).

### v0.2.5

- **ROADMAP state**: Mermaid class `active` (`H["v0.2.5<br/>Math engine"]:::active`); section heading ``## v0.2.5 — Math engine foundations`` (no status label in heading).
- **BACKLOG tickets (`Milestone: v0.2.5`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-3001 | Engine error module (dossier WP1) | `done` |
| THME-3002 | Exact dynamic-programming allocator behind `thme search --solver` (dossier WP2) | `blocked` |
| THME-3004 | Shared basis module: one Walsh-Hadamard transform (dossier WP5) | `done` |
| THME-3006 | Number-format conversions with stated error bounds (dossier WP3) | `done` |
| THME-3007 | Dependency-free SVG visual engine (dossier WP9) | `done` |
| THME-3033 | Isometric figure kit in Rust (anatomy port) | `done` |
| THME-3039 | Animated live-layers figure | `ready` |
| THME-3043 | Evaluate the `neutron-iso` Rust port of the anatomy kit | `triage` |

- **Counts per Status value**: `done`: 5, `blocked`: 1, `ready`: 1, `triage`: 1 (total: 8).
- **Verdict**: `consistent` (ROADMAP says `active`, and 3 of 8 tickets are not `done`).

### v0.2.6

- **ROADMAP state**: Mermaid class `active` (`I["v0.2.6<br/>Epiphanes model format"]:::active`); section heading ``## v0.2.6 — Epiphanes model format`` (no status label in heading).
- **BACKLOG tickets (`Milestone: v0.2.6`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-3078 | `.epiph` flexible encodings: any tensor layout, safely extensible | `triage` |
| THME-3011 | Measured bits per weight from the written artifact (placeholder) | `triage` |
| THME-3012 | `.epiph` v1.0: open container with full model metadata | `done` |
| THME-3013 | `.epiph`: pack tensors in their selected formats | `done` |
| THME-3014 | `.epiph`: tokenizer and GGUF metadata attachments | `done` |
| THME-3015 | `.epiph`: lossless payload compression | `done` |
| THME-3016 | `.epiph`: GPU/CPU placement plan | `done` |
| THME-3017 | Trademark search and filing for the format name | `blocked (owner: funding and registration)` |
| THME-3019 | Signed `.epiph` files | `done` |

- **Counts per Status value**: `done`: 6, `triage`: 2, `blocked (owner: funding and registration)`: 1 (total: 9).
- **Verdict**: `consistent` (ROADMAP says `active`, and 3 of 9 tickets are not `done`).

### v0.2.7

- **ROADMAP state**: Mermaid class `active` (`J["v0.2.7<br/>Security and trust"]:::active`); section heading ``## v0.2.7 — Security and trust`` (no status label in heading).
- **BACKLOG tickets (`Milestone: v0.2.7`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-3005 | SPDX headers on every source file | `done` |
| THME-3008 | Permission and ownership record for reverse-engineering features (dossier WP19) | `triage` |
| THME-3018 | Static model security scanner (`thme scan`) | `done` |
| THME-3020 | Developer sandbox for untrusted code | `shelved (owner: opportunities and the commercial sector)` |
| THME-3021 | Guardrail detection and evaluation (read-only) | `triage` |
| THME-3022 | Scanner: more formats and compressed pickles | `done` |
| THME-3045 | Remove GPL-2.0 derived GPU-unlock code; plan an original Rust unlock engine | `done` |
| THME-3046 | Original Rust GPU-unlock engine (placeholder) | `triage` |

- **Counts per Status value**: `done`: 4, `triage`: 3, `shelved (owner: opportunities and the commercial sector)`: 1 (total: 8).
- **Verdict**: `consistent` (ROADMAP says `active`, and 4 of 8 tickets are not `done`).

### v0.2.8

- **ROADMAP state**: Mermaid class `next` (`K["v0.2.8<br/>Quality gate, benchmarks"]:::next`); section heading ``## v0.2.8 — Quality gate and benchmarks`` (no status label in heading; 8 of 9 sub-items are marked `**Done**` and `v0.2.8.3` is marked `**Imported mode done** ...; native and bridged wait for model runs`).
- **BACKLOG tickets (`Milestone: v0.2.8`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-3077 | Quality gate v2: a margin that fits the model's perplexity (two regimes) | ``done in code (27 November 2026); the default gate stays v1, v2 is chosen with `--gate` `` |
| THME-3000 | Consolidate the dossier v3.1 claim register under docs/research/ | `done` |
| THME-3003 | One versioned quality gate; retire the stale 0.25 tolerance (dossier WP20) | `done` |
| THME-3009 | Versioned schemas under docs/schemas/ (dossier B.6.4) | `done` |
| THME-3024 | Benchmark registry: task files and result records | `done` |
| THME-3027 | Benchmark metrics library | `done` |
| THME-3028 | Benchmark statistics library | `done` |
| THME-3029 | Three-state quality gate | `done` |
| THME-3030 | Benchmark contamination check | `done` |
| THME-3031 | Benchmark runner: imported mode | `done` |
| THME-3032 | Benchmark charts | `done` |

- **Counts per Status value**: `done`: 10, ``done in code (27 November 2026); the default gate stays v1, v2 is chosen with `--gate` ``: 1 (total: 11).
- **Verdict**: `consistent` (ROADMAP says `next`, and `THME-3077` is qualified as `done in code ...; the default gate stays v1` rather than plain `done`; note however that 10 of 11 tickets are `done`, zero tickets are `triage`/`ready`/`in progress`/`blocked`, and 8 of 9 ROADMAP sub-items are `Done`, so if `THME-3077` is treated as done this milestone would be `roadmap too pessimistic`).

### v0.2.9

- **ROADMAP state**: Mermaid class `next` (`L["v0.2.9<br/>Translation layer,<br/>convergence"]:::next`); section heading ``## v0.2.9 — Translation layer and v0.3 convergence`` (no status label in heading).
- **BACKLOG tickets (`Milestone: v0.2.9`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-3068 | llama.cpp compatibility test needs a configured runtime | `triage` |

- **Counts per Status value**: `triage`: 1 (total: 1).
- **Verdict**: `consistent` (ROADMAP says `next`, and its single ticket is `triage`).

### v0.3.0

- **ROADMAP state**: Mermaid class `active` (`M["v0.3.0<br/>Plug and play workspace"]:::active`); section heading ``## v0.3.0 — Plug and Play Workspace`` (no status label in heading).
- **BACKLOG tickets (`Milestone: v0.3.0`)**:

| Ticket ID | Short title | Status (verbatim) |
| :--- | :--- | :--- |
| THME-3037 | Tokenizer and chat templates | `done` |
| THME-3038 | Studio chat tab | `ready` |
| THME-3040 | `thme discover`: find the models already on this machine | `ready` |
| THME-3041 | `thme doctor` and a guided first run | `done` |
| THME-3042 | Bridges to the tools people already use | `triage` |
| THME-3055 | VRAM and adapter facts in `HostFacts` *(in Archive)* | `triage` |
| THME-3056 | Memory model for a run *(in Archive)* | `triage` |
| THME-3057 | `RunPlan` type and decision order, CPU-only *(in Archive)* | `triage` |
| THME-3058 | Shrink loop *(in Archive)* | `triage` |
| THME-3059 | Roofline decode estimate per device *(in Archive)* | `triage` |
| THME-3060 | CPU probe *(in Archive)* | `triage` |
| THME-3061 | GPU probe and CPU-wins rule *(in Archive)* | `triage` |
| THME-3062 | Hybrid placement through `epiph::placement` *(in Archive)* | `triage` |
| THME-3063 | Plan cache *(in Archive)* | `triage` |
| THME-3064 | Wire `DeviceArg::Auto` and Studio selector to the plan *(in Archive)* | `triage` |
| THME-3065 | `--verify` predicted vs measured report *(in Archive)* | `triage` |
| THME-3066 | Windows shared-memory guard *(in Archive)* | `triage` |

- **Counts per Status value**: `triage`: 13 (1 in active backlog, 12 in `## Archive` below `<!-- backlog:end -->`), `done`: 2, `ready`: 2 (total: 17).
- **Verdict**: `consistent` (ROADMAP says `active`, and 15 of 17 tickets, or 3 of 5 inside `backlog:start..end`, are not `done`).

---

## BACKLOG Tickets with No `Status` Line or an Empty One

- Between `<!-- backlog:start -->` and `<!-- backlog:end -->`: **0** tickets lack a `Status` line or have an empty `Status` line.
- Anywhere in `BACKLOG.md` with an empty `Status` line: **0**.
- Under `## Archive` (Sprints 1 through 13): **85** tickets have no `Status` line (`THME-1003` is the sole Sprint 1 through 13 ticket that has a `- **Status**:` line). Sprints 1 through 8 carry `(Done)` or `[COMPLETED]` in their titles, whereas Sprints 9 through 13 do not.

| Ticket ID | Title | Section |
| :--- | :--- | :--- |
| THME-101 | Full GGUF v2/v3 Quantization Type Decoder (Done) | Sprint 1 (v0.1.2) |
| THME-102 | Truncated File & Malformed Header Recovery (Done) | Sprint 1 (v0.1.2) |
| THME-103 | Workspace Crate Modularization (`thme-core`, `thme-cli`, `thme-gui`) (Done) | Sprint 1 (v0.1.2) |
| THME-201 | AVX2 / AVX-512 / ARM NEON Walsh-Hadamard Transform (Done) | Sprint 2 (v0.1.3) |
| THME-202 | Criterion Micro-Benchmark Suite in CI (Done) | Sprint 2 (v0.1.3) |
| THME-301 | External Perplexity Evaluation Bridge (Done) | Sprint 3 (v0.1.4) |
| THME-302 | Automated Anti-Degradation Quality Gate (Done) | Sprint 3 (v0.1.4) |
| THME-401 | Model Context Protocol (MCP) Server (`thme mcp`) (Done) | Sprint 4 (v0.1.5) |
| THME-402 | VS Code Extension & GitHub Copilot Integration (Done) | Sprint 4 (v0.1.5) |
| THME-403 | PyTorch & CUDA Ecosystem Interoperability Bridge (Done) | Sprint 4 (v0.1.5) |
| THME-404 | Universal 8 Specialized AI Model Paradigm Converter (Done) | Sprint 4 (v0.1.5) |
| THME-501 | Draft Model Manifold Synthesizer (Done) | Sprint 5 (v0.1.6) |
| THME-502 | Multi-Token Tree Speculation Engine (Done) | Sprint 5 (v0.1.6) |
| THME-601 | Hardware Target Specifications & Machine Balance Database (Done) | Sprint 6 (v0.1.7) |
| THME-602 | Linear Projection & Attention Operational Arithmetic Intensity Profiler (Done) | Sprint 6 (v0.1.7) |
| THME-603 | SwiGLU FFN & FlashAttention-2 Arithmetic Intensity Mathematical Models (Done) | Sprint 6 (v0.1.7) |
| THME-604 | 4-Tier Memory Hierarchy Cache Simulator (L1/L2/L3/DRAM) (Done) | Sprint 6 (v0.1.7) |
| THME-605 | Working Set Size & Cache Eviction Latency Degradation Model (Done) | Sprint 6 (v0.1.7) |
| THME-606 | Empirical Roofline Boundary Curve & Memory/Compute Classifier (Done) | Sprint 6 (v0.1.7) |
| THME-607 | Dynamic TDP Power & Thermal Throttling Projection Engine (Done) | Sprint 6 (v0.1.7) |
| THME-608 | CLI Subcommand `thme roofline` & MCP Tool `thme_profile_roofline` (Done) | Sprint 6 (v0.1.7) |
| THME-609 | Desktop Studio Tab 4 Interactive SVG/egui Roofline Deck (Done) | Sprint 6 (v0.1.7) |
| THME-701 | Vectorized BitNet b1.58 Ternary Bitstream Packer (Done) | Sprint 7 (v0.1.8) |
| THME-702 | Vectorized Ternary Bitstream Unpacker with FP16 Scale Product (Done) | Sprint 7 (v0.1.8) |
| THME-703 | Direct Pure Rust GGUF v3 Binary Writer & 32-Byte Alignment (Done) | Sprint 7 (v0.1.8) |
| THME-704 | GGUF Metadata Key-Value Reconstitution & Synchronization Engine (Done) | Sprint 7 (v0.1.8) |
| THME-705 | Direct Pure Rust SafeTensors Binary Serializer & 8-Byte Prefix (Done) | Sprint 7 (v0.1.8) |
| THME-706 | Low-Memory Streaming Chunked File Writer Engine (Done) | Sprint 7 (v0.1.8) |
| THME-707 | Calibrated Per-Channel & Per-Group MSE-Optimal Scale Search (Done) | Sprint 7 (v0.1.8) |
| THME-708 | CLI Subcommand `thme export` & Desktop Studio Tab 1 Exporter Wizard (Done) | Sprint 7 (v0.1.8) |
| THME-709 | Downstream Runtime Compatibility Testbed (Done) | Sprint 7 (v0.1.8) |
| THME-801 | Co-Surgery Closed-Loop State Machine & Controller [COMPLETED] | Sprint 8 (v0.1.9) |
| THME-802 | Teacher LLM Topology Analysis Prompt Synthesizer [COMPLETED] | Sprint 8 (v0.1.9) |
| THME-803 | Headless JSON-RPC & Subprocess Teacher Agent Communication Bridge [COMPLETED] | Sprint 8 (v0.1.9) |
| THME-804 | Multi-Objective 3D Pareto Optimizer [COMPLETED] | Sprint 8 (v0.1.9) |
| THME-805 | Pareto Hypervolume Indicator Convergence Metric [COMPLETED] | Sprint 8 (v0.1.9) |
| THME-806 | Automated Self-Healing Parameter Relaxation Heuristic [COMPLETED] | Sprint 8 (v0.1.9) |
| THME-807 | Anti-Degradation Quality Gate Watchdog & Rollback Engine [COMPLETED] | Sprint 8 (v0.1.9) |
| THME-808 | CLI Subcommand `thme co-surgery` & Progress Telemetry Stream [COMPLETED] | Sprint 8 (v0.1.9) |
| THME-809 | Desktop Studio Tab 2 Closed-Loop Co-Surgery Dashboard [COMPLETED] | Sprint 8 (v0.1.9) |
| THME-901 | SmolLM-135M Open-Weights End-to-End Verification Pipeline | Sprint 9 (v0.2.0, superseded) |
| THME-902 | LLaMA-3.2-1B Open-Weights End-to-End Verification Pipeline | Sprint 9 (v0.2.0, superseded) |
| THME-903 | Automated Reproducible Benchmark Script Suite | Sprint 9 (v0.2.0, superseded) |
| THME-904 | Published Benchmark Dossier Generator | Sprint 9 (v0.2.0, superseded) |
| THME-905 | Multi-Platform LTO Stripped Binary Release Packaging | Sprint 9 (v0.2.0, superseded) |
| THME-906 | Minisign & GPG Cryptographic Artifact Signing Pipeline | Sprint 9 (v0.2.0, superseded) |
| THME-907 | SHA-256 Checksum Manifest Generation | Sprint 9 (v0.2.0, superseded) |
| THME-908 | Comprehensive Architecture Playbook, User Manual & Man Pages | Sprint 9 (v0.2.0, superseded) |
| THME-909 | End-to-End Clean-Machine Verification Test Harness | Sprint 9 (v0.2.0, superseded) |
| THME-910 | Post-Refactoring Risk Register & Technical Debt Amortization (Pre-Release Gate) | Sprint 9 (v0.2.0, superseded) |
| THME-1001 | Hardware Abstraction Layer (HAL) Core Traits | Sprint 10 (former v0.2.1, deferred) |
| THME-1002 | NVIDIA CUDA Driver API Backend via Pure Rust `cudarc` | Sprint 10 (former v0.2.1, deferred) |
| THME-1004 | Apple Metal 3 Performance Shaders Backend | Sprint 10 (former v0.2.1, deferred) |
| THME-1005 | Fused BitNet b1.58 GEMV / GEMM Kernel | Sprint 10 (former v0.2.1, deferred) |
| THME-1006 | Fused RMSNorm Micro-Kernel | Sprint 10 (former v0.2.1, deferred) |
| THME-1007 | Fused SwiGLU Activation Micro-Kernel | Sprint 10 (former v0.2.1, deferred) |
| THME-1008 | Fused Rotary Position Embedding (RoPE) Kernel | Sprint 10 (former v0.2.1, deferred) |
| THME-1009 | CLI Subcommand `thme run` Direct Inference | Sprint 10 (former v0.2.1, deferred) |
| THME-1101 | Linux Direct Asynchronous I/O Engine (`io_uring`) | Sprint 11 (former v0.2.2, deferred) |
| THME-1102 | Windows Direct Asynchronous I/O Engine (Overlapped I/O) | Sprint 11 (former v0.2.2, deferred) |
| THME-1103 | Sequential Layer Streaming & Eviction Engine | Sprint 11 (former v0.2.2, deferred) |
| THME-1104 | Double-Buffered Prefetch Pipeline | Sprint 11 (former v0.2.2, deferred) |
| THME-1105 | Pinned Memory Staging Buffers (`mlock` / `VirtualLock`) | Sprint 11 (former v0.2.2, deferred) |
| THME-1106 | REA-LLM Tiered Heterogeneous Memory Placement Manager | Sprint 11 (former v0.2.2, deferred) |
| THME-1107 | Dynamic VRAM Memory Accountant | Sprint 11 (former v0.2.2, deferred) |
| THME-1108 | 70B Model in 8 GB VRAM End-to-End Testbed | Sprint 11 (former v0.2.2, deferred) |
| THME-1109 | PCIe Bandwidth Saturation & Disk I/O Telemetry Monitor | Sprint 11 (former v0.2.2, deferred) |
| THME-1201 | Intra-Node Tensor Parallelism (TP) Projection Splitter | Sprint 12 (former v0.2.3, deferred) |
| THME-1202 | Inter-Node Pipeline Parallelism (PP) & Layer Partitioner | Sprint 12 (former v0.2.3, deferred) |
| THME-1203 | 1F1B (One-Forward-One-Backward) Pipeline Bubble Minimizer | Sprint 12 (former v0.2.3, deferred) |
| THME-1204 | Pure Rust Asynchronous TCP Ring All-Reduce | Sprint 12 (former v0.2.3, deferred) |
| THME-1205 | Hardware RDMA / RoCE Network Fallback Engine | Sprint 12 (former v0.2.3, deferred) |
| THME-1206 | Multicast / mDNS Cluster Node Discovery & Capability Handshake | Sprint 12 (former v0.2.3, deferred) |
| THME-1207 | Continuous 100 ms Heartbeat & Dead-Node Watchdog | Sprint 12 (former v0.2.3, deferred) |
| THME-1208 | Automated Dynamic Layer Re-Sharding Failover | Sprint 12 (former v0.2.3, deferred) |
| THME-1209 | CLI Subcommand `thme cluster` & Cluster Formation Wizard | Sprint 12 (former v0.2.3, deferred) |
| THME-1301 | High-Throughput Asynchronous WebSocket Event Streaming Daemon | Sprint 13 (former v0.2.4, deferred) |
| THME-1302 | JSON-RPC 2.0 Method Router & Remote Protocol Dispatcher | Sprint 13 (former v0.2.4, deferred) |
| THME-1303 | Cryptographic HMAC-SHA256 Surgery Recipe Attestation | Sprint 13 (former v0.2.4, deferred) |
| THME-1304 | Autonomous CLI Agent Adapter | Sprint 13 (former v0.2.4, deferred) |
| THME-1305 | CI/CD Orchestration Harness Adapter | Sprint 13 (former v0.2.4, deferred) |
| THME-1306 | Desktop Studio Remote Daemon Connection Bridge | Sprint 13 (former v0.2.4, deferred) |
| THME-1307 | IDE & GitHub Copilot Extension Bridge | Sprint 13 (former v0.2.4, deferred) |
| THME-1308 | Real-Time System Telemetry & Resource Pressure Broadcasting | Sprint 13 (former v0.2.4, deferred) |
| THME-1309 | CLI Subcommand `thme daemon` & Universal Daemon Controller | Sprint 13 (former v0.2.4, deferred) |

---

## Digest

Across the 11 milestones (`v0.2.0` through `v0.3.0`):

- **`consistent`**: 10 (`v0.2.0`, `v0.2.2`, `v0.2.3`, `v0.2.4`, `v0.2.5`, `v0.2.6`, `v0.2.7`, `v0.2.8`, `v0.2.9`, `v0.3.0`)
- **`no tickets`**: 1 (`v0.2.1`)
- **`roadmap too optimistic`**: 0
- **`roadmap too pessimistic`**: 0
