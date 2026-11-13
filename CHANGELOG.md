# Changelog

All notable changes to **THME (Topological Heterogeneous Manifold Engine)** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - v0.2.0 Measured Core

### License
- **THME is now dual-licensed: AGPL-3.0-only or a commercial license from Epiphanes**
  ([LICENSING.md](LICENSING.md)). Source obtained before 2026-11-13, up to and including commit
  `d161bdb`, stays available under Apache-2.0. Contributions are accepted under the contributor
  terms in CONTRIBUTING.md, and `deny.toml` keeps dependencies permissive.

### Added
- Reference CPU forward pass (`src/inference/`): LLaMA family and GPT-NeoX (Pythia), AVX2/FMA
  kernels with a portable fallback, layer schedules and per-depth readouts.
- Measured precision round trips for FP16, BF16, Q8_0, Q4_0 and ternary (TQ1_0), using the
  exporter's own block encoders and the parser's decoders.
- `scripts/prepare_eval_tokens.py`: WikiText-2 test tokenization with corpus, tokenizer and
  token-file hashes.
- `examples/precision_sweep.rs` (uniform, per-role, per-layer, recipe and random-placement
  variants) and `examples/depth_probe.rs` (exit readouts, repeated and skipped layers).
- `scripts/generate_charts.py`: light and dark SVG charts rendered only from measured reports.
- `thme validate --report <verification_report.json>` gates a measured report.
- `thme evaluate MODEL --recipe R --tokens T`: measures any recipe with the forward pass; refuses
  operations it cannot run unless `--allow-partial`, which lists them as not evaluated.
- `thme search MODEL --tokens T --budget B`: mixed-precision search on a cached, measured cost
  table with greedy allocation, whole-candidate refinement, a calibration/validation split,
  random matched-placement controls and `--emit-recipe`.
- Pluggable co-surgery quality evaluation (`QualityEvaluator`): `thme co-surgery MODEL --tokens`
  uses `MeasuredEvaluator` (reference forward pass); the analytic proxy remains the labelled fallback.
- BYOA in recipes: `WeightContract` classification, `RotatedQuantize` (Hadamard rotate, quantize,
  rotate back), built-ins `thme.hadamard_ternary` and `thme.hadamard_int4`, `thme evaluate --algorithms`.
- `thme search --pareto-budgets` (measured size-vs-quality front) and `--test-tokens` (held-out slice);
  Pareto chart and dossier section.
- Studio recipe workbench in the Live 3D tab: recipe-applied runs and a measured gate.
- Studio **Live 3D** tab (`src/gui/live.rs`): real-time 3D view of any LLaMA-family or GPT-NeoX
  model during evaluation, driven by a forward-pass probe (`src/inference/probe.rs`) that records
  per-tensor output activity, residual updates, dead MLP units and non-finite values. Includes a
  software 3D renderer (orbit, zoom, hover inspection) and a labelled synthetic demo model.
- Manga brand system: mascot, README banner with screentone shading and speed lines, ink icon
  (Studio window icon), default **Manga Night** Studio theme with a paper **Manga Ink** variant, ink-framed charts, and a computed
  Calabi-Yau mark (`scripts/generate_brand_art.ps1`, `scripts/generate_logo.py`).
- Studio capture mode (`THME_GUI_CAPTURE_DIR`) and `scripts/capture_studio.ps1`: real
  screenshots of every Studio tab under `assets/screenshots/`.
- Measured reports for SmolLM-135M and Pythia-14M under `docs/benchmarks/`.
- `docs/research/THME_measured_findings_2026-11.md` and `docs/subsystems.md`.

### Changed
- Production workflow: pinned toolchain (`rust-toolchain.toml`); CI with hygiene, lint, three-OS
  tests, MSRV, measured-claims regeneration, cargo-deny and a non-blocking latest-stable preview;
  hardened release pipeline (tag/version and changelog checks, gates before build, four targets,
  per-file and combined checksums); Dependabot; PR and issue templates (bug, feature, experiment);
  CODEOWNERS; `docs/WORKFLOW.md`; active backlog in BACKLOG.md with `scripts/sync_github_backlog.py`.
- `examples/production_verification.rs` runs a real perplexity gate on the source weights, the
  re-imported GGUF export and the surgery recipe. Without weights and tokens it reports
  NOT EVALUATED instead of a number, and it exits non-zero when a candidate fails.
- `scripts/generate_dossier.py` renders committed reports only and refuses to run without a
  measured evaluation.
- README rewritten around measured results, an evidence table and real screenshots.
- ROADMAP restructured: v0.2.0 Measured core, v0.2.1 Recipe and BYOA lab, v0.2.2 Formats and
  calibration, v0.2.3 Recovery, v0.2.4 Runtime. GPU, streaming, clustering and gateway move to v0.3.
- Tensor role detection recognises GPT-NeoX names and a fused `AttentionQkv` role.
- `ModelEvaluator::evaluate_weights` documented as an analytic proxy.
- `thme validate` and the MCP quality tool label supplied perplexities as not measured by THME.

### Removed
- Hand-written benchmark figures (H100, RTX 4090, M3 Max, MI300X, Xeon tables and the
  LLaMA-3.2-1B row); none were measured.
- Drawn showcase GIFs and their generators (`examples/generate_showcase_gifs.rs`,
  `scripts/generate_showcase_gifs.py`, `scripts/generate_live_benchmark_gif.py`); they did not
  depict the application.

### Fixed
- SafeTensors tensors are ordered by file offset; the safetensors crate returns hash order, which
  made random-control placements and cached search plans irreproducible across runs.
- MSRV corrected to 1.89, the highest `rust-version` in the locked dependency tree (the code itself
  already needed 1.87 for `usize::is_multiple_of`).
- The previous quality gate compared a weight sample with a scaled copy of itself, so it could
  not fail.
- Studio: glyphs the font cannot render (arrows, combining marks) replaced; Layer Tweaker dataflow
  cards no longer overlap their own labels; the co-surgery state machine no longer unwraps.

---
## [0.1.9] - 2026-11-11

### Added
- **Closed-Loop Teacher-Student Co-Surgery Baseline (`src/surgery/`)**:
  - Autonomous closed-loop state machine & controller (`src/surgery/closed_loop.rs`) governing iterative `ProfileSensitivity` $\to$ `SynthesizeRecipe` $\to$ `ApplySurgery` $\to$ `ValidateQuality` $\to$ `EvaluatePareto` $\to$ `Commit`/`Rollback` state transitions.
  - Teacher LLM topology analysis prompt synthesizer (`src/surgery/prompt_synth.rs`) formatting top-K layer sensitivity, SVD decay rates, and memory reach profiles.
  - Headless JSON-RPC & subprocess Teacher agent communication bridge (`src/surgery/agent_bridge.rs`) for external agent orchestration and mock testing.
  - Multi-objective 3D Pareto optimizer & hypervolume metric (`src/surgery/pareto_3d.rs`) tracking trade-offs across $\Delta\text{PPL}$, size reduction ratio, and speedup multipliers.
  - Automated self-healing parameter relaxation heuristic (`src/surgery/relaxation.rs`) backing off aggressive precision on sensitive layers.
  - Anti-degradation quality gate watchdog & rollback coordinator (`src/surgery/watchdog.rs`) restoring verified model checkpoints.
  - CLI subcommand `thme co-surgery` (`src/cli/co_surgery.rs`) with real-time per-step terminal telemetry stream and JSON reporting.
  - Desktop Studio Tab 9 Closed-Loop Co-Surgery dashboard (`src/gui/app.rs`) with budget controls, live telemetry logs, and Pareto checkpoint history.

---

## [0.1.8] - 2026-11-11

### Added
- **Binary Quantization Exporter Suite (`src/exporter/`)**:
  - Vectorized BitNet b1.58 ternary packer and unpacker (`src/exporter/bitnet_pack.rs`) with 4 weights per byte and scale product de-quantization.
  - Pure-Rust GGUF v3 binary writer (`src/exporter/gguf_writer.rs`) with 32-byte alignment and metadata serialization.
  - GGUF architecture metadata reconstitution engine (`src/exporter/metadata.rs`) across all 8 paradigm architectures.
  - Pure-Rust SafeTensors binary serializer (`src/exporter/safetensors_writer.rs`) with 8-byte length prefix and JSON metadata table.
  - Low-memory bounded streaming chunked file writer (`src/exporter/streaming_writer.rs`) with concurrent SHA-256 calculation.
  - Calibrated per-channel and per-group MSE-optimal scale search (`src/exporter/calibration.rs`).
  - CLI subcommand `thme export` (`src/cli/export.rs`) and Desktop Studio export wizard controls.
  - Comprehensive downstream runtime compatibility test suite (`tests/test_exporter_compatibility.rs`).

---

## [0.1.7] - 2026-11-10

### Added
- **Hardware-Calibrated Roofline & Micro-Kernel Profiler (`src/profiling/`)**:
  - Hardware specifications database for H100 SXM5, RTX 4090, Apple Silicon, AMD MI300X, and Intel Xeon (`src/profiling/hardware.rs`).
  - Operational arithmetic intensity analysis for linear GEMM, SDPA, SwiGLU FFN, and FlashAttention-2 (`src/profiling/intensity.rs`).
  - 4-tier cache simulator (L1, L2, L3, DRAM) and working set eviction model (`src/profiling/cache_sim.rs`).
  - Empirical roofline boundary curve generator and memory vs compute classifier (`src/profiling/roofline.rs`).
  - Dynamic TDP thermal and DVFS throttling projection engine (`src/profiling/thermal.rs`).
  - CLI subcommand `thme roofline` (`src/cli/roofline.rs`), MCP tool `thme_profile_roofline`, and Tab 4 GUI deck.

---

## [0.1.6] - 2026-11-10

### Added
- **Speculative Decoding & Draft Model Synthesizer Engine (`src/speculative/`)**:
  - `DraftSynthesizer` (`src/speculative/synthesizer.rs`): Automated draft model extraction supporting Prefix Layers, Strided Subsampling, Low-Rank Projection, and BitNet b1.58 Ternary Discretization.
  - `SpeculativeEngine` (`src/speculative/engine.rs`): Multi-token speculative decoding execution and verification engine modeling rejection sampling, token acceptance probability ($\alpha_{\text{spec}}$), tokens emitted per pass ($\mathbb{E}[k] + 1$), latency cost ratio ($r = \frac{T_{\text{draft}}}{T_{\text{target}}}$), and effective wall-clock latency speedup ($S$).
  - `SpeculativeTreeTopology` (`src/speculative/tree.rs`): Multi-branch tree speculation topology with 2D causal tree attention mask synthesis (Medusa / Eagle style) and longest accepted path selection.
- **CLI `thme speculate` (`src/cli/speculate.rs`, `src/cli/mod.rs`)**:
  - Command-line tool for speculative decoding evaluation with lookahead horizon ($\gamma$), temperature, BitNet draft toggle, tree decoding, JSON reporting, and draft model export.
- **Model Context Protocol (MCP) Tool `thme_simulate_speculative_decoding` (`src/cli/mcp.rs`)**:
  - Exposes programmatic speculative decoding simulation to AI coding agents in VS Code and GitHub Copilot.
- **Desktop Studio GUI Speculative Cockpit (`src/gui/app.rs`)**:
  - Added interactive Speculative Decoding & Multi-Token Draft Synthesizer Deck in Tab 4 (Full Benchmarks) with live sliders, telemetry stat cards, simulation runner, and one-click draft model registration.
- **Declarative Surgery Recipe Support (`src/surgery/recipe.rs`)**:
  - Added `TweakOperation::ExtractDraftModel` to execute draft model extraction within Teacher LLM recipes.
- **Speculative Decoding Test Suite (`tests/test_speculative_decoding.rs`)**:
  - 5 comprehensive tests for prefix extraction, ternary BitNet draft models, speculative speedup simulation, tree topology attention masks, and recipe surgery. Total test count expanded to 81 passing tests.

---

## [0.1.5] - 2026-11-10

### Added
- **Universal 8 Specialized AI Model Paradigm Converter (`src/ir/paradigm.rs`)**:
  - Full support for ingesting, detecting, and cross-converting models across 8 specialized AI paradigms:
    - **LLM** (Large Language Model - Autoregressive Causal Transformer)
    - **LCM** (Large Concept Model - SONAR Concept Space & Latent Diffusion)
    - **LAM** (Large Action Model - Multimodal Intent Perception & Neuro-Symbolic Memory)
    - **MoE** (Mixture of Experts - Dynamic Router Gating & Top-K Experts)
    - **VLM** (Vision-Language Model - ViT Vision Towers & Cross-Modal Projectors)
    - **SLM** (Small Language Model - GQA, Manifold Pruning & BitNet b1.58 Ternary Discretization)
    - **MLM** (Masked Language Model - Token Masking & Bidirectional Context Highway)
    - **SAM** (Segment Anything Model - Prompt Encoders, Feature Correlation & Mask Decoders)
  - Mathematical tensor transformation engine supporting arbitrary cross-conversion between all 8 paradigms.
  - Sparse upcycling engine partitioning dense MLPs into parallel expert matrices and synthesized router gating tensors.
  - BitNet b1.58 ternary discretization ($\{-1, 0, +1\}$) delivering up to 85.3% storage savings for edge deployment.
- **CLI `thme convert` (`src/cli/convert.rs`, `src/cli/mod.rs`)**:
  - Command-line utility to cross-convert any loaded model or baseline across all 8 paradigms with customizable expert count, Top-$K$ routing, target hidden dimensions, layer counts, BitNet ternary toggle, and declarative recipe saving.
- **Model Context Protocol (MCP) Tool `thme_convert_paradigm` (`src/cli/mcp.rs`)**:
  - Exposes programmatic paradigm conversion to AI coding agents in VS Code and GitHub Copilot.
- **Desktop Studio Paradigm Converter UI (`src/gui/app.rs`)**:
  - Integrated into Tab 1 (Models Manager) with detected active paradigm badge, 8-paradigm selector grid, parameter controls, live conversion report, and immediate benchmarkability.
- **Declarative Surgery Recipe Operation (`src/surgery/recipe.rs`)**:
  - Added `TweakOperation::ConvertParadigm` enabling declarative recipe-driven cross-paradigm surgery.
- **Paradigm Conversion Test Suite (`tests/test_model_paradigm_conversion.rs`)**:
  - 5 comprehensive unit and integration tests including full 8-model round-trip testing. Total test suite expanded to 76 passing tests.
- **PyTorch & CUDA Ecosystem Interoperability Bridge (`src/interop/mod.rs`, `src/interop/pytorch.rs`)**:
  - Standalone PyTorch CUDA script generator (`train_cuda.py`) integrating THME's Pareto precision allocations with standard PyTorch 2.x, CUDA Tensor Cores, PEFT LoRA, QLoRA (BitsAndBytes 4-bit), Unsloth fast GPU kernels, and DeepSpeed ZeRO-2/3.
  - Hugging Face `config.json` generator matching `transformers` causal LM architecture specs for LLaMA, Mistral, Gemma, Qwen, and DeepSeek.
  - Native pure Rust PEFT LoRA adapter ingestion and zero-copy in-engine low-rank weight merging: $W_{\text{effective}} = W_0 + \frac{\alpha}{r}(B \times A)$.
- **CLI `thme export` / `thme interop` (`src/cli/interop.rs`)**:
  - Command-line generation of PyTorch CUDA harnesses, QLoRA scripts, and Hugging Face configuration files.
- **PyTorch & CUDA Studio Deck in GUI (Tab 7 & Tab 8)**:
  - 1-click generation of PyTorch CUDA training scripts and Hugging Face configs directly from the desktop studio.
- **Universal Model Context Protocol (MCP) Server (`thme mcp`, `src/cli/mcp.rs`)**:
  - Implements standard 2024-11-05 JSON-RPC protocol over stdin/stdout, allowing AI coding agents (VS Code Copilot, Claude Desktop, Cursor, Antigravity) to inspect and optimize models programmatically.
  - Exposes 7 tools: `thme_inspect_model`, `thme_auto_decide`, `thme_apply_surgery`, `thme_validate_quality`, `thme_simulate_token_flow`, `thme_run_experiment`, `thme_micro_train`.
- **VS Code & GitHub Copilot Extension (`editors/vscode/`)**:
  - Packaged extension (`thme-vscode-0.1.5.vsix`) with Activity Bar sidebar views for Active Model & Layers, Pareto Precision Plan, REA-LLM & EWHA Telemetry, and Copilot Live Tools.
  - Native Language Model Tools integration via `vscode.lm.registerTool`: `thme_copilot_inspect`, `thme_copilot_surgery`, and `thme_copilot_validate` enabling conversational Copilot debugging and surgery.
  - Auto-discovery workspace configuration `.vscode/mcp.json`.
- **Interoperability Test Suite (`tests/test_interop.rs`)**:
  - Tests for PyTorch script generation (LoRA, QLoRA, Unsloth, DeepSpeed ZeRO-2/3), Hugging Face config generation, and PEFT LoRA mathematical weight merging. Total test count expanded to 71 passing tests.
- **High-Definition Showcase & Marketing GIF Suite (`assets/`, `scripts/generate_showcase_gifs.py`)**:
  - `assets/live_neural_benchmark.gif` (Updated to v0.1.5): 920x520 widescreen multi-panel demonstration of all 9 studio tabs, 8x8 weight slice matrix, 16-bar spectral kurtosis histogram, and 13-stage Transformer forward token flow.
  - `assets/scientific_math_optimization.gif`: Demonstrates scientific and mathematical formulations in action (Fast Walsh-Hadamard Transform orthogonal rotation, BitNet b1.58 ternary discretization, EWHA energy accounting, and Pareto DP optimization) delivering a $+2.18\times$ throughput speedup, $-68.5\%$ energy reduction, and near-zero quality degradation ($\Delta PPL \le 0.032$).
  - `assets/layer_deconstruction_surgery.gif`: Showcases deep layer anatomy deconstruction across 7 sub-block nodes (Input RMSNorm, Sequence Mixer, Residual Highway 1, Post-Mixer RMSNorm, SwiGLU MLP, Residual Highway 2, and Early-Exit Gate) with live declarative recipe execution.
  - `assets/pytorch_cuda_interop.gif`: Visualizes automated `train_cuda.py` script generation, QLoRA 4-bit BitsAndBytes setup, and in-engine PEFT LoRA adapter zero-copy weight merging ($W = W_0 + \frac{\alpha}{r} B \times A$).

### Changed
- **Professional Engineering UI Refactoring**:
  - Removed all emojis from visual elements across the Native Desktop Studio GUI, CLI, documentation, and VS Code extension in favor of clean, high-precision engineering typography.

---

## [0.1.4] - 2026-11-10

### Added
- **Empirical Quality Gate & Perplexity Verification Bridge (`src/quality/evaluator.rs`)**:
  - Analytical and external evaluation harness supporting standard benchmark corpora (WikiText-2, C4, Penn Treebank, Synthetic High-Entropy).
  - Perplexity degradation computation: $\Delta PPL = PPL_{\text{candidate}} - PPL_{\text{baseline}}$ and percentage representation distortion.
  - Automated rollback quality gate: asserts $\Delta PPL \le \epsilon$, returning `PerplexityGateDecision::Accept` or `PerplexityGateDecision::RollbackRequired` with actionable guidance.
  - External runner bridge with stdout parser for external harnesses like `llama-perplexity`.
- **CLI Validate Enhancements (`src/cli/validate.rs`)**:
  - Added `--corpus`, `--baseline-ppl`, `--candidate-ppl`, `--max-ppl-delta`, and `--json` flags to both `thme validate` and `llm-tweaker validate`.
  - Emits formatted quality summaries or structured JSON payloads for CI pipelines and automated Teacher LLM surgery loops.
- **Robust Orthogonal Rotation Slicing (`src/transform/byoa.rs`)**:
  - Clamped non-power-of-2 `block_size` to the nearest lower power of 2, preventing silent bypass of the Fast Walsh-Hadamard Transform butterfly loop.
- **Quality Evaluator Test Suite (`tests/test_quality_evaluator.rs`)**:
  - Verified cross-entropy and perplexity scoring across corpora, gate acceptance/rejection boundaries, and empty weight error handling.

---

## [0.1.3] - 2026-11-10

### Added
- **Multi-Format Dataset Ingestion & Fast Tokenizer (`src/dataset/mod.rs`)**:
  - Ingests and normalizes arbitrary training, fine-tuning, and calibration datasets from plain text (`.txt`), JSON Lines (`.jsonl`), JSON arrays (`.json`), and CSV/TSV files.
  - Native multi-schema JSON parsers auto-detecting OpenAI chat messages (`[{"role": "...", "content": "..."}]`), Alpaca instructions (`instruction`, `input`, `output`), ShareGPT formats, and raw text records.
  - High-speed Fowler-Noll-Vo hash projection tokenizer with length distribution statistics, vocabulary diversity estimations, and batch slicing (`get_batch`, `get_flat_batch`).
- **Non-Conventional Experimentation Laboratory (`src/experiment/mod.rs`)**:
  - Research suite for mathematical exploratory transformations directly applied to model weights:
    1. *Stochastic Noise Injection*: Simulates analog hardware drift, thermal fluctuations, and low-voltage DRAM margins.
    2. *Spectral Energy Reshaping*: Warps singular value spectra via power exponent $\sigma_i^\gamma$ to compress or accentuate representations.
    3. *Complex Phase Modulation*: Unitary complex angle rotation $\cos(\omega j + \phi)$ across weight channels.
    4. *Continuous Activation Warping*: Non-linear continuous manifold interpolation between SwiGLU, GELU, and harmonic sine functions.
    5. *Stochastic DropBlock Sparsity*: Structured block-level zeroing for hardware-aligned sparsity benchmarking.
  - Analytical metrics calculation tracking original vs. transformed L2 norm, MSE, SNR (dB), and sparsity delta.
- **On-Device Micro-Trainer & Adaptation Engine (`src/trainer/mod.rs`)**:
  - Lightweight parameter training and calibration loop for adapting model weights to ingested datasets.
  - Multiple loss objectives: Autoregressive Cross-Entropy, Teacher Knowledge Distillation (with temperature $\tau$), and EWHA Energy-Regularized loss (with penalty parameter $\lambda_E$).
  - Optimizers: AdamW with decoupled weight decay, SignSGD (for 1-bit / BitNet b1.58 ternary weights), and standard SGD with momentum.
  - Real-time step telemetry recording Loss, Perplexity ($PPL = \exp(\mathcal{L})$), throughput (tokens/sec), and estimated energy (mJ).
- **Tab 8 Developer Lab & Training Studio (`src/gui/app.rs`)**:
  - Dedicated GUI tab providing dataset file loading, synthetic calibration generation, live sample previews, interactive experimentation sliders, target layer selection, step/epoch training execution, and live convergence telemetry.
- **Layer Anatomy Deconstruction & Architecture Disassembly (`src/ir/layer_decomp.rs`)**:
  - Deep inspection of internal layer composition (LayerNorm, QKV Attention, Attention Out, MLP Gate/Up, MLP Down, Post-Mixer Norm, Residual Connections).
  - Component parameter counting, memory footprint calculation, active routing indicator, and hybrid conversion topology dispatch.
  - Visual Anatomy & Dataflow Map in Tab 2 of Desktop Studio GUI.
- **Bring Your Own Algorithm (BYOA) Engine (`src/transform/byoa.rs`)**:
  - Full parametric custom algorithm definition (`AlgorithmDefinition`), category classifications (`Quantization`, `LinearTransformation`, `SparsityPruning`, `SequenceMixer`, `FeatureDistillation`), and transformation rules (`QuantizeCodebook`, `AbsmeanTernary`, `OrthogonalRotation`, `StructuredSparsity`, `SmoothQuantOutlier`, `AffineProject`).
  - Extensible `AlgorithmRegistry` pre-populated with 5 verified presets (BitNet b1.58, 2:4 Structured Sparsity, Walsh-Hadamard WHT, SmoothQuant, Gated DeltaNet).
  - Dynamic in-place weight transformation (`apply_to_weights`) computing MSE, L2 norms, SNR, and sparsity ratio.
  - JSON specification schema supporting import, export, and runtime registration without rebuilding core engine.
  - Full integration into `AutoDecider` with quality gate checking (`decide_for_parameter_with_registry`).
- **Open Weights Rebuilder & Architectural Scaffold Engine (`src/ir/rebuilder.rs`)**:
  - Topological layer and architecture blueprints (`ArchitectureBlueprint`, `LayerBlueprint`) supporting hybrid sequences (GQA Attention, Gated DeltaNet, Phase Recurrence, Mamba-2, Sliding Window).
  - Scratch model synthesis (`ModelRebuilder::build_scaffold`) producing fully populated `ModelIR` tensors.
  - Open-weights remapping engine (`ModelRebuilder::rebuild_from_open_weights`) transferring weights from source checkpoints (LLaMA, Mistral, Qwen, DeepSeek) into new topologies with dimensional projection (`RemapStrategy::LowRankProject`, `ZeroPad`, `Truncate`).
- **REA-LLM & EWHA Telemetry Framework (`src/transform/rea.rs`)**:
  - Implements the Execution-Weighted Heterogeneous Allocation mathematical framework from the THME research specification.
  - Multi-path early exit routing profile ($80/20$ scenario) computing exact layer reach probabilities $q_l$.
  - Formal channel representation mixes $(\alpha_l \text{ Binary}, \beta_l \text{ Ternary}, \gamma_l \text{ Phase}, \text{FP16 fallback})$.
  - Analytic memory traffic accountant ($W_{\text{bytes/token}} \approx \sum_l q_l \cdot W_{l,\text{active}}$) and per-token energy accountant ($E_{\text{token}} = \sum_l q_l \cdot E_{l,\text{active}}$).
- **Desktop Studio GUI Enhancements (`src/gui/app.rs`)**:
  - **Tab 0 (Master Dashboard)**: Unified 4-card cockpit with real-time REA-LLM depth fraction ($\bar{f}$), memory traffic, and energy indicators.
  - **Tab 1 (Models Manager)**: Added "Rebuild Architecture From Scratch" wizard modal for assembling custom layer sequences, setting checkpoint depths, and remapping active model weights.
  - **Tab 3 (Distillation & Matrix Engine)**: Interactive BYOA Registry table, Custom Algorithm Designer form, one-click JSON Import/Export, and live 8x8 weight slice tester.
  - **Tab 4 (Full Benchmarks)**: Dedicated REA-LLM EWHA Telemetry and layerwise reach/representation breakdown grid.
- **Declarative Surgery Extensions (`src/surgery/recipe.rs`)**:
  - Added `TweakOperation::ApplyCustomAlgorithm` and `TweakOperation::RebuildArchitecture` to the declarative surgery pipeline with full audit trails.
- **Comprehensive Integration Test Suites**:
  - `tests/test_byoa.rs`: Built-in algorithm verification, JSON import/export roundtrip, transformation mechanics, and AutoDecider custom algorithm dispatch.
  - `tests/test_rea_llm.rs`: Reach profile validation, representation mix properties, memory traffic & energy calculations, and constrained EWHA optimization.
  - `tests/test_rebuilder.rs`: Scaffold generation, open-weights remapping across dimensions, and declarative surgery recipe execution.

---

## [0.1.2] - 2026-11-10

### Added
- **Full GGUF v2/v3 Quantization Matrix (`GgmlType`)**: Added complete type enumerations for `F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, K-quants (`Q2_K`..`Q8_K`), and I-quants (`IQ2_XXS`..`IQ4_XS`, `IQ1_M`).
- **Zero-Allocation Block Decoders (`GgufBlockDecoder`)**: Unpacks raw block payloads for `Q8_0`, `Q4_0`, `Q4_1`, and `Q5_0` with exact floating-point scale and offset reconstitution.
- **Resilient GGUF Stream Recovery**: Truncated or corrupt GGUF files now recover all intact tensors parsed up to the point of corruption rather than terminating with panic.
- **Infallible Parser Fuzz Suite (`tests/test_fuzz_parsers.rs`)**: Exhaustive prefix-truncation and bit-mutation property tests guaranteeing zero parser panics on corrupted or malicious inputs.
- **GGUF Quant Unit Tests (`tests/test_gguf_quant_types.rs`)**: Validates block unpackers and real weight profile statistics on quantized GGUF tensors.
- **High-Definition Studio Showcase GIF (`assets/live_neural_benchmark.gif`)**: 920x520 widescreen multi-panel animation showcasing all 8 studio tabs, 8x8 weight slice matrix heatmap, 16-bar distribution histogram, and 13-stage Transformer forward token stream.

### Fixed & Improved
- **Weighted Majority Layer Precision Reporting**: `src/allocator/pareto.rs` now aggregates format distribution by parameter volume (`numel`) rather than indexing the first tensor (`tensors[0]`).
- **Fast Walsh-Hadamard Dimension Validation**: `MatrixMath::fast_walsh_hadamard_transform` now explicitly asserts and returns error for non-power-of-two dimensions instead of silent no-ops.
- **Unicode & Mojibake Cleansing**: Eliminated double-UTF8 mojibake sequences across GUI autotuner telemetry, Greek mathematical symbols ($\alpha, \beta, \gamma, \mu, \sigma, \tau, \epsilon$), and iconography.
- **Dependency Hygiene**: Removed unused dependencies (`thiserror`, `serde_yaml`, `indicatif`) from `Cargo.toml`.

### Security & Correctness
- Added explicit `// SAFETY:` justifications on all memory-mapped file operations in `src/parser/gguf.rs` and `src/parser/safetensors.rs`.
- Added strict length bounds checking on GGUF string and array headers to guard against allocation exhaustion attacks.

---

## [0.1.1] - 2026-11-10

### Added
- **Formal CI Matrix**: Cross-platform GitHub Actions workflow covering `ubuntu-latest`, `windows-latest`, and `macos-latest` with `rustfmt`, `clippy -D warnings`, and full test execution.
- **Automated Path-Leak Guard**: CI check rejecting absolute developer paths or user profile patterns across git-tracked assets.
- **Release Automation Workflow**: Tag-driven (`v*`) stripped cross-platform binary releases.
- **JSON Schema for Surgery Recipes**: Defined in `docs/schemas/surgery-recipe-v1.json` with runnable reference recipe `examples/recipes/hybrid_mamba_deltanet_recipe.json`.
- **Engineering Playbook & AGENTS.md**: Standardized non-negotiables, infallible parsing requirements, and release gate protocols.
- **Roadmap & Backlog Ledger**: Full milestone tracking in `ROADMAP.md` and actionable task breakdown in `BACKLOG.md`.

### Changed
- Project rebranded to **THME** by **Epiphanes**, retaining `llm-tweaker` as backwards-compatible CLI alias.
- Added release profile symbol stripping (`strip = "symbols"`, `debug = false`) in `Cargo.toml` and `.cargo/config.toml`.
- Resolved all Clippy linter warnings across library, binaries, and integration test suites.

---

## [0.1.0] - 2026-11-10

### Added
- **Universal Model Importer**: Zero-copy memory-mapped SafeTensors and GGUF v2/v3 parsers with automatic fallback architecture deduction.
- **Topological Sensitivity Profiling**: SVD energy decay, condition number analysis, and Fisher information proxy estimation.
- **Pareto Precision Allocator**: Dynamic programming budget optimization under quality gates ($Q_{\text{candidate}} \ge Q_{\text{baseline}} - \epsilon$).
- **Live Desktop Studio GUI**: Native 60 FPS `egui`/`eframe` dashboard with 5 distinct color themes (Obsidian Cyan, Cyberpunk Neon, Matrix Emerald, Nordic Frost, Solarized Amber).
- **Matrix & Distillation Engine**: Truncated SVD, Walsh-Hadamard orthogonal rotation, SmoothQuant per-channel scaling, BitNet b1.58 ternary quantization, and 2:4 structured sparsity.
- **AutoDecider**: Parameter-level algorithm selector optimizing bit-width, energy preservation, and compression ratio.
- **Token Telemetry Simulator**: 13-stage Transformer forward propagation simulator with real-time throughput metrics, batch scaling, and KV-cache bounds.
- **History & Audit Log**: Immutable checkpoint rollback engine and cryptographically indexed audit stream.
- **Teacher LLM Surgery**: Declarative JSON recipe executor for layer replacement, sequence mixer swapping, and weight pruning.
