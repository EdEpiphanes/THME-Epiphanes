# Changelog

All notable changes to **THME (Topological Heterogeneous Manifold Engine)** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
