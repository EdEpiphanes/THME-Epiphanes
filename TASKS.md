# THME Task & Workflow Verification Ledger

> **Topological Heterogeneous Manifold Engine (THME)** by **Epiphanes**  
> Backwards-Compatible CLI Alias: `llm-tweaker`  
> Repository: `https://github.com/EdEpiphanes/THME-Epiphanes`  
> Last Updated: 2026-11-10

---

## 1. Quick Verification Cheatsheet

Run these commands anytime to verify the health, correctness, and hygiene of the workspace:

```bash
# 1. Format verification (strictly zero diffs)
cargo fmt --check

# 2. Strict linter (zero warnings tolerated)
cargo clippy --all-targets -- -D warnings

# 3. Full test suite execution (all 71 integration & unit tests)
cargo test

# 4. Subsystem-specific test runs:
cargo test --test test_interop         # PyTorch, CUDA & PEFT LoRA adapter bridge
cargo test --test test_dataset         # Multi-format dataset ingestion & tokenization
cargo test --test test_experiment      # Non-conventional experimentation laboratory
cargo test --test test_trainer         # On-device micro-training & adaptation engine
cargo test --test test_byoa            # BYOA registry & weight transformation
cargo test --test test_rea_llm         # REA-LLM EWHA routing & energy accounting
cargo test --test test_rebuilder       # Open-weights architecture rebuilder & surgery
cargo test --test test_layer_deconstruction # Layer anatomy deconstruction & routing
cargo test --test test_gguf_quant_types # GGUF block decoders & quant matrix
cargo test --test test_fuzz_parsers    # Infallible prefix-truncation parser fuzz suite
cargo test --test test_matrix_distill  # SVD, Hadamard, SmoothQuant, BitNet, Sparsity
cargo test --test test_ir              # Model IR, layer indexing, Pareto planner
cargo test --test test_safetensors     # SafeTensors memory-mapped loader

# 5. Automated Path-Leak Guard (must return 0 matches)
git grep -inE "(C:\\\\Users\\\\[a-zA-Z0-9_-]+|/home/[a-zA-Z0-9_-]+)"
```

---

## 2. Completed Milestones & Verification Ledger

### ✅ v0.1.0 — Core Foundations & Studio GUI
- [x] **Zero-Copy SafeTensors & GGUF Parser**
  - **Source**: `src/parser/safetensors.rs`, `src/parser/gguf.rs`, `src/parser/mod.rs`
  - **Verification**: `cargo test --test test_safetensors && cargo test parser::gguf`
  - **Status**: Verified Complete. Memory-mapped (`memmap2`) header parsing without cloning weight buffers.
- [x] **Layer Sensitivity Profiler & Weight Statistics**
  - **Source**: `src/profiling/weight_stats.rs`, `src/profiling/sensitivity.rs`
  - **Verification**: `cargo test profiling::weight_stats`
  - **Status**: Verified Complete. Fast `f16`/`bf16` decoders, kurtosis, sparsity, outlier ratio, and 32-bin histograms.
- [x] **Pareto Precision Allocator & Dynamic Programming Engine**
  - **Source**: `src/allocator/pareto.rs`, `src/allocator/precision.rs`
  - **Verification**: `cargo test test_model_ir_and_pareto_planner`
  - **Status**: Verified Complete. Multi-constraint DP knapsack allocating FP16, INT8, INT4, Ternary under quality gate $\Delta Q \le \epsilon$.
- [x] **Matrix Distillation & Linear Transformation Suite**
  - **Source**: `src/transform/mod.rs`
  - **Verification**: `cargo test transform::tests`
  - **Status**: Verified Complete. Truncated SVD, Fast Walsh-Hadamard Transform (FWHT), SmoothQuant, BitNet b1.58 ternary discretization, and 2:4 structured sparsity.
- [x] **Native 60 FPS Desktop Studio GUI (`egui`/`eframe`)**
  - **Source**: `src/gui/app.rs`, `src/bin/gui.rs`
  - **Verification**: `cargo check --bin thme-gui`
  - **Status**: Verified Complete. Master Dashboard, 8 specialized tabs, 5 visual themes, 8×8 weight slice heatmaps, and interactive parameter controls.
- [x] **Live 13-Stage Token Flow Benchmarker**
  - **Source**: `src/profiling/token_flow.rs`
  - **Verification**: `cargo test profiling::token_flow`
  - **Status**: Verified Complete. Real-time token generator, KV-cache footprint calculator, and layer waterfall visualizer.
- [x] **Declarative Surgery Engine & Audit Log**
  - **Source**: `src/surgery/recipe.rs`, `src/surgery/mod.rs`
  - **Verification**: `cargo test surgery::recipe`
  - **Status**: Verified Complete. Reversible JSON surgery recipes, SHA-256 state tracking, and chronological audit trails.

---

### ✅ v0.1.1 — Path Hygiene, CI Matrix & Standards
- [x] **Rebranding to THME under Epiphanes**
  - **Source**: `Cargo.toml`, `AGENTS.md`, `README.md`, `src/bin/llm_tweaker.rs`
  - **Verification**: CLI alias `llm-tweaker` redirects cleanly to `thme`.
  - **Status**: Verified Complete.
- [x] **3-OS GitHub Actions CI Matrix**
  - **Source**: `.github/workflows/ci.yml`
  - **Verification**: Runs on `ubuntu-latest`, `windows-latest`, `macos-latest` with `cargo check`, `cargo test`, `cargo clippy -D warnings`, and `cargo fmt --check`.
  - **Status**: Verified Complete.
- [x] **Automated Path-Leak Guard in CI**
  - **Source**: `.github/workflows/ci.yml`
  - **Verification**: Regex search for Windows/Linux user paths rejecting matching commits.
  - **Status**: Verified Complete.
- [x] **Release Packaging & Binary Size Stripping**
  - **Source**: `Cargo.toml`, `.cargo/config.toml`, `.github/workflows/release.yml`
  - **Verification**: Release builds strip debug symbols; binaries adhere to `< 15 MB` (GUI) and `< 10 MB` (CLI).
  - **Status**: Verified Complete.
- [x] **JSON Schema for Surgery Recipes**
  - **Source**: `docs/schemas/surgery-recipe-v1.json`, `examples/recipes/hybrid_mamba_deltanet_recipe.json`
  - **Verification**: Validated against JSON Schema Draft-07 standard.
  - **Status**: Verified Complete.

---

### ✅ v0.1.2 — Format Correctness & Fuzz Testing Suite
- [x] **Full GGUF v2/v3 Quantization Matrix (`GgmlType`)**
  - **Source**: `src/parser/gguf.rs`
  - **Verification**: `cargo test --test test_gguf_quant_types` (6/6 tests passing)
  - **Status**: Verified Complete. Supports `F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, K-quants (`Q2_K`..`Q8_K`), and I-quants (`IQ1_M`..`IQ4_XS`).
- [x] **Zero-Allocation GGUF Block Decoders (`GgufBlockDecoder`)**
  - **Source**: `src/parser/gguf.rs`
  - **Verification**: `test_q4_0_block_unpacking`, `test_q4_1_block_unpacking`, `test_q5_0_block_unpacking`, `test_q8_0_block_unpacking`
  - **Status**: Verified Complete. Exact floating-point scale and offset reconstitution.
- [x] **Graceful Stream Recovery on Truncated Files**
  - **Source**: `src/parser/gguf.rs`, `src/parser/safetensors.rs`
  - **Verification**: Salvages intact tensors up to the corruption point with structured diagnostics.
  - **Status**: Verified Complete.
- [x] **Infallible Parser Fuzz Suite**
  - **Source**: `tests/test_fuzz_parsers.rs`
  - **Verification**: `cargo test --test test_fuzz_parsers` (Prefix-truncation and bit mutation across 10,000 steps without panics)
  - **Status**: Verified Complete.
- [x] **High-Definition Showcase Visuals**
  - **Source**: `assets/live_neural_benchmark.gif`
  - **Status**: Verified Complete. 920×520 multi-panel widescreen demonstration.

---

### ✅ v0.1.3 — Bring Your Own Algorithm (BYOA), Open Weights & REA-LLM Engine
- [x] **Bring Your Own Algorithm (BYOA) Registry & Execution Engine**
  - **Source**: `src/transform/byoa.rs`, `src/transform/mod.rs`
  - **Verification**: `cargo test --test test_byoa` (4/4 tests passing)
  - **Features**:
    - Extensible `AlgorithmRegistry` pre-populated with 5 verified presets (BitNet b1.58, 2:4 Sparsity, Hadamard WHT, SmoothQuant, Gated DeltaNet).
    - 6 Transformation rules: `QuantizeCodebook`, `AbsmeanTernary`, `OrthogonalRotation`, `StructuredSparsity`, `SmoothQuantOutlier`, `AffineProject`.
    - In-place float weight slice transformation (`apply_to_weights`) computing MSE, L2 norms, SNR, and sparsity ratio.
    - JSON import/export (`AlgorithmDefinition::from_json`, `to_json`).
    - Dynamic integration into Pareto `AutoDecider::decide_for_parameter_with_registry`.
  - **Status**: Verified Complete.
- [x] **Open Weights & Rebuilding From Scratch Engine**
  - **Source**: `src/ir/rebuilder.rs`, `src/ir/mod.rs`
  - **Verification**: `cargo test --test test_rebuilder` (3/3 tests passing)
  - **Features**:
    - `ArchitectureBlueprint` and `LayerBlueprint` supporting hybrid sequences (GQA, Gated DeltaNet, Phase Recurrence, Mamba-2, Sliding Window).
    - Scaffold synthesis (`ModelRebuilder::build_scaffold`) assembling clean `ModelIR` from scratch.
    - Open-weights remapping (`ModelRebuilder::rebuild_from_open_weights`) transferring weights from source checkpoints (LLaMA, Mistral, Qwen, DeepSeek) into new topologies with dimensional projection (`LowRankProject`, `ZeroPad`, `Truncate`).
    - Declarative surgery extensions: `TweakOperation::ApplyCustomAlgorithm`, `TweakOperation::RebuildArchitecture`.
  - **Status**: Verified Complete.
- [x] **REA-LLM & EWHA Telemetry Framework**
  - **Source**: `src/transform/rea.rs`, `src/transform/mod.rs`
  - **Verification**: `cargo test --test test_rea_llm` (4/4 tests passing)
  - **Features**:
    - Implementation of Execution-Weighted Heterogeneous Allocation (EWHA) mathematical framework from the research specification (`docs/research/THME_Research_Programme_v2.pdf`).
    - Two-path $80/20$ early-exit routing reach profile ($q_l = 1.0$ surface zone, $q_l = 0.20$ deep zone, $\bar{f} = 44\%$ depth fraction).
    - Representation mix allocations across channel groups ($\alpha_l$ Binary, $\beta_l$ Ternary, $\gamma_l$ Phase, FP16 fallback).
    - Analytic memory traffic accountant: $W_{\text{bytes/token}} \approx \sum_l q_l \cdot W_{l,\text{active}}$.
    - Per-token energy accountant: $E_{\text{token}} = \sum_l q_l \cdot E_{l,\text{active}}$.
  - **Status**: Verified Complete.
- [x] **Layer Anatomy Deconstruction & Dataflow Routing Engine**
  - **Source**: `src/ir/layer_decomp.rs`, `src/ir/mod.rs`, `src/gui/app.rs`
  - **Verification**: `cargo test --test test_layer_deconstruction` (2/2 tests passing)
  - **Features**:
    - Complete topological disassembly of any layer down to its constituent sub-blocks: Input RMSNorm, Sequence Mixer ($W_q, W_k, W_v, W_o$), Residual Highway 1, Post-Mixer RMSNorm, Feed-Forward SwiGLU MLP ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$), Residual Highway 2, and Routing & Early-Exit Gateway.
    - Full detection and classification of layer topologies: `StandardMonolithic`, `HybridDualMixer`, `TribridTriMixer`, `MambaSsmHybrid`, and `CustomByoaTopology`.
    - Active forward dataflow routing paths traced for every matrix (e.g. `RoPE -> KV-Cache`, `SwiGLU Elementwise Mult`, `Residual Highway 2`).
    - Interactive Visual Layer Anatomy & Dataflow Map in Tab 2 with direct parameter inspection focus and sub-block BYOA converter.
  - **Status**: Verified Complete.
- [x] **Desktop Studio GUI Updates (60 FPS Native)**
  - **Source**: `src/gui/app.rs`
  - **Verification**: `cargo check --bin thme-gui`
  - **Features**:
    - **Tab 0 (Master Cockpit)**: 4-card status grid displaying active architecture, AutoDecider matrix plan, REA-LLM EWHA telemetry, and live token flow.
    - **Tab 1 (Models Manager)**: Added "Rebuild Architecture From Scratch" wizard modal.
    - **Tab 3 (Distillation & Matrix Engine)**: Interactive BYOA Registry table, Custom Algorithm Designer form, one-click JSON Import/Export, and live 8×8 weight slice tester.
    - **Tab 4 (Full Benchmarks)**: Dedicated REA-LLM EWHA Telemetry and layerwise representation breakdown.
  - **Status**: Verified Complete.
- [x] **Multi-Format Dataset Ingestion & Fast Tokenizer**
  - **Source**: `src/dataset/mod.rs`
  - **Verification**: `cargo test --test test_dataset` (5/5 tests passing)
  - **Features**:
    - Ingests arbitrary training and calibration datasets from plain text (`.txt`), JSON Lines (`.jsonl`), JSON array (`.json`), CSV/TSV tables, and synthetic sequences.
    - Automatic detection and extraction from OpenAI messages, Alpaca instructions, and raw text records.
    - Fast hash projection tokenizer with length distribution statistics, vocabulary diversity estimations, and batch slicing (`get_batch`, `get_flat_batch`).
  - **Status**: Verified Complete.
- [x] **Non-Conventional Experimentation Laboratory Engine**
  - **Source**: `src/experiment/mod.rs`
  - **Verification**: `cargo test --test test_experiment` (5/5 tests passing)
  - **Features**:
    - 5 mathematical experimental transformations: Stochastic Noise Injection (simulating analog hardware drift & voltage margins), Spectral Energy Reshaping ($\sigma_i^\gamma$), Complex Phase Modulation ($\cos(\omega j + \phi)$), Continuous Activation Warping, and Stochastic DropBlock Sparsity.
    - In-place tensor transformation with analytical metrics: Original vs Transformed L2 norm, Mean Squared Error, Signal-to-Noise Ratio (dB), and Sparsity Delta %.
  - **Status**: Verified Complete.
- [x] **On-Device Micro-Trainer & Parameter Adaptation Engine**
  - **Source**: `src/trainer/mod.rs`
  - **Verification**: `cargo test --test test_trainer` (3/3 tests passing)
  - **Features**:
    - In-engine parameter adaptation loop executing simulated forward passes, gradient updates, and convergence tracking.
    - Loss objectives: Autoregressive Cross-Entropy, Teacher Knowledge Distillation ($\tau$), and EWHA Energy-Regularized loss ($\lambda_E$).
    - Optimizers: AdamW (decoupled weight decay), SignSGD (for 1-bit and ternary weights), and standard SGD with momentum.
    - Telemetry tracking: Step loss, Perplexity ($PPL$), throughput (tok/sec), and estimated energy (mJ).
  - **Status**: Verified Complete.
- [x] **Tab 8 Developer Lab & Training Studio GUI**
  - **Source**: `src/gui/app.rs`
  - **Verification**: `cargo check --bin thme-gui`
  - **Features**:
    - Integrated Tab 8 ("🧪 Dev Lab & Training") featuring Dataset Ingestion Deck, Non-Conventional Experimentation Lab, and Micro-Training & Adaptation Studio.
  - **Status**: Verified Complete.

---

### ✅ v0.1.4 — Empirical Quality Gate Evaluation & Perplexity Verification Bridge
- [x] **Empirical Perplexity Evaluator (`src/quality/evaluator.rs`)**
  - **Verification**: `cargo test --test test_quality_evaluator` (3/3 tests passing)
  - **Features**:
    - Multi-corpus representation degradation calculation for WikiText-2, C4, Penn Treebank, and Synthetic calibration.
    - Analytical cross-entropy and perplexity ($PPL = \exp(\mathcal{L})$) calculation based on weight variance, distortion, and outlier distributions.
    - External tool subprocess bridge with stdout regex parsing (supports `llama-perplexity`).
  - **Status**: Verified Complete.
- [x] **Automated Anti-Degradation Rollback Gate**
  - **Features**:
    - Asserts $\Delta PPL = PPL_{\text{candidate}} - PPL_{\text{baseline}} \le \epsilon$.
    - Returns `PerplexityGateDecision::Accept` or `PerplexityGateDecision::RollbackRequired` with actionable guidance.
  - **Status**: Verified Complete.
- [x] **CLI `thme validate` Command Enhancements**
  - **Features**:
    - Supports `--corpus`, `--baseline-ppl`, `--candidate-ppl`, `--max-ppl-delta`, and `--json`.
    - Returns non-zero exit code on quality violations to protect automated CI/Teacher LLM pipelines.
  - **Status**: Verified Complete.

### ✅ v0.1.5 — Surgery Daemon, IDE Extension & GitHub Copilot Integration
- [x] **Universal Model Context Protocol (MCP) Server (`thme mcp`)**
  - **Source**: `src/cli/mcp.rs`, `src/cli/mod.rs`
  - **Verification**: `cargo test --test test_mcp`
  - **Status**: Verified Complete. Standard 2024-11-05 JSON-RPC MCP server over stdin/stdout exposing 7 tools (`thme_inspect_model`, `thme_auto_decide`, `thme_apply_surgery`, `thme_validate_quality`, `thme_simulate_token_flow`, `thme_run_experiment`, `thme_micro_train`).
- [x] **VS Code & GitHub Copilot Extension Package (`editors/vscode/`)**
  - **Source**: `editors/vscode/package.json`, `editors/vscode/src/extension.ts`, `.vscode/mcp.json`
  - **Verification**: `npx @vscode/vsce package --no-dependencies` (zero warnings, generates `thme-vscode-0.1.5.vsix`)
  - **Status**: Verified Complete. 4 Explorer sidebar views, Copilot Language Model Tools (`thme_copilot_inspect`, `thme_copilot_surgery`, `thme_copilot_validate`), and auto-wired MCP workspace config.
- [x] **PyTorch & CUDA Ecosystem Interoperability Bridge (`src/interop/mod.rs`)**
  - **Source**: `src/interop/pytorch.rs`, `src/cli/interop.rs`
  - **Verification**: `cargo test --test test_interop` (3/3 passed)
  - **Status**: Verified Complete. Pure Rust code generation for PyTorch CUDA scripts (`train_cuda.py`), Hugging Face `config.json` generator, and PEFT LoRA adapter merge math ($W = W_0 + \frac{\alpha}{r} B \times A$).

---

## 3. Upcoming Roadmap & Sprints (v0.1.6 to v0.2.0)

---

### ⏳ Milestone v0.1.6: Speculative Decoding & Draft Model Synthesizer
*Goal: Synthesize compact draft models from early exit checkpoints and provide speculative verification telemetry.*

- [ ] **[THME-601] Early-Exit Draft Manifold Synthesizer**
  - **Target**: `src/profiling/speculation.rs`
  - **Deliverable**: Synthesizes a fast draft sub-model directly from the pre-checkpoint layers ($l < l_{cp}$).
  - **Features**: Standalone draft scaffold with shared embedding/head projections.
- [ ] **[THME-602] Speculative Token Verification Simulator**
  - **Target**: `src/profiling/token_flow.rs`
  - **Deliverable**: Simulates $K$-token draft candidate verification against target model, calculating empirical acceptance rate $\alpha_{\text{spec}}$ and effective latency speedup.

---

### ⏳ Milestone v0.1.7: Hardware-Calibrated Roofline & Micro-Kernel Profiler
*Goal: Provide empirical hardware Roofline models and multi-level cache hierarchy simulations.*

- [ ] **[THME-701] Arithmetic Intensity & Memory Bandwidth Roofline Model**
  - **Target**: `src/profiling/roofline.rs`
  - **Deliverable**: Analytical curves graphing FLOPs/byte against peak theoretical and measured memory bandwidth across target architectures (NVIDIA Ada/Blackwell, Apple Silicon M-series, Intel/AMD AVX-512).
- [ ] **[THME-702] Multi-Level Cache Hierarchy Simulator**
  - **Target**: `src/profiling/cache_sim.rs`
  - **Deliverable**: L1, L2, L3, and HBM/DDR5 cache miss penalty modeling predicting stalls during low-bit weight dequantization.

---

### ⏳ Milestone v0.1.8: BitNet b1.58 Calibrated Scale Quantization Exporter
*Goal: Export modified and distilled models directly into deployable GGUF / SafeTensors files with BitNet b1.58 ternary weights and calibrated scales.*

- [ ] **[THME-801] Binary GGUF & SafeTensors Exporter**
  - **Target**: `src/exporter/gguf.rs`, `src/exporter/safetensors.rs`
  - **Deliverable**: Serializes in-memory `ModelIR` tensors back to standard `.gguf` and `.safetensors` files directly loadable by `llama.cpp` and Ollama.
  - **Features**: Packs BitNet b1.58 ternary weights (-1, 0, +1) into 2-bit container blocks with per-channel scale factors.

---

### ⏳ Milestone v0.1.9: Automated Teacher-Student Co-Surgery & Distillation Pipeline
*Goal: Autonomous closed-loop optimization daemon where Teacher LLM iteratively generates surgery recipes and THME benchmarks quality until Pareto convergence.*

- [ ] **[THME-901] Closed-Loop Co-Surgery Daemon**
  - **Target**: `src/surgery/co_surgery.rs`
  - **Deliverable**: Iterative loop querying Teacher LLM for layer substitutions, executing surgery, benchmarking $\Delta PPL$ on WikiText-2, and accepting/rejecting with rollback.
- [ ] **[THME-902] Multi-Objective Pareto Convergence Monitor**
  - **Target**: `src/quality/pareto_monitor.rs`
  - **Deliverable**: Convergence tracking across 3 axes: $\Delta PPL$, EWHA energy consumption, and memory bandwidth reduction.

---

### ⏳ Milestone v0.2.0: Production Release & Real Open-Weights Demonstration
*Goal: Full production deployment demonstration on open weights (SmolLM-135M / LLaMA-3.2-1B) with published checkpoints and benchmarks.*

- [ ] **[THME-1001] End-to-End Open-Weights Reproducible Demonstration**
  - **Target**: `examples/reproducible_surgery_demo.rs`
  - **Deliverable**: Complete reproducible script that loads an open-weights model, executes BYOA & EWHA restructuring, and exports the optimized model with verified perplexity before and after.
- [ ] **[THME-1002] v0.2.0 Release Packaging & Distribution**
  - **Target**: Tag `v0.2.0` release workflow.
  - **Deliverable**: Pre-compiled multi-platform release binaries, signed attestation, updated documentation, and user guide.

---

### ⏳ Milestone v0.2.1: Direct GPU Driver Execution Engine
*Goal: Pure Rust execution on NVIDIA CUDA, Vulkan 1.3, and Apple Metal 3 bypassing external Python/PyTorch runtimes.*

- [ ] **[THME-1101] Direct Pure Rust GPU Driver Engine**
  - **Target**: `src/hardware/drivers/`
  - **Deliverable**: Direct driver execution via `cudarc` (CUDA Driver API), Vulkan 1.3 SPIR-V shaders, and Metal 3 Performance Shaders.
- [ ] **[THME-1102] Unified Hardware Abstraction Layer (HAL)**
  - **Target**: `src/hardware/hal.rs`
  - **Deliverable**: `HardwareDevice`, `DeviceMemoryBuffer`, and `KernelLaunchConfig` traits with automated compute backend selection.

---

### ⏳ Milestone v0.2.2: Tiered Heterogeneous Memory & Out-of-Core NVMe SSD Streaming
*Goal: AirLLM-style sequential layer streaming enabling 70B parameter inference and surgery on 8 GB consumer GPUs.*

- [ ] **[THME-1103] Out-of-Core NVMe SSD Layer Streaming Engine**
  - **Target**: `src/storage/nvme_stream.rs`
  - **Deliverable**: Asynchronous direct I/O via `io_uring` (Linux) and Overlapped I/O (Windows) with double-buffered layer prefetching.
- [ ] **[THME-1104] Tiered Heterogeneous Memory Manager**
  - **Target**: `src/memory/tiered.rs`
  - **Deliverable**: Three-tier memory management (GPU VRAM <-> Host RAM <-> NVMe SSD) allocating layers according to EWHA reach probability $q_l$.

---

### ⏳ Milestone v0.2.3: Distributed Heterogeneous Multi-Node Clustering
*Goal: Distributed Pipeline & Tensor Parallelism across mixed hardware nodes with pure Rust async communication.*

- [ ] **[THME-1105] Heterogeneous Pipeline & Tensor Parallelism**
  - **Target**: `src/cluster/parallel.rs`
  - **Deliverable**: Intra-node Tensor Parallelism across mixed local GPUs and inter-node Pipeline Parallelism across cluster nodes.
- [ ] **[THME-1106] Pure Rust Async TCP/RDMA Ring All-Reduce**
  - **Target**: `src/cluster/comm.rs`
  - **Deliverable**: Asynchronous ring all-reduce communication primitive with node heartbeats, fault tolerance, and dynamic re-sharding.

---

### ⏳ Milestone v0.2.4: Universal Agentic AI Connectivity Gateway
*Goal: Universal gateway connecting CLI agents, CI/CD harnesses, desktop apps, and IDE copilots.*

- [ ] **[THME-1107] Universal Agentic Gateway Daemon & WebSocket Streaming**
  - **Target**: `src/daemon/gateway.rs`
  - **Deliverable**: Headless JSON-RPC 2.0 daemon and high-throughput WebSocket telemetry stream with HMAC-SHA256 surgery attestation.
- [ ] **[THME-1108] Universal Agent Adapter Matrix**
  - **Target**: `src/daemon/adapters/`
  - **Deliverable**: Standardized client adapters for Autonomous CLI Agents (Claude Code, Gemini CLI, Aider), CI/CD Harnesses, Desktop Studio GUI, and IDEs (VS Code, Copilot).

---

## 4. Long-Term Backlog (Deferred to v0.3.0+)

These advanced initiatives are intentionally scheduled after the hardware execution and agentic gateway engines (v0.2.1 – v0.2.4) are established:

1. **Mixture-of-Experts (MoE) Dynamic Specialist Routing**: Dynamic expert sensitivity profiling and routing topology restructuring for models like Mixtral and DeepSeek-V2/V3.
2. **Continuous Online RL Adaptation**: Real-time layer manifold adjustment guided by continuous reinforcement learning feedback loops.
