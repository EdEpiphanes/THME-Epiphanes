# THME Roadmap: v0.1 to v0.2

> **Topological Heterogeneous Manifold Engine (THME)** by **Epiphanes**
> Roadmap governance and release progression from initial foundations to verified production deployment.

---

## Strategic Overview

The mission of THME is to provide a mathematically principled, verifiable framework for Large Language Model memory profiling, layer sensitivity estimation, Pareto precision allocation, automated Teacher LLM architecture surgery, and high-performance heterogeneous hardware execution.

The path from v0.1.0 to v0.2.0 focuses strictly on **correctness, honesty, reproducible measurements, and verified transformation**. High-performance hardware execution (direct GPU drivers, out-of-core NVMe SSD streaming, multi-node clustering, and the Universal Agentic AI Gateway) is formally integrated across **v0.2.1 through v0.2.4**, followed by advanced dynamic Mixture-of-Experts routing in **v0.3.0+**.

---

## Milestone Breakdown

### Phase 1: Foundations & Release Engineering
- [x] **v0.1.0 — Core Foundations**
  - Zero-copy SafeTensors and GGUF v2/v3 memory-mapped parsing.
  - Layer topological sensitivity estimation (SVD energy decay, condition number, Fisher proxy).
  - Dynamic programming Pareto precision allocator under quality gates ($Q \ge Q_0 - \epsilon$).
  - Native 60 FPS Desktop Studio GUI (`egui`/`eframe`) with 5 visual color schemes.
  - Matrix distillation engine: Truncated SVD, Walsh-Hadamard rotation, SmoothQuant, BitNet b1.58, 2:4 sparsity.
  - Parameter AutoDecider based on kurtosis, spectral decay, and layer depth.
  - Live 13-stage Transformer forward token telemetry simulator.
  - Checkpoint history rollback engine and audit logging.
  - Teacher LLM surgery engine via declarative JSON recipes.

- [x] **v0.1.1 — Path Hygiene, CI & Standards**
  - Rebranding to **THME** under **Epiphanes** with `llm-tweaker` backwards compatibility.
  - Enforcement of the Engineering Playbook (`AGENTS.md`) and non-negotiables.
  - Multi-platform GitHub Actions CI matrix (`ubuntu-latest`, `windows-latest`, `macos-latest`).
  - Automated CI Path-Leak Guard to reject machine paths and developer usernames.
  - Release packaging workflow on `v*` tags with stripped binaries (`strip = "symbols"`).
  - Zero-warning linter pass (`cargo clippy --all-targets -- -D warnings`).
  - JSON schema specification for surgery recipes (`docs/schemas/surgery-recipe-v1.json`).

---

### Phase 2: Format Correctness & Telemetry Hardening
- [x] **v0.1.2 — Format Correctness & Fuzz Testing**
  - Implement parsing for full GGUF v2/v3 quantization types (`Q4_K_M`, `Q5_K_S`, `Q8_0`, `IQ4_XS`, `IQ3_XXS`).
  - Graceful recovery and diagnostic error reporting on truncated or corrupt model headers.
  - Split core into workspace crates: `thme-core` (pure algorithms & IR), `thme-cli`, and `thme-gui`.
  - Continuous fuzz testing targets using `cargo-fuzz` / `honggfuzz` for both SafeTensors and GGUF parsers.
  - Publish `thme-core` crate to `crates.io`.

- [x] **v0.1.3 — Bring Your Own Algorithm (BYOA), Open Weights Rebuilder & REA-LLM Engine**
  - Custom algorithm registry (`AlgorithmRegistry`), JSON import/export, and in-place weight transformations.
  - Architecture scaffold synthesizer (`ModelRebuilder`) with open-weights dimensional remapping (`LowRankProject`, `ZeroPad`).
  - REA-LLM & EWHA framework implementation: $80/20$ reach profile $q_l$, per-token memory traffic $W_{\text{bytes/token}}$, and per-token energy $E_{\text{token}}$.
  - Desktop Studio GUI integrations: Tab 0 unified EWHA cockpit, Tab 1 Rebuilder wizard, Tab 3 BYOA registry with live 8x8 slice tester, and Tab 4 EWHA telemetry grid.
  - Criterion micro-benchmark suite and CPU SIMD acceleration kernels.

---

### Phase 3: Empirical Quality Verification & Universal Integration
- [x] **v0.1.4 — Empirical Quality Gate Evaluation & Perplexity Verification Bridge**
  - Real perplexity evaluation bridge supporting standard test sets (WikiText-2, C4, PTB) and local token scoring harness.
  - Perplexity degradation computation: $\Delta PPL = PPL_{\text{candidate}} - PPL_{\text{baseline}}$ with automated quality gate decision ($\Delta PPL \le \epsilon$).
  - Automated rollback trigger: if degradation exceeds tolerance, rollback to last baseline checkpoint immediately and emit structured audit diagnostics.
  - `thme validate` CLI command with `--corpus`, `--max-ppl-delta`, and JSON output flags.

- [x] **v0.1.5 — Surgery Daemon, IDE Extensions, PyTorch/CUDA Bridge & 8 AI Models Matrix**
  - Universal 8 Specialized AI Model Paradigm Converter (`src/ir/paradigm.rs`):
    - Full detection, ingestion, and cross-conversion matrix across 8 model paradigms: LLM, LCM, LAM, MoE, VLM, SLM, MLM, SAM.
    - Sparse upcycling engine partitioning dense MLPs into parallel expert matrices and synthesized router gating tensors.
    - BitNet b1.58 ternary discretization delivering up to 85.3% footprint reduction for mobile and edge deployments.
    - CLI `thme convert` command with full customization flags (`--experts`, `--top-k`, `--hidden`, `--layers`, `--bitnet`, `--save-recipe`).
    - Declarative surgery recipe support (`TweakOperation::ConvertParadigm`) and Desktop Studio GUI converter in Tab 1.
  - Headless Model Context Protocol (MCP) server (`thme mcp`) exposing 8 specialized LLM surgical tools including `thme_convert_paradigm`.
  - Packaged Visual Studio Code extension (`editors/vscode/thme-vscode-0.1.5.vsix`) with 4 Explorer sidebar panels.
  - GitHub Copilot Language Model Tools integration (`thme_copilot_inspect`, `thme_copilot_surgery`, `thme_copilot_validate`).
  - PyTorch & CUDA fine-tuning interoperability bridge (`src/interop/`):
    - Synthesizes standalone, self-contained `train_cuda.py` scripts with LoRA, QLoRA (BitsAndBytes 4-bit), Unsloth Fast, and DeepSpeed ZeRO-2/3.
    - Hugging Face `config.json` generator for LLaMA, Mistral, Gemma, Qwen, and DeepSeek model families.
    - Pure Rust PEFT LoRA adapter checkpoint parser and zero-copy weight merger ($W = W_0 + \frac{\alpha}{r} B \times A$).
    - `thme export` / `thme interop` CLI command for instant pipeline transitions.

- [/] **v0.1.6 — Speculative Decoding & Draft Model Synthesizer** [IN PROGRESS]
  - Draft model manifold synthesizer extracting lightweight sub-models from target model early-exit checkpoints or prefix layers ($0..K$).
  - Multi-token tree speculation verification engine calculating speculative acceptance rate $\alpha_{\text{spec}}$, expected accepted tokens $\mathbb{E}[k]$, and effective latency speedup $S = \frac{\mathbb{E}[k] + 1}{1 + \gamma \frac{T_{\text{draft}}}{T_{\text{target}}}}$.
  - Tree-based speculative attention mask synthesis supporting multi-candidate branching (Medusa/Eagle style speculative trees).
  - CLI `thme speculate` subcommand with full telemetry output and draft model export.
  - IDE & Agent MCP tool `thme_simulate_speculative_decoding` for automated agent profiling.
  - Studio GUI Benchmarks (Tab 4) live speculative simulation deck with acceptance curve and speedup multiplier gauges.

---

### Phase 4: Hardware Profiling, Serialization & Closed-Loop Surgery
- [ ] **v0.1.7 — Hardware-Calibrated Roofline & Micro-Kernel Profiler**
  - Empirical Roofline model graphing Arithmetic Intensity (FLOPs/byte) against memory bandwidth saturation curves across target hardware (NVIDIA Ada/Blackwell, Apple Silicon M-series, Intel/AMD AVX-512).
  - Multi-level memory hierarchy cache simulator (L1, L2, L3, HBM3/DDR5) predicting cache hit rates and memory latency penalties.
  - Power and thermal throttling projection curves for continuous edge inference workloads.

- [ ] **v0.1.8 — BitNet b1.58 Calibrated Scale Quantization Exporter**
  - Direct binary disk serialization for modified and distilled models into deployable GGUF and SafeTensors files.
  - BitNet b1.58 ternary weight packing (2 bits per weight: -1, 0, +1 packed 4 weights per byte) with calibrated per-channel scale factors.
  - Header reconstitution and metadata synchronization ensuring exported artifacts are directly loadable in standard runtimes (`llama.cpp`, Ollama, vLLM).

- [ ] **v0.1.9 — Automated Teacher-Student Co-Surgery & Distillation Pipeline**
  - Autonomous closed-loop optimization daemon: Teacher LLM analyzes layer sensitivity, generates declarative surgery recipes, THME applies them, and runs the evaluation bridge.
  - Multi-objective Pareto auto-tuner balancing $\Delta PPL$, EWHA energy, and inference memory traffic.
  - Self-healing recovery: automatically retries or adjusts quantization parameters when quality thresholds are breached.

---

### Phase 5: Production Core Release
- [ ] **v0.2.0 — Production-Ready Verified Release**
  - Verified production deployment demonstration on a real open-weights model (e.g. SmolLM-135M / LLaMA-3.2-1B) with published before-and-after checkpoints.
  - End-to-end reproducible script generating published benchmark tables, perplexity figures, and energy measurements.
  - Multi-platform signed release binaries, comprehensive documentation, and video walkthrough.

---

### Phase 6: Tiered Hardware Execution & Universal Agentic Connectivity
- [ ] **v0.2.1 — Direct GPU Driver Execution Engine (Vulkan / Metal / CUDA Driver API)**
  - Direct pure Rust GPU driver execution engine bypassing heavy external runtimes:
    - NVIDIA CUDA Driver API via `cudarc` (bypassing the Python runtime).
    - Cross-platform Vulkan 1.3 compute shaders via SPIR-V for Windows and Linux.
    - Apple Metal 3 Performance Shaders for macOS Apple Silicon unified memory.
  - Unified hardware abstraction layer: `HardwareDevice`, `DeviceMemoryBuffer`, `KernelLaunchConfig`.
  - Fused low-bit matrix multiplication micro-kernels, RMSNorm, and SwiGLU activation kernels.

- [ ] **v0.2.2 — Tiered Heterogeneous Memory & Out-of-Core NVMe SSD Streaming**
  - Three-tier hierarchical memory pipeline: GPU VRAM $\leftrightarrow$ Host RAM $\leftrightarrow$ NVMe SSD direct asynchronous I/O (`io_uring` on Linux, Overlapped I/O on Windows).
  - AirLLM-style sequential layer weight streaming enabling 70B parameter inference and surgery on 8 GB consumer GPUs.
  - Zero-copy pinned memory staging buffers and background double-buffering prefetch queue.

- [ ] **v0.2.3 — Distributed Heterogeneous Multi-Node Clustering (Pipeline & Tensor Parallelism)**
  - Intra-node Tensor Parallelism (TP) across mixed GPUs (e.g., discrete NVIDIA RTX + integrated Apple Silicon / AMD).
  - Inter-node Pipeline Parallelism (PP) over standard Ethernet / InfiniBand / Thunderbolt networks.
  - Pure Rust asynchronous TCP/RDMA ring all-reduce communication primitives.
  - Dynamic fault tolerance: heartbeat monitors, automated layer re-sharding on node failure, and cluster health telemetry.

- [ ] **v0.2.4 — Universal Agentic AI Connectivity Gateway (CLI, Harness, Desktop, IDE)**
  - Headless JSON-RPC 2.0 daemon and high-performance WebSocket streaming gateway (`thme daemon`).
  - Universal Agent Adapter Matrix:
    - Autonomous CLI Agents (Claude Code, Gemini CLI, Aider, OpenHands).
    - CI/CD & Orchestration Harnesses (Harness CI/CD, GitHub Actions, GitLab CI, Airflow).
    - Native Desktop Studio App (`thme-gui`).
    - IDEs & Copilots (VS Code, Cursor, Windsurf, GitHub Copilot Language Model Tools).
  - Bidirectional event streaming: layer progress, token throughput, thermal throttling, memory pressure.
  - Cryptographic HMAC-SHA256 surgery recipe verification and provenance attestation.

---

## Long-Term Backlog (Deferred to v0.3.0+)

These advanced algorithmic initiatives are documented in the backlog and intentionally deferred to subsequent major milestones:
- **Continuous Online RL Adaptation**: Real-time layer manifold adjustment guided by continuous reinforcement learning feedback loops.
- **Topological Hyper-Graph Neural Architecture Search**: Automated micro-circuit rewiring using topological data analysis (TDA) persistent homology.
