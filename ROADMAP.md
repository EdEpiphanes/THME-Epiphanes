# THME Roadmap: v0.1 to v0.2

> **Topological Heterogeneous Manifold Engine (THME)** by **Epiphanes**
> Roadmap governance and release progression from initial foundations to verified production deployment.

---

## Strategic Overview

The mission of THME is to provide a mathematically principled, verifiable framework for Large Language Model memory profiling, layer sensitivity estimation, Pareto precision allocation, and automated Teacher LLM architecture surgery.

As defined in the **Engineering Playbook**, the path from v0.1.0 to v0.2.0 focuses strictly on **correctness, honesty, reproducible measurements, and verified transformation**. High-complexity execution runtimes (Vulkan / Metal / CUDA kernels, 70B out-of-core streaming, and Mixture-of-Experts routing) are deliberately scheduled for **v0.3.0+** so that the core analytical and surgical engines are hardened first.

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

### Phase 3: Empirical Quality Verification & Distributed Surgery
- [ ] **v0.1.4 — Empirical Quality Gate Evaluation & Perplexity Verification Bridge**
  - Real perplexity evaluation bridge supporting standard test sets (WikiText-2, C4, PTB) and local token scoring harness.
  - Perplexity degradation computation: $\Delta PPL = PPL_{\text{candidate}} - PPL_{\text{baseline}}$ with automated quality gate decision ($\Delta PPL \le \epsilon$).
  - Automated rollback trigger: if degradation exceeds tolerance, rollback to last baseline checkpoint immediately and emit structured audit diagnostics.
  - `thme validate` CLI command with `--corpus`, `--max-ppl-delta`, and JSON output flags.

- [ ] **v0.1.5 — Surgery Daemon & Programmatic Remote API**
  - Headless daemon mode (`thme daemon`) exposing JSON-RPC 2.0 and REST endpoints.
  - Remote Teacher LLM integration: automated agents can inspect model IR, receive AutoDecider recommendations, and submit surgery recipes programmatically over HTTP.
  - Cryptographic attestation and HMAC-SHA256 signature on executed surgery recipes to guarantee provenance.

- [ ] **v0.1.6 — Speculative Decoding & Draft Model Synthesizer**
  - Draft model manifold synthesizer extracting lightweight sub-models from early-exit checkpoints (e.g., first 4 layers of 16-layer model).
  - Multi-token tree speculation verification engine calculating acceptance rate $\alpha_{\text{spec}}$ and effective latency speedup.
  - Verification telemetry integrated into Studio GUI Benchmarks tab.

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

### Phase 5: Production Release
- [ ] **v0.2.0 — Production-Ready Verified Release**
  - Verified production deployment demonstration on a real open-weights model (e.g. SmolLM-135M / LLaMA-3.2-1B) with published before-and-after checkpoints.
  - End-to-end reproducible script generating published benchmark tables, perplexity figures, and energy measurements.
  - Multi-platform signed release binaries, comprehensive documentation, and video walkthrough.

---

## Long-Term Backlog (Deferred to v0.3.0+)

These advanced initiatives are documented in the backlog and intentionally deferred to subsequent major milestones:
- **Hardware GPU Execution Backends**: Native Vulkan compute, Apple Metal Performance Shaders, and CUDA custom kernels.
- **Out-of-Core 70B Streaming**: AirLLM-style sequential layer paging from NVMe SSD into constrained VRAM (e.g., 70B inference in 8 GB VRAM).
- **Mixture-of-Experts (MoE) Routing**: Dynamic expert sensitivity profiling and routing topology re-structuring for models like Mixtral and DeepSeek-V2/V3.
