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

- [ ] **v0.1.3 — Real Measurement & Hardware Kernels**
  - CPU SIMD acceleration kernels: AVX2, AVX-512, and ARM NEON for matrix transformations (Hadamard, BitNet quantization, 2:4 sparsity).
  - Criterion micro-benchmark suite (`benches/`) running in CI.
  - Strict separation of live simulated telemetry from real hardware measurements in GUI and CLI reports.

---

### Phase 3: Empirical Quality Verification & Distributed Surgery
- [ ] **v0.1.4 — Empirical Quality Gate Evaluation**
  - Real perplexity evaluation bridge via external evaluators (e.g., `llama.cpp` perplexity tool) and local scoring harness.
  - Automated degradation verification against baseline unquantized models across standard test sets (WikiText-2, C4).
  - `thme validate` CLI command asserting zero quality degradation violations before model commitment.

- [ ] **v0.1.5 — Surgery Daemon & Remote API**
  - Headless daemon mode (`thme daemon`) exposing JSON-RPC and REST endpoints.
  - Remote Teacher LLM integration: automated agents can inspect model IR, receive AutoDecider recommendations, and submit surgery recipes programmatically over HTTP.
  - Cryptographic attestation and HMAC signature on executed surgery recipes.

---

### Phase 4: Production Release
- [ ] **v0.2.0 — Production-Ready Verified Release**
  - Verified Quantization Exporter: export modified and distilled models directly into deployable GGUF / SafeTensors files with BitNet b1.58 ternary weights and calibrated scales.
  - Reproducible end-to-end demonstration on a real open-weights model (e.g., LLaMA-3.2-1B, Qwen-2.5-0.5B, or SmolLM-135M) with verified before-and-after perplexity measurements.
  - Multi-platform signed release binaries, comprehensive documentation, and video walkthrough.

---

## Long-Term Backlog (Deferred to v0.3.0+)

These advanced initiatives are documented in the backlog and intentionally deferred to subsequent major milestones:
- **Hardware GPU Execution Backends**: Native Vulkan compute, Apple Metal Performance Shaders, and CUDA custom kernels.
- **Out-of-Core 70B Streaming**: AirLLM-style sequential layer paging from NVMe SSD into constrained VRAM (e.g., 70B inference in 8 GB VRAM).
- **Mixture-of-Experts (MoE) Routing**: Dynamic expert sensitivity profiling and routing topology re-structuring for models like Mixtral and DeepSeek-V2/V3.
