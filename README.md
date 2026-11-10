# LLM-Tweaker / THME

<p align="center">
  <img src="assets/live_neural_benchmark.gif" alt="THME Desktop Studio - Live Neural Network Token Flow and Telemetry Benchmarker" width="840" />
</p>

<p align="center">
  <strong>Topological Heterogeneous Manifold Engine (THME) and LLM-Tweaker Studio</strong><br>
  <em>Open-source framework for Large Language Model memory profiling, zero-copy weight analysis, live token flow telemetry, and automated Teacher LLM architecture surgery.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-2021%20Edition-orange?logo=rust&style=flat-square" alt="Rust 2021" />
  <img src="https://img.shields.io/badge/GUI-eframe%20%2F%20egui%20(Native%2060%20FPS)-06b6d4?style=flat-square" alt="GUI eframe" />
  <img src="https://img.shields.io/badge/Formats-SafeTensors%20%2B%20GGUF%20v2%2Fv3-10b981?style=flat-square" alt="Formats" />
  <img src="https://img.shields.io/badge/Memory-Zero--Copy%20MMap-f59e0b?style=flat-square" alt="Zero-Copy" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/Tests-11%20Passed-success?style=flat-square" alt="Tests" />
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#desktop-studio-showcase">Desktop Studio</a> •
  <a href="#cli-tools">CLI Tools</a> •
  <a href="#teacher-llm-surgery">Teacher LLM Surgery</a> •
  <a href="#mathematical-foundation">Research Paper</a> •
  <a href="#production-roadmap">Production Roadmap</a> •
  <a href="#quickstart">Quickstart</a>
</p>

---

## Overview

**LLM-Tweaker (THME)** profiles and optimizes Large Language Models using heterogeneous precision allocation and sequential execution. Rather than applying a single uniform bit-width across an entire model, THME measures the topological sensitivity of each attention projection and feed-forward block, building a Pareto-optimal precision plan constrained by quality gates ($Q_{\text{candidate}} \ge Q_{\text{baseline}} - \epsilon$).

Built in pure Rust without external runtime dependencies:
- **Universal model importer**: Zero-copy memory-mapped parsing of GGUF v2/v3 and SafeTensors files or sharded directories, with fallback architecture deduction when `config.json` is absent.
- **Distillation and matrix transformation engine**: Implementations of Truncated SVD, Walsh-Hadamard orthogonal rotation, SmoothQuant per-channel scaling, BitNet b1.58 absmean ternary quantization, and 2:4 semi-structured sparsity.
- **Automated parameter decision engine**: Analyzes spectral energy, condition number, and outlier kurtosis for each tensor, assigning the mathematically optimal algorithm per parameter.
- **Layer tweaker and section inspector**: Adjusts layer sequence mixers, precision allocations, scale factors, bias offsets, and pruning thresholds, with interactive 8x8 weight slice matrices and 32-bin histograms.
- **Live token flow and full benchmark suite**: Simulates token generation across 13 Transformer stages with 60 FPS telemetry, batch scaling analysis ($B=1 \dots 64$), and context window KV cache projections.
- **History checkpoints and audit trail**: Records immutable snapshots for instant configuration rollback and a searchable, structured compliance audit stream.
- **Teacher LLM surgery engine**: Declarative JSON recipe format allowing automated agents to adjust layer precisions, replace sequence mixers, and prune components.
- **Standalone native desktop studio**: Compiles to a 5.8 MB zero-dependency executable ([`LLM-Tweaker-Studio.exe`](file:///LLM-Tweaker-Studio.exe)).

---

## Desktop Studio Showcase

Launch the native desktop application on Windows, macOS, or Linux:
```bash
thme gui
# Or run the dedicated binaries:
thme-gui
./LLM-Tweaker-Studio.exe
```

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  THME v2.0  UNIVERSAL LLM RESTRUCTURER, MATRIX ENGINE & STUDIO                          [📁 Load Model] │
│  Llama-3-8B [GGUF • LLAMA] (32 Layers • Hidden Dim 4096 • 8030M Params)                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Master Dashboard] [Models] [Layer Tweaker] [Matrix & Distill] [Benchmarks] [History] [Audit] [Export]  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  THROUGHPUT: 44.2 tok/s  |  VRAM STORAGE: 3820 MB  |  TRAFFIC/TOK: 28.4 MB  |  DEPTH: 82%  | EXIT: 80%  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### The 8 Interactive Studio Workspaces

| Workspace Tab | Focus | Core Capabilities |
|---|---|---|
| **Master Dashboard** | Unified command cockpit | Consolidates model health cards, live neural token flow animation, layer sensitivity curves, quick optimization actions, and recent history and audit events. |
| **Models Manager** | Repository & inventory | Multi-model slot manager with GGUF and SafeTensors importers, synthetic model generators, and a sortable tensor inventory registry. |
| **Layer Tweaker & Sections** | Fine-grained surgery | Drill down into any layer: tweak sequence mixers (GQA, Gated DeltaNet, Phase Recurrence, Sliding Window, Mamba-2), precision mix ($\alpha, \beta, \gamma$), scale, bias, and pruning. Inspect Attention Q/K/V/O, MLP Gate/Up/Down, and Norm sections with 8x8 weight slices and 32-bin histograms. |
| **Distillation & Matrix Engine** | Linear algebra transformations | Implements Truncated SVD, Walsh-Hadamard rotation, SmoothQuant, BitNet b1.58, 2:4 sparsity, and Knowledge Distillation feature alignment. Includes the **Automated Parameter Decider** that evaluates every parameter tensor and executes global Pareto transformation plans. |
| **Full Benchmarks** | Multi-dimension profiling | Live 13-stage Transformer flow simulator with animated telemetry, batch size scaling ($B = 1, 4, 8, 16, 32, 64$), context length scaling ($512 \dots 8192$), and layer latency waterfalls. |
| **History & Checkpoints** | Audit and rollback | Immutable snapshots of tuning sessions, before/after scorecard diffs, one-click state restoration, and JSON history report export. |
| **Audit Log** | Structured event stream | Filterable chronological event log with severity classification (`OPTIMIZE`, `DISTILL`, `SURGERY`, `BENCHMARK`, `INFO`, `WARN`), query search, and CSV export. |
| **Export & Surgery** | Model artifacts & JSON recipes | Exports tuned models to SafeTensors, GGUF, or THME bundles. Includes a multiline JSON specification editor and executor for Teacher LLM restructuring recipes. |

---

## CLI Tools

LLM-Tweaker provides dual-binary CLI support (`thme` or `llm-tweaker`):

### Inspect model weights
Scan local GGUF or SafeTensors files zero-copy:
```bash
thme inspect models/llama-3-8b.gguf
thme inspect /path/to/huggingface-dir --json
```

```text
=== THME Model Inspection Summary ===
┌───────────────────────┬──────────────────────────┐
│ Property              │ Value                    │
├───────────────────────┼──────────────────────────┤
│ Model Name            │ llama-3-8b               │
│ Architecture          │ Llama-family             │
│ Total Parameters      │ 8.03 B (8030261248)      │
│ Raw Weight Size       │ 14.96 GB                 │
│ Hidden Size           │ 4096                     │
│ Number of Layers      │ 32                       │
│ Attention Heads (Q)   │ 32                       │
│ Key-Value Heads (KV)  │ 8 (Grouped-Query GQA)    │
│ Head Dimension        │ 128                      │
│ Vocabulary Size       │ 128256                   │
│ Max Context Length    │ 8192                     │
│ Total Tensors         │ 291                      │
└───────────────────────┴──────────────────────────┘
```

### Profile layer sensitivity and KV cache
Project memory scaling across context windows from 1k to 128k+ tokens:
```bash
thme profile models/llama-3-8b.gguf --batch-size 1 --context-lengths 2048,8192,32768,131072
```

### Generate Pareto precision allocation
Compute a precision plan targeting a specific memory budget:
```bash
thme plan models/llama-3-8b.gguf --budget-gb 8.0 --output optimization_plan.json
```

### Apply Teacher LLM surgery recipe
Execute transformation recipes created by automated Teacher LLMs:
```bash
thme tweak --model models/llama-3-8b.gguf --recipe recipes/deltanet_hybrid.json --output models/restructured.json
```

### Evaluate quality gates
Validate candidate scores against baseline benchmarks with directional checking (lower-is-better for Perplexity; higher-is-better for Accuracy):
```bash
thme validate --baseline 85.0 --candidate 84.4 --max-degradation 1.0
thme validate --suite-file configs/quality_gates.yaml
```

---

## Teacher LLM Surgery

External AI agents can programmatically restructure imported models by writing declarative JSON recipes:

```json
{
  "name": "THME-Optimal-Heterogeneous-Recipe",
  "author": "Teacher-LLM-AutoTuner",
  "target_model": "Llama-3-8B",
  "version": "1.0",
  "description": "Restructures dense Transformer layers into hybrid DeltaNet/Phase recurrence and mixed-precision representation.",
  "operations": [
    {
      "type": "SetPrecision",
      "params": {
        "start_layer": 0,
        "end_layer": 1,
        "role": null,
        "format": "FP16"
      }
    },
    {
      "type": "SetPrecision",
      "params": {
        "start_layer": 2,
        "end_layer": 28,
        "role": "MlpGate",
        "format": "Ternary"
      }
    },
    {
      "type": "ReplaceMixer",
      "params": {
        "layer": 1,
        "mixer_type": "gated_deltanet"
      }
    },
    {
      "type": "ConfigureEarlyExit",
      "params": {
        "checkpoint_layer": 8,
        "confidence_threshold": 0.85,
        "target_exit_ratio": 80.0
      }
    },
    {
      "type": "QuantizeKvCache",
      "params": {
        "format": "INT8"
      }
    }
  ],
  "expected_speedup": 2.15,
  "expected_memory_savings_pct": 68.5
}
```

Applying this recipe computes instant diffs:
```text
[Surgery Started] Applying recipe 'THME-Optimal-Heterogeneous-Recipe' by Teacher-LLM-AutoTuner
Applied precision FP16 to layers 0..=1 (role: None)
Applied precision Ternary to layers 2..=28 (role: Some(MlpGate))
Replaced Layer 1 sequence mixer with 'gated_deltanet'
Configured Early-Exit at Layer 8 (Threshold: 0.85, Exit Target: 80%)
Quantized KV Cache global format to INT8
[Surgery Complete] 29 tensors restructured. Storage saved: 4120.4 MB (64.2%), Estimated Speedup: 2.18x
```

---

## Mathematical Foundation

The scientific architecture is documented in [`docs/research/THME_Research_Programme_v2.pdf`](docs/research/THME_Research_Programme_v2.pdf).

### KV cache memory formula
Autoregressive generation stores historical Key and Value representations. THME computes this footprint using:
$$M_{\mathrm{KV}} = 2 \times L \times B \times S \times H_{\mathrm{KV}} \times D_{\mathrm{head}} \times \text{bytes}$$
where $L$ is layer count, $B$ is batch size, $S$ is context sequence length, $H_{\mathrm{KV}}$ is key-value head count (distinguishing Grouped-Query Attention from Multi-Head Attention), and $D_{\mathrm{head}}$ is head dimension.

### Adaptive speculative early-exit routing
Tokens evaluate confidence at checkpoint layer $L_{\mathrm{exit}}$. Layer reach probability is:
$$q_l = \begin{cases} 1.0 & \text{if } l \leq L_{\mathrm{exit}} \\ 1.0 - p_{\mathrm{exit}} & \text{if } l > L_{\mathrm{exit}} \end{cases}$$
Mean computational depth fraction is $\bar{f} = \frac{1}{N} \sum_{l=0}^{N-1} q_l$, reducing memory bandwidth consumption accordingly.

### Layerwise parabolic sensitivity
Perturbation tests indicate that initial projection layers and terminal output heads are more sensitive to quantization noise, while intermediate layers tolerate lower precision:
$$s(l) = s_{\mathrm{base}} + 4 \cdot \Delta_{\mathrm{depth}} \cdot \left(\frac{l}{N - 1} - 0.5\right)^2$$

---

## Production Roadmap

```mermaid
flowchart TD
    v010["v0.1.0 Baseline: Pure-Rust Engine & Desktop Studio (Completed)"]
    v011["v0.1.1 Format Matrix: Complete GGUF v2/v3 & Out-of-Core Safety"]
    v012["v0.1.2 CPU SIMD Engine: AVX-512 & ARM NEON Kernels"]
    v013["v0.1.3 GPU Backends: Vulkan / WebGPU / Metal / CUDA Acceleration"]
    v014["v0.1.4 AirLLM Streaming: 70B Models on 8GB VRAM with Async Prefetch"]
    v015["v0.1.5 Teacher LLM Daemon: Autonomous Agent JSON-RPC Protocol"]
    v016["v0.1.6 Quant Exporter: BitNet b1.58 & SmoothQuant Serialization"]
    v017["v0.1.7 Quality Gate Suite: Automated Perplexity & MMLU Benchmarking"]
    v018["v0.1.8 MoE Specialist: DeepSeek-V2/V3 & Mixtral Router Tweaking"]
    v019["v0.1.9 Production Release: Automated Cross-Platform CI/CD & Signed Binaries"]

    v010 --> v011 --> v012 --> v013 --> v014 --> v015 --> v016 --> v017 --> v018 --> v019
```

- **v0.1.0:** Zero-copy SafeTensors and GGUF parser, Pareto precision allocator, directional Quality Gate, Desktop Studio GUI with 6 tabs, live token flow telemetry, and headless surgery CLI.
- **v0.1.1:** Exhaustive GGML quantization types (`Q2_K`, `Q3_K`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `IQ2_XXS`), sharded split GGUF, and truncated file recovery.
- **v0.1.2:** AVX-512, AVX2, and ARM NEON SIMD micro-kernels for real CPU dot-product latency and layer-wise activation SNR measurement.
- **v0.1.3:** `wgpu` (Vulkan/Metal) and CUDA hardware backends for on-device GPU layer profiling and PCIe transfer measurement.
- **v0.1.4:** Sequential AirLLM asynchronous double-buffered disk-to-VRAM weight streamer allowing 70B models to run on 8GB VRAM.
- **v0.1.5:** Headless daemon (`thme daemon --port 8080`) exposing JSON-RPC and REST endpoints for autonomous model surgery agents.
- **v0.1.6:** Quantization writer exporting serialized `.gguf` and `.safetensors` files packed with BitNet b1.58 ternary and calibrated scales.
- **v0.1.7:** In-engine Perplexity evaluator on WikiText-2 and zero-shot MMLU benchmarking suite.
- **v0.1.8:** Sparse Mixture-of-Experts (MoE) support for DeepSeek-V2/V3 and Mixtral 8x22B with expert pruning and routing analysis.
- **v0.1.9:** Multi-platform packaging (Windows MSI/exe, Linux AppImage, macOS universal DMG), fuzz testing, and verified release binaries.

---

## Quickstart

### Prerequisites
- [Rust Toolchain](https://rustup.rs/) (edition 2021, Rust >= 1.80)

### Installation
```bash
git clone https://github.com/TechTroniX/LLM-Tweaker.git
cd LLM-Tweaker
cargo build --release
```

The compiled binaries will be available in:
- `target/release/thme` (Primary CLI)
- `target/release/llm-tweaker` (Alias binary)
- `target/release/thme-gui` (Standalone Native Desktop Studio)
- [`LLM-Tweaker-Studio.exe`](file:///LLM-Tweaker-Studio.exe) (Root standalone executable)

### Run Tests
```bash
cargo test
```

---

## Contributing & Community

LLM-Tweaker is an open-source project. We welcome contributions, RFCs, bug reports, and research extensions:
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for architecture guidelines and coding standards.
- Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Security policy is documented in [SECURITY.md](SECURITY.md).

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
