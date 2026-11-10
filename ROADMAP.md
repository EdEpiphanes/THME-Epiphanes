# THME Roadmap: v0.1 to v0.2

> **Topological Heterogeneous Manifold Engine (THME)** by **Epiphanes**
> Roadmap governance and release progression from initial foundations to verified production deployment and heterogeneous hardware execution.

---

## Strategic Overview

The mission of THME is to provide a mathematically principled, verifiable framework for Large Language Model memory profiling, layer sensitivity estimation, Pareto precision allocation, automated Teacher LLM architecture surgery, and high-performance heterogeneous hardware execution.

The progression spans six deliberate phases:
1. **Foundations & Standards (v0.1.0 – v0.1.1):** Zero-copy parsers, topology estimators, Pareto precision allocators, CI path-leak guards, and cross-platform automation.
2. **Format Correctness & Telemetry (v0.1.2 – v0.1.3):** Full GGUF v2/v3 quant matrix, BYOA algorithm registry, REA-LLM EWHA energy routing, and architecture rebuilder.
3. **Quality Gates & Interoperability (v0.1.4 – v0.1.6):** Empirical perplexity evaluation bridges, Model Context Protocol (MCP) server, VS Code extension, PyTorch/CUDA LoRA interop, Universal 8 AI Models Converter, and Speculative Decoding Synthesizer.
4. **Hardware Profiling & Closed-Loop Surgery (v0.1.7 – v0.1.9):** Hardware-calibrated Roofline models, BitNet b1.58 binary serialization, and autonomous Teacher-Student co-surgery loops.
5. **Production Core Release (v0.2.0):** End-to-end verified small-model open-weights demonstrations, reproducible benchmarks, and signed release binaries.
6. **Tiered Hardware Execution & Agentic Connectivity (v0.2.1 – v0.2.4):** Direct GPU driver execution (Vulkan/Metal/CUDA), out-of-core NVMe SSD streaming, multi-node clustering, and the Universal Agentic AI Gateway daemon.

---

## Complete Milestone Progression

```
+--------------------------------------------------------------------------------------------------------------------+
|                                              THME ROADMAP PROGRESSION                                              |
+--------------------------------------------------------------------------------------------------------------------+
| [x] v0.1.0  Core Foundations (Zero-copy parsing, SVD decay, Pareto allocator, 60 FPS studio GUI)                  |
| [x] v0.1.1  Standards & Hygiene (Rebrand, 3-OS CI matrix, path-leak guard, zero warnings, recipe schema)          |
| [x] v0.1.2  Format Correctness (Full GGUF quant matrix, resilient stream recovery, parser fuzzing)                 |
| [x] v0.1.3  BYOA & REA-LLM (Algorithm registry, open weights rebuilder, EWHA energy telemetry, micro-trainer)     |
| [x] v0.1.4  Quality Gate (External perplexity bridge, WikiText-2 / C4 scoring, automated rollback gating)         |
| [x] v0.1.5  Surgery Daemon & 8 AI Models (MCP server, VS Code extension, Copilot tools, PyTorch bridge, 8 Models)|
| [x] v0.1.6  Speculative Decoding (Draft manifold extraction, multi-token tree verification, Leviathan speedup)    |
| [ ] v0.1.7  Roofline Profiler (Arithmetic intensity, hardware ceilings, 4-tier cache hierarchy simulation) ACTIVE |
| [ ] v0.1.8  Quant Exporter (Direct binary GGUF v3 & SafeTensors writer, BitNet b1.58 2-bit packing)        PLAN   |
| [ ] v0.1.9  Closed-Loop Surgery (Teacher-Student co-surgery daemon, multi-objective Pareto, self-healing)  PLAN   |
| [ ] v0.2.0  Production Release (SmolLM-135M / LLaMA-3.2-1B open-weights demo, signed release binaries)    TARGET |
| [ ] v0.2.1  Direct GPU Drivers (Vulkan 1.3 SPIR-V, Metal 3, CUDA Driver API via pure Rust cudarc)           SCHED  |
| [ ] v0.2.2  Out-of-Core NVMe Streaming (3-tier VRAM/RAM/SSD hierarchy, io_uring, 70B inference in 8 GB)     SCHED  |
| [ ] v0.2.3  Multi-Node Clustering (Intra-node TP, inter-node PP, async TCP/RDMA ring all-reduce)              SCHED  |
| [ ] v0.2.4  Universal Agentic Gateway (JSON-RPC/WebSocket daemon, HMAC-SHA256 attestation, adapter matrix) SCHED  |
+--------------------------------------------------------------------------------------------------------------------+
```

---

## Detailed Milestone Specifications & To-Do Lists

### Phase 1: Foundations & Release Engineering

#### [x] v0.1.0 — Core Foundations (Completed)
- [x] Zero-copy SafeTensors and GGUF v2/v3 memory-mapped parsing via `memmap2`.
- [x] Layer topological sensitivity estimation: SVD spectral energy decay ($\sigma_i / \sigma_0$), condition number ($\kappa = \sigma_{\max} / \sigma_{\min}$), and Fisher information proxy.
- [x] Dynamic programming Pareto precision allocator under strict quality degradation bounds ($Q \ge Q_0 - \epsilon$).
- [x] Native 60 FPS Desktop Studio GUI (`egui`/`eframe`) with 5 visual color palettes.
- [x] Matrix distillation engine: Truncated SVD, Walsh-Hadamard rotation, SmoothQuant, BitNet b1.58, 2:4 structured sparsity.
- [x] Parameter AutoDecider based on layer kurtosis, spectral decay, and depth.
- [x] Live 13-stage Transformer forward token flow simulator.
- [x] Checkpoint history rollback engine and immutable audit ledger.
- [x] Teacher LLM surgery engine via declarative JSON recipes.

#### [x] v0.1.1 — Path Hygiene, CI & Standards (Completed)
- [x] Rebranding to **THME** under **Epiphanes** with `llm-tweaker` backwards-compatible CLI alias.
- [x] Enforcement of the Engineering Playbook (`AGENTS.md`) and non-negotiable guidelines.
- [x] Multi-platform GitHub Actions CI matrix (`ubuntu-latest`, `windows-latest`, `macos-latest`).
- [x] Automated CI Path-Leak Guard rejecting machine paths and developer usernames.
- [x] Release packaging workflow on `v*` tags with stripped binaries (`strip = "symbols"`).
- [x] Zero-warning linter pass (`cargo clippy --all-targets -- -D warnings`).
- [x] JSON schema specification for surgery recipes (`docs/schemas/surgery-recipe-v1.json`).

---

### Phase 2: Format Correctness & Telemetry Hardening

#### [x] v0.1.2 — Format Correctness & Fuzz Testing (Completed)
- [x] Parsing for full GGUF v2/v3 quantization matrix (28 GGML types including `Q4_K_M`, `Q5_K_S`, `Q8_0`, `IQ4_XS`, `IQ3_XXS`).
- [x] Zero-allocation block decoders extracting per-block scales and minimums during weight reading.
- [x] Graceful recovery and diagnostic error reporting on truncated or corrupt model headers (`THMEError::CorruptedHeader`).
- [x] Modular workspace structure decoupling pure IR and algorithms from GUI dependencies.
- [x] 1,000-iteration infallible parser fuzz test suite ensuring zero panics on arbitrary malformed inputs.

#### [x] v0.1.3 — Bring Your Own Algorithm (BYOA), Open Weights Rebuilder & REA-LLM Engine (Completed)
- [x] Bring Your Own Algorithm (`AlgorithmRegistry`) with custom transformation registration and JSON import/export.
- [x] Architecture scaffold synthesizer (`ModelRebuilder`) supporting dimensional remapping (`LowRankProject`, `ZeroPad`).
- [x] REA-LLM & EWHA framework: $80/20$ reach profile $q_l$, per-token memory traffic $W_{\text{bytes/token}}$, and per-token energy consumption $E_{\text{token}}$.
- [x] Layer anatomy deconstruction isolating Attention ($Q, K, V, O$) and MLP ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) blocks.
- [x] Multi-format dataset ingestion supporting JSONL, raw text, and tokenized arrays.
- [x] Desktop Studio GUI integrations: Tab 0 unified EWHA cockpit, Tab 1 Rebuilder wizard, Tab 3 BYOA registry with live 8x8 slice tester, Tab 4 EWHA telemetry grid.

---

### Phase 3: Empirical Quality Verification & Universal Integration

#### [x] v0.1.4 — Empirical Quality Gate Evaluation & Perplexity Verification Bridge (Completed)
- [x] External perplexity evaluation bridge supporting standard test sets (WikiText-2, C4, PTB) and local token scoring harnesses.
- [x] Perplexity degradation calculation: $\Delta PPL = PPL_{\text{candidate}} - PPL_{\text{baseline}}$ with automated quality gate decision ($\Delta PPL \le \epsilon$).
- [x] Automated rollback trigger: immediate checkpoint restoration when degradation exceeds tolerance, emitting structured audit diagnostics.
- [x] CLI `thme validate` command with `--corpus`, `--max-ppl-delta`, and JSON output flags.

#### [x] v0.1.5 — Surgery Daemon, IDE Extensions, PyTorch Bridge & 8 AI Models Matrix (Completed)
- [x] Headless Model Context Protocol (MCP) server (`thme mcp`) exposing 8 specialized LLM surgical tools.
- [x] Packaged Visual Studio Code extension (`editors/vscode/thme-vscode-0.1.5.vsix`) with 4 Explorer sidebar panels.
- [x] GitHub Copilot Language Model Tools integration (`thme_copilot_inspect`, `thme_copilot_surgery`, `thme_copilot_validate`).
- [x] PyTorch & CUDA fine-tuning interoperability bridge (`src/interop/`):
  - Pure Rust generator for `train_cuda.py` supporting LoRA, QLoRA (BitsAndBytes 4-bit), Unsloth Fast, and DeepSpeed ZeRO-2/3.
  - Hugging Face `config.json` generator for LLaMA, Mistral, Gemma, Qwen, and DeepSeek model families.
  - PEFT LoRA adapter checkpoint parser and zero-copy weight merger ($W = W_0 + \frac{\alpha}{r} B \times A$).
  - CLI `thme export` / `thme interop` command.
- [x] Universal 8 Specialized AI Model Paradigm Converter (`src/ir/paradigm.rs`):
  - Detection, ingestion, and cross-conversion matrix across 8 paradigms: LLM, LCM, LAM, MoE, VLM, SLM, MLM, SAM.
  - Sparse upcycling engine partitioning dense MLPs into parallel expert matrices and synthesized router gating tensors.
  - BitNet b1.58 ternary discretization delivering up to 85.3% footprint reduction for mobile and edge deployments.
  - CLI `thme convert` command with full parameter customization.
  - Declarative surgery recipe support (`TweakOperation::ConvertParadigm`) and Studio GUI converter in Tab 1.

#### [x] v0.1.6 — Speculative Decoding & Draft Model Synthesizer (Completed)
- [x] Draft model manifold synthesizer (`src/speculative/synthesizer.rs`):
  - Sub-model manifold extraction via prefix layers ($0..K$), strided layer sampling, low-rank projection, or BitNet b1.58 ternary discretization.
  - Automatic head re-anchoring, vocabulary mapping, and dimensional scaling.
  - Parameter reduction calculation (e.g. 76.8% memory savings) and latency cost ratio computation $r = \frac{T_{\text{draft}}}{T_{\text{target}}}$.
- [x] Multi-token speculative rejection sampling verification engine (`src/speculative/engine.rs`):
  - Speculative candidate verification under temperature-calibrated rejection sampling: $u \le \min\left(1, \frac{p(x)}{q(x)}\right)$.
  - Verification telemetry: mean acceptance rate $\alpha_{\text{spec}}$, emitted tokens per step $\mathbb{E}[k] + 1$, and Leviathan effective wall-clock speedup:
    $$S = \frac{\mathbb{E}[k] + 1}{1 + \gamma \cdot r}$$
  - Dynamic horizon search identifying optimal draft token budget $\gamma^* = \arg\max_\gamma S(\gamma)$.
- [x] Tree-based speculation topology (`src/speculative/tree.rs`):
  - Multi-candidate branching topology (Medusa/Eagle style speculative trees).
  - 2D causal tree attention mask synthesis enforcing strict causal masking across tree paths.
- [x] CLI `thme speculate` subcommand with full telemetry output, tree options, and draft model export.
- [x] IDE & Agent MCP tool `thme_simulate_speculative_decoding`.
- [x] Studio GUI Benchmarks (Tab 4) live speculative simulation deck with acceptance curve and speedup multiplier gauges.
- [x] Declarative surgery recipe support (`TweakOperation::ExtractDraftModel`).

---

### Phase 4: Hardware Profiling, Serialization & Closed-Loop Surgery

#### [ ] v0.1.7 — Hardware-Calibrated Roofline & Micro-Kernel Profiler (Active)
Provides an empirical, hardware-calibrated operational Roofline model and multi-tier memory hierarchy cache simulator. Diagnoses whether Transformer layers are memory-bandwidth bound or compute-bound under varying hardware targets, batch sizes, and sequence lengths.

- **Mathematical Foundations**:
  - Operational Arithmetic Intensity:
    $$I = \frac{\text{Total Floating Point Operations (FLOPs)}}{\text{Total Memory Traffic (Bytes)}} \quad \left[\frac{\text{FLOPs}}{\text{byte}}\right]$$
  - Layer Arithmetic Intensity Formulas:
    - Linear Projection ($X W$ with $X \in \mathbb{R}^{B \times D_{\text{in}}}$, $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$):
      $$I_{\text{linear}} = \frac{2 B D_{\text{in}} D_{\text{out}}}{B D_{\text{in}} b_x + D_{\text{in}} D_{\text{out}} b_w + B D_{\text{out}} b_y}$$
      At batch size $B=1$ (autoregressive generation), $I_{\text{linear}} \approx \frac{2}{b_w} \approx 1.0 \text{ FLOPs/byte}$ for FP16, rendering linear layers strictly memory-bound.
    - Standard Scaled Dot-Product Attention (Sequence length $S$, Head dimension $d$):
      $$\text{FLOPs}_{\text{attn}} = 4 B H S^2 d, \quad \text{Bytes}_{\text{attn}} = 2 B H S^2 b_{\text{score}} + 4 B H S d b_{\text{kv}}$$
    - FlashAttention-2 Tiled Attention (SRAM tile size $M$):
      $$\text{Bytes}_{\text{flash}} = \mathcal{O}\left(B H S d \cdot \frac{S d}{M}\right) \implies I_{\text{flash}} \gg I_{\text{standard}}$$
    - SwiGLU FFN Block ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ with intermediate size $D_{\text{ffn}} = \frac{8}{3} D$):
      $$\text{FLOPs}_{\text{ffn}} = 6 B D D_{\text{ffn}}, \quad \text{Bytes}_{\text{ffn}} = 3 D D_{\text{ffn}} b_w + B (2 D + 2 D_{\text{ffn}}) b_{\text{act}}$$
  - Hardware Operational Ceiling Knee:
    $$I^* = \frac{\text{Peak Compute Performance (TFLOP/s)}}{\text{Peak Memory Bandwidth (GB/s)}} \quad \left[\frac{\text{FLOPs}}{\text{byte}}\right]$$
  - Attainable Performance Boundary:
    $$P(I) = \min\left(P_{\text{peak}}, I \times B_{\text{peak}}\right) \quad \left[\text{TFLOP/s}\right]$$
  - Multi-Tier Memory Latency & Bandwidth Degradation:
    $$T_{\text{layer}} = \sum_{k \in \{\text{L1}, \text{L2}, \text{L3}, \text{DRAM}\}} \left( M_k \cdot \text{Latency}_k + \frac{\text{Volume}_k}{\text{Bandwidth}_k} \right)$$
    where $M_k$ is the cache miss count at tier $k$.

- **To-Do List**:
  - [ ] **Hardware Target Database (`src/profiling/hardware.rs`)**:
    - Predefined hardware specifications: NVIDIA H100 SXM5 (1,979 TFLOP/s FP16 Tensor, 3,350 GB/s HBM3), NVIDIA RTX 4090 (330 TFLOP/s FP16 Tensor, 1,008 GB/s GDDR6X), Apple M3 Max (38 TFLOP/s FP16, 400 GB/s Unified), Apple M2 Ultra (54 TFLOP/s FP16, 800 GB/s Unified), AMD Instinct MI300X (2,610 TFLOP/s FP16, 5,300 GB/s HBM3), Intel Xeon Platinum 8490H (AVX-512, 10 TFLOP/s, 307 GB/s DDR5).
    - User-definable custom hardware specs via JSON config (`--hardware-config <path>`).
  - [ ] **Layer-by-Layer Operational Intensity Profiler (`src/profiling/intensity.rs`)**:
    - Compute exact FLOP counts and memory transfer volumes for each layer type (Linear, RMSNorm, Multi-Head Attention, SwiGLU, Rotary Embeddings).
    - Support variable batch size $B \in [1, 512]$ and sequence length $S \in [1, 32768]$.
    - Account for KV-cache read/write traffic during decode phase vs prefill phase.
  - [ ] **4-Tier Memory Hierarchy Cache Simulator (`src/profiling/cache_sim.rs`)**:
    - Simulation tiers: L1 Data (128 KB, 1 cycle), L2 Cache (4-16 MB, 15 cycles), L3/SLC (32-96 MB, 50 cycles), Main Memory (HBM3/GDDR6/DDR5, 200 cycles).
    - Calculate working set size for each operation and predict cache line eviction rates.
    - Estimate effective memory latency penalty based on cache miss rates.
  - [ ] **Empirical Roofline Model Engine (`src/profiling/roofline.rs`)**:
    - Generate log-log Roofline plot coordinates: peak compute ceiling, memory bandwidth slope, operational boundary knee $I^*$.
    - Map each model layer as a coordinate $(I_l, P_l)$ on the Roofline chart.
    - Categorize layers into `MemoryBound` (below knee $I < I^*$) vs `ComputeBound` (at or above knee $I \ge I^*$).
    - Output optimization suggestions (e.g. "Layer 12 MLP is memory-bound; apply BitNet b1.58 or 2:4 sparsity to halve memory traffic").
  - [ ] **Thermal & Power Scaling Projection (`src/profiling/thermal.rs`)**:
    - Model TDP dissipation: dynamic power $P_{\text{dyn}} = C V^2 f \alpha_{\text{activity}}$ plus static leakage power.
    - Predict thermal throttling onset under continuous sustained token generation vs burst prefill.
  - [ ] **CLI Subcommand `thme roofline` (`src/cli/roofline.rs`)**:
    - Flags: `--model <path>`, `--hardware <id>`, `--batch <N>`, `--seq-len <S>`, `--phase <prefill|decode>`, `--json`.
    - Print ASCII Roofline diagram with plotted layer operational points.
  - [ ] **Headless MCP Tool `thme_profile_roofline` (`src/cli/mcp.rs`)**:
    - Expose Roofline profiling and bottleneck diagnosis to IDEs and AI coding agents.
  - [ ] **Desktop Studio GUI Roofline Deck (Tab 4)**:
    - Interactive log-log Roofline plot using `egui_plot` with hover tooltips displaying layer names, intensity, and efficiency percentage.
    - Interactive sliders for batch size, sequence length, and hardware target selection.
  - [ ] **Comprehensive Test Suite (`tests/test_roofline_profiler.rs`)**:
    - Unit tests validating intensity math, cache hit estimation, and Roofline categorization.

---

#### [ ] v0.1.8 — BitNet b1.58 Calibrated Scale Quantization Exporter (Planned)
Enables direct binary disk serialization of optimized models into deployable GGUF v3 and SafeTensors files, featuring BitNet b1.58 ternary packed weight serialization (2 bits per weight: $-1, 0, +1$ packed 4 weights per byte) with calibrated per-channel scale factors.

- **Mathematical Foundations**:
  - Weight Discretization:
    $$\alpha = \frac{1}{n m} \sum_{i=1}^n \sum_{j=1}^m |W_{ij}|, \quad \tilde{W}_{ij} = \text{Round}\left(\text{Clip}\left(\frac{W_{ij}}{\alpha}, -1, 1\right)\right) \in \{-1, 0, +1\}$$
  - 2-Bit Packing Format (4 ternary values per `u8` byte):
    $$\text{byte} = \sum_{k=0}^3 \text{code}(\tilde{W}_k) \ll (2k)$$
    Encoding mapping: $\text{code}(-1) = \text{0b00}$, $\text{code}(0) = \text{0b01}$, $\text{code}(+1) = \text{0b10}$, $\text{0b11}$ reserved.
  - Scale Factor Calibration:
    Per-channel MSE-optimal scale search:
    $$\alpha_i^* = \arg\min_{\alpha} \sum_{j=1}^m \left( W_{ij} - \alpha \cdot \text{Round}\left(\text{Clip}\left(\frac{W_{ij}}{\alpha}, -1, 1\right)\right) \right)^2$$
  - Memory Footprint Reduction:
    $$\text{Compression Ratio} = \frac{16 \text{ bits/weight}}{2 \text{ bits/weight}} = 8.0\times \quad (\approx 87.5\% \text{ reduction})$$

- **To-Do List**:
  - [ ] **BitNet b1.58 Vectorized Bitstream Packer (`src/exporter/bitnet_pack.rs`)**:
    - SIMD-accelerated packing converting slices of `i8` values $\in \{-1, 0, +1\}$ into packed `u8` buffers.
    - Vectorized unpacker reconstructing de-quantized floating point values with scale multiplication.
  - [ ] **Scale Factor Calibration Engine (`src/exporter/calibration.rs`)**:
    - Support per-tensor, per-channel (row-wise), and per-block (group size 64/128) scale calculation.
    - Implement MSE-optimal scale refinement via 1D grid search.
  - [ ] **Direct Pure Rust GGUF v3 Binary Writer (`src/exporter/gguf_writer.rs`)**:
    - Serialize GGUF v3 binary specification from pure Rust without external C/C++ dependencies:
      - Magic header `GGUF` (`0x46554747`).
      - Version integer (`3`).
      - Tensor count and key-value metadata count.
      - Standard metadata key-values: `general.architecture`, `general.name`, `llama.block_count`, `llama.context_length`, `llama.embedding_length`, `llama.feed_forward_length`, `llama.attention.head_count`.
      - Tensor info array: name string, dimension array, type ID (`GGML_TYPE_Q8_0`, custom BitNet type), offset byte.
      - 32-byte memory alignment padding.
      - Continuous raw weight payload buffer.
  - [ ] **Direct Pure Rust SafeTensors Binary Writer (`src/exporter/safetensors_writer.rs`)**:
    - Write SafeTensors format:
      - 8-byte unsigned little-endian JSON header length $N$.
      - UTF-8 JSON header string with tensor metadata (`dtype`, `shape`, `data_offsets`).
      - Memory-aligned binary tensor data payload.
  - [ ] **Surgery Pipeline Exporter Integration (`src/exporter/mod.rs`)**:
    - Direct export from in-memory `ModelIR` following any sequence of surgery operations (pruning, SVD, BitNet, LoRA merge).
    - Streaming writer for large models avoiding full-model duplicate RAM allocation.
  - [ ] **Downstream Runtime Compatibility Verification (`tests/test_exporter_compatibility.rs`)**:
    - Automated test writing a synthetic GGUF model and loading it via `llama.cpp` / `ollama` test fixtures.
    - Checksum and bitwise equivalence validation on round-trip serialization.
  - [ ] **CLI Subcommand `thme export` (`src/cli/export.rs`)**:
    - Flags: `--input <model_path>`, `--format <gguf|safetensors>`, `--quant <bitnet-b1.58|q8_0|fp16>`, `--output <path>`, `--verify`.
  - [ ] **Studio GUI Exporter Wizard (Tab 1 Models Manager)**:
    - Export dialog with format selection, target bit-width, destination path picker, live progress bar, and SHA-256 verification badge.

---

#### [ ] v0.1.9 — Automated Teacher-Student Co-Surgery & Distillation Pipeline (Planned)
Autonomous closed-loop optimization daemon: an external or local Teacher LLM analyzes layer sensitivity, generates declarative surgery recipes, THME applies them, verifies quality via the perplexity bridge, and iteratively tunes the Pareto frontier.

- **Mathematical Foundations**:
  - Multi-Objective Loss Function:
    $$\min_{\theta} \mathcal{J}(\theta) = \lambda_{\text{ppl}} \Delta PPL(\theta) + \lambda_{\text{mem}} W_{\text{token}}(\theta) + \lambda_{\text{energy}} E_{\text{token}}(\theta) - \lambda_{\text{speed}} S_{\text{speedup}}(\theta)$$
    subject to:
    $$\Delta PPL(\theta) \le \epsilon_{\max}, \quad \text{Memory}(\theta) \le M_{\text{budget}}$$
  - Pareto Hypervolume Maximization:
    Iterative sampling of candidate configurations $\theta = (r_1..r_L, b_1..b_L, p_1..p_L)$ maximizing the non-dominated Pareto frontier volume:
    $$\mathcal{S}^* = \left\{ \theta \in \Theta \mid \nexists \theta' \text{ such that } \theta' \succ \theta \right\}$$
  - Self-Healing Step Function:
    If validation rejects candidate $\theta^{(k)}$ due to $\Delta PPL > \epsilon_{\max}$:
    $$\theta^{(k+1)} = \text{Relax}(\theta^{(k)}, \nabla_{\theta} \text{Sensitivity})$$
    where high-sensitivity layers revert to higher precision while low-sensitivity layers absorb additional compression.

- **To-Do List**:
  - [ ] **Co-Surgery Optimization State Machine (`src/surgery/closed_loop.rs`)**:
    - Multi-stage loop: Inspect Topology $\to$ Synthesize Recipe $\to$ Apply Surgery $\to$ Evaluate $\Delta PPL \to$ Pareto Update $\to$ Accept/Rollback.
    - Configurable termination criteria: target parameter budget achieved, max iterations reached, or Pareto frontier convergence.
  - [ ] **Multi-Objective Pareto Solver (`src/optimization/pareto_solver.rs`)**:
    - 3D Pareto frontier tracker: Storage/Memory reduction vs Perplexity degradation vs Telemetry speedup.
    - Dominated solution pruning and Pareto hypervolume metric tracking.
  - [ ] **Teacher LLM Prompt Synthesizer & Adapter (`src/surgery/teacher_agent.rs`)**:
    - Synthesizes compact, structured prompt context describing top-K sensitive layers, spectral decay slopes, and EWHA reach profiles.
    - Accepts structured JSON surgery recipes from Teacher models (via CLI stdin, MCP tool, or HTTP API).
  - [ ] **Automated Self-Healing Rollback & Parameter Relaxation**:
    - Automatic fallback logic restoring last known good checkpoint when quality gates trip.
    - Parameter relaxation heuristic backing off pruning or quantization on the single most sensitive layer identified by audit diagnostics.
  - [ ] **CLI Subcommand `thme co-surgery` (`src/cli/co_surgery.rs`)**:
    - Flags: `--model <path>`, `--target-reduction <%>`, `--max-ppl-delta <float>`, `--max-iterations <N>`, `--teacher <command|api>`, `--auto-accept`.
  - [ ] **Desktop Studio GUI Closed-Loop Dashboard (Tab 2 Surgery)**:
    - Live iteration timeline showing candidate proposals, real-time $\Delta PPL$ bars, EWHA energy reduction gauges, and accepted/rejected badges.
  - [ ] **Integration Test Suite (`tests/test_co_surgery.rs`)**:
    - Simulated Teacher agent driving 5 iterations of automated surgery with verified convergence.

---

### Phase 5: Production Core Release

#### [ ] v0.2.0 — Production-Ready Verified Release (Target)
The verified production milestone demonstrating end-to-end profiling, optimization, export, and verification on real open-weights models (SmolLM-135M / LLaMA-3.2-1B) with published reproducible results.

- **To-Do List**:
  - [ ] **Production Model Verification Suite (`examples/production_verification.rs`)**:
    - Automated test fixture ingesting genuine open-weights checkpoints (SmolLM-135M and LLaMA-3.2-1B).
    - Complete end-to-end pipeline: SafeTensors ingest $\to$ layer sensitivity profiling $\to$ Pareto BitNet/SVD allocation $\to$ GGUF export $\to$ WikiText-2 perplexity evaluation.
  - [ ] **Reproducible Benchmark Script & Dossier Generator (`scripts/benchmark_suite.py` / `.sh`)**:
    - Generates markdown and CSV report containing: baseline vs optimized parameter count, VRAM consumption, decode tokens/sec, WikiText-2 PPL ($\Delta PPL \le 0.45$), and energy per token.
  - [ ] **Multi-Platform Release Packaging**:
    - Strip symbols, LTO optimization (`lto = "fat"`, `codegen-units = 1`).
    - Pre-compiled binaries for:
      - Linux: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`
      - Windows: `x86_64-pc-windows-msvc`
      - macOS: `x86_64-apple-darwin`, `aarch64-apple-darwin` (Apple Silicon)
  - [ ] **Cryptographic Binary Signing & Integrity**:
    - Minisign / GPG signatures on all release binaries.
    - SHA-256 checksum manifest (`SHA256SUMS.txt`).
  - [ ] **Official Documentation, Man Pages & Walkthrough**:
    - Comprehensive User Manual (`docs/manual.md`).
    - Unix man pages (`docs/man/thme.1`).
    - High-resolution architectural diagrams and showcase walkthroughs.

---

### Phase 6: Tiered Hardware Execution & Universal Agentic Connectivity

#### [ ] v0.2.1 — Direct GPU Driver Execution Engine (Vulkan / Metal / CUDA Driver API) (Scheduled)
Direct pure Rust GPU driver execution engine bypassing external Python runtimes, C++ frameworks, or PyTorch wheels. Communicates directly with hardware drivers via pure Rust bindings.

- **Architectural Specifications**:
  - NVIDIA CUDA Driver API via pure Rust `cudarc`:
    - Direct calls to `libcuda.so` / `nvcuda.dll` bypassing CUDA runtime (`libcudart`) and Python.
    - JIT compilation of embedded PTX kernels or pre-compiled CUBIN binaries.
    - Asynchronous CUDA streams and event synchronization.
  - Vulkan 1.3 Compute Shaders via `ash`:
    - Cross-platform GPU execution across Linux, Windows, and Android.
    - Pre-compiled SPIR-V compute shaders for low-bit matrix multiplication and activation layers.
    - Vulkan pipeline barriers and memory descriptors.
  - Apple Metal 3 via `metal-rs`:
    - Native execution on macOS and iOS Apple Silicon unified memory architectures.
    - Metal Performance Shaders (MPS) graph integration and custom MSL compute shaders.
  - Fused Low-Bit Execution Kernels:
    - Fused BitNet b1.58 GEMM ($Y = \tilde{W} X$ with 2-bit unpacked on-the-fly into tensor registers).
    - Fused RMSNorm: $\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$ in a single thread block pass.
    - Fused SwiGLU: $\text{SwiGLU}(x) = (x W_{\text{gate}} \cdot \text{sigmoid}(x W_{\text{gate}})) \odot (x W_{\text{up}})$.

- **To-Do List**:
  - [ ] **Hardware Abstraction Layer (HAL) Traits (`src/hardware/hal.rs`)**:
    - Define traits: `HardwareDevice`, `DeviceMemoryBuffer`, `CommandQueue`, `KernelLaunchConfig`.
  - [ ] **CUDA Driver Backend (`src/hardware/cuda.rs`)**:
    - Context initialization, unified memory allocation (`cuMemAllocManaged`), async stream dispatch.
  - [ ] **Vulkan 1.3 Compute Backend (`src/hardware/vulkan.rs`)**:
    - Instance/device creation, physical device queue enumeration, SPIR-V pipeline loader.
  - [ ] **Apple Metal 3 Backend (`src/hardware/metal.rs`)**:
    - Metal device creation, command buffer queue, shared memory buffer allocation without memory copies.
  - [ ] **Micro-Kernel Library (`src/hardware/kernels/`)**:
    - Fused BitNet b1.58 GEMV / GEMM kernel.
    - Fused RMSNorm kernel.
    - Fused SwiGLU activation kernel.
    - Rotary Position Embedding (RoPE) kernel.
  - [ ] **CLI Subcommand `thme run` (`src/cli/run.rs`)**:
    - Execute model inference directly via selected GPU backend: `thme run --model <path> --backend <cuda|vulkan|metal|cpu> --prompt "..."`.

---

#### [ ] v0.2.2 — Tiered Heterogeneous Memory & Out-of-Core NVMe SSD Streaming (Scheduled)
Three-tier hierarchical memory pipeline (GPU VRAM $\leftrightarrow$ Host RAM $\leftrightarrow$ NVMe SSD direct asynchronous I/O) enabling 70B parameter model inference and architecture surgery on consumer systems with 8 GB VRAM.

- **Architectural Specifications**:
  - Asynchronous Direct I/O Engine:
    - Linux: `io_uring` direct disk block access bypassing OS page cache buffer copies.
    - Windows: Overlapped I/O with `FILE_FLAG_NO_BUFFERING` via Windows Win32 API.
    - PCIe 4.0/5.0 NVMe streaming achieving $6,000 - 12,000 \text{ MB/s}$ sustained sequential read throughput.
  - Layer-by-Layer Sequential Execution (AirLLM-Style):
    - Load layer $k$ weights from NVMe into pinned host memory $\to$ DMA transfer to GPU VRAM $\to$ execute layer forward pass $\to$ reuse VRAM buffer for layer $k+1$.
  - Double-Buffering Asynchronous Prefetch Pipeline:
    - Stream layer $k+1$ from NVMe into staging buffer in parallel while GPU compute core processes layer $k$.
  - REA-LLM Tier Placement:
    - Hot layers ($q_l = 1.0$, early attention) permanently pinned in GPU VRAM.
    - Warm layers ($q_l \ge 0.50$, middle layers) cached in Host RAM.
    - Deep layers ($q_l \le 0.20$) streamed on-demand from NVMe SSD.

- **To-Do List**:
  - [ ] **Direct I/O Storage Engine (`src/storage/direct_io.rs`)**:
    - Linux `io_uring` backend with submission/completion ring management.
    - Windows Overlapped I/O backend with completion ports (`IOCP`).
  - [ ] **Layer Streaming Prefetch Queue (`src/storage/layer_streamer.rs`)**:
    - Double-buffered prefetch pipeline with async futures.
    - Pin memory pages (`mlock` on Unix, `VirtualLock` on Windows) for direct GPU DMA.
  - [ ] **Tiered Memory Accountant (`src/memory/tiered_allocator.rs`)**:
    - Real-time VRAM/RAM/NVMe capacity budgeting and layer placement policy.
    - Automated eviction of inactive layer activations.
  - [ ] **70B Model Inference Benchmark (`tests/test_out_of_core_streaming.rs`)**:
    - Synthetic 70B model streaming forward pass completing inside an 8 GB memory constraint.

---

#### [ ] v0.2.3 — Distributed Heterogeneous Multi-Node Clustering (Pipeline & Tensor Parallelism) (Scheduled)
Distributed execution across heterogeneous nodes (e.g. Mac Studio M2 Ultra + Linux workstation with RTX 4090 + Windows laptop) with pure Rust asynchronous TCP/RDMA ring all-reduce communication.

- **Architectural Specifications**:
  - Intra-Node Tensor Parallelism (TP):
    - Split column-parallel Linear ($W_{\text{gate}}, W_{\text{up}}, W_Q, W_K, W_V$) and row-parallel Linear ($W_{\text{down}}, W_O$) across local GPUs.
  - Inter-Node Pipeline Parallelism (PP):
    - Partition model layers $[0..L-1]$ into $P$ sequential stages assigned to distributed cluster nodes over standard network links.
    - 1F1B (One-Forward-One-Backward) schedule minimizing activation pipeline bubbles.
  - Pure Rust Ring All-Reduce (`src/cluster/comm.rs`):
    - Asynchronous peer-to-peer ring topology over non-blocking TCP sockets.
    - Fallback to RDMA / RoCE (InfiniBand verbs) where high-speed interconnect hardware is detected.
  - Fault Tolerance & Dynamic Re-Sharding:
    - Heartbeat pings (100 ms interval); automatic re-sharding of remaining layers upon node disconnection.

- **To-Do List**:
  - [ ] **Cluster Node Discovery & Handshake (`src/cluster/discovery.rs`)**:
    - Multicast / mDNS node discovery and capability exchange (VRAM size, compute TFLOPs).
  - [ ] **Ring All-Reduce Implementation (`src/cluster/ring.rs`)**:
    - Pure Rust Scatter-Reduce and All-Gather stages over async streams.
  - [ ] **Pipeline Parallelism Scheduler (`src/cluster/pipeline.rs`)**:
    - Activation token buffer routing between sequential node stages.
  - [ ] **Fault Tolerance Monitor (`src/cluster/watchdog.rs`)**:
    - Heartbeat detection, dead-node failover, and layer reallocation.
  - [ ] **CLI Subcommand `thme cluster` (`src/cli/cluster.rs`)**:
    - Commands: `thme cluster init`, `thme cluster join <master_ip>`, `thme cluster status`.

---

#### [ ] v0.2.4 — Universal Agentic AI Connectivity Gateway (CLI, Harness, Desktop, IDE) (Scheduled)
High-performance headless JSON-RPC 2.0 daemon and bidirectional WebSocket event streaming gateway connecting autonomous AI agents, CI/CD harnesses, desktop apps, and IDE copilots.

- **Architectural Specifications**:
  - Headless JSON-RPC 2.0 / WebSocket Daemon:
    - High-throughput asynchronous server (`thme daemon --port 9090`) handling concurrent agent requests.
    - Bi-directional event stream publishing: per-token throughput, layer execution progress, thermal throttling, memory pressure.
  - Universal Agent Adapter Matrix:
    - **Autonomous CLI Agents**: Standard MCP tool schemas and CLI subprocess bridge for Claude Code, Gemini CLI, Aider, OpenHands.
    - **CI/CD & Orchestration Harnesses**: Automated validation, delta-PPL quality gating, and deployment gates for Harness CI/CD, GitHub Actions, GitLab CI.
    - **Desktop Studio Application**: Native WebSockets connection between `thme-gui` and background daemons running on remote GPU servers.
    - **IDEs & AI Copilots**: Enhanced extension backends for VS Code, Cursor, Windsurf, and GitHub Copilot Language Model Tools.
  - Cryptographic Recipe Attestation:
    - HMAC-SHA256 digital signatures on all executed surgery recipes ensuring verified provenance and zero unauthorized weight tampering.

- **To-Do List**:
  - [ ] **WebSocket Streaming Gateway Daemon (`src/daemon/server.rs`)**:
    - Async WebSocket event bus broadcasting real-time model telemetry and forward execution stats.
  - [ ] **JSON-RPC 2.0 Method Router (`src/daemon/rpc.rs`)**:
    - Remote procedure execution for all THME capabilities: inspect, autodecide, surgery, validate, speculate, roofline, export.
  - [ ] **HMAC-SHA256 Recipe Attestation (`src/daemon/attestation.rs`)**:
    - Cryptographic signing and validation of surgery recipe JSON payloads.
  - [ ] **Agent Adapter Connectors (`src/daemon/adapters/`)**:
    - Adapters for Harness pipelines, GitHub Actions workflows, CLI agent protocols.
  - [ ] **CLI Subcommand `thme daemon` (`src/cli/daemon.rs`)**:
    - Flags: `--port <N>`, `--bind <ip>`, `--auth-token <secret>`, `--allow-remote-surgery`.
  - [ ] **Integration Test Suite (`tests/test_agentic_gateway.rs`)**:
    - Concurrent WebSocket client connections executing surgery, receiving streaming telemetry, and validating attestation signatures.

---

## Long-Term Research Initiatives (Deferred to v0.3.0+)

These advanced algorithmic directions are tracked in the architecture backlog and intentionally deferred to subsequent major milestones:
- **Topological Hyper-Graph Neural Architecture Search**: Automated micro-circuit rewiring using topological data analysis (TDA) persistent homology to detect and prune redundant manifold loops.
- **Continuous Online RL Adaptation**: Real-time layer manifold adjustment guided by continuous reinforcement learning feedback loops during live conversational inference.
- **Dynamic Mixture-of-Experts Sub-Manifold Synthesis**: Real-time conditional routing dynamically spawning expert sub-networks based on input token semantic entropy.
