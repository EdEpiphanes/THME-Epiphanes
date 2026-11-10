# THME Engineering Backlog (v0.1.2 -> v0.2.0)

> **Epiphanes Engineering Backlog**
> Actionable ticket ledger tracking engineering issues, acceptance criteria, and completion conditions through v0.2.0.

---

## Sprint 1: Format Correctness & Fuzzing (Milestone v0.1.2) [COMPLETED]

### [THME-101] Full GGUF v2/v3 Quantization Type Decoder (Done)
- **Component**: `thme::parser::gguf`
- **Problem**: Current GGUF reader detects quant type IDs but lacks full block header decoders for `Q4_K_M`, `Q5_K_S`, `Q8_0`, and `IQ4_XS` super-blocks.
- **Acceptance Criteria**:
  - Implement zero-allocation block structures for all standard llama.cpp quant types.
  - Compute exact per-block scale and min values during weight extraction.
  - Validate against standard test fixtures without runtime allocations.
- **Done Means**: `cargo test test_gguf_all_quant_types` passes for synthetic headers representing each GGUF type.

### [THME-102] Truncated File & Malformed Header Recovery (Done)
- **Component**: `thme::parser`
- **Problem**: Corrupted downloads or partial transfers can cause premature parser abortion without actionable diagnostic contexts.
- **Acceptance Criteria**:
  - Parser returns structured `THMEError::CorruptedHeader` with byte offset and expected vs actual size.
  - Implement partial header recovery to salvage intact layer metadata from damaged files.
  - Never panic under any slice bounds.
- **Done Means**: Property test feeding 100,000 random byte mutations to SafeTensors and GGUF parsers yields 0 panics.

### [THME-103] Workspace Crate Modularization (`thme-core`, `thme-cli`, `thme-gui`)
- **Component**: Cargo Workspace
- **Problem**: Monolithic crate structure couples lightweight headless algorithms with heavy `egui`/`eframe` dependencies.
- **Acceptance Criteria**:
  - Extract pure IR, parsers, and math algorithms into `crates/thme-core`.
  - Extract terminal commands into `crates/thme-cli`.
  - Retain desktop studio in `crates/thme-gui`.
- **Done Means**: `cargo check -p thme-core --no-default-features` builds in < 5 seconds without GUI dependencies.

---

## Sprint 2: Real Telemetry & SIMD Vector Kernels (Milestone v0.1.3)

### [THME-201] AVX2 / AVX-512 / ARM NEON Walsh-Hadamard Transform
- **Component**: `thme::transform::hadamard`
- **Problem**: Scalar FWHT (Fast Walsh-Hadamard Transform) is single-threaded and CPU-bound for large hidden dimensions ($D=4096$).
- **Acceptance Criteria**:
  - Implement vector butterfly stages using portable SIMD or explicit AVX2 / NEON intrinsics.
  - Fall back gracefully to scalar implementation on non-AVX systems.
  - Ensure zero numerical deviation from scalar baseline ($|y_{\text{simd}} - y_{\text{scalar}}| < 10^{-6}$).
- **Done Means**: Benchmark shows $\ge 4.5\times$ speedup on 4096-dim vectors with AVX2 enabled.

### [THME-202] Criterion Micro-Benchmark Suite in CI
- **Component**: `benches/`
- **Problem**: Telemetry figures currently rely on simulator estimates rather than empirical hardware benchmarks.
- **Acceptance Criteria**:
  - Add `benches/bench_matrix_ops.rs` measuring FWHT, SVD power iteration, and BitNet discretization.
  - Add `benches/bench_token_throughput.rs` measuring single-token forward simulation latency.
  - Integrate criterion reports into CI artifacts.
- **Done Means**: `cargo bench` executes in CI and generates automated regression comparison markdown.

---

## Sprint 3: Empirical Quality Verification (Milestone v0.1.4) [COMPLETED]

### [THME-301] External Perplexity Evaluation Bridge (Done)
- **Component**: `thme::quality::evaluator`
- **Problem**: Quality gates require ground-truth perplexity ($PPL$) evaluation on benchmark corpora (WikiText-2, C4).
- **Acceptance Criteria**:
  - Implement IPC/subprocess bridge to `llama.cpp/perplexity` or standalone inference runner.
  - Stream progress and token loss in real-time to GUI Quality tab.
  - Calculate empirical degradation $\Delta PPL = PPL_{\text{opt}} - PPL_{\text{base}}$.
- **Done Means**: `thme validate --model <path> --test-set wikitext2` runs end-to-end and outputs structured pass/fail gate verdict.

### [THME-302] Automated Anti-Degradation Quality Gate (Done)
- **Component**: `thme::quality::gate`
- **Problem**: Models must automatically reject optimizations that exceed allowable perplexity degradation ($\epsilon = 2.0\%$).
- **Acceptance Criteria**:
  - Enforce rollback if candidate exceeds $\epsilon$.
  - Output Pareto frontier curve comparing storage reduction vs $PPL$ impact.
- **Done Means**: Quality gate test triggers automatic rollback when degradation threshold is violated.

---

## Sprint 4: Surgery Daemon, IDE Extensions & PyTorch Bridge (Milestone v0.1.5) [COMPLETED]

### [THME-401] Model Context Protocol (MCP) Server (`thme mcp`) (Done)
- **Component**: `thme::cli::mcp`
- **Problem**: Autonomous agents, IDEs, and Teacher LLMs need standard JSON-RPC tools to inspect model topology and apply surgery.
- **Acceptance Criteria**:
  - Standard JSON-RPC 2024-11-05 protocol implementation over stdin/stdout.
  - Expose 7 specialized tools: inspect model, autodecide, apply surgery, validate quality, simulate token flow, run experiment, micro-train.
- **Done Means**: `cargo test --test test_mcp` verifies tool discovery, schema generation, and execution.

### [THME-402] VS Code Extension & GitHub Copilot Integration (Done)
- **Component**: `editors/vscode`
- **Problem**: Developers need native IDE access to model topologies, surgery recipes, and Copilot tools.
- **Acceptance Criteria**:
  - Packaged extension (`thme-vscode-0.1.5.vsix`) with 4 explorer sidebars.
  - GitHub Copilot Language Model Tools integration (`thme_copilot_inspect`, `thme_copilot_surgery`, `thme_copilot_validate`).
- **Done Means**: Extension packages cleanly with zero warnings and installs in VS Code.

### [THME-403] PyTorch & CUDA Ecosystem Interoperability Bridge (Done)
- **Component**: `thme::interop`
- **Problem**: Practitioners requiring traditional fine-tuning need seamless migration to and from PyTorch/CUDA environments.
- **Acceptance Criteria**:
  - Pure Rust generator for `train_cuda.py` supporting LoRA, QLoRA (BitsAndBytes 4-bit), Unsloth Fast, and DeepSpeed ZeRO-2/3.
  - Hugging Face `config.json` generator for major architectures (LLaMA, Mistral, Gemma, Qwen, DeepSeek).
  - PEFT LoRA adapter zero-copy weight merger ($W = W_0 + \frac{\alpha}{r} B \times A$).
- **Done Means**: `cargo test --test test_interop` passes for script generation, configs, and LoRA math.

### [THME-404] Universal 8 Specialized AI Model Paradigm Converter (Done)
- **Component**: `thme::ir::paradigm`, `thme::cli::convert`
- **Problem**: Ingesting and cross-converting diverse model architectures (LLM, LCM, LAM, MoE, VLM, SLM, MLM, SAM) requires universal tensor manifold transformations.
- **Acceptance Criteria**:
  - Implement heuristic and structural detection across all 8 specialized AI paradigms.
  - Universal cross-conversion matrix with sparse upcycling (MoE), BitNet b1.58 ternary discretization (SLM), vision tower augmentation (VLM/SAM), neuro-symbolic intent routing (LAM), latent concept space projection (LCM), bidirectional masking (MLM), and dense consolidation (LLM).
  - CLI `thme convert` command with full parameter customization.
  - Desktop Studio GUI converter integrated in Tab 1 (Models Manager).
  - Declarative surgery recipe support (`TweakOperation::ConvertParadigm`) and MCP tool `thme_convert_paradigm`.
- **Done Means**: `cargo test --test test_model_paradigm_conversion` passes for detection, recipe surgery, and all 8 model round-trip conversions.

---

## Sprint 5: Speculative Decoding & Draft Synthesizer (Milestone v0.1.6) [COMPLETED]

### [THME-501] Draft Model Manifold Synthesizer (Done)
- **Component**: `thme::speculative::synthesizer`
- **Problem**: Generating tokens with multi-billion parameter models autoregressively is bottlenecked by memory bandwidth; smaller draft models are needed to propose candidate tokens.
- **Acceptance Criteria**:
  - Extract lightweight prefix draft sub-manifolds (e.g. 2 to 4 layers) from parent target models.
  - Support dimension downscaling and BitNet b1.58 ternary discretization for low-latency CPU/edge execution.
  - Calculate empirical parameter reductions, memory savings, and latency cost ratios $r = \frac{T_{\text{draft}}}{T_{\text{target}}}$.
- **Done Means**: `DraftSynthesizer::extract_draft_model` generates valid `ModelIR` sub-networks with matching vocabularies and heads.

### [THME-502] Multi-Token Tree Speculation Engine (Done)
- **Component**: `thme::speculative::engine`, `thme::speculative::tree`
- **Problem**: Validating speculative candidate sequences requires mathematically sound rejection sampling and multi-branch tree attention.
- **Acceptance Criteria**:
  - Implement speculative rejection sampling with temperature scaling.
  - Compute acceptance rate $\alpha_{\text{spec}}$, emitted tokens per step $\mathbb{E}[k] + 1$, and Leviathan effective wall-clock speedup $S = \frac{\mathbb{E}[k]+1}{1 + \gamma \cdot r}$.
  - Multi-candidate tree topology synthesis with causal 2D tree attention mask generation.
  - CLI `thme speculate` command and MCP tool `thme_simulate_speculative_decoding`.
  - Desktop Studio GUI Tab 4 Speculative Deck with live stat gauges and simulation runner.
- **Done Means**: `cargo test --test test_speculative_decoding` passes 5/5 tests and verifies effective speedups $> 1.5\times$.

---

## Sprint 6: Production Exporter & Release (Milestone v0.2.0)

### [THME-601] Quantization Model Exporter (GGUF / SafeTensors)
- **Component**: `thme::exporter`
- **Problem**: Modifying layer precisions and weights currently operates in-memory; users need to export the optimized model to disk.
- **Acceptance Criteria**:
  - Serialize modified tensors into valid GGUF and SafeTensors files.
  - Write BitNet b1.58 ternary packed bitstreams and per-channel FP16 scale vectors.
  - Generated GGUF files load and execute cleanly in downstream runtimes (`llama.cpp`, Ollama).
- **Done Means**: Exported model from `thme export` successfully runs text generation inference.

### [THME-502] End-to-End Small Model Demonstration
- **Component**: `examples/`
- **Problem**: Users need a verifiable, reproducible demo of a real model being profiled, optimized, and evaluated.
- **Acceptance Criteria**:
  - Provide automated download and profiling script for a real open-weights model (e.g. SmolLM-135M or Qwen-2.5-0.5B).
  - Produce public verification report showing exact compression ratio, throughput gain, and zero-shot accuracy preservation.
- **Done Means**: Script runs on clean machine in < 3 minutes and outputs verified comparison table.

---

## Sprint 6: Tiered Hardware Execution & Universal Agentic Connectivity (Milestones v0.2.1 – v0.2.4)

### [THME-1101] Direct Pure Rust GPU Driver Engine
- **Component**: `thme::hardware::drivers`
- **Problem**: Relying on external runtimes introduces heavy dependencies; THME requires direct driver communication.
- **Acceptance Criteria**:
  - NVIDIA CUDA Driver API integration via `cudarc` (pure Rust, bypasses Python).
  - Vulkan 1.3 SPIR-V compute pipeline for Windows and Linux.
  - Apple Metal 3 Performance Shaders integration for Apple Silicon unified memory.
- **Done Means**: Device discovery enumerates GPUs and runs a benchmark vector-add kernel on CUDA, Vulkan, and Metal.

### [THME-1102] Unified Hardware Abstraction Layer
- **Component**: `thme::hardware::hal`
- **Problem**: Heterogeneous backends have distinct memory and execution paradigms.
- **Acceptance Criteria**:
  - Traits: `HardwareDevice`, `DeviceMemoryBuffer`, `KernelLaunchConfig`.
  - Unified memory allocation, host-to-device transfers, and kernel dispatch.
- **Done Means**: Layer forward pass executes identically across CPU, CUDA, and Vulkan backends.

### [THME-1103] Out-of-Core NVMe SSD Layer Streaming Engine
- **Component**: `thme::storage::nvme_stream`
- **Problem**: Large 70B parameter models exceed consumer GPU VRAM capacity.
- **Acceptance Criteria**:
  - Asynchronous direct I/O: `io_uring` on Linux, Overlapped I/O on Windows.
  - Sequential layer streaming (AirLLM-style) directly from NVMe SSD (PCIe 4.0/5.0 at 6,000+ MB/s).
  - Double-buffered prefetch queue: compute layer $N$ while asynchronously loading layer $N+1$.
- **Done Means**: Full 70B parameter forward pass completes in 8 GB VRAM without out-of-memory errors.

### [THME-1104] Tiered Heterogeneous Memory Manager
- **Component**: `thme::memory::tiered`
- **Problem**: Optimal performance requires intelligent paging between VRAM, Host RAM, and NVMe SSD.
- **Acceptance Criteria**:
  - Automated tier placement based on layer reach probability $q_l$: hot layers ($q_l = 1.0$) pinned in VRAM, warm layers in Host RAM, deep layers ($q_l = 0.20$) streamed from SSD.
  - Zero-copy memory-mapped staging buffers.
- **Done Means**: Memory accountant maintains strict target VRAM budget under arbitrary layer depths.

### [THME-1105] Heterogeneous Multi-Node Pipeline & Tensor Parallelism
- **Component**: `thme::cluster::parallel`
- **Problem**: Distributed execution across mixed clusters (e.g. Mac Studio + Linux RTX) requires hybrid parallelism.
- **Acceptance Criteria**:
  - Intra-node Tensor Parallelism (TP) for split matrix multiplications.
  - Inter-node Pipeline Parallelism (PP) assigning sequential layer partitions across cluster nodes.
- **Done Means**: Distributed forward pass executes across two networked physical nodes.

### [THME-1106] Pure Rust Async TCP/RDMA Ring All-Reduce
- **Component**: `thme::cluster::comm`
- **Problem**: Multi-node synchronization requires low-latency communication without MPI or NCCL dependencies.
- **Acceptance Criteria**:
  - Pure Rust asynchronous Ring All-Reduce over TCP sockets with RDMA / RoCE fallback.
  - Automatic heartbeats, network partition detection, and dynamic layer re-sharding on node departure.
- **Done Means**: 100 MB gradient buffer synchronizes across 4 nodes with verified bitwise consistency.

### [THME-1107] Universal Agentic Gateway Daemon & WebSocket Streaming
- **Component**: `thme::daemon::gateway`
- **Problem**: Autonomous AI agents need continuous real-time telemetry and programmatic control.
- **Acceptance Criteria**:
  - Headless JSON-RPC 2.0 daemon and WebSocket event stream (`thme daemon --port 9090`).
  - Cryptographic HMAC-SHA256 signature verification on executed surgery recipes.
  - Real-time telemetry broadcast: token throughput, layer progress, thermal and memory pressure.
- **Done Means**: Client agent receives continuous streaming telemetry and dispatches recipe over WebSocket.

### [THME-1108] Universal Agent Adapter Matrix
- **Component**: `thme::daemon::adapters`
- **Problem**: Diverse agent ecosystems (CLI, CI/CD harnesses, Desktop apps, IDE copilots) need native connectors.
- **Acceptance Criteria**:
  - Adapters for: Autonomous CLI Agents (Claude Code, Gemini CLI, Aider), Orchestration Harnesses (Harness CI/CD, GitHub Actions), Native Studio Desktop (`thme-gui`), and IDEs (VS Code, Copilot).
- **Done Means**: All 4 client archetypes connect, inspect, and execute surgery recipes through standardized adapters.
