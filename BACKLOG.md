# THME Engineering Backlog (v0.1.2 -> v0.2.4)

> **Epiphanes Engineering Backlog**
> Actionable ticket ledger tracking engineering issues, acceptance criteria, and completion conditions through v0.2.4.

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

### [THME-103] Workspace Crate Modularization (`thme-core`, `thme-cli`, `thme-gui`) (Done)
- **Component**: Cargo Workspace
- **Problem**: Monolithic crate structure couples lightweight headless algorithms with heavy `egui`/`eframe` dependencies.
- **Acceptance Criteria**:
  - Extract pure IR, parsers, and math algorithms into `crates/thme-core`.
  - Extract terminal commands into `crates/thme-cli`.
  - Retain desktop studio in `crates/thme-gui`.
- **Done Means**: `cargo check -p thme-core --no-default-features` builds in < 5 seconds without GUI dependencies.

---

## Sprint 2: Real Telemetry & SIMD Vector Kernels (Milestone v0.1.3) [COMPLETED]

### [THME-201] AVX2 / AVX-512 / ARM NEON Walsh-Hadamard Transform (Done)
- **Component**: `thme::transform::hadamard`
- **Problem**: Scalar FWHT (Fast Walsh-Hadamard Transform) is single-threaded and CPU-bound for large hidden dimensions ($D=4096$).
- **Acceptance Criteria**:
  - Implement vector butterfly stages using portable SIMD or explicit AVX2 / NEON intrinsics.
  - Fall back gracefully to scalar implementation on non-AVX systems.
  - Ensure zero numerical deviation from scalar baseline ($|y_{\text{simd}} - y_{\text{scalar}}| < 10^{-6}$).
- **Done Means**: Benchmark shows $\ge 4.5\times$ speedup on 4096-dim vectors with AVX2 enabled.

### [THME-202] Criterion Micro-Benchmark Suite in CI (Done)
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
  - Expose 8 specialized tools: inspect model, autodecide, apply surgery, validate quality, simulate token flow, run experiment, micro-train, convert paradigm.
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

## Sprint 6: Hardware-Calibrated Roofline & Micro-Kernel Profiler (Milestone v0.1.7) [ACTIVE]

### [THME-601] Hardware Target Specifications & Machine Balance Database
- **Component**: `thme::profiling::hardware`
- **Problem**: Calculating Roofline boundaries requires calibrated peak compute (TFLOP/s) and peak memory bandwidth (GB/s) parameters for diverse GPUs and CPUs.
- **Acceptance Criteria**:
  - Predefined profile library: NVIDIA H100 SXM5, NVIDIA RTX 4090, Apple M3 Max, Apple M2 Ultra, AMD Instinct MI300X, Intel Xeon Platinum 8490H (AVX-512).
  - Compute machine balance knee $I^* = \frac{\text{Peak TFLOP/s}}{\text{Peak Bandwidth GB/s}}$ for each hardware entry.
  - Support external JSON hardware specification loading via CLI flag `--hardware-config`.
- **Done Means**: `cargo test test_hardware_database` validates specs, units, and knee values across all predefined hardware profiles.

### [THME-602] Layer Operational Arithmetic Intensity Profiler
- **Component**: `thme::profiling::intensity`
- **Problem**: Determining whether layers are compute-bound or memory-bound requires exact FLOP and byte transfer tracking per Transformer sub-block.
- **Acceptance Criteria**:
  - Compute exact FLOPs and memory traffic for Attention ($Q, K, V, O$ projections, attention matrix math) and SwiGLU FFN ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$).
  - Distinguish batch size $B=1$ autoregressive decode traffic (strictly memory-bound) from $B \ge 64$ prompt prefill traffic (compute-bound).
  - Track KV cache memory traffic across varying context lengths $S \in [1, 32768]$.
- **Done Means**: Intensity calculator outputs exact FLOP/byte values for all layers of a 32-layer LLaMA architecture.

### [THME-603] 4-Tier Memory Hierarchy Cache Simulator
- **Component**: `thme::profiling::cache_sim`
- **Problem**: Transformer layers exhibit varying working set sizes that fit into or spill out of L1, L2, L3/SLC, or main memory, incurring steep latency penalties.
- **Acceptance Criteria**:
  - Model 4 cache tiers: L1 Data (128 KB, 1 cycle), L2 (4-16 MB, 15 cycles), L3/SLC (32-96 MB, 50 cycles), Main Memory (HBM3/GDDR6/DDR5, 200 cycles).
  - Calculate working set size per layer and predict cache line eviction rates.
  - Estimate effective memory latency penalty based on cache miss rates.
- **Done Means**: Cache simulator accurately predicts higher latency degradation on layers whose working sets exceed L2 cache capacity.

### [THME-604] Empirical Roofline Model Generator, CLI `thme roofline`, and Tab 4 Deck
- **Component**: `thme::profiling::roofline`, `thme::cli::roofline`, `thme::gui::app`
- **Problem**: Developers need visual and CLI feedback mapping each layer against hardware ceilings to guide precision pruning.
- **Acceptance Criteria**:
  - Generate log-log Roofline plot data: compute ceiling, bandwidth ceiling, and plotted layer coordinates $(I_l, P_l)$.
  - Classify layers as `MemoryBound` vs `ComputeBound` and emit concrete optimization recommendations.
  - Implement CLI `thme roofline` with ASCII diagram and JSON output.
  - Implement headless MCP tool `thme_profile_roofline`.
  - Integrate interactive Roofline plot in Desktop Studio Tab 4 with hover tooltips and hardware selection dropdown.
- **Done Means**: `cargo test --test test_roofline_profiler` passes and verifies correct classification across batch sizes.

---

## Sprint 7: BitNet b1.58 Calibrated Scale Quantization Exporter (Milestone v0.1.8) [PLANNED]

### [THME-701] BitNet b1.58 Vectorized Bitstream Packer
- **Component**: `thme::exporter::bitnet_pack`
- **Problem**: Storing 1.58-bit ternary weights ($\{-1, 0, +1\}$) requires dense bit packing (2 bits per weight, 4 weights per byte) with scale extraction.
- **Acceptance Criteria**:
  - Implement SIMD bit-packing converting slices of ternary values into packed `u8` buffers.
  - Implement unpacking routines with per-channel FP16 scale multiplication.
  - Per-channel MSE-optimal scale factor search $\arg\min_\alpha \|W - \alpha \tilde{W}\|_F^2$.
- **Done Means**: Round-trip pack and unpack achieves bitwise fidelity and $\ge 8.0\times$ storage reduction.

### [THME-702] Pure Rust GGUF v3 Binary Writer & Metadata Synchronizer
- **Component**: `thme::exporter::gguf_writer`
- **Problem**: Deploying optimized models to runtime engines (`llama.cpp`, Ollama) requires direct serialization to GGUF v3 binary files from pure Rust.
- **Acceptance Criteria**:
  - Serialize GGUF v3 header magic (`GGUF`), version 3, tensor count, and kv metadata count.
  - Serialize architecture metadata (`general.architecture`, `llama.block_count`, etc.).
  - Enforce 32-byte alignment padding for tensor data payloads.
  - Stream tensor weight buffers without in-memory duplicate allocations.
- **Done Means**: Serialized GGUF file passes header validation and tensor index integrity checks.

### [THME-703] Pure Rust SafeTensors Serializer & Streaming Output
- **Component**: `thme::exporter::safetensors_writer`
- **Problem**: Deploying to Hugging Face, vLLM, or PyTorch ecosystems requires standard SafeTensors serialization.
- **Acceptance Criteria**:
  - Write 8-byte little-endian header length prefix.
  - Write formatted JSON metadata block with data offsets, shapes, and dtypes.
  - Stream raw binary tensor payload buffers aligned to 8-byte boundaries.
- **Done Means**: Serialized SafeTensors file loads directly via `safetensors` reader with identical tensor shapes and values.

### [THME-704] CLI `thme export` and Downstream Runtime Testbed
- **Component**: `thme::cli::export`
- **Problem**: End-users need a one-command export tool to write optimized models to disk.
- **Acceptance Criteria**:
  - CLI `thme export --input <model> --format <gguf|safetensors> --quant <bitnet-b1.58|q8_0|fp16> --output <path>`.
  - Desktop Studio GUI export wizard in Tab 1 with progress bar and SHA-256 verification.
  - Test fixture running inference on exported GGUF artifact.
- **Done Means**: Exported model successfully runs text generation inference in verification testbed.

---

## Sprint 8: Automated Teacher-Student Co-Surgery Pipeline (Milestone v0.1.9) [PLANNED]

### [THME-801] Closed-Loop Co-Surgery State Machine & Iteration Controller
- **Component**: `thme::surgery::closed_loop`
- **Problem**: Optimizing multi-layer architectures manually is slow; an autonomous feedback loop is required to iteratively propose, apply, and evaluate surgery recipes.
- **Acceptance Criteria**:
  - Implement iterative state machine: Profile Topology $\to$ Synthesize Recipe $\to$ Apply Transformation $\to$ Evaluate $\Delta PPL \to$ Pareto Update $\to$ Commit/Rollback.
  - Support configurable stopping conditions: target parameter budget, max iterations, or Pareto frontier stabilization.
- **Done Means**: State machine completes 10 automated iterations without manual intervention.

### [THME-802] Multi-Objective 3D Pareto Optimizer
- **Component**: `thme::optimization::pareto_solver`
- **Problem**: Compressing models involves trade-offs between perplexity degradation ($\Delta PPL$), inference memory traffic ($W_{\text{bytes/token}}$), and energy consumption ($E_{\text{token}}$).
- **Acceptance Criteria**:
  - Implement 3D Pareto dominance tracker over candidate parameter configurations.
  - Compute Pareto hypervolume indicator to measure optimization progress.
  - Output non-dominated Pareto optimal model checkpoints.
- **Done Means**: Optimizer discovers and tracks the non-dominated Pareto frontier across 50 simulated candidates.

### [THME-803] Self-Healing Rollback Watchdog & Parameter Relaxation Engine
- **Component**: `thme::surgery::self_healing`
- **Problem**: Over-aggressive compression can destroy model coherence ($\Delta PPL > \epsilon_{\max}$), requiring automated recovery.
- **Acceptance Criteria**:
  - Automated checkpoint rollback when validation loss exceeds quality gate tolerance.
  - Parameter relaxation algorithm: automatically restores precision on the most sensitive layer while applying compression to lower-sensitivity layers.
- **Done Means**: Self-healing test triggers rollback on bad candidate and successfully converges on relaxed alternative.

### [THME-804] CLI `thme co-surgery` & Desktop Studio Tab 2 Dashboard
- **Component**: `thme::cli::co_surgery`, `thme::gui::app`
- **Problem**: Operators need visibility into closed-loop progress and real-time loss curves.
- **Acceptance Criteria**:
  - CLI `thme co-surgery --model <path> --target-reduction 50% --max-ppl-delta 0.5 --iterations 10`.
  - Desktop Studio GUI Tab 2 dashboard showing iteration history, Pareto scatter plot, and live quality bars.
- **Done Means**: CLI and GUI display live progress during automated co-surgery runs.

---

## Sprint 9: Production Core Release (Milestone v0.2.0) [TARGET]

### [THME-901] Open-Weights Production Testbed (SmolLM-135M / LLaMA-3.2-1B)
- **Component**: `examples/production_verification.rs`
- **Problem**: Release claims require rigorous, reproducible verification on genuine open-weights checkpoints.
- **Acceptance Criteria**:
  - Automated script downloading and validating SmolLM-135M and LLaMA-3.2-1B checkpoints.
  - End-to-end execution: SafeTensors ingest $\to$ sensitivity profiling $\to$ Pareto BitNet b1.58 surgery $\to$ GGUF export $\to$ WikiText-2 PPL validation.
- **Done Means**: Full pipeline executes in < 3 minutes on a standard developer machine and passes quality gate ($\Delta PPL \le 0.45$).

### [THME-902] Reproducible Verification Script & Benchmark Dossier
- **Component**: `scripts/benchmark_suite.py`, `docs/benchmark_dossier.md`
- **Problem**: Benchmark results must be completely transparent, hardware-specified, and reproducible by third parties.
- **Acceptance Criteria**:
  - Script generates markdown and CSV comparison tables: parameter count, memory footprint, tokens/sec, WikiText-2 PPL, and EWHA energy.
  - Explicit hardware attribution (GPU model, driver version, memory bandwidth).
- **Done Means**: Generated dossier matches reported release specifications with zero manual edits.

### [THME-903] Release Artifact Signing & Multi-Platform Packaging
- **Component**: Release CI Pipeline
- **Problem**: Production binaries must be secure, tamper-evident, and cross-platform.
- **Acceptance Criteria**:
  - Cryptographic Minisign / GPG signatures on all release binaries.
  - SHA-256 manifest (`SHA256SUMS.txt`).
  - Pre-compiled stripped binaries for Linux (x86_64, aarch64), Windows (x64), and macOS (Apple Silicon, Intel).
- **Done Means**: CI release workflow produces verified signed assets on release tag.

---

## Sprint 10: Direct GPU Driver Execution Engine (Milestone v0.2.1) [SCHEDULED]

### [THME-1001] Unified Hardware Abstraction Layer (HAL) Traits
- **Component**: `thme::hardware::hal`
- **Problem**: Direct driver execution across CUDA, Vulkan, and Metal requires a uniform memory and kernel dispatch interface.
- **Acceptance Criteria**:
  - Define core traits: `HardwareDevice`, `DeviceMemoryBuffer`, `CommandQueue`, `KernelLaunchConfig`.
  - Unified memory allocation, host-to-device transfers, and kernel launch semantics.
- **Done Means**: Generic layer forward test executes identically across mock CPU, CUDA, and Vulkan backends.

### [THME-1002] NVIDIA CUDA Driver API Backend via Pure Rust `cudarc`
- **Component**: `thme::hardware::cuda`
- **Problem**: External PyTorch runtimes introduce gigabytes of dependencies; THME requires direct driver communication.
- **Acceptance Criteria**:
  - Pure Rust CUDA Driver API integration using `cudarc` (bypasses Python runtime).
  - JIT compilation of embedded PTX kernels or pre-compiled CUBIN modules.
  - Asynchronous stream dispatch and event synchronization.
- **Done Means**: Vector-add and matrix-multiply kernels execute on NVIDIA GPU without Python or PyTorch installed.

### [THME-1003] Vulkan 1.3 SPIR-V Compute Pipeline Backend
- **Component**: `thme::hardware::vulkan`
- **Problem**: Cross-platform GPU execution across Linux, Windows, and AMD/Intel GPUs requires standard Vulkan compute shaders.
- **Acceptance Criteria**:
  - Initialize Vulkan 1.3 compute pipelines via `ash`.
  - Load pre-compiled SPIR-V shaders for GEMM and activation layers.
  - Descriptor set binding and pipeline barrier memory synchronization.
- **Done Means**: Forward pass executes successfully on Vulkan compute device.

### [THME-1004] Apple Metal 3 Performance Shaders Backend
- **Component**: `thme::hardware::metal`
- **Problem**: Apple Silicon unified memory requires native Metal 3 integration for zero-copy memory transfers.
- **Acceptance Criteria**:
  - Metal device initialization and command queue dispatch via `metal-rs`.
  - Zero-copy shared memory buffer allocation between CPU and GPU.
  - Integration with Metal Performance Shaders (MPS) and custom MSL compute shaders.
- **Done Means**: Forward pass executes on Apple Silicon Mac with unified memory zero-copy buffers.

### [THME-1005] Fused Low-Bit Micro-Kernels (BitNet GEMM, RMSNorm, SwiGLU)
- **Component**: `thme::hardware::kernels`
- **Problem**: Separate kernel launches for normalization and activation introduce memory round-trips.
- **Acceptance Criteria**:
  - Fused BitNet b1.58 GEMV / GEMM kernel unpacking 2-bit ternary weights in register files.
  - Fused RMSNorm kernel computing variance and scaling in a single thread block pass.
  - Fused SwiGLU activation kernel combining gate and up projections.
- **Done Means**: Fused kernels achieve $\ge 1.8\times$ speedup over separate unfused operations.

---

## Sprint 11: Out-of-Core NVMe SSD Layer Streaming (Milestone v0.2.2) [SCHEDULED]

### [THME-1101] Direct Asynchronous I/O Engine (`io_uring` & Overlapped I/O)
- **Component**: `thme::storage::direct_io`
- **Problem**: OS page cache introduces double-buffering overhead during high-speed SSD model streaming.
- **Acceptance Criteria**:
  - Linux: direct asynchronous disk block reading via `io_uring` with submission/completion queues.
  - Windows: Overlapped I/O with `FILE_FLAG_NO_BUFFERING` via Win32 completion ports (`IOCP`).
  - Achieve $> 6,000 \text{ MB/s}$ sequential throughput on PCIe 4.0/5.0 NVMe drives.
- **Done Means**: Benchmark reads 10 GB model tensor data with zero OS buffer caching and verified data integrity.

### [THME-1102] Sequential Layer Streaming & Double-Buffered Prefetch Pipeline
- **Component**: `thme::storage::layer_streamer`
- **Problem**: 70B parameter models require 140 GB of storage; consumer GPUs have 8-16 GB VRAM.
- **Acceptance Criteria**:
  - AirLLM-style sequential layer loading: load layer $k$ into pinned RAM $\to$ DMA to GPU $\to$ execute $\to$ reuse buffer for $k+1$.
  - Double-buffered prefetch pipeline: stream layer $k+1$ from SSD in parallel while GPU computes layer $k$.
- **Done Means**: Layer streaming forward pass completes with GPU compute overlapping disk I/O.

### [THME-1103] REA-LLM Tiered Heterogeneous Memory Placement Manager
- **Component**: `thme::memory::tiered_allocator`
- **Problem**: Heterogeneous systems have distinct memory tiers with differing latency and capacity.
- **Acceptance Criteria**:
  - Tier placement policy driven by layer reach probability $q_l$: hot layers in VRAM, warm layers in Host RAM, deep layers on NVMe SSD.
  - Zero-copy pinned memory staging buffers.
- **Done Means**: Memory manager enforces hard 8 GB VRAM limit while executing full model forward pass.

### [THME-1104] 70B Model in 8 GB VRAM End-to-End Testbed
- **Component**: `tests/test_out_of_core_streaming.rs`
- **Problem**: Validating out-of-core streaming requires end-to-end multi-layer execution under memory constraints.
- **Acceptance Criteria**:
  - Ingest synthetic 70B parameter topology (80 layers, $D=8192$).
  - Execute full autoregressive token step on host with strictly limited 8 GB VRAM footprint.
- **Done Means**: Full forward step succeeds without out-of-memory aborts or segmentation faults.

---

## Sprint 12: Distributed Heterogeneous Multi-Node Clustering (Milestone v0.2.3) [SCHEDULED]

### [THME-1201] Intra-Node Tensor Parallelism & Inter-Node Pipeline Parallelism
- **Component**: `thme::cluster::parallel`
- **Problem**: Scaling across heterogeneous nodes requires combining Tensor Parallelism locally and Pipeline Parallelism across the network.
- **Acceptance Criteria**:
  - Intra-node Tensor Parallelism splitting attention and MLP projections across local accelerators.
  - Inter-node Pipeline Parallelism distributing layer partitions across networked nodes.
  - 1F1B schedule minimizing activation pipeline bubbles.
- **Done Means**: Distributed forward pass splits layers across 2 nodes and produces identical output tokens.

### [THME-1202] Pure Rust Asynchronous TCP/RDMA Ring All-Reduce
- **Component**: `thme::cluster::ring`
- **Problem**: Multi-node communication must avoid heavy MPI or NCCL runtime dependencies.
- **Acceptance Criteria**:
  - Pure Rust Ring All-Reduce over asynchronous non-blocking TCP streams.
  - Hardware RDMA / RoCE fallback when InfiniBand hardware is available.
  - Bandwidth saturation $> 85\%$ of theoretical network link capacity.
- **Done Means**: Multi-node tensor synchronization test verifies bitwise consistency across 4 nodes.

### [THME-1203] Cluster Heartbeat Watchdog & Dynamic Fault-Tolerant Re-Sharding
- **Component**: `thme::cluster::watchdog`
- **Problem**: Nodes in heterogeneous ad-hoc clusters may disconnect or fail unexpectedly.
- **Acceptance Criteria**:
  - Continuous heartbeat monitoring with 100 ms timeout window.
  - Automatic layer re-sharding redistributing orphaned layers to remaining active nodes.
- **Done Means**: Simulated node drop triggers automated re-sharding and forward pass recovers without termination.

---

## Sprint 13: Universal Agentic AI Connectivity Gateway (Milestone v0.2.4) [SCHEDULED]

### [THME-1301] High-Performance WebSocket Streaming Daemon & JSON-RPC Router
- **Component**: `thme::daemon::server`
- **Problem**: Autonomous AI agents, CI/CD harnesses, desktop apps, and IDEs require concurrent programmatic control and live telemetry.
- **Acceptance Criteria**:
  - Headless daemon (`thme daemon --port 9090`) handling concurrent JSON-RPC 2.0 requests.
  - Real-time WebSocket streaming broadcasting per-token progress, layer execution, and thermal metrics.
- **Done Means**: Concurrent clients connect, invoke surgery methods, and receive real-time telemetry streams.

### [THME-1302] Cryptographic HMAC-SHA256 Surgery Recipe Attestation
- **Component**: `thme::daemon::attestation`
- **Problem**: Remote agent surgery poses security risks; recipes must be verified and tamper-proof.
- **Acceptance Criteria**:
  - Cryptographic HMAC-SHA256 signing of surgery recipe payloads with shared secrets or public keys.
  - Daemon rejects unauthenticated or altered recipe JSON payloads with structured error responses.
- **Done Means**: Tampered recipe payload is rejected with `AttestationVerificationFailed` error.

### [THME-1303] Universal Agent Adapter Matrix (CLI, CI/CD, Desktop, IDEs)
- **Component**: `thme::daemon::adapters`
- **Problem**: Diverse agent ecosystems require dedicated integration bridges.
- **Acceptance Criteria**:
  - Autonomous CLI Agent adapter (Claude Code, Gemini CLI, Aider).
  - CI/CD Orchestration Harness adapter (Harness CI/CD, GitHub Actions).
  - Desktop Studio WebSocket connector (`thme-gui` remote daemon connection).
  - IDE extension connector (VS Code, Copilot LM Tools).
- **Done Means**: All 4 client archetypes connect, authenticate, and execute surgery recipes via standard adapters.
