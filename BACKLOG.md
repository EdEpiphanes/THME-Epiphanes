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
- **Component**: `thme::profiling::hardware` (`src/profiling/hardware.rs`)
- **Problem**: Calculating Roofline boundaries requires calibrated peak compute (TFLOP/s) and peak memory bandwidth (GB/s) parameters across heterogeneous hardware platforms (NVIDIA, Apple Silicon, AMD, Intel).
- **Acceptance Criteria**:
  - Implement predefined hardware specifications: NVIDIA H100 SXM5 (989 TFLOP/s FP16, 3.35 TB/s HBM3), NVIDIA RTX 4090 (82.6 TFLOP/s FP16, 1.01 TB/s GDDR6X), Apple M3 Max (31.2 TFLOP/s FP16, 400 GB/s Unified), Apple M2 Ultra (54.4 TFLOP/s FP16, 800 GB/s Unified), AMD Instinct MI300X (1307 TFLOP/s FP16, 5.3 TB/s HBM3), and Intel Xeon Platinum 8490H (5.0 TFLOP/s AVX-512, 307 GB/s DDR5).
  - Compute machine balance knee $I^* = \frac{\text{Peak TFLOP/s}}{\text{Peak Bandwidth GB/s}}$ [FLOPs/byte] for each device.
  - Support external JSON hardware specification loading via CLI flag `--hardware-config <path>`.
- **Production Pipeline / CI/CD Impact**: Establishes authoritative reference ceilings for automated benchmark regression gates and multi-hardware CI target profiling.
- **Done Means**: `cargo test test_hardware_database` passes and validates specs, units, and knee values across all 6 predefined hardware profiles.

### [THME-602] Linear Projection & Attention Operational Arithmetic Intensity Profiler
- **Component**: `thme::profiling::intensity` (`src/profiling/intensity.rs`)
- **Problem**: Determining whether attention projections and matrix multiplications are memory-bandwidth-bound or compute-bound requires tracking exact FLOPs and byte transfers across decode and prefill phases.
- **Acceptance Criteria**:
  - Compute FLOP counts and memory traffic for linear projections ($W_q, W_k, W_v, W_o$) based on tensor shapes, sequence lengths, and precision types.
  - Compute Scaled Dot-Product Attention FLOPs ($2 B H S^2 D_{\text{head}}$) and quadratic activation transfer volumes.
  - Distinguish $B=1$ autoregressive decode traffic (memory-bound, $I \approx 1.0$) from $B \ge 64$ prompt prefill traffic (compute-bound, $I \gg I^*$).
  - Model KV cache memory traffic across context window lengths $S \in [1, 32768]$.
- **Production Pipeline / CI/CD Impact**: Feeds layer classification into the Pareto optimizer to prevent over-optimizing compute-bound layers while neglecting memory-bound bottlenecks.
- **Done Means**: `cargo test test_linear_attention_intensity` passes, confirming exact FLOP and byte calculations for attention layers of LLaMA and Mistral models.

### [THME-603] SwiGLU FFN & FlashAttention-2 Arithmetic Intensity Mathematical Models
- **Component**: `thme::profiling::intensity` (`src/profiling/intensity.rs`)
- **Problem**: Modern LLMs rely heavily on SwiGLU feed-forward networks (accounting for ~65% of total parameters) and tiled FlashAttention-2, which require specialized non-linear intensity models.
- **Acceptance Criteria**:
  - Model SwiGLU FFN FLOP count ($6 B D D_{\text{ffn}}$) and memory traffic ($3 D D_{\text{ffn}} b_w + B(2D + 2D_{\text{ffn}}) b_{\text{act}}$).
  - Model elementwise SiLU activation and Hadamard product arithmetic intensity.
  - Model FlashAttention-2 SRAM tiling memory reduction factor ($\mathcal{O}(B H S^2 d^2 / M_{\text{SRAM}})$) compared to standard attention.
- **Production Pipeline / CI/CD Impact**: Informs precision pruning suggestions for feed-forward blocks in automated surgery recipes and benchmark reports.
- **Done Means**: `cargo test test_swiglu_flashattn_intensity` passes, verifying correct intensity curves for SwiGLU FFN across various hidden dimensions.

### [THME-604] 4-Tier Memory Hierarchy Cache Simulator (L1/L2/L3/DRAM)
- **Component**: `thme::profiling::cache_sim` (`src/profiling/cache_sim.rs`)
- **Problem**: Transformer layers exhibit distinct tensor working set sizes that fit into or spill out of on-chip caches, introducing multi-cycle latency penalties not captured by simple DRAM bandwidth equations.
- **Acceptance Criteria**:
  - Model 4 memory hierarchy tiers: L1 Data (128 KB, 1 cycle / 0.5 ns), L2 Cache (4-16 MB, 15 cycles / 4 ns), L3 / System-Level Cache (32-96 MB, 50 cycles / 15 ns), and Main Memory / VRAM (HBM3/GDDR6/DDR5, 200 cycles / 60 ns).
  - Simulate hit/miss latency penalties: $T = \sum_{k} (M_k \cdot \text{Latency}_k + \frac{\text{Volume}_k}{\text{Bandwidth}_k})$.
  - Track active tensor footprint across layer transitions to determine residency.
- **Production Pipeline / CI/CD Impact**: Provides synthetic cache hit/miss latency projections in headless CI environments without requiring physical hardware performance counters.
- **Done Means**: `cargo test test_cache_simulator_hierarchy` passes and demonstrates stepped latency degradation when tensor footprints exceed L1, L2, and L3 boundaries.

### [THME-605] Working Set Size & Cache Eviction Latency Degradation Model
- **Component**: `thme::profiling::cache_sim` (`src/profiling/cache_sim.rs`)
- **Problem**: Transformer layers with large intermediate activations cause cache thrashing, evicting KV cache pages and model weights, leading to unpredictable latency spikes.
- **Acceptance Criteria**:
  - Calculate working set size $W_{\text{active}} = \text{size}(W_{\text{layer}}) + \text{size}(\text{Activations}) + \text{size}(\text{KV}_{\text{page}})$.
  - Estimate cache line eviction probability using LRU / pseudo-LRU working set simulation.
  - Compute effective memory latency penalty scaling factor based on predicted eviction rates.
- **Production Pipeline / CI/CD Impact**: Enables surgery recipes to identify layers that can fit completely into L2/L3 cache upon BitNet b1.58 ternary quantization.
- **Done Means**: `cargo test test_working_set_eviction_model` validates that quantizing a layer below L2 cache capacity reduces simulated memory stalls by $\ge 60\%$.

### [THME-606] Empirical Roofline Boundary Curve & Memory/Compute Classifier
- **Component**: `thme::profiling::roofline` (`src/profiling/roofline.rs`)
- **Problem**: Practitioners need automated, mathematical classification of each model layer to determine whether precision reduction or kernel fusion is the appropriate optimization.
- **Acceptance Criteria**:
  - Compute attainable performance ceiling: $P(I) = \min(P_{\text{peak}}, I \times B_{\text{peak}})$.
  - Classify layers as `MemoryBound` ($I < I^*$) or `ComputeBound` ($I \ge I^*$).
  - Output efficiency metric $\eta = \frac{P_{\text{observed}}}{P(I)}$ and compute distance to knee $|I - I^*|$.
  - Emit structured optimization recommendations: suggest low-bit quantization for `MemoryBound` and kernel fusion/tensor core acceleration for `ComputeBound`.
- **Production Pipeline / CI/CD Impact**: Enforces automated profiling quality gates, preventing optimizations that target the wrong hardware bottleneck.
- **Done Means**: `cargo test test_roofline_classifier` passes, correctly labeling decode vs prefill across all 32 layers of LLaMA-8B.

### [THME-607] Dynamic TDP Power & Thermal Throttling Projection Engine
- **Component**: `thme::profiling::thermal` (`src/profiling/thermal.rs`)
- **Problem**: Sustained compute-bound operations trigger thermal throttling and TDP power clamping, degrading sustained token throughput compared to burst performance.
- **Acceptance Criteria**:
  - Model dynamic power dissipation $P_{\text{dyn}} = C V^2 f \alpha + P_{\text{static}}$.
  - Project thermal accumulation over time: $T(t) = T_{\text{ambient}} + (T_{\max} - T_{\text{ambient}})(1 - e^{-t/\tau_{\text{thermal}}})$.
  - Predict thermal throttle frequency scaling factor $\beta_{\text{throttle}} \in [0.6, 1.0]$ when $T(t) > T_{\text{crit}}$.
  - Compute sustained vs peak token throughput projections.
- **Production Pipeline / CI/CD Impact**: Integrates thermal and energy constraints into mobile and edge deployment verification benchmarks.
- **Done Means**: `cargo test test_thermal_throttling_model` passes, verifying simulated frequency back-off under sustained high-TDP workloads.

### [THME-608] CLI Subcommand `thme roofline` & MCP Tool `thme_profile_roofline`
- **Component**: `thme::cli::roofline` (`src/cli/roofline.rs`), `thme::cli::mcp` (`src/cli/mcp.rs`)
- **Problem**: Terminal users and autonomous MCP agents need seamless programmatic access to Roofline profiling and JSON reports.
- **Acceptance Criteria**:
  - CLI command `thme roofline --model <path> --hardware <device> --batch-size <B> --seq-len <S>`.
  - Generate ASCII log-log Roofline plot with plotted layer markers and hardware ceiling lines.
  - Support `--json` flag outputting complete telemetry schema (layer coordinates, classifications, recommendations).
  - Implement MCP tool `thme_profile_roofline` for IDE and agent interaction.
- **Production Pipeline / CI/CD Impact**: Enables CI pipelines to capture Roofline telemetry artifacts and agents to inspect model bottlenecks via JSON-RPC.
- **Done Means**: `thme roofline --help` executes cleanly, and unit tests in `tests/test_roofline_profiler.rs` verify CLI and MCP outputs.

### [THME-609] Desktop Studio Tab 4 Interactive SVG/egui Roofline Deck
- **Component**: `thme::gui::app` (`src/gui/app.rs`)
- **Problem**: Visual studio users require an interactive, real-time graphical representation of the Roofline model with dynamic parameter manipulation.
- **Acceptance Criteria**:
  - Integrate interactive Roofline deck in Tab 4 (Benchmarks) using `egui` native rendering.
  - Provide interactive dropdowns for predefined hardware profiles and custom JSON configs.
  - Live sliders for Batch Size ($B = 1 \dots 128$) and Context Length ($S = 128 \dots 32768$) dynamically shifting layer markers.
  - Hover tooltips displaying exact FLOPs, memory transfer bytes, arithmetic intensity, and optimization advice per layer.
- **Production Pipeline / CI/CD Impact**: Provides graphical validation deck for interactive release demos and user research workflows.
- **Done Means**: GUI builds with zero clippy warnings, runs at 60 FPS, and renders the Roofline plot with responsive hover tooltips.

---

## Sprint 7: Binary Quantization Exporter (Milestone v0.1.8) [PLANNED]

### [THME-701] Vectorized BitNet b1.58 Ternary Bitstream Packer
- **Component**: `thme::exporter::bitnet_pack` (`src/exporter/bitnet_pack.rs`)
- **Problem**: Storing 1.58-bit ternary weights ($\{-1, 0, +1\}$) requires dense 2-bit packing (4 weights per byte) with SIMD acceleration to minimize export latency.
- **Acceptance Criteria**:
  - Implement 2-bit encoding: $-1 \mapsto 00_2$, $0 \mapsto 01_2$, $+1 \mapsto 10_2$ ($11_2$ reserved).
  - Vectorized packing kernel compressing slices of `i8` or `f32` ternary states into contiguous `u8` buffers.
  - Support AVX2 / NEON / portable SIMD intrinsics with scalar fallback.
  - Achieve $\ge 8.0\times$ storage reduction compared to FP16 baselines.
- **Production Pipeline / CI/CD Impact**: Core serialization kernel ensuring fast export times in CI packaging jobs.
- **Done Means**: `cargo test test_bitnet_vectorized_packer` passes, verifying exact bitwise round-trip packing fidelity.

### [THME-702] Vectorized Ternary Bitstream Unpacker with FP16 Scale Product
- **Component**: `thme::exporter::bitnet_pack` (`src/exporter/bitnet_pack.rs`)
- **Problem**: Export verification and downstream decoding require high-speed unpacking of 2-bit representations back into floating-point tensors with scale restoration.
- **Acceptance Criteria**:
  - Zero-allocation unpacking routine converting packed `u8` bytes into `f32`/`f16` activation buffers.
  - Vectorized fused unpack-and-multiply applying per-channel scale factors $\alpha_i$ on-the-fly.
  - Guarantee zero loss of numerical accuracy compared to pre-packed ternary values.
- **Production Pipeline / CI/CD Impact**: Used by CI verification harnesses to validate exported artifacts before deployment.
- **Done Means**: `cargo test test_ternary_unpacker_fidelity` passes across 100,000 randomized weight matrices.

### [THME-703] Direct Pure Rust GGUF v3 Binary Writer & 32-Byte Alignment
- **Component**: `thme::exporter::gguf_writer` (`src/exporter/gguf_writer.rs`)
- **Problem**: Deploying models to `llama.cpp` and Ollama requires writing compliant GGUF v3 binary files from pure Rust without external runtime dependencies.
- **Acceptance Criteria**:
  - Serialize GGUF header: magic `GGUF` (`0x46554747`), version 3, tensor count, and key-value metadata count.
  - Write string, array, and numeric metadata records following GGUF v3 specification.
  - Serialize tensor info table (name, dimensions, GGML type, data offset).
  - Enforce strict 32-byte alignment padding before tensor data payloads.
- **Production Pipeline / CI/CD Impact**: Enables THME to produce first-class deployable GGUF files in CI release workflows.
- **Done Means**: `cargo test test_gguf_v3_writer` passes, and output files pass GGUF header parser validation.

### [THME-704] GGUF Metadata Key-Value Reconstitution & Synchronization Engine
- **Component**: `thme::exporter::metadata` (`src/exporter/metadata.rs`)
- **Problem**: Restructuring layers or modifying precision alters model architecture metadata, which must be synchronized into the GGUF header to prevent downstream runtime crashes.
- **Acceptance Criteria**:
  - Automatically update `general.architecture`, `llama.block_count`, `llama.embedding_length`, `llama.feed_forward_length`, and `llama.attention.head_count`.
  - Inject tokenizer vocabulary metadata, BOS/EOS token IDs, and RoPE frequency scaling parameters.
  - Reconcile parameter changes resulting from layer pruning or MoE sparse upcycling.
- **Production Pipeline / CI/CD Impact**: Prevents metadata drift and corrupt model configurations in published model checkpoints.
- **Done Means**: `cargo test test_gguf_metadata_sync` passes, confirming metadata integrity across all 8 paradigm architectures.

### [THME-705] Direct Pure Rust SafeTensors Binary Serializer & 8-Byte Prefix
- **Component**: `thme::exporter::safetensors_writer` (`src/exporter/safetensors_writer.rs`)
- **Problem**: Exporting to PyTorch, Hugging Face `transformers`, and vLLM requires writing compliant SafeTensors binary archives.
- **Acceptance Criteria**:
  - Write 8-byte little-endian unsigned integer indicating JSON header length.
  - Construct formatted, sorted JSON metadata map with tensor data offsets, shapes, and dtypes (`F32`, `F16`, `BF16`, `I8`).
  - Serialize binary tensor payloads aligned to 8-byte boundaries.
  - Optional `__metadata__` block embedding THME surgery recipe provenance and git commit hash.
- **Production Pipeline / CI/CD Impact**: Supplies standard model weights directly to Hugging Face Hub upload pipelines.
- **Done Means**: `cargo test test_safetensors_serializer` passes, and exported files load successfully via the standard `safetensors` crate.

### [THME-706] Low-Memory Streaming Chunked File Writer Engine
- **Component**: `thme::exporter::streaming` (`src/exporter/streaming.rs`)
- **Problem**: Exporting multi-billion parameter models (e.g. 8B to 70B) in one in-memory buffer causes Out-Of-Memory (OOM) crashes on developer workstations.
- **Acceptance Criteria**:
  - Implement buffered chunked streaming writer operating with bounded $\le 64 \text{ MB}$ RAM footprint.
  - Quantize, pack, and stream tensors sequentially layer-by-layer directly to disk.
  - Maintain backpressure and asynchronous chunk flushing.
  - Compute running SHA-256 checksum during streaming.
- **Production Pipeline / CI/CD Impact**: Allows resource-constrained CI runners (7 GB RAM) to export large model weights reliably.
- **Done Means**: `cargo test test_streaming_chunked_writer` exports an 8B synthetic model without exceeding 64 MB of resident memory.

### [THME-707] Calibrated Per-Channel & Per-Group MSE-Optimal Scale Search
- **Component**: `thme::exporter::calibration` (`src/exporter/calibration.rs`)
- **Problem**: Standard absmean scale factors ($\alpha = \frac{1}{N}\|W\|_1$) can cause significant quantization error on outlier distributions; MSE optimization is needed.
- **Acceptance Criteria**:
  - Implement grid search / Golden Section search finding $\alpha^* = \arg\min_\alpha \|W - \alpha \tilde{W}\|_F^2$.
  - Support row-wise, column-wise, and group-wise ($G = 128$) scale factor calibration.
  - Fall back to analytical absmean when matrix variance is uniform.
- **Production Pipeline / CI/CD Impact**: Improves representation fidelity of exported 1.58-bit models, reducing perplexity penalty by up to 0.35 points.
- **Done Means**: `cargo test test_mse_scale_calibration` passes, demonstrating lower Frobenius reconstruction error than baseline absmean.

### [THME-708] CLI Subcommand `thme export` & Desktop Studio Tab 1 Exporter Wizard
- **Component**: `thme::cli::export` (`src/cli/export.rs`), `thme::gui::app` (`src/gui/app.rs`)
- **Problem**: Users need intuitive CLI and graphical controls to export models with chosen formats, bit-widths, and destination paths.
- **Acceptance Criteria**:
  - CLI command `thme export --input <model> --output <path> --format <gguf|safetensors> --quant <bitnet-b1.58|q8_0|fp16>`.
  - CLI progress bar showing per-layer quantization and disk write throughput (MB/s).
  - Desktop Studio Tab 1 Export Wizard modal with target format selectors, scale options, and SHA-256 digest display.
- **Production Pipeline / CI/CD Impact**: Standard entry point for model deployment automation in production CI/CD workflows.
- **Done Means**: `thme export --help` displays clean options, and unit tests verify CLI argument parsing and GUI modal state.

### [THME-709] Downstream Runtime Compatibility Testbed
- **Component**: `tests/test_exporter_compatibility.rs`
- **Problem**: Exported binary artifacts must be validated against downstream runtimes (`llama.cpp`, Ollama, vLLM) to prevent shipping unparseable models.
- **Acceptance Criteria**:
  - Integration test fixture loading exported GGUF files with external or mock `llama.cpp` parser.
  - Verify tensor names, dimensions, and type tags match standard architecture conventions.
  - Verify forward pass numerical stability (zero NaN or Inf values).
- **Production Pipeline / CI/CD Impact**: Serves as the mandatory release quality gate for all quantization exporter updates.
- **Done Means**: `cargo test --test test_exporter_compatibility` passes, verifying 100% format compliance across test fixtures.

---

## Sprint 8: Closed-Loop Teacher-Student Co-Surgery (Milestone v0.1.9) [PLANNED]

### [THME-801] Co-Surgery Closed-Loop State Machine & Controller
- **Component**: `thme::surgery::closed_loop` (`src/surgery/closed_loop.rs`)
- **Problem**: Manually tuning heterogeneous layer precision is labor-intensive; an autonomous closed-loop state machine is needed to orchestrate iterative surgery.
- **Acceptance Criteria**:
  - State machine tracking states: `Idle` $\to$ `ProfileSensitivity` $\to$ `SynthesizeRecipe` $\to$ `ApplySurgery` $\to$ `ValidateQuality` $\to$ `EvaluatePareto` $\to$ `CommitOrRollback`.
  - Support configurable loop budgets: max iterations ($N_{\max}$), target parameter reduction (%), or target perplexity delta ($\Delta PPL_{\max}$).
  - Emit structured telemetry events at each state transition.
- **Production Pipeline / CI/CD Impact**: Enables unattended, continuous model compression runs in nightly optimization pipelines.
- **Done Means**: `cargo test test_closed_loop_state_machine` completes 10 simulated iterations and transitions to `Completed`.

### [THME-802] Teacher LLM Topology Analysis Prompt Synthesizer
- **Component**: `thme::surgery::prompt_synth` (`src/surgery/prompt_synth.rs`)
- **Problem**: External Teacher LLMs require concise, mathematically rich context summarizing layer sensitivities, spectral energy decay, and memory bottlenecks to formulate effective surgery recipes.
- **Acceptance Criteria**:
  - Formulate structured markdown/JSON prompt payloads containing model architecture summary, top-K sensitive layers, condition numbers, and EWHA reach profiles.
  - Include declarative recipe schema specification and strict generation constraints in the prompt.
  - Keep prompt token footprint compact ($\le 2,048$ tokens).
- **Production Pipeline / CI/CD Impact**: Standardizes prompt engineering across LLM backends (Anthropic Claude, Google Gemini, OpenAI, local models).
- **Done Means**: `cargo test test_prompt_synthesizer` passes, verifying correct markdown and JSON prompt structure.

### [THME-803] Headless JSON-RPC & Subprocess Teacher Agent Communication Bridge
- **Component**: `thme::surgery::agent_bridge` (`src/surgery/agent_bridge.rs`)
- **Problem**: The co-surgery engine needs a reliable communication channel to invoke external AI agents (via CLI subprocess, HTTP REST, or MCP JSON-RPC).
- **Acceptance Criteria**:
  - Subprocess bridge executing external CLI agents (`claude`, `gemini`, `curl`).
  - Stdin/stdout pipe handler with timeout protection and stderr logging.
  - Parse generated JSON surgery recipes with automated schema validation.
- **Production Pipeline / CI/CD Impact**: Decouples the Rust execution engine from external AI agent providers.
- **Done Means**: `cargo test test_agent_bridge` passes with mock agent returning valid surgery recipes.

### [THME-804] Multi-Objective 3D Pareto Optimizer
- **Component**: `thme::optimization::pareto_solver` (`src/optimization/pareto_solver.rs`)
- **Problem**: Compression decisions involve trade-offs across 3 competing objectives: perplexity degradation ($\Delta PPL$), memory traffic ($W_{\text{bytes/token}}$), and wall-clock speedup ($S$).
- **Acceptance Criteria**:
  - Track 3D objective vectors $\mathbf{y} = (\Delta PPL, W_{\text{token}}, -S)$.
  - Implement Pareto dominance relation: $\mathbf{a} \prec \mathbf{b}$ iff $a_i \le b_i \ \forall i$ and $\exists j: a_j < b_j$.
  - Maintain dynamic archive of non-dominated Pareto optimal checkpoints.
- **Production Pipeline / CI/CD Impact**: Generates optimal candidate frontiers for deployment engineers to pick based on hardware constraints.
- **Done Means**: `cargo test test_3d_pareto_optimizer` discovers and maintains the non-dominated set across 50 simulated candidates.

### [THME-805] Pareto Hypervolume Indicator Convergence Metric
- **Component**: `thme::optimization::hypervolume` (`src/optimization/hypervolume.rs`)
- **Problem**: Autonomous co-surgery needs an objective mathematical stopping criterion to determine when further iterations yield diminishing returns.
- **Acceptance Criteria**:
  - Calculate hypervolume indicator $\mathcal{S}(\mathcal{P}, \mathbf{r})$ bounded by reference point $\mathbf{r}$.
  - Compute relative hypervolume gain $\Delta \mathcal{S}_k = \frac{\mathcal{S}_k - \mathcal{S}_{k-1}}{\mathcal{S}_{k-1}}$.
  - Trigger termination when $\Delta \mathcal{S}_k < \epsilon_{\text{conv}}$ for 3 consecutive iterations.
- **Production Pipeline / CI/CD Impact**: Eliminates wasteful compute by stopping automated optimization runs once convergence is reached.
- **Done Means**: `cargo test test_hypervolume_convergence` correctly halts simulated optimization upon frontier stabilization.

### [THME-806] Automated Self-Healing Parameter Relaxation Heuristic
- **Component**: `thme::surgery::self_healing` (`src/surgery/self_healing.rs`)
- **Problem**: Aggressive quantization proposals can breach quality tolerances; rather than discarding the entire recipe, fine-grained relaxation should recover quality.
- **Acceptance Criteria**:
  - Identify the single layer contributing highest perplexity degradation.
  - Relax that layer to higher precision (e.g. Ternary $\to$ INT8 or INT8 $\to$ FP16) while keeping other layers compressed.
  - Re-evaluate candidate quality before triggering full rollback.
- **Production Pipeline / CI/CD Impact**: Increases recipe success rate by $> 40\%$ in automated overnight optimization pipelines.
- **Done Means**: `cargo test test_self_healing_relaxation` demonstrates recovery of breached quality gate without discarding valid optimizations.

### [THME-807] Anti-Degradation Quality Gate Watchdog & Rollback Engine
- **Component**: `thme::surgery::watchdog` (`src/surgery/watchdog.rs`)
- **Problem**: Models must never enter production in a degraded or broken state; instant rollback is mandatory when validation fails.
- **Acceptance Criteria**:
  - Watchdog monitoring validation output after every surgery step.
  - If $\Delta PPL > \epsilon_{\max}$ or NaN/Inf is detected, trigger immediate zero-cost pointer restoration to prior snapshot.
  - Record structured audit event (`SurgeryRollback`) with diagnostic failure reasons.
- **Production Pipeline / CI/CD Impact**: Critical safety gate preventing corrupt checkpoints from advancing in the release pipeline.
- **Done Means**: `cargo test test_co_surgery_watchdog_rollback` verifies immediate state restoration upon injected catastrophic degradation.

### [THME-808] CLI Subcommand `thme co-surgery` & Progress Telemetry Stream
- **Component**: `thme::cli::co_surgery` (`src/cli/co_surgery.rs`)
- **Problem**: DevOps engineers need a terminal command to launch autonomous co-surgery runs with real-time logging and progress reporting.
- **Acceptance Criteria**:
  - CLI `thme co-surgery --model <path> --agent <cmd> --max-iterations <N> --target-reduction <%> --max-ppl-delta <float>`.
  - Live terminal progress stream: iteration counter, proposed recipe diff, validation $\Delta PPL$, Pareto status.
  - Support `--non-interactive` flag emitting JSONL events for CI log ingestion.
- **Production Pipeline / CI/CD Impact**: Primary integration command for continuous model surgery cron jobs and GitHub Actions workflows.
- **Done Means**: `thme co-surgery --help` displays correct flags, and unit tests verify command dispatch.

### [THME-809] Desktop Studio Tab 2 Closed-Loop Co-Surgery Dashboard
- **Component**: `thme::gui::app` (`src/gui/app.rs`)
- **Problem**: Interactive users need visual oversight of autonomous co-surgery sessions with live graphs and manual override buttons.
- **Acceptance Criteria**:
  - Interactive co-surgery panel in Tab 2 (Layer Tweaker & Surgery).
  - Real-time iteration history table, loss curve plot, and 3D Pareto scatter projection.
  - Start / Pause / Stop buttons and manual intervention controls.
- **Production Pipeline / CI/CD Impact**: Visual monitoring interface for researchers and internal demonstrations.
- **Done Means**: Desktop Studio compiles cleanly and renders the Co-Surgery dashboard with responsive controls.

---

## Sprint 9: Production Core Release (Milestone v0.2.0) [TARGET]

### [THME-901] SmolLM-135M Open-Weights End-to-End Verification Pipeline
- **Component**: `examples/production_verification.rs`
- **Problem**: Release validation requires end-to-end proof-of-correctness on genuine, publicly available open weights.
- **Acceptance Criteria**:
  - Download and ingest genuine SmolLM-135M checkpoint (SafeTensors).
  - Execute sensitivity profiling, Pareto BitNet b1.58 surgery, and GGUF export.
  - Validate perplexity on WikiText-2, ensuring $\Delta PPL \le 0.45$.
  - Complete execution within 3 minutes on standard CI runner.
- **Production Pipeline / CI/CD Impact**: Mandatory integration test gate required before tagging v0.2.0.
- **Done Means**: `cargo test --example production_verification -- smollm-135m` passes all quality assertions.

### [THME-902] LLaMA-3.2-1B Open-Weights End-to-End Verification Pipeline
- **Component**: `examples/production_verification.rs`
- **Problem**: Verifying scalability to billion-parameter architectures requires a full pipeline run on LLaMA-3.2-1B.
- **Acceptance Criteria**:
  - Ingest LLaMA-3.2-1B checkpoint under bounded memory constraints ($\le 4 \text{ GB}$ RAM).
  - Execute layer surgery and export valid GGUF and SafeTensors weights.
  - Verify generation output coherence on sample prompts.
- **Production Pipeline / CI/CD Impact**: Validates that THME algorithms scale correctly beyond toy models to modern frontier architectures.
- **Done Means**: `cargo test --example production_verification -- llama-3.2-1b` runs to completion and verifies tensor integrity.

### [THME-903] Automated Reproducible Benchmark Script Suite
- **Component**: `scripts/benchmark_suite.py`, `scripts/benchmark_suite.sh`
- **Problem**: External researchers and users need one-command scripts to reproduce all published release benchmarks on their own hardware.
- **Acceptance Criteria**:
  - Multi-platform benchmark script (Python 3 and Bash) with zero external library dependencies.
  - Benchmarks: ingest throughput (MB/s), SVD profiling latency, Pareto allocation time, export speed, and token simulation.
  - Collect hardware telemetry (CPU model, RAM, OS, cache sizes).
- **Production Pipeline / CI/CD Impact**: Generates benchmark artifacts uploaded to GitHub Actions release assets.
- **Done Means**: Running `./scripts/benchmark_suite.sh` on clean environment completes in $< 5$ minutes and outputs `benchmark_results.json`.

### [THME-904] Published Benchmark Dossier Generator
- **Component**: `docs/benchmark_dossier.md`, `scripts/generate_dossier.py`
- **Problem**: Benchmark results must be formatted into transparent, publication-grade documentation with clear hardware attribution.
- **Acceptance Criteria**:
  - Synthesize benchmark data into Markdown and CSV tables comparing baseline vs THME-optimized models.
  - Include parameter counts, memory footprint, throughput (tok/s), WikiText-2 PPL, and EWHA energy (mJ/tok).
  - Include strict hardware attribution (NVIDIA H100, RTX 4090, Apple M3 Max, AMD MI300X, Intel Xeon).
- **Production Pipeline / CI/CD Impact**: Publishes empirical performance documentation alongside release announcements.
- **Done Means**: Generated `docs/benchmark_dossier.md` passes format check and contains complete empirical tables.

### [THME-905] Multi-Platform LTO Stripped Binary Release Packaging
- **Component**: `.github/workflows/release.yml`, `Cargo.toml`
- **Problem**: Production binaries must be optimized for size and speed across all supported operating systems and architectures.
- **Acceptance Criteria**:
  - Configure Release profile with Fat LTO (`lto = "fat"`), single codegen unit (`codegen-units = 1`), and panic abort (`panic = "abort"`).
  - Strip debug symbols: binary sizes $\le 10 \text{ MB}$ for CLI and $\le 15 \text{ MB}$ for GUI.
  - Build matrices for Linux (x86_64, aarch64), Windows (x64), and macOS (Apple Silicon, Intel).
- **Production Pipeline / CI/CD Impact**: Automates multi-platform asset compilation and packaging upon release tag push.
- **Done Means**: Release workflow builds all 5 target platform archives and validates binary footprint limits.

### [THME-906] Minisign & GPG Cryptographic Artifact Signing Pipeline
- **Component**: `.github/workflows/release.yml`, `scripts/sign_artifacts.sh`
- **Problem**: Binary releases must have cryptographic provenance to guarantee authenticity and prevent supply-chain tampering.
- **Acceptance Criteria**:
  - Sign release tarballs and executables using Minisign (`.minisig`) and GPG (`.asc`).
  - Publish public signing key in repository root (`SECURITY.md`).
  - Automated signature verification check in release workflow.
- **Production Pipeline / CI/CD Impact**: Establishes a tamper-evident software supply chain for enterprise adopters.
- **Done Means**: `minisign -Vm <archive> -p epiphanes.pub` verifies signatures with exit code 0.

### [THME-907] SHA-256 Checksum Manifest Generation
- **Component**: `.github/workflows/release.yml`
- **Problem**: Users and package managers require standard SHA-256 checksum manifests to verify download integrity.
- **Acceptance Criteria**:
  - Generate `SHA256SUMS.txt` manifest covering all release archives and standalone binaries.
  - Automated verification step verifying `sha256sum -c SHA256SUMS.txt`.
- **Production Pipeline / CI/CD Impact**: Standard release asset referenced by Homebrew, Winget, and Arch AUR package formulas.
- **Done Means**: Generated `SHA256SUMS.txt` verifies successfully across all built artifacts.

### [THME-908] Comprehensive Architecture Playbook, User Manual & Man Pages
- **Component**: `docs/manual.md`, `docs/man/thme.1`
- **Problem**: Production users and system administrators need exhaustive documentation and standard Unix man pages.
- **Acceptance Criteria**:
  - Authoritative User Manual (`docs/manual.md`) covering all CLI commands, GUI tabs, and JSON recipe schemas.
  - Standard troff Unix man page (`docs/man/thme.1`) installable to `/usr/share/man/man1`.
  - Architecture Playbook documenting internal data structures and extension points.
- **Production Pipeline / CI/CD Impact**: Verified by documentation linters in CI to prevent broken links or outdated command flags.
- **Done Means**: `man ./docs/man/thme.1` renders cleanly without groff warnings.

### [THME-909] End-to-End Clean-Machine Verification Test Harness
- **Component**: `.github/workflows/clean_machine_test.yml`
- **Problem**: Binaries must execute on clean operating system installations without hidden dynamic library dependencies or environment assumptions.
- **Acceptance Criteria**:
  - Ephemeral CI runner with clean Ubuntu, Windows, and macOS images.
  - Download release binary, execute `thme --version`, run synthetic profiling, and export model.
  - Verify zero missing shared library errors (`ldd` / Dependency Walker check).
- **Production Pipeline / CI/CD Impact**: Final quality gate certifying production readiness before public release announcement.
- **Done Means**: Clean machine CI workflow passes 100% across all 3 OS platforms.

---

## Sprint 10: Direct GPU Driver Execution Engine (Milestone v0.2.1) [SCHEDULED]

### [THME-1001] Hardware Abstraction Layer (HAL) Core Traits
- **Component**: `thme::hardware::hal` (`src/hardware/hal.rs`)
- **Problem**: Supporting multiple GPU backends (CUDA, Vulkan, Metal) requires a unified trait architecture for buffer allocation, memory copies, and kernel launches.
- **Acceptance Criteria**:
  - Define core traits: `GpuDevice`, `DeviceBuffer`, `StreamQueue`, `ComputeKernel`, and `KernelLaunchConfig`.
  - Unified host-to-device, device-to-host, and device-to-device memory copy abstractions.
  - Device capability querying (compute units, shared memory per block, warp size).
- **Production Pipeline / CI/CD Impact**: Foundational trait contract decoupling backend implementations from high-level model execution.
- **Done Means**: `cargo test test_hal_traits` passes with mock CPU backend implementing full HAL interface.

### [THME-1002] NVIDIA CUDA Driver API Backend via Pure Rust `cudarc`
- **Component**: `thme::hardware::cuda` (`src/hardware/cuda.rs`)
- **Problem**: External PyTorch runtimes introduce gigabytes of dependencies; THME requires direct driver communication via `libcuda.so` / `nvcuda.dll`.
- **Acceptance Criteria**:
  - Integrate `cudarc` for driver API initialization and context management.
  - Support JIT PTX compilation and loading pre-compiled CUBIN modules.
  - Asynchronous stream execution and CUDA event synchronization.
- **Production Pipeline / CI/CD Impact**: Enables high-performance NVIDIA GPU execution without requiring Python or PyTorch installed.
- **Done Means**: `cargo test test_cuda_driver_backend` initializes CUDA context and executes vector-add kernel on available NVIDIA GPU.

### [THME-1003] Vulkan 1.3 SPIR-V Compute Pipeline Backend
- **Component**: `thme::hardware::vulkan` (`src/hardware/vulkan.rs`)
- **Problem**: Cross-platform GPU execution across Linux, Windows, and AMD/Intel GPUs requires standard Vulkan compute shaders.
- **Acceptance Criteria**:
  - Initialize Vulkan 1.3 compute instance and device queue via `ash`.
  - Load embedded SPIR-V shader modules for matrix multiplication and activation layers.
  - Manage descriptor sets, storage buffers, and pipeline execution barriers.
- **Production Pipeline / CI/CD Impact**: Delivers zero-dependency GPU acceleration on non-NVIDIA hardware in standard CI testbeds.
- **Done Means**: `cargo test test_vulkan_compute_backend` successfully runs matrix multiply on Vulkan compute device.

### [THME-1004] Apple Metal 3 Performance Shaders Backend
- **Component**: `thme::hardware::metal` (`src/hardware/metal.rs`)
- **Problem**: Apple Silicon Macs feature high-bandwidth unified memory that requires native Metal 3 Performance Shaders to achieve zero-copy execution.
- **Acceptance Criteria**:
  - Metal device creation and command queue management via `metal-rs`.
  - Allocate shared CPU-GPU storage buffers (`MTLStorageModeShared`) with zero memory copies.
  - Dispatch custom MSL compute shaders and integrate Metal Performance Shaders (MPS) matrix kernels.
- **Production Pipeline / CI/CD Impact**: Enables native, highly efficient inference and surgery on macOS developer laptops.
- **Done Means**: `cargo test test_metal_backend` passes on macOS runner, verifying zero-copy execution.

### [THME-1005] Fused BitNet b1.58 GEMV / GEMM Kernel
- **Component**: `thme::hardware::kernels::bitnet_gemm` (`src/hardware/kernels/bitnet_gemm.rs`)
- **Problem**: Unpacking 2-bit ternary weights to global memory before GEMM introduces severe memory bandwidth overhead; unpacking must happen inside GPU registers.
- **Acceptance Criteria**:
  - CUDA, Vulkan, and Metal compute kernels unpacking 2-bit weights directly within register files.
  - Replace multiplication with conditional addition/subtraction: $y += (w == 1) ? x : ((w == -1) ? -x : 0)$.
  - Achieve $\ge 2.5\times$ speedup over un-quantized FP16 GEMV during batch size $B=1$ decode.
- **Production Pipeline / CI/CD Impact**: Core inference speedup engine for deployable BitNet models.
- **Done Means**: `cargo bench --bench bench_bitnet_gemm` demonstrates $\ge 2.5\times$ throughput improvement.

### [THME-1006] Fused RMSNorm Micro-Kernel
- **Component**: `thme::hardware::kernels::rmsnorm` (`src/hardware/kernels/rmsnorm.rs`)
- **Problem**: Separate reduction and scaling passes for RMSNorm incur unnecessary global memory round-trips.
- **Acceptance Criteria**:
  - Single-pass thread block cooperative reduction computing root mean square: $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$.
  - Fused weight scaling $y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$ writing directly to activation output.
- **Production Pipeline / CI/CD Impact**: Reduces layer transition latency across all Transformer blocks.
- **Done Means**: `cargo test test_fused_rmsnorm` passes and verifies bitwise alignment with reference implementation.

### [THME-1007] Fused SwiGLU Activation Micro-Kernel
- **Component**: `thme::hardware::kernels::swiglu` (`src/hardware/kernels/swiglu.rs`)
- **Problem**: SwiGLU requires computing $\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)$, which wastes memory bandwidth if intermediate activations are written to DRAM.
- **Acceptance Criteria**:
  - Fused elementwise kernel computing $\text{SiLU}(a) = \frac{a}{1 + e^{-a}}$ and multiplying by $b$ in local registers.
  - Zero global memory allocation for intermediate activation buffers.
- **Production Pipeline / CI/CD Impact**: Accelerates MLP execution by $\ge 35\%$ across modern LLaMA-style architectures.
- **Done Means**: `cargo test test_fused_swiglu` verifies numerical correctness within $10^{-5}$ tolerance.

### [THME-1008] Fused Rotary Position Embedding (RoPE) Kernel
- **Component**: `thme::hardware::kernels::rope` (`src/hardware/kernels/rope.rs`)
- **Problem**: Applying RoPE to $Q$ and $K$ matrices requires complex sinusoidal rotation that can bottleneck attention pre-processing.
- **Acceptance Criteria**:
  - Fused in-place 2D rotation kernel: $\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix} \mapsto \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$.
  - Support precomputed frequency tables and dynamic RoPE scaling (YaRN, Linear).
- **Production Pipeline / CI/CD Impact**: Streamlines attention projection pipeline in pure Rust driver execution.
- **Done Means**: `cargo test test_fused_rope` verifies exact rotation matching Hugging Face LLaMA reference.

### [THME-1009] CLI Subcommand `thme run` Direct Inference
- **Component**: `thme::cli::run` (`src/cli/run.rs`)
- **Problem**: Users need a direct CLI command to run LLM inference using native GPU drivers without Python.
- **Acceptance Criteria**:
  - CLI `thme run --model <path> --backend <cuda|vulkan|metal|cpu> --prompt <text> --max-tokens <N>`.
  - Token generation loop with live tok/s throughput and latency metrics.
  - Automatic hardware backend detection and graceful fallback to CPU SIMD.
- **Production Pipeline / CI/CD Impact**: Primary command for standalone inference benchmarks and edge deployment.
- **Done Means**: `thme run --help` executes cleanly, and integration test runs end-to-end token generation on test fixture.

---

## Sprint 11: Tiered Out-of-Core NVMe SSD Streaming (Milestone v0.2.2) [SCHEDULED]

### [THME-1101] Linux Direct Asynchronous I/O Engine (`io_uring`)
- **Component**: `thme::storage::direct_io` (`src/storage/direct_io.rs`)
- **Problem**: Standard OS file reading incurs page cache buffering and context switch overhead, capping disk throughput below PCIe 4.0/5.0 line rates.
- **Acceptance Criteria**:
  - Implement `io_uring` submission and completion queue polling via `io-uring` crate.
  - Use `O_DIRECT` flag to bypass OS kernel buffer cache and read directly into user-space buffers.
  - Achieve $> 6,000 \text{ MB/s}$ sustained sequential read throughput on NVMe SSDs.
- **Production Pipeline / CI/CD Impact**: Establishes high-speed storage streaming backbone for Linux servers.
- **Done Means**: `cargo test test_io_uring_direct_io` passes on Linux runner, reading 1 GB test file with zero page cache caching.

### [THME-1102] Windows Direct Asynchronous I/O Engine (Overlapped I/O)
- **Component**: `thme::storage::direct_io` (`src/storage/direct_io.rs`)
- **Problem**: Windows developers require native unbuffered disk streaming matching Linux `io_uring` throughput.
- **Acceptance Criteria**:
  - Implement Win32 Overlapped I/O using `FILE_FLAG_NO_BUFFERING` and `FILE_FLAG_OVERLAPPED`.
  - Asynchronous completion handling via I/O Completion Ports (`IOCP`).
  - Enforce sector-aligned memory buffer allocation (`VirtualAlloc` on 4096-byte boundaries).
- **Production Pipeline / CI/CD Impact**: Enables high-speed SSD model streaming on Windows developer workstations.
- **Done Means**: `cargo test test_windows_overlapped_io` passes on Windows runner with verified direct I/O throughput.

### [THME-1103] Sequential Layer Streaming & Eviction Engine
- **Component**: `thme::storage::layer_streamer` (`src/storage/layer_streamer.rs`)
- **Problem**: Executing a 70B parameter model requires 140 GB of weights, exceeding consumer GPU VRAM; layers must be loaded and evicted sequentially.
- **Acceptance Criteria**:
  - Layer-by-layer forward execution: stream layer $k$ from disk $\to$ transfer to GPU $\to$ execute forward step $\to$ release buffer for layer $k+1$.
  - Maintain single-layer active memory ceiling in VRAM.
- **Production Pipeline / CI/CD Impact**: Enables massive model execution in memory-constrained CI environments.
- **Done Means**: `cargo test test_layer_streaming_execution` completes forward pass across 32 layers using single-layer memory budget.

### [THME-1104] Double-Buffered Prefetch Pipeline
- **Component**: `thme::storage::prefetch` (`src/storage/prefetch.rs`)
- **Problem**: Sequential loading stalls GPU compute while waiting for disk reads; I/O and computation must be overlapped.
- **Acceptance Criteria**:
  - Double-buffered queue: asynchronous disk thread prefetches layer $k+1$ while GPU compute stream executes layer $k$.
  - Synchronize streams using non-blocking barriers.
  - Eliminate $\ge 80\%$ of disk I/O latency stalls during continuous autoregressive generation.
- **Production Pipeline / CI/CD Impact**: Maximizes throughput during out-of-core model surgery and profiling.
- **Done Means**: `cargo test test_double_buffered_prefetch` demonstrates concurrent disk reading and mock layer compute.

### [THME-1105] Pinned Memory Staging Buffers (`mlock` / `VirtualLock`)
- **Component**: `thme::memory::pinned` (`src/memory/pinned.rs`)
- **Problem**: GPU DMA transfers from pageable host RAM require intermediate copies into pinned memory, reducing transfer bandwidth.
- **Acceptance Criteria**:
  - Allocate page-locked host memory buffers using `mlock` on Unix and `VirtualLock` on Windows.
  - Enable direct DMA transfers from host buffer to GPU VRAM at full PCIe bandwidth ($\sim 31.5 \text{ GB/s}$ on PCIe 4.0 x16).
- **Production Pipeline / CI/CD Impact**: Eliminates host memory copy bottlenecks during layer streaming.
- **Done Means**: `cargo test test_pinned_memory_allocation` verifies lock status and zero-copy DMA transfer capabilities.

### [THME-1106] REA-LLM Tiered Heterogeneous Memory Placement Manager
- **Component**: `thme::memory::tiered_allocator` (`src/memory/tiered_allocator.rs`)
- **Problem**: Different model layers exhibit different reach probabilities $q_l$; memory placement should be tiered strategically across VRAM, Host RAM, and NVMe SSD.
- **Acceptance Criteria**:
  - Placement policy: hot layers ($q_l > 0.8$) permanently pinned in VRAM, warm layers ($0.4 \le q_l \le 0.8$) staged in Host RAM, deep layers ($q_l < 0.4$) streamed from NVMe SSD.
  - Dynamically adapt layer placement based on runtime reach telemetry.
- **Production Pipeline / CI/CD Impact**: Optimizes execution speed under strict hardware memory constraints.
- **Done Means**: `cargo test test_tiered_memory_placement` verifies correct layer distribution across simulated 3-tier hierarchy.

### [THME-1107] Dynamic VRAM Memory Accountant
- **Component**: `thme::memory::accountant` (`src/memory/accountant.rs`)
- **Problem**: Out-of-core execution without strict VRAM accounting risks hitting hard CUDA OOM errors.
- **Acceptance Criteria**:
  - Track active VRAM allocations in real-time (model weights, KV cache, activation buffers).
  - Enforce user-defined hard ceiling (e.g. `--max-vram 8GB`), throttling prefetch queue if VRAM headroom is $< 512 \text{ MB}$.
- **Production Pipeline / CI/CD Impact**: Prevents catastrophic OOM aborts in CI and production deployments.
- **Done Means**: `cargo test test_vram_accountant_ceiling` strictly caps allocated memory under stress testing.

### [THME-1108] 70B Model in 8 GB VRAM End-to-End Testbed
- **Component**: `tests/test_out_of_core_streaming.rs`
- **Problem**: Verifying out-of-core streaming requires full multi-layer autoregressive execution on a genuine 70B parameter topology under an 8 GB memory limit.
- **Acceptance Criteria**:
  - Generate synthetic 70B parameter model topology (80 layers, $D=8192$, $D_{\text{ffn}}=28672$).
  - Execute end-to-end forward token step strictly bounded within 8 GB VRAM footprint.
  - Verify numerical consistency with standard un-streamed reference output.
- **Production Pipeline / CI/CD Impact**: Primary proof that THME enables large-model inference on consumer GPUs.
- **Done Means**: `cargo test --test test_out_of_core_streaming` passes and verifies $< 8 \text{ GB}$ peak VRAM usage.

### [THME-1109] PCIe Bandwidth Saturation & Disk I/O Telemetry Monitor
- **Component**: `thme::profiling::io_monitor` (`src/profiling/io_monitor.rs`)
- **Problem**: Operators need visibility into PCIe bus saturation, disk read queue depths, and streaming bottlenecks.
- **Acceptance Criteria**:
  - Real-time telemetry: PCIe transfer rate (GB/s), NVMe read throughput (MB/s), and I/O wait latency fraction ($f_{\text{wait}} \in [0, 1]$).
  - Expose metrics to CLI status bars and Desktop Studio Tab 4 benchmark deck.
- **Production Pipeline / CI/CD Impact**: Identifies storage and bus bottlenecks during hardware profiling.
- **Done Means**: `cargo test test_io_telemetry_monitor` validates accurate metric computation and reporting.

---

## Sprint 12: Distributed Heterogeneous Multi-Node Clustering (Milestone v0.2.3) [SCHEDULED]

### [THME-1201] Intra-Node Tensor Parallelism (TP) Projection Splitter
- **Component**: `thme::cluster::tensor_parallel` (`src/cluster/tensor_parallel.rs`)
- **Problem**: Multi-GPU workstations require splitting Attention and MLP projection matrices across local accelerators to reduce per-device memory footprint and latency.
- **Acceptance Criteria**:
  - Column-parallel linear projections for $W_q, W_k, W_v$ and MLP $W_{\text{gate}}, W_{\text{up}}$.
  - Row-parallel linear projections for $W_o$ and MLP $W_{\text{down}}$ with local partial sum reduction.
  - Zero numerical divergence from single-device forward pass.
- **Production Pipeline / CI/CD Impact**: Enables multi-GPU scaling on single compute nodes.
- **Done Means**: `cargo test test_tensor_parallel_projections` passes, verifying exact output match across 2 and 4 simulated devices.

### [THME-1202] Inter-Node Pipeline Parallelism (PP) & Layer Partitioner
- **Component**: `thme::cluster::pipeline_parallel` (`src/cluster/pipeline_parallel.rs`)
- **Problem**: Distributing multi-layer models across networked nodes requires partitioning consecutive layer blocks across heterogeneous machines.
- **Acceptance Criteria**:
  - Dynamic layer partitioning algorithm assigning layers to nodes based on compute capacity and network link speed.
  - Activation communication pipeline forwarding intermediate activation tensors between nodes.
- **Production Pipeline / CI/CD Impact**: Allows clustering mixed machines (e.g. desktop + server) to run massive models.
- **Done Means**: `cargo test test_pipeline_parallel_partitioning` splits 32 layers across 3 nodes with balanced execution.

### [THME-1203] 1F1B (One-Forward-One-Backward) Pipeline Bubble Minimizer
- **Component**: `thme::cluster::schedule_1f1b` (`src/cluster/schedule_1f1b.rs`)
- **Problem**: Naive pipeline parallelism leaves nodes idle while waiting for activations (pipeline bubbles); a 1F1B schedule is needed.
- **Acceptance Criteria**:
  - Implement 1F1B micro-batch scheduling interleaving forward and backward passes.
  - Reduce pipeline bubble fraction to $\frac{p - 1}{m}$ where $p$ is pipeline depth and $m$ is micro-batch count.
- **Production Pipeline / CI/CD Impact**: Maximizes cluster accelerator utilization during distributed micro-training.
- **Done Means**: `cargo test test_1f1b_scheduler` verifies micro-batch ordering and idle time minimization.

### [THME-1204] Pure Rust Asynchronous TCP Ring All-Reduce
- **Component**: `thme::cluster::ring_allreduce` (`src/cluster/ring_allreduce.rs`)
- **Problem**: Distributed communication must avoid heavy MPI or NCCL runtime dependencies, providing pure Rust network communication.
- **Acceptance Criteria**:
  - Asynchronous Ring All-Reduce over non-blocking TCP socket streams using `tokio`.
  - Scatter-Reduce and All-Gather phases transferring $2 \frac{N-1}{N} S$ bytes per node.
  - Achieve $> 85\%$ network bandwidth utilization on 10 GbE / 100 GbE connections.
- **Production Pipeline / CI/CD Impact**: Pure Rust distributed primitives running anywhere without complex cluster runtime configuration.
- **Done Means**: `cargo test test_ring_allreduce_tcp` passes across 4 simulated local nodes.

### [THME-1205] Hardware RDMA / RoCE Network Fallback Engine
- **Component**: `thme::cluster::rdma` (`src/cluster/rdma.rs`)
- **Problem**: High-end HPC clusters feature InfiniBand / RoCE networking; THME should leverage kernel-bypass RDMA when available.
- **Acceptance Criteria**:
  - Detect InfiniBand / RoCE hardware interfaces via `libibverbs` FFI bindings.
  - Establish RDMA Queue Pairs (QP) for zero-copy memory transfers.
  - Fall back transparently to TCP ring all-reduce if RDMA hardware is not present.
- **Production Pipeline / CI/CD Impact**: Delivers enterprise-grade multi-node scaling on HPC clusters.
- **Done Means**: `cargo test test_rdma_detection_and_fallback` verifies fallback behavior and queue pair setup.

### [THME-1206] Multicast / mDNS Cluster Node Discovery & Capability Handshake
- **Component**: `thme::cluster::discovery` (`src/cluster/discovery.rs`)
- **Problem**: Ad-hoc clustering requires machines on a local subnet to automatically discover each other without manual IP configuration.
- **Acceptance Criteria**:
  - mDNS / UDP multicast beacon broadcasting THME node ID, compute capability (TFLOP/s), and available VRAM.
  - Dynamic capability handshake establishing cluster topology and leader election.
- **Production Pipeline / CI/CD Impact**: Zero-configuration cluster setup for local network workstations.
- **Done Means**: `cargo test test_cluster_mdns_discovery` discovers peers and exchanges hardware capability payloads.

### [THME-1207] Continuous 100 ms Heartbeat & Dead-Node Watchdog
- **Component**: `thme::cluster::watchdog` (`src/cluster/watchdog.rs`)
- **Problem**: Nodes in heterogeneous or ad-hoc clusters may disconnect, reboot, or experience network drops during long runs.
- **Acceptance Criteria**:
  - Periodic heartbeat ping every 100 ms between cluster nodes.
  - Detect dead or unresponsive nodes after 3 missed heartbeats (300 ms).
  - Emit cluster health alert and trigger failover protocol.
- **Production Pipeline / CI/CD Impact**: Protects distributed training and inference jobs from silent network hangs.
- **Done Means**: `cargo test test_cluster_heartbeat_watchdog` detects simulated node failure within 300 ms.

### [THME-1208] Automated Dynamic Layer Re-Sharding Failover
- **Component**: `thme::cluster::reshard` (`src/cluster/reshard.rs`)
- **Problem**: When a node disconnects, its assigned model layers become orphaned; the cluster must redistribute them without terminating the job.
- **Acceptance Criteria**:
  - Automated re-sharding algorithm redistributing orphaned layers to remaining healthy nodes.
  - Re-establish communication pipelines and resume forward pass from current checkpoint.
- **Production Pipeline / CI/CD Impact**: Guarantees fault-tolerant execution in volatile distributed environments.
- **Done Means**: `cargo test test_dynamic_resharding` demonstrates forward pass recovery after dropping 1 of 3 nodes.

### [THME-1209] CLI Subcommand `thme cluster` & Cluster Formation Wizard
- **Component**: `thme::cli::cluster` (`src/cli/cluster.rs`)
- **Problem**: Operators need a command-line interface to initialize, join, and inspect distributed clusters.
- **Acceptance Criteria**:
  - CLI `thme cluster init --port <port>`, `thme cluster join --leader <ip:port>`, and `thme cluster status`.
  - Terminal dashboard showing active nodes, hardware specifications, assigned layers, and network latency matrix.
- **Production Pipeline / CI/CD Impact**: Primary CLI for managing distributed THME deployments in data centers and local labs.
- **Done Means**: `thme cluster --help` executes cleanly, unit tests verify cluster command dispatch.

---

## Sprint 13: Universal Agentic AI Connectivity Gateway (Milestone v0.2.4) [SCHEDULED]

### [THME-1301] High-Throughput Asynchronous WebSocket Event Streaming Daemon
- **Component**: `thme::daemon::server` (`src/daemon/server.rs`)
- **Problem**: Remote agents, IDEs, and dashboards require real-time bidirectional WebSocket streaming of per-token progress, layer execution, and telemetry.
- **Acceptance Criteria**:
  - Headless daemon (`thme daemon --port 9090`) based on `tokio-tungstenite`.
  - Broadcast structured JSON event streams: `TokenGenerated`, `LayerExecuted`, `SurgeryApplied`, `ThermalWarning`.
  - Support $> 100$ concurrent client WebSocket connections with $< 1 \text{ ms}$ broadcast latency.
- **Production Pipeline / CI/CD Impact**: Core event hub connecting all external clients to running THME instances.
- **Done Means**: `cargo test test_websocket_streaming_daemon` verifies concurrent client connections and message broadcasting.

### [THME-1302] JSON-RPC 2.0 Method Router & Remote Protocol Dispatcher
- **Component**: `thme::daemon::router` (`src/daemon/router.rs`)
- **Problem**: External tools need a standardized remote procedure call API to invoke THME operations (inspect, surgery, validate, export).
- **Acceptance Criteria**:
  - JSON-RPC 2.0 specification compliant router supporting batch requests and error codes.
  - Dispatch methods: `thme_inspectModel`, `thme_applySurgery`, `thme_validateQuality`, `thme_exportModel`, `thme_profileRoofline`.
- **Production Pipeline / CI/CD Impact**: Provides stable, language-agnostic API for third-party developer integrations.
- **Done Means**: `cargo test test_jsonrpc_method_router` validates JSON-RPC request and response payloads.

### [THME-1303] Cryptographic HMAC-SHA256 Surgery Recipe Attestation
- **Component**: `thme::daemon::attestation` (`src/daemon/attestation.rs`)
- **Problem**: Exposing model surgery over a network creates security vulnerabilities; recipes must be authenticated and tamper-proof.
- **Acceptance Criteria**:
  - HMAC-SHA256 signature verification on incoming JSON surgery recipe payloads using pre-shared keys or bearer tokens.
  - Reject unsigned or altered payloads with `401 Unauthorized` / `AttestationFailed` error.
  - Rate limiting and request throttling per authenticated identity.
- **Production Pipeline / CI/CD Impact**: Enforces software supply-chain security and prevents unauthorized model modifications in shared environments.
- **Done Means**: `cargo test test_hmac_recipe_attestation` rejects tampered recipes and accepts validly signed payloads.

### [THME-1304] Autonomous CLI Agent Adapter
- **Component**: `thme::daemon::adapters::cli_agent` (`src/daemon/adapters/cli_agent.rs`)
- **Problem**: Coding agents (Claude Code, Gemini CLI, Aider, OpenHands) need formatted tool schemas and direct execution wrappers.
- **Acceptance Criteria**:
  - Generate agent tool configuration blocks compatible with popular coding agent frameworks.
  - Structured error messages suggesting specific remediation steps when surgery operations fail.
- **Production Pipeline / CI/CD Impact**: Enables autonomous coding agents to perform automated model surgery directly from CLI prompts.
- **Done Means**: `cargo test test_cli_agent_adapter` verifies agent schema generation and tool calling.

### [THME-1305] CI/CD Orchestration Harness Adapter
- **Component**: `thme::daemon::adapters::cicd` (`src/daemon/adapters/cicd.rs`)
- **Problem**: Enterprise CI/CD systems (Harness CI/CD, GitHub Actions, GitLab CI) need structured step outputs, JUnit reports, and exit code mappings.
- **Acceptance Criteria**:
  - Emit GitHub Actions workflow commands (`::set-output`, `::error`, `::notice`).
  - Generate standard JUnit XML test reports from model validation and perplexity benchmarks.
- **Production Pipeline / CI/CD Impact**: Seamlessly integrates THME quality gates into enterprise CI/CD deployment pipelines.
- **Done Means**: `cargo test test_cicd_adapter_reports` verifies JUnit XML generation and workflow command formatting.

### [THME-1306] Desktop Studio Remote Daemon Connection Bridge
- **Component**: `thme::gui::remote_client` (`src/gui/remote_client.rs`)
- **Problem**: Users running `thme-gui` on lightweight laptops want to connect to remote GPU servers hosting THME daemon instances.
- **Acceptance Criteria**:
  - Remote connection modal in Desktop Studio: enter host IP, port, and authentication token.
  - Remote state synchronization mirroring remote model inventory, telemetry, and surgery state in the local GUI.
- **Production Pipeline / CI/CD Impact**: Expands Desktop Studio usability to remote headless GPU clouds.
- **Done Means**: `cargo test test_gui_remote_bridge` establishes WebSocket session and synchronizes model state.

### [THME-1307] IDE & GitHub Copilot Extension Bridge
- **Component**: `thme::daemon::adapters::ide_bridge` (`src/daemon/adapters/ide_bridge.rs`)
- **Problem**: VS Code, Cursor, and Windsurf extensions require live daemon connectivity to stream model topology changes into editor sidebars.
- **Acceptance Criteria**:
  - Bi-directional bridge connecting editor language servers to THME daemon.
  - Live workspace telemetry updating VS Code tree views upon layer surgery.
- **Production Pipeline / CI/CD Impact**: Closes the loop between developer code editors and active model surgery daemons.
- **Done Means**: `cargo test test_ide_bridge_synchronization` verifies event passing between editor client and daemon.

### [THME-1308] Real-Time System Telemetry & Resource Pressure Broadcasting
- **Component**: `thme::daemon::telemetry_broadcaster` (`src/daemon/telemetry_broadcaster.rs`)
- **Problem**: Remote monitoring systems (Grafana, Datadog) need real-time streams of VRAM pressure, GPU utilization, and token throughput.
- **Acceptance Criteria**:
  - Stream metrics in OpenTelemetry / Prometheus format: `thme_tokens_per_second`, `thme_vram_used_bytes`, `thme_layer_latency_seconds`.
  - Publish metrics over HTTP endpoint `/metrics`.
- **Production Pipeline / CI/CD Impact**: Standard observability integration for enterprise model serving infrastructure.
- **Done Means**: `cargo test test_telemetry_prometheus_exporter` verifies `/metrics` response format and values.

### [THME-1309] CLI Subcommand `thme daemon` & Universal Daemon Controller
- **Component**: `thme::cli::daemon` (`src/cli/daemon.rs`)
- **Problem**: System administrators require a dedicated CLI command to launch, configure, and daemonize the gateway server.
- **Acceptance Criteria**:
  - CLI `thme daemon --port <port> --bind <ip> --token <secret> --cors <origins>`.
  - Support graceful shutdown on SIGINT/SIGTERM, draining active connections.
  - Built-in logging with `--verbose` and structured JSON logs.
- **Production Pipeline / CI/CD Impact**: Primary system service entry point for systemd / docker container deployments.
- **Done Means**: `thme daemon --help` displays clean options, and integration tests verify server start and graceful shutdown.
