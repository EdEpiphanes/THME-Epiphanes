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

## Sprint 3: Empirical Quality Verification (Milestone v0.1.4)

### [THME-301] External Perplexity Evaluation Bridge
- **Component**: `thme::quality::evaluator`
- **Problem**: Quality gates require ground-truth perplexity ($PPL$) evaluation on benchmark corpora (WikiText-2, C4).
- **Acceptance Criteria**:
  - Implement IPC/subprocess bridge to `llama.cpp/perplexity` or standalone inference runner.
  - Stream progress and token loss in real-time to GUI Quality tab.
  - Calculate empirical degradation $\Delta PPL = PPL_{\text{opt}} - PPL_{\text{base}}$.
- **Done Means**: `thme validate --model <path> --test-set wikitext2` runs end-to-end and outputs structured pass/fail gate verdict.

### [THME-302] Automated Anti-Degradation Quality Gate
- **Component**: `thme::quality::gate`
- **Problem**: Models must automatically reject optimizations that exceed allowable perplexity degradation ($\epsilon = 2.0\%$).
- **Acceptance Criteria**:
  - Enforce rollback if candidate exceeds $\epsilon$.
  - Output Pareto frontier curve comparing storage reduction vs $PPL$ impact.
- **Done Means**: Quality gate test triggers automatic rollback when degradation threshold is violated.

---

## Sprint 4: Surgery Daemon & API (Milestone v0.1.5)

### [THME-401] Headless Surgery Daemon (`thme daemon`)
- **Component**: `thme::daemon`
- **Problem**: Teacher LLMs running on remote compute clusters need an API to query model IR and dispatch surgery recipes.
- **Acceptance Criteria**:
  - Lightweight async server listening on localhost port (configurable via `--port`).
  - Endpoints: `GET /v1/model/ir`, `POST /v1/surgery/apply`, `GET /v1/telemetry/stream`.
  - JSON Schema validation on all incoming surgery payloads.
- **Done Means**: Remote curl request sends recipe JSON, modifies loaded model IR, and receives audit verification response.

---

## Sprint 5: Production Exporter & Release (Milestone v0.2.0)

### [THME-501] Quantization Model Exporter (GGUF / SafeTensors)
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
