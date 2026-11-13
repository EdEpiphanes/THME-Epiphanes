# THME Quickstart Guide: Zero-Copy Profiling, Precision Planning, and Studio Workflow

## Executive Summary

The Topological Heterogeneous Manifold Engine (THME) is a high-performance, pure Rust framework engineered for zero-copy Large Language Model (LLM) memory profiling, topological layer sensitivity estimation, Pareto precision allocation, and automated Teacher LLM surgery. 

This guide walks through setting up the repository from source, generating a verifiable synthetic model fixture, inspecting tensor topologies without allocating raw weights in heap memory, profiling memory scaling across expanding context windows, computing an optimal multi-precision allocation plan, validating perplexity quality gates, and launching the native 60 FPS desktop studio.

---

## Prerequisites and Environment Setup

Before starting, ensure your host environment meets the following specifications:

| Requirement | Minimum Specification | Recommended Specification |
|:---|:---|:---|
| Operating System | Linux (Ubuntu 22.04+), macOS (12+), Windows 10/11 | Linux x86_64 or macOS Apple Silicon |
| Rust Toolchain | Rust 2021 Edition (version >= 1.89) | Stable channel (`rustc --version` >= 1.89) |
| Build Tool | Cargo (bundled with Rust) | Latest stable Cargo |
| System Memory | 4 GB available RAM | 8 GB+ RAM for larger model topologies |
| C/C++ Compiler | Optional (pure Rust implementation) | GCC / Clang / MSVC for OS integrations |

Verify your active Rust toolchain installation:

```bash
rustc --version
cargo --version
```

If Rust is not yet installed on your system, install it using `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

---

## Step 1: Building THME from Source

Clone the repository and compile the workspace binaries using release optimizations:

```bash
git clone https://github.com/EdEpiphanes/THME-Epiphanes.git
cd THME-Epiphanes
cargo build --release
```

The build process produces three core binaries in `./target/release/`:
- `thme`: Primary unified command-line interface.
- `llm-tweaker`: Backwards-compatible alias for existing automated pipelines.
- `thme-gui`: Standalone native desktop studio (powered by `eframe` and `egui`).

Verify that the CLI binary is compiled and functional:

```bash
./target/release/thme --help
```

Expected output:

```text
THME: Topological Heterogeneous Manifold Engine (v0.1.5)
Zero-copy LLM memory profiling, layer sensitivity estimation, Pareto precision allocation, and automated Teacher LLM surgery.

Usage: thme <COMMAND>

Commands:
  gui        Launch the 60 FPS interactive desktop studio GUI [alias: ui]
  inspect    Scan and inspect model metadata and tensor topology zero-copy
  profile    Profile layer manifold sensitivity and KV-cache context memory scaling
  plan       Compute heterogeneous precision allocation plan across Pareto frontiers
  validate   Empirical quality gate checker with delta-PPL verification [alias: check]
  tweak      Execute declarative Teacher LLM surgery recipes [alias: surgery]
  report     Export executive optimization and audit reports
  convert    Cross-convert model architecture across 8 specialized AI paradigms
  speculate  Synthesize draft models and evaluate speculative decoding acceleration
  roofline   Profile layer arithmetic intensity against empirical hardware ceilings
  export     Generate PyTorch CUDA fine-tuning harnesses and Hugging Face configs [alias: interop]
  mcp        Run the Model Context Protocol (MCP) JSON-RPC daemon [alias: daemon]
  help       Print this message or the help of the given subcommand(s)
```

---

## Step 2: Generating a Synthetic Model Fixture

To validate profiling and surgery commands without downloading multi-gigabyte production weights, THME includes a mock model generator in `examples/generate_mock_model.rs`. This creates a lightweight, fully compliant 4-layer LLaMA architecture with realistic weight statistics:

```bash
cargo run --example generate_mock_model
```

Expected output:

```text
Mock LLaMA created successfully at examples/mock_llama
```

This generates `examples/mock_llama/` containing:
- `config.json`: Model configuration specifying 4 layers, hidden dimension 256, 8 Query heads, 4 Key-Value heads, vocabulary size 32,000, and max context 4,096.
- `model.safetensors`: Binary SafeTensors file (24.2 MB) containing 30 tensor buffers across attention, MLP, and normalization layers.

---

## Step 3: Zero-Copy Model Inspection

Inspect the structural topology and parameter layout of the model:

```bash
./target/release/thme inspect examples/mock_llama
```

Expected output:

```text
Scanning model metadata at: "examples/mock_llama"

=== THME Model Inspection Summary ===
╭──────────────────────┬───────────────────╮
│ Property             ┆ Value             │
╞══════════════════════╪═══════════════════╡
│ Model Name           ┆ mock_llama        │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Architecture         ┆ Llama-family      │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Total Parameters     ┆ 0.01 B (12124416) │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Raw Weight Size      ┆ 0.02 GB           │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Hidden Size          ┆ 256               │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Number of Layers     ┆ 4                 │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Attention Heads (Q)  ┆ 8                 │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Key-Value Heads (KV) ┆ 4                 │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Head Dimension       ┆ 32                │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Vocabulary Size      ┆ 32000             │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Max Context Length   ┆ 4096              │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Total Tensors        ┆ 30                │
╰──────────────────────┴───────────────────╯
```

For programmatic evaluation, pass the `--json` flag to receive structured JSON output.

---

## Step 4: Layer Sensitivity Profiling and KV-Cache Projections

Estimate layer-by-layer sensitivity and project Key-Value cache memory requirements as context lengths expand from 1,024 to 131,072 tokens:

```bash
./target/release/thme profile examples/mock_llama --batch-size 1
```

Expected output:

```text
Profiling manifold sensitivity & memory for: "examples/mock_llama"

=== Topological Layer Sensitivity Analysis ===
╭──────────┬────────────┬─────────────┬───────────────────┬───────────────────────╮
│ Layer    ┆ Parameters ┆ Original MB ┆ Sensitivity Score ┆ Recommended Precision │
╞══════════╪════════════╪═════════════╪═══════════════════╪═══════════════════════╡
│ Layer 00 ┆ 0.98 M     ┆ 1.9 MB      ┆ 0.418             ┆ Q4_0                  │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Layer 01 ┆ 0.98 M     ┆ 1.9 MB      ┆ 0.283             ┆ Ternary (1.58b)       │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Layer 02 ┆ 0.98 M     ┆ 1.9 MB      ┆ 0.239             ┆ Ternary (1.58b)       │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Layer 03 ┆ 0.98 M     ┆ 1.9 MB      ┆ 0.283             ┆ Ternary (1.58b)       │
╰──────────┴────────────┴─────────────┴───────────────────┴───────────────────────╯

=== Autoregressive KV-Cache Memory Projections ===
╭────────────────┬────────────┬─────────────┬───────────────────────────┬──────────────────────────╮
│ Context Tokens ┆ Batch Size ┆ FP16 Memory ┆ FP8 Memory                ┆ Q4 Memory (Est.)         │
╞════════════════╪════════════╪═════════════╪═══════════════════════════╪══════════════════════════╡
│ 1024           ┆ 1          ┆ 2.00 MB     ┆ 1.00 MB (50.0% savings)   ┆ 0.50 MB (75.0% savings)  │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 2048           ┆ 1          ┆ 4.00 MB     ┆ 2.00 MB (50.0% savings)   ┆ 1.00 MB (75.0% savings)  │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 4096           ┆ 1          ┆ 8.00 MB     ┆ 4.00 MB (50.0% savings)   ┆ 2.00 MB (75.0% savings)  │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 8192           ┆ 1          ┆ 16.00 MB    ┆ 8.00 MB (50.0% savings)   ┆ 4.00 MB (75.0% savings)  │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 16384          ┆ 1          ┆ 32.00 MB    ┆ 16.00 MB (50.0% savings)  ┆ 8.00 MB (75.0% savings)  │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 32768          ┆ 1          ┆ 64.00 MB    ┆ 32.00 MB (50.0% savings)  ┆ 16.00 MB (75.0% savings) │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 65536          ┆ 1          ┆ 128.00 MB   ┆ 64.00 MB (50.0% savings)  ┆ 32.00 MB (75.0% savings) │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 131072         ┆ 1          ┆ 256.00 MB   ┆ 128.00 MB (50.0% savings) ┆ 64.00 MB (75.0% savings) │
╰────────────────┴────────────┴─────────────┴───────────────────────────┴──────────────────────────╯
```

---

## Step 5: Computing a Pareto Precision Allocation Plan

Generate an optimal heterogeneous precision plan given a hardware memory budget:

```bash
./target/release/thme plan examples/mock_llama --budget-gb 0.015 --output plan.json
```

Expected output:

```text
Computing heterogeneous precision Pareto plan for: "examples/mock_llama"

=== Heterogeneous Precision Allocation Plan ===
╭─────────────────────────────┬────────────╮
│ Metric                      ┆ Value      │
╞═════════════════════════════╪════════════╡
│ Target Model                ┆ mock_llama │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Original Size               ┆ 0.02 GB    │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Planned Compressed Size     ┆ 0.01 GB    │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Compression Ratio           ┆ 2.54x      │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Estimated Quality Retention ┆ 97.3%      │
╰─────────────────────────────┴────────────╯

Optimization plan successfully written to: "plan.json"
```

The resulting `plan.json` specifies exact per-layer quantizations (such as retaining sensitive input/output layers in Q4 while quantizing middle layers to BitNet b1.58 ternary representation).

---

## Step 6: Validating Quality Gates with Delta-PPL

THME enforces an automated empirical quality gate that evaluates Perplexity degradation ($\Delta \text{PPL}$) against standardized evaluation corpora (WikiText-2, C4, PTB):

$$\Delta \text{PPL} = \text{PPL}_{\text{candidate}} - \text{PPL}_{\text{baseline}} \le \epsilon$$

Execute a verification check:

```bash
./target/release/thme validate --corpus wikitext2 --baseline-ppl 5.84 --candidate-ppl 5.92 --max-ppl-delta 0.10
```

Expected output:

```text
--------------------------------------------------
  THME Empirical Perplexity Gate (Milestone v0.1.4)
--------------------------------------------------
  Baseline PPL   : 5.8400
  Candidate PPL  : 5.9200
  Δ PPL          : +0.0800
  Allowed Limit  : 0.1000
  Degradation    : +1.37%
  Decision       : Accept
  Summary        : PASSED: ΔPPL = +0.080 (Degradation: +1.37%) within tolerance limit (≤ 0.100). Candidate accepted.
  Action         : Commit model changes and proceed with export or next surgery stage.
--------------------------------------------------
```

If the candidate degradation exceeds the allowed threshold, the quality gate outputs `Decision: Reject` and triggers an automated rollback watchdog to prevent degraded weights from proceeding into downstream quantization or deployment.

---

## Step 7: Launching the Desktop Studio GUI

To inspect models and simulate surgeries visually, launch the native 60 FPS studio:

```bash
./target/release/thme gui
```

Or run the dedicated GUI binary directly:

```bash
./target/release/thme-gui
```

The desktop studio provides nine specialized tabs:
1. **Model Topology & Universal Paradigm Converter**: Real-time layer viewer, parameter breakdown, and cross-paradigm converter.
2. **Topological Sensitivity & Pareto Frontier**: Interactive sensitivity graphs and budget-constrained Pareto curve slider.
3. **Teacher LLM Surgery & Recipe Engine**: Visual diff audit of modified sequence mixers, attention head pruning, and precision changes.
4. **Hardware Roofline & Speculative Deck**: Interactive log-log roofline plot across target devices, FlashAttention-2 comparison, and speculative decoding draft simulator.
5. **Autoregressive Token Flow Simulator**: Real-time per-token memory traffic (bytes/token) and EWHA energy dissipation (mJ/token).
6. **Empirical Quality Gate & Delta-PPL Watchdog**: Live perplexity validation curves against WikiText-2 and C4.
7. **PyTorch / CUDA Training Bridge**: Interactive LoRA/QLoRA configuration and one-click script exporter.
8. **On-Device Micro-Training Deck**: In-engine parameter adaptation with CE and Distillation loss curves.
9. **Model Context Protocol (MCP) Daemon & Telemetry**: Real-time JSON-RPC request and response log monitor.

---

## CLI Parameter Reference

| Subcommand | Flag | Type | Default | Description |
|:---|:---|:---|:---|:---|
| `inspect` | `<MODEL_PATH>` | Path | None | Path to model file (`.safetensors` or `.gguf`) or directory |
| `inspect` | `--json` | Flag | false | Output report in machine-readable JSON |
| `profile` | `<MODEL_PATH>` | Path | None | Target model to profile |
| `profile` | `-b, --batch-size` | Integer | 1 | Inference batch size $B$ |
| `profile` | `-c, --context-lengths` | String | "1024,2048,4096,8192,16384,32768,65536,131072" | Comma-separated list of context window lengths |
| `profile` | `-o, --output` | Path | None | Destination JSON path for sensitivity data |
| `plan` | `<MODEL_PATH>` | Path | None | Target model to plan precision for |
| `plan` | `-b, --budget-gb` | Float | None | Target hardware memory ceiling in gigabytes |
| `plan` | `-m, --max-degradation`| Float | 0.05 | Maximum allowable quality degradation $\epsilon$ (default 5%) |
| `plan` | `-o, --output` | Path | "optimization_plan.json" | Destination file for the generated recipe |
| `validate` | `--baseline-ppl` | Float | None | Baseline reference perplexity score |
| `validate` | `--candidate-ppl`| Float | None | Candidate model perplexity score after transformation |
| `validate` | `--max-ppl-delta`| Float | 0.25 | Maximum acceptable increase in perplexity |
| `validate` | `--corpus` | String | "wikitext2" | Evaluation corpus (`wikitext2`, `c4`, `ptb`, `synthetic`) |
| `validate` | `--json` | Flag | false | Output structured verdict as JSON |

---

## Troubleshooting and Common Issues

### Issue 1: Model Path Missing or Unreadable
- **Symptom**: `Error: Failed to open model file: No such file or directory`.
- **Cause**: The path provided does not exist, or the folder does not contain a supported `model.safetensors` or `.gguf` file.
- **Remedy**: Generate the fixture first using `cargo run --example generate_mock_model`, or verify that the directory contains `config.json` and a valid tensor file.

### Issue 2: Budget Exceeds Model Baseline Size
- **Symptom**: `Optimization plan shows 1.00x compression`.
- **Cause**: The specified `--budget-gb` is larger than the uncompressed model size.
- **Remedy**: Set `--budget-gb` to a value strictly lower than the model's uncompressed footprint (e.g., 0.010 GB for the 0.02 GB mock model).

### Issue 3: Perplexity Delta Exceeded
- **Symptom**: `Decision: Reject: ΔPPL exceeds tolerance limit`.
- **Cause**: The candidate precision was reduced too aggressively, resulting in accuracy loss above the `--max-ppl-delta` limit.
- **Remedy**: Increase the target memory budget in `thme plan` or relax `--max-ppl-delta` if the downstream task is latency-critical.

---

## Next Steps and Related Guides

Continue exploring THME through the specialized subsystem guides:
- [Speculative Decoding Tutorial](speculative_decoding.md): Multi-token draft synthesis, BitNet ternary acceleration, and Medusa tree decoding.
- [Roofline Profiling Tutorial](roofline_profiling.md): Hardware ceilings, arithmetic intensity, and 4-tier cache latency simulation.
- [PyTorch & CUDA Interoperability Guide](pytorch_cuda_interop.md): Exporting `train_cuda.py`, Hugging Face configs, and in-engine LoRA weight merging.
- [Model Context Protocol (MCP) Setup](mcp_copilot_setup.md): Connecting THME to GitHub Copilot and Claude Desktop via JSON-RPC.
- [Universal 8 AI Model Paradigm Converter](paradigm_converter.md): Cross-converting architectures between LLM, MoE, SLM, and multimodal models.
- [README Overview](../README.md): Central architectural documentation and production roadmap.
