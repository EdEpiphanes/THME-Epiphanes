# THME (Topological Heterogeneous Manifold Engine) — User Manual

> Official Documentation and CLI Reference for THME by Epiphanes.  
> Version: `0.2.0` | Repository: `https://github.com/EdEpiphanes/THME-Epiphanes`

---

## 1. Executive Summary

THME (Topological Heterogeneous Manifold Engine, backwards-compatible CLI alias `llm-tweaker`) is a pure-Rust framework for zero-copy LLM memory profiling, layer sensitivity estimation, Pareto precision optimization, speculative decoding drafting, hardware roofline boundary analysis, and autonomous closed-loop co-surgery.

---

## 2. Command Line Interface (CLI)

```bash
thme <SUBCOMMAND> [OPTIONS]
```

### 2.1 Subcommands Reference

Generated from `thme --help` on 27 November 2026; run `thme <subcommand> --help` for each one's options.

| Subcommand | Alias | Purpose |
| :--- | :--- | :--- |
| `gui` | `ui` | Launch the THME Studio (the separate `thme-gui` program next to this one) |
| `inspect` | - | Inspect model weights, tensor shapes, and architecture zero-copy |
| `profile` | - | Profile layer sensitivity and KV-cache memory scaling |
| `plan` | - | Generate heterogeneous precision Pareto allocation plan |
| `pathfinder` | - | Plan run settings (context, KV precision, predicted memory) from a model header |
| `evaluate` | - | Measure a surgery recipe's perplexity cost with the reference forward pass |
| `forward` | - | Run authentic reference forward pass on token sequence and output logits |
| `chat` | - | Chat with a model using incremental KV-cached greedy decoding |
| `search` | - | Search for a mixed-precision recipe from measured costs under a size budget |
| `calibrate` | - | Record per-input-channel activation statistics on calibration tokens |
| `validate` | `check` | Run quality gate check between baseline and candidate |
| `tweak` | `surgery` | Apply Teacher LLM surgery recipe or restructuring plan |
| `report` | - | View or export optimization plan reports |
| `convert` | - | Cross-convert model topology across 8 specialized AI paradigms (LLM, LCM, LAM, MoE, VLM, SLM, MLM, SAM) |
| `cluster` | - | Classify multi-PC interconnects (Thunderbolt / Ethernet / Wi-Fi) and partition layers |
| `speculate` | - | Synthesize draft models and evaluate multi-token speculative decoding speedups |
| `roofline` | - | Profile model layers against empirical hardware roofline ceilings and cache hierarchy |
| `export` | - | Export a source-backed model as GGUF or SafeTensors |
| `epiph` | - | Pack or verify Epiphanes model files (.epiph): weights plus all their metadata |
| `scan` | - | Security scan of a model file or folder for code execution, hidden payloads and malformed headers |
| `discover` | - | List models already on this machine (Hugging Face, Ollama, LM Studio, llama.cpp); read-only |
| `doctor` | - | Say what works on this machine (OS, memory, disk, GPU adapter, models) and what to try next |
| `bench` | - | Benchmark registry: list and check task files and result records |
| `figure` | - | Draw a model as an isometric figure: one tray per layer, one block per tensor |
| `interop` | - | Generate PyTorch / CUDA fine-tuning scripts or Hugging Face configurations |
| `mcp` | `daemon` | Run Model Context Protocol (MCP) server for VS Code, Copilot & AI agents |
| `co-surgery` | `cosurgery` | Run autonomous Closed-Loop Teacher-Student Co-Surgery optimization |
| `telemetry` | - | Query real-time CPU, RAM, and discrete GPU hardware telemetry |
| `tokenize` | - | Encode text with a model's tokenizer.json and decode it back |
| `chat-template` | - | Render a model's own chat template for a JSON list of messages |
| `download` | - | Download a model from the Hugging Face hub, then run the safety scan on it |

---

## 3. Desktop Studio GUI Guide

Launch with:

```bash
thme gui
```

The GUI consists of 10 specialized workspaces:

- **Tab 0: Master Dashboard**: Global cockpit consolidating memory savings, perplexity deltas, and token simulation.
- **Tab 1: Models & Weights**: Ingestion wizard supporting drag-and-drop SafeTensors, GGUF and .epiph and architecture rebuilder.
- **Tab 2: Layer Tweaker**: Fine-grained deconstruction of layer attention, MLPs, reaching profiles, and alpha/beta mixing.
- **Tab 3: Matrix Engine**: Spectral SVD decomposition, Hadamard transformations, and SmoothQuant analysis.
- **Tab 4: Benchmarks & Roofline**: Interactive log-log Roofline plot with hardware selector and batch sliders.
- **Tab 5: History & Checkpoints**: Immutable snapshot history with zero-cost state rollbacks.
- **Tab 6: Audit Log**: Structured compliance trail recording all surgery operations.
- **Tab 7: Surgery & Export**: Declarative recipe editor and GGUF v3 / SafeTensors binary export wizard.
- **Tab 8: Dev Lab & Training**: Multi-dataset ingestion and in-engine weight adaptation micro-trainer.
- **Tab 9: Co-Surgery**: Autonomous Teacher-Student co-surgery closed loop with live Pareto checkpoint ledger.

---

## 4. Declarative Recipe Schema

Surgery recipes are structured declarative JSON documents:

```json
{
  "name": "BitNet-Pareto-SmolLM",
  "version": "1.0.0",
  "author": "THME Engineer",
  "description": "Applies Ternary BitNet b1.58 to intermediate MLP layers",
  "target_model": "SmolLM-135M",
  "expected_speedup": 1.45,
  "expected_memory_savings_pct": 52.0,
  "operations": [
    {
      "type": "SetPrecision",
      "start_layer": 2,
      "end_layer": 28,
      "role": "MlpDown",
      "format": "Ternary"
    }
  ]
}
```

---

## 5. Security & Verification

Verify release asset integrity using:

```bash
sha256sum -c SHA256SUMS.txt
```
