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

| Subcommand | Alias | Purpose |
| :--- | :--- | :--- |
| `gui` | `ui` | Launch the 60 FPS interactive Native Desktop Studio GUI. |
| `inspect` | — | Zero-copy inspection of SafeTensors and GGUF tensors, shapes, and header metadata. |
| `profile` | — | Layerwise topological sensitivity analysis, condition numbers, and KV-cache scaling. |
| `plan` | — | Dynamic programming heterogeneous Pareto precision allocation plan generator. |
| `validate` | `check` | Perplexity gate verification against baseline across WikiText-2, C4, PTB. |
| `tweak` | `surgery`| Apply declarative JSON surgery recipes to model architectures in-place. |
| `report` | — | Export optimization and sensitivity summary reports in Markdown or JSON. |
| `convert` | — | Cross-convert model topology across 8 specialized AI paradigms (LLM, MoE, SLM, etc.). |
| `speculate` | — | Extract prefix manifolds and evaluate multi-token speculative draft speedups. |
| `roofline` | — | Profile operational arithmetic intensity against empirical hardware roofs. |
| `export` | — | Export source-backed model as GGUF v3 (32-byte aligned) or SafeTensors. |
| `interop` | — | Generate PyTorch fine-tuning scripts and Hugging Face configuration files. |
| `mcp` | `daemon` | Launch Model Context Protocol JSON-RPC server for VS Code and Copilot. |
| `co-surgery`| `cosurgery` | Run closed-loop autonomous Teacher-Student co-surgery optimization loop. |

---

## 3. Desktop Studio GUI Guide

Launch with:
```bash
thme gui
```

The GUI consists of 10 specialized workspaces:
- **Tab 0: Master Dashboard**: Global cockpit consolidating memory savings, perplexity deltas, and token simulation.
- **Tab 1: Models & Weights**: Ingestion wizard supporting drag-and-drop GGUF/SafeTensors and architecture rebuilder.
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
