# Universal 8 AI Model Paradigm Converter: Cross-Paradigm Manifold Restructuring and Surgery Recipes

## Executive Summary

Modern machine learning architectures are bifurcating beyond monolithic causal transformers into specialized operational paradigms: Mixture of Experts (MoE) for high parameter capacity with bounded inference compute, Small Language Models (SLM) for edge execution, Vision-Language Models (VLM) for multimodal grounding, and Large Action Models (LAM) for autonomous agentic reasoning.

THME incorporates a Universal Model Paradigm Converter (`thme convert`) that programmatically restructures model manifolds across 8 specialized AI paradigms:
1. **LLM**: Large Language Model (Autoregressive causal token generator)
2. **LCM**: Large Concept Model (Sentence segmentation, SONAR concept space, latent diffusion)
3. **LAM**: Large Action Model (Perception system, intent router, neuro-symbolic memory, action planner)
4. **MoE**: Mixture of Experts (Router gating mechanism, sparse expert MLP bank, Top-K dispatcher)
5. **VLM**: Vision-Language Model (Vision patch encoder, cross-attention multimodal projector, fusion LM)
6. **SLM**: Small Language Model (Compact tokenizer, efficient GQA, BitNet b1.58 ternary MLP, edge KV cache)
7. **MLM**: Masked Language Model (Bidirectional encoder attention, [MASK] token prediction head)
8. **SAM**: Segment Anything Model (Spatial visual patch attention, prompt encoder, mask decoder)

---

## Architectural Taxonomy of the 8 Specialized Paradigms

| Paradigm Code | Full Architectural Name | Structural Pipeline Stages | Core Topological Mutations | Primary Deployment Target |
|:---|:---|:---|:---|:---|
| `llm` | Large Language Model | Causal Self-Attention -> SwiGLU FFN -> Autoregressive Head | Dense $N$-layer transformer baseline | Server clusters, cloud APIs |
| `moe` | Mixture of Experts | Router Gate -> Expert MLP Bank -> Top-K Dispatcher -> Combiner | Expands dense FFN into $E$ parallel experts with Softmax gating | Cloud scale inference with bounded latency |
| `slm` | Small Language Model | Compact Tokenizer -> GQA -> BitNet b1.58 MLP -> Edge Cache | Layer pruning, hidden projection, ternary discretization ($\{-1, 0, +1\}$) | On-device, edge microcontrollers, mobile |
| `vlm` | Vision-Language Model | Vision Encoder -> Multimodal Projector -> Fusion Backbone | Injects visual patch projection matrices into transformer prefix | Multimodal reasoning, document OCR, robotics |
| `lam` | Large Action Model | Perception -> Intent Router -> Neuro-Symbolic Memory -> Action Planner | Injects tool invocation heads and policy trajectory scoring tensors | Agentic execution, web automation, tool calling |
| `lcm` | Large Concept Model | Sentence Segmenter -> SONAR Concept Space -> Latent Diffusion | Converts discrete token vocabulary into continuous concept embeddings | Semantic search, cross-lingual abstract reasoning |
| `mlm` | Masked Language Model | Bidirectional Attention -> LayerNorm -> MLM Head | Removes causal lower-triangular attention mask | Classification, embeddings, reranking |
| `sam` | Segment Anything Model | Spatial Patch Attention -> Prompt Encoder -> Mask Decoder | Converts 1D sequence tokens into 2D spatial feature grids | Computer vision segmentation, zero-shot visual masking |

---

## Prerequisites and Environment Setup

Ensure the THME binary is compiled and available:

```bash
cargo build --release
```

To test conversions against a local model fixture, generate the mock LLaMA model:

```bash
cargo run --example generate_mock_model
```

If the `--model` flag is omitted, `thme convert` automatically initializes the 12-layer synthetic baseline (`synthetic-llama-baseline`: 12 layers, hidden size 2048, 747.6M parameters).

---

## Step 1: Automated Architecture Detection

When given an arbitrary model file or directory, THME scans tensor tensor names, shapes, and normalization layers zero-copy to classify its existing architectural paradigm:

```bash
./target/release/thme convert --model examples/mock_llama --to moe
```

Expected output:

```text
Scanning model metadata at: "examples/mock_llama"
Detected Source Architecture: Large Language Model (LLM) | Tensors: 30 | Hidden Dim: 256
```

THME identifies that `mock_llama` contains standard causal attention projections and SwiGLU feed-forward networks, classifying it as a dense Large Language Model (LLM).

---

## Step 2: Converting Dense LLM to Sparse Mixture of Experts (MoE)

Sparse Mixture of Experts models (such as Mixtral, DeepSeek-V2/V3, or DBRX) route each token through a router gating mechanism to a sparse subset of $K$ experts out of $E$ total experts, keeping inference compute bounded while expanding total parameter capacity.

Convert the synthetic LLM baseline to an 8-expert, Top-2 routing MoE:

```bash
./target/release/thme convert --to moe --experts 8 --top-k 2
```

Expected output:

```text
Using default 12-layer synthetic LLM baseline for conversion test.
Detected Source Architecture: Large Language Model (LLM) | Tensors: 86 | Hidden Dim: 2048

============================================================
   THME UNIVERSAL MODEL PARADIGM CONVERSION REPORT
============================================================
Target Paradigm  : Mixture of Experts (MoE)
Key Pipeline     : Router Gate -> Expert MLP Bank -> Top-K Dispatcher -> Weighted Combiner
------------------------------------------------------------
Conversion Summary: LLM -> MoE
- Parameters: 747634688 -> 3654483968 (+388.8%)
- Footprint: 1426.00 MB -> 6970.38 MB (-388.8% savings)
- Tensors Added: 264 | Restructured: 36
- Details: Successfully converted model from LLM to MoE. Parameters: 747634688 -> 3654483968 (+388.8%), Tensors Restructured: 300.
============================================================
```

### Topological Transformation Mechanics
1. **Router Gating Allocation**: For each layer $l \in [0, L-1]$, THME injects a router gating tensor $W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times E}$.
2. **Expert Replication**: The dense gate, up, and down MLP projections are split across $E$ independent expert weights.
3. **Inference Compute**: Although total model parameters expand by +388.8% (from 747M to 3.65B), active forward compute remains bounded because only $K=2$ experts execute per token.

---

## Step 3: Compacting to Edge BitNet b1.58 Ternary SLM

To deploy a large model onto resource-constrained edge hardware, convert it into a Small Language Model (SLM) by pruning layers, downscaling the hidden dimension, and quantizing MLP weights to BitNet b1.58 ternary format:

```bash
./target/release/thme convert --to slm --layers 6 --hidden 1024 --bitnet --save-recipe slm_recipe.json
```

Expected output:

```text
Using default 12-layer synthetic LLM baseline for conversion test.
Detected Source Architecture: Large Language Model (LLM) | Tensors: 86 | Hidden Dim: 2048
Saved declarative surgery recipe to: "slm_recipe.json"

============================================================
   THME UNIVERSAL MODEL PARADIGM CONVERSION REPORT
============================================================
Target Paradigm  : Small Language Model (SLM)
Key Pipeline     : Compact Tokenizer -> Efficient GQA -> BitNet b1.58 MLP -> Edge Cache
------------------------------------------------------------
Conversion Summary: LLM -> SLM
- Parameters: 747634688 -> 109838336 (-85.3%)
- Footprint: 1426.00 MB -> 209.50 MB (+85.3% savings)
- Tensors Added: 0 | Restructured: 62
- Details: Successfully converted model from LLM to SLM. Parameters: 747634688 -> 109838336 (-85.3%), Tensors Restructured: 62.
============================================================
```

The resulting model achieves an **85.3% memory reduction**, shrinking from 1,426 MB down to 209.5 MB, making it suitable for edge devices with restricted memory budgets.

---

## Step 4: Transforming to Multimodal VLM and Agentic LAM

### Converting to Vision-Language Model (VLM)

Transform an autoregressive language backbone into a multimodal model capable of processing visual patch tokens alongside text:

```bash
./target/release/thme convert --to vlm
```

Expected output:

```text
Using default 12-layer synthetic LLM baseline for conversion test.
Detected Source Architecture: Large Language Model (LLM) | Tensors: 86 | Hidden Dim: 2048

============================================================
   THME UNIVERSAL MODEL PARADIGM CONVERSION REPORT
============================================================
Target Paradigm  : Vision-Language Model (VLM)
Key Pipeline     : Vision Encoder -> Text Encoder -> Multimodal Projector -> Fusion LM
------------------------------------------------------------
Conversion Summary: LLM -> VLM
- Parameters: 747634688 -> 774897664 (+3.6%)
- Footprint: 1426.00 MB -> 1478.00 MB (-3.6% savings)
- Tensors Added: 25 | Restructured: 0
- Details: Successfully converted model from LLM to VLM. Parameters: 747634688 -> 774897664 (+3.6%), Tensors Restructured: 25.
============================================================
```

THME injects vision patch projection matrices (25 additional tensors, +3.6% parameter delta), aligning visual patch feature dimensions with the transformer hidden manifold.

### Converting to Large Action Model (LAM)

Transform a model for agentic tool use and API trajectory execution:

```bash
./target/release/thme convert --to lam
```

Expected output:

```text
Using default 12-layer synthetic LLM baseline for conversion test.
Detected Source Architecture: Large Language Model (LLM) | Tensors: 86 | Hidden Dim: 2048

============================================================
   THME UNIVERSAL MODEL PARADIGM CONVERSION REPORT
============================================================
Target Paradigm  : Large Action Model (LAM)
Key Pipeline     : Perception System -> Intent Router -> Neuro-Symbolic Memory -> Action Planner
------------------------------------------------------------
Conversion Summary: LLM -> LAM
- Parameters: 747634688 -> 747735040 (+0.0%)
- Footprint: 1426.00 MB -> 1426.19 MB (-0.0% savings)
- Tensors Added: 3 | Restructured: 0
- Details: Successfully converted model from LLM to LAM. Parameters: 747634688 -> 747735040 (+0.0%), Tensors Restructured: 3.
============================================================
```

THME injects neuro-symbolic intent routing heads and policy action trajectory evaluation tensors into the output stage.

---

## Step 5: Exporting Declarative Surgery Recipes

When performing architectural conversions, pass `--save-recipe <FILE>` to export the exact mathematical transformation specification as a declarative JSON recipe conforming to the `surgery-recipe-v1.json` schema.

Inspect the exported recipe:

```json
{
  "name": "paradigm-convert-to-slm",
  "author": "THME Paradigm Synthesizer",
  "target_model": "synthetic-llama-baseline",
  "version": "0.1.5",
  "description": "Declarative cross-paradigm surgery recipe converting LLM to SLM",
  "operations": [
    {
      "type": "ConvertParadigm",
      "params": {
        "target_paradigm": "SLM",
        "num_experts": 4,
        "top_k": 2,
        "target_hidden_size": 1024,
        "target_layers": 6,
        "apply_bitnet": true
      }
    }
  ],
  "expected_speedup": 1.35,
  "expected_memory_savings_pct": 85.308556
}
```

This recipe can be checked into version control, executed in automated CI pipelines using `thme tweak -m model.safetensors -r recipe.json`, or shared with team members.

---

## Step 6: Interactive Conversion in Studio GUI Tab 1

For interactive conversion and visual inspection:

1. Launch the studio:
   ```bash
   ./target/release/thme gui
   ```
2. In **Tab 1: Model Topology & Universal Paradigm Converter**:
   - The left panel displays the active model topology (layers, tensor shapes, parameters).
   - In the right panel (**Universal 8 Paradigm Converter**), select the target paradigm from the radio buttons:
     `LLM` • `LCM` • `LAM` • `MoE` • `VLM` • `SLM` • `MLM` • `SAM`
   - Adjust the contextual parameter sliders:
     - For `MoE`: Configure **Number of Experts** (2 to 64) and **Top-K Routing** (1 to 4).
     - For `SLM`: Configure **Target Layers**, **Target Hidden Dimension**, and toggle **BitNet b1.58 Ternary Discretization**.
   - Click **Execute Paradigm Restructuring** to view the live parameter delta, tensor diff, and memory savings bar.
   - Click **Export Surgery Recipe** to serialize the configured transformation to disk.

---

## CLI Parameter Reference

| Flag | Short | Type | Default | Description |
|:---|:---|:---|:---|:---|
| `--model` | `-m` | Path | None | Path to model file (`.safetensors` or `.gguf`) or directory. Defaults to synthetic baseline |
| `--to` | `-t` | String | None | Target paradigm code (`llm`, `lcm`, `lam`, `moe`, `vlm`, `slm`, `mlm`, `sam`) |
| `--experts` | | Integer | 4 | Number of experts when converting to MoE |
| `--top-k` | | Integer | 2 | Top-K active routing experts per token when converting to MoE |
| `--hidden` | | Integer | None | Target hidden dimension when downscaling or pruning to SLM |
| `--layers` | | Integer | None | Target layer count when pruning to SLM |
| `--bitnet` | | Flag | false | Discretize weights to BitNet b1.58 ternary format ($\{-1, 0, +1\}$) |
| `--save-recipe` | | Path | None | Destination path to save declarative surgery recipe JSON |
| `--json` | | Flag | false | Output conversion report in machine-readable JSON |

---

## Troubleshooting and Common Issues

### Issue 1: Unrecognized Target Paradigm
- **Symptom**: `Error: Unknown target paradigm 'transformer'. Supported paradigms: llm, lcm, lam, moe, vlm, slm, mlm, sam`.
- **Cause**: An invalid or unsupported paradigm identifier was passed to `--to`.
- **Remedy**: Specify one of the 8 supported paradigm identifiers: `llm`, `lcm`, `lam`, `moe`, `vlm`, `slm`, `mlm`, or `sam`.

### Issue 2: Excessive MoE Parameter Growth
- **Symptom**: Out-of-memory or excessive file size after MoE conversion.
- **Cause**: High expert counts (e.g., `--experts 16` or `32`) replicate feed-forward parameters linearly across all layers.
- **Remedy**: Lower `--experts` to 4 or 8, or prune layers using `--layers` prior to MoE upcycling.

---

## Next Steps and Related Guides

- [Speculative Decoding Tutorial](speculative_decoding.md): Accelerate inference for converted SLM and MoE architectures.
- [Roofline Profiling Tutorial](roofline_profiling.md): Analyze memory-bound versus compute-bound ceilings across hardware devices.
- [PyTorch & CUDA Interoperability Guide](pytorch_cuda_interop.md): Export fine-tuning scripts and merge LoRA adapters.
- [Quickstart Guide](quickstart.md): Zero-copy inspection, layer sensitivity profiling, and quality gates.
- [README Overview](../README.md): Central architectural documentation and production roadmap.
