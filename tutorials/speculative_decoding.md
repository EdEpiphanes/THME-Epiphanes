# Speculative Decoding with THME: Multi-Token Draft Synthesis, BitNet Ternary Acceleration, and Tree Decoding

## Executive Summary

Autoregressive Large Language Model (LLM) generation is strictly memory-bandwidth bound during token generation (batch size $B=1$). Speculative decoding breaks this memory wall by using a lightweight draft model to speculate $\gamma$ candidate tokens per step, followed by a single parallel forward evaluation pass through the target model.

THME provides an end-to-end speculative decoding engine that synthesizes lightweight draft sub-manifolds directly from target models without separate multi-billion token pretraining runs. The engine supports prefix layer extraction, hidden dimension projection, BitNet b1.58 ternary discretization of MLP blocks, Medusa/Eagle style multi-branch tree speculative decoding with 2D causal masks, and automated horizon optimization based on empirical acceptance statistics.

---

## Mathematical Foundations and Mechanics

### 1. The Leviathan Speculative Speedup Model

The effective latency speedup multiplier $S$ of speculative decoding is formulated by the Leviathan equation:

$$S = \frac{\mathbb{E}[k] + 1}{1 + \gamma \cdot r}$$

Where:
- $\gamma$: Speculative lookahead horizon (number of draft tokens emitted per verification pass).
- $r$: Latency cost ratio of the draft model relative to the target model:
  $$r = \frac{T_{\text{draft}}}{T_{\text{target}}} < 1.0$$
- $\mathbb{E}[k]$: Expected number of accepted draft tokens per cycle, parameterized by the mean acceptance probability $\alpha$:
  $$\mathbb{E}[k] = \sum_{i=1}^{\gamma} \alpha^i = \frac{\alpha (1 - \alpha^{\gamma})}{1 - \alpha}$$
- $\mathbb{E}[k] + 1$: Total tokens produced per cycle (the $k$ accepted tokens plus 1 recovery token sampled from the target model distribution).

### 2. Speculative Rejection Sampling Probability

To guarantee mathematically identical output distributions to the target model, THME evaluates candidate draft tokens using speculative rejection sampling:

$$u \le \min\left(1, \frac{p(x_i)}{q(x_i)}\right), \quad u \sim \mathcal{U}(0, 1)$$

Where:
- $q(x_i)$: Probability distribution over vocabulary emitted by the draft model.
- $p(x_i)$: Probability distribution over vocabulary computed by the target model.
- If $u \le \frac{p(x_i)}{q(x_i)}$, token $x_i$ is accepted.
- If rejected at index $k+1$, generation for that cycle halts. A residual token is resampled from the normalized positive difference $(p(x) - q(x))^+$, completely eliminating draft model hallucination drift.

### 3. Lookahead Horizon Trade-Off

Increasing the lookahead horizon $\gamma$ provides diminishing returns. While $\mathbb{E}[k]$ asymptotes toward $\frac{\alpha}{1-\alpha}$, draft latency overhead accumulates linearly: $1 + \gamma \cdot r$. If $r$ is large or $\alpha$ is low, $S < 1.0$ (speculative deceleration). THME automatically solves for the optimal horizon:

$$\gamma^* = \arg\max_{\gamma \in \mathbb{N}^+} \frac{\sum_{i=1}^{\gamma} \alpha^i + 1}{1 + \gamma \cdot r}$$

---

## Prerequisites and Setup

Before executing speculative decoding synthesis, verify that THME is built:

```bash
cargo build --release
```

Ensure a target model is available, such as the mock LLaMA fixture generated in `examples/mock_llama`:

```bash
cargo run --example generate_mock_model
```

If no target model path is specified, `thme speculate` automatically initializes a 12-layer synthetic LLM baseline (`synthetic-llama-baseline`: 12 layers, hidden size 2048, 747.6M parameters).

---

## Step 1: Synthesizing a Prefix Draft Model

To construct a draft model without training from scratch, THME extracts the first $L_{\text{draft}}$ prefix layers from the target manifold while retaining the embedding and final language model projection head.

Run speculative simulation with 4 prefix layers and a lookahead horizon of 5:

```bash
./target/release/thme speculate --draft-layers 4 --horizon 5 --temperature 0.7
```

Expected output:

```text
Using default 12-layer synthetic LLM baseline as Target Model.
Target Model: 'synthetic-llama-baseline' | Layers: 12 | Hidden Dim: 2048 | Params: 747634688

============================================================
   THME SPECULATIVE DECODING & DRAFT SYNTHESIS REPORT
============================================================
Draft Model Synthesis Report: 'synthetic-llama-baseline' -> 'synthetic-llama-baseline-draft-4l-2048d'
- Layers: 12 -> 4 (66.7% reduction)
- Hidden Dim: 2048 -> 2048
- Parameters: 747634688 -> 336592896 (55.0% reduction)
- Memory Footprint: 1426.00 MB -> 642.00 MB (55.0% savings)
- Latency Cost Ratio (r = T_draft/T_target): 0.333
- Strategy: Synthesized draft sub-manifold using Prefix Layer Extraction across 4 layers (hidden=2048).
------------------------------------------------------------
Speculative Decoding Telemetry Report (Horizon gamma=5)
- Total Tokens Emitted       : 55
- Mean Acceptance Rate alpha : 24.0%
- Tokens Generated / Pass    : 2.20 tok/pass
- Draft Latency Cost Ratio r : 0.333
- Effective Latency Speedup  : 0.83x
- Memory Bandwidth Savings   : 0.0%
- Status                     : Draft Latency Overhead Exceeds Acceptance Gain
------------------------------------------------------------
Optimal Lookahead Recommendation: gamma = 1 tokens (Estimated Speedup: 1.24x)
============================================================
```

In this unoptimized baseline run, the latency ratio $r = 0.333$ is too high for a 5-token horizon with a 24.0% acceptance rate, causing the draft overhead to exceed the verification gain.

---

## Step 2: Accelerating Draft Models with BitNet b1.58 Discretization

To make speculative decoding profitable, the latency cost ratio $r$ must be minimized. Applying BitNet b1.58 ternary discretization ($\{-1, 0, +1\}$) to the draft model's feed-forward MLP projections drastically reduces draft memory bandwidth and execution latency without significantly harming token acceptance rates:

```bash
./target/release/thme speculate --draft-layers 4 --bitnet --horizon 4
```

Expected output:

```text
Using default 12-layer synthetic LLM baseline as Target Model.
Target Model: 'synthetic-llama-baseline' | Layers: 12 | Hidden Dim: 2048 | Params: 747634688

============================================================
   THME SPECULATIVE DECODING & DRAFT SYNTHESIS REPORT
============================================================
Draft Model Synthesis Report: 'synthetic-llama-baseline' -> 'synthetic-llama-baseline-draft-4l-2048d'
- Layers: 12 -> 4 (66.7% reduction)
- Hidden Dim: 2048 -> 2048
- Parameters: 747634688 -> 336592896 (55.0% reduction)
- Memory Footprint: 1426.00 MB -> 411.00 MB (71.2% savings)
- Latency Cost Ratio (r = T_draft/T_target): 0.217
- Strategy: Synthesized draft sub-manifold using Ternary BitNet b1.58 Prefix across 4 layers (hidden=2048).
------------------------------------------------------------
Speculative Decoding Telemetry Report (Horizon gamma=4)
- Total Tokens Emitted       : 55
- Mean Acceptance Rate alpha : 30.0%
- Tokens Generated / Pass    : 2.20 tok/pass
- Draft Latency Cost Ratio r : 0.217
- Effective Latency Speedup  : 1.18x
- Memory Bandwidth Savings   : 15.2%
- Status                     : Moderate Acceleration Gain
------------------------------------------------------------
Optimal Lookahead Recommendation: gamma = 2 tokens (Estimated Speedup: 1.43x)
============================================================
```

By reducing the memory footprint from 642 MB down to 411 MB (71.2% savings), the latency cost ratio drops from $0.333$ to $0.217$. Speculation now yields a positive speedup of **1.18x**, with an optimal horizon of $\gamma = 2$ achieving **1.43x**.

---

## Step 3: Multi-Branch Tree Speculative Decoding

Standard linear speculation validates a single sequence of $\gamma$ tokens. If the first token is rejected, all subsequent tokens are discarded. Multi-branch tree speculation (similar to Medusa and Eagle) emits multiple candidate token paths simultaneously in a tree topology and validates them in a single parallel target forward pass using a custom 2D causal attention mask.

Enable tree speculation:

```bash
./target/release/thme speculate --bitnet --tree --steps 20
```

Expected output:

```text
Using default 12-layer synthetic LLM baseline as Target Model.
Target Model: 'synthetic-llama-baseline' | Layers: 12 | Hidden Dim: 2048 | Params: 747634688

============================================================
   THME SPECULATIVE DECODING & DRAFT SYNTHESIS REPORT
============================================================
Draft Model Synthesis Report: 'synthetic-llama-baseline' -> 'synthetic-llama-baseline-draft-4l-2048d'
- Layers: 12 -> 4 (66.7% reduction)
- Hidden Dim: 2048 -> 2048
- Parameters: 747634688 -> 336592896 (55.0% reduction)
- Memory Footprint: 1426.00 MB -> 411.00 MB (71.2% savings)
- Latency Cost Ratio (r = T_draft/T_target): 0.217
- Strategy: Synthesized draft sub-manifold using Ternary BitNet b1.58 Prefix across 4 layers (hidden=2048).
------------------------------------------------------------
Speculative Decoding Telemetry Report (Horizon gamma=5)
- Total Tokens Emitted       : 44
- Mean Acceptance Rate alpha : 24.0%
- Tokens Generated / Pass    : 2.20 tok/pass
- Draft Latency Cost Ratio r : 0.217
- Effective Latency Speedup  : 1.06x
- Memory Bandwidth Savings   : 5.3%
- Status                     : Moderate Acceleration Gain
------------------------------------------------------------
Tree-Based Speculation (Medusa/Eagle Multi-Branch Topology):
- Total Tree Candidate Nodes : 16
- Longest Accepted Path Nodes: 3
- Accepted Branch Tokens     : 2
- Tree Acceptance Rate       : 66.7%
------------------------------------------------------------
Optimal Lookahead Recommendation: gamma = 2 tokens (Estimated Speedup: 1.43x)
============================================================
```

Notice that the tree acceptance rate reaches **66.7%**, significantly outperforming linear acceptance by exploring top-2 candidate token branches concurrently.

---

## Step 4: Exporting Synthesized Draft Models for Inference

To deploy the synthesized draft model to external runtimes (such as vLLM, llama.cpp, or Hugging Face pipelines), export the draft model Intermediate Representation to JSON:

```bash
./target/release/thme speculate --model examples/mock_llama --draft-layers 2 --bitnet --export-draft models/mock_draft.json
```

Expected output:

```text
Scanning model metadata at: "examples/mock_llama"
Target Model: 'mock_llama' | Layers: 4 | Hidden Dim: 256 | Params: 12124416
Exported draft model IR to: "models/mock_draft.json"
```

The exported `mock_draft.json` file contains:
- Complete structural metadata (reduced layer count, hidden dimension, attention head ratios).
- Precision mappings for every tensor (e.g., Ternary format for MLP blocks, FP16 for normalization).
- Tensor byte offset references directly mapped from the original SafeTensors archive, enabling zero-copy weight loading in production runtimes.

---

## Step 5: Programmatic Telemetry via JSON for CI/CD Gates

Automated continuous integration pipelines can capture speculative decoding metrics in JSON format:

```bash
./target/release/thme speculate --bitnet --horizon 3 --json
```

Expected output:

```json
{
  "synthesis": {
    "description": "Synthesized draft sub-manifold using Ternary BitNet b1.58 Prefix across 4 layers (hidden=2048).",
    "draft_byte_size": 430965504,
    "draft_hidden_size": 2048,
    "draft_layers": 4,
    "draft_model_name": "synthetic-llama-baseline-draft-4l-2048d",
    "draft_params": 336592896,
    "latency_cost_ratio": 0.21666666666666667,
    "memory_savings_pct": 71.17806925512798,
    "parameter_reduction_pct": 54.97896213183731,
    "target_byte_size": 1495269376,
    "target_hidden_size": 2048,
    "target_layers": 12,
    "target_model_name": "synthetic-llama-baseline",
    "target_params": 747634688
  },
  "telemetry": {
    "effective_speedup_multiplier": 1.3090909090909093,
    "latency_cost_ratio": 0.21666666666666667,
    "lookahead_horizon": 3,
    "mean_acceptance_rate": 0.38666666666666666,
    "mean_accepted_per_step": 1.16,
    "mean_tokens_per_pass": 2.16,
    "memory_traffic_savings_pct": 23.611111111111114,
    "steps": [ ... ],
    "total_accepted_tokens": 29,
    "total_drafted_tokens": 75,
    "total_steps_simulated": 25,
    "total_tokens_emitted": 54
  },
  "tree_speculation": null
}
```

Use `jq` to validate speedup criteria within automated testing workflows:

```bash
./target/release/thme speculate --bitnet --horizon 3 --json | jq '.telemetry.effective_speedup_multiplier >= 1.0'
```

---

## Step 6: Studio GUI Tab 4 Speculative Deck Walkthrough

For visual inspection and real-time tuning:

1. Launch the studio:
   ```bash
   ./target/release/thme gui
   ```
2. Navigate to **Tab 4: Hardware Roofline & Speculative Deck**.
3. In the right panel (**Speculative Decoding & Multi-Token Draft Synthesizer**):
   - Adjust the **Draft Model Layers** slider (e.g., from 12 down to 4).
   - Toggle **BitNet b1.58 Ternary MLP** to observe instant drops in latency ratio $r$.
   - Drag the **Lookahead Horizon $\gamma$** slider from 1 to 8 and observe the real-time Leviathan speedup curve plot.
   - Toggle **Multi-Branch Tree Speculation** to visualize the candidate branching tree and 2D causal attention mask.
   - Click **Export Draft Model IR** to serialize the configured draft specification to disk.

---

## CLI Parameter Reference

| Flag | Short | Type | Default | Description |
|:---|:---|:---|:---|:---|
| `--model` | `-m` | Path | None | Target model path (`.safetensors` or `.gguf`). If omitted, uses synthetic baseline |
| `--draft-layers` | `-l` | Integer | 4 | Number of prefix layers to extract for the draft model |
| `--draft-hidden` | | Integer | None | Downscaled hidden dimension for the draft model (optional projection) |
| `--bitnet` | | Flag | false | Discretize draft MLP projections to BitNet b1.58 ternary representation |
| `--horizon` | `-g` | Integer | 5 | Speculative lookahead horizon $\gamma$ (draft tokens emitted per cycle) |
| `--temperature` | `-t` | Float | 0.7 | Sampling temperature $\tau$ (0.0 for greedy decoding, 0.7 for standard sampling) |
| `--steps` | `-s` | Integer | 25 | Number of speculative forward steps to simulate |
| `--tree` | | Flag | false | Evaluate multi-branch tree speculation (Medusa/Eagle tree attention) |
| `--export-draft` | | Path | None | Destination JSON path to export the synthesized draft model IR |
| `--json` | | Flag | false | Output telemetry in machine-readable JSON |

---

## Troubleshooting and Common Issues

### Issue 1: Speculative Deceleration ($S < 1.0$)
- **Symptom**: `Status: Draft Latency Overhead Exceeds Acceptance Gain (Speedup: 0.83x)`.
- **Cause**: The draft model has too many layers relative to the target, or the lookahead horizon $\gamma$ is too high for the model's acceptance rate.
- **Remedy**:
  1. Reduce `--draft-layers` (e.g., from 6 down to 2 or 4).
  2. Enable `--bitnet` to accelerate draft forward execution.
  3. Set `--horizon` to the value recommended in `Optimal Lookahead Recommendation`.

### Issue 2: Low Token Acceptance Rate ($\alpha < 20\%$)
- **Symptom**: `Mean Acceptance Rate alpha : 12.0%`.
- **Cause**: High sampling temperature, or the draft model hidden size was reduced too drastically, destroying representation alignment.
- **Remedy**: Lower the sampling temperature with `-t 0.2` or omit `--draft-hidden` to maintain full dimensional alignment with the target model's projection heads.

---

## Next Steps and Related Guides

- [Roofline Profiling Tutorial](roofline_profiling.md): Analyze memory-bound versus compute-bound ceilings on NVIDIA H100 and consumer GPUs.
- [Universal 8 AI Paradigm Converter](paradigm_converter.md): Convert dense LLMs to edge-ready BitNet SLMs.
- [Quickstart Guide](quickstart.md): Zero-copy inspection, layer sensitivity profiling, and quality gates.
- [README Overview](../README.md): Central architectural documentation and production roadmap.
