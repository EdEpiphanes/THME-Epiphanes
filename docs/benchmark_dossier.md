# THME Benchmark Dossier

> Measured results for THME (Topological Heterogeneous Manifold Engine).
> Every number below was produced by a real forward pass on the host listed under Provenance.
> Nothing in this document is simulated, projected, or estimated unless the row says so.

## 1. Perplexity Quality Gate

### SmolLM-135M

- Corpus: WikiText-2 (raw) test split, complete split
- Scored tokens: 303,118 in 297 windows of 1024
- Gate tolerance: ΔPPL <= 0.45

| Weights | Perplexity | Δ PPL | Cross-entropy (nats) | Gate |
| :--- | ---: | ---: | ---: | :--- |
| Source weights exactly as stored on disk | 19.7985 |  | 2.9856 | baseline |
| Exported GGUF artifact re-imported and decoded (uniform Q8_0) | 19.8326 | +0.0341 | 2.9873 | PASSED [ACCEPT] |
| Recipe 'Production-Pareto-Smollm135m' applied in memory: each tensor round-tripped through its selected format | 48.7059 | +28.9075 | 3.8858 | FAILED [ROLLBACK] |

## 2. Measured Precision Sensitivity

Model: SmolLM-135M. Each row is a separate real forward pass over the first 16,384 corpus tokens (context 1024), so absolute values differ from the full-split figures in section 1. Tensors a row does not name keep their stored values. No retraining or calibration is applied. Bits per weight are nominal arithmetic, not artifact sizes.

Baseline perplexity on this subset: 17.9730

**Uniform formats**

| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |
| :--- | ---: | ---: | ---: | ---: |
| Every tensor as FP16 | 17.973 | -0.0000 | 100.0% | 16.00 |
| Every linear tensor as Q8_0 | 17.978 | +0.0046 | 78.9% | 13.06 |
| Every linear tensor as ternary (TQ1_0) | 1.59e+07 | +15,877,276.1175 | 78.9% | 7.99 |

**One tensor role ternarized at a time**

| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |
| :--- | ---: | ---: | ---: | ---: |
| Only attention_q tensors ternary, all layers | 56.830 | +38.8570 | 7.4% | 29.75 |
| Only attention_out tensors ternary, all layers | 66.572 | +48.5986 | 7.4% | 29.75 |
| Only attention_k tensors ternary, all layers | 75.165 | +57.1921 | 2.5% | 31.25 |
| Only mlp_up tensors ternary, all layers | 92.720 | +74.7475 | 19.7% | 26.00 |
| Only mlp_gate tensors ternary, all layers | 98.018 | +80.0449 | 19.7% | 26.00 |
| Only mlp_down tensors ternary, all layers | 188.018 | +170.0453 | 19.7% | 26.00 |
| Only attention_v tensors ternary, all layers | 376.654 | +358.6815 | 2.5% | 31.25 |

**Recipe versus random placement with the same tensor counts and shapes**

| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |
| :--- | ---: | ---: | ---: | ---: |
| Production recipe: mlp_down ternary and attention_q Q8_0 in layers 1..=28 | 43.431 | +25.4578 | 25.3% | 24.74 |
| 28 ternary and 28 Q8_0 tensors of the recipe's shapes, placed at random (seed 1) | 100.258 | +82.2851 | 25.3% | 24.74 |
| 28 ternary and 28 Q8_0 tensors of the recipe's shapes, placed at random (seed 2) | 191.211 | +173.2381 | 25.3% | 24.74 |
| 28 ternary and 28 Q8_0 tensors of the recipe's shapes, placed at random (seed 3) | 71.279 | +53.3058 | 25.3% | 24.74 |

Recipe Δ +25.4578; random placements Δ mean +102.9430 (min +53.3058, max +173.2381, n=3). The recipe is better than every random placement tested. With n=3 this is exploratory, not a statistical claim.

**All linear tensors of one layer ternarized at a time**

| Layer | Perplexity | Δ PPL |
| ---: | ---: | ---: |
| 0 | 1,447.395 | +1,429.4225 |
| 1 | 22.625 | +4.6518 |
| 2 | 20.442 | +2.4688 |
| 3 | 20.492 | +2.5190 |
| 4 | 19.196 | +1.2229 |
| 5 | 18.954 | +0.9812 |
| 6 | 18.890 | +0.9175 |
| 7 | 19.223 | +1.2501 |
| 8 | 19.416 | +1.4431 |
| 9 | 19.035 | +1.0619 |
| 10 | 19.277 | +1.3038 |
| 11 | 20.658 | +2.6849 |
| 12 | 20.559 | +2.5863 |
| 13 | 24.793 | +6.8202 |
| 14 | 23.175 | +5.2022 |
| 15 | 19.209 | +1.2356 |
| 16 | 19.896 | +1.9235 |
| 17 | 20.149 | +2.1763 |
| 18 | 21.026 | +3.0533 |
| 19 | 19.352 | +1.3790 |
| 20 | 19.424 | +1.4513 |
| 21 | 19.824 | +1.8512 |
| 22 | 19.716 | +1.7427 |
| 23 | 22.452 | +4.4792 |
| 24 | 19.546 | +1.5729 |
| 25 | 19.755 | +1.7822 |
| 26 | 21.608 | +3.6355 |
| 27 | 19.633 | +1.6597 |
| 28 | 21.671 | +3.6978 |
| 29 | 31.032 | +13.0594 |

Additivity check: the per-layer deltas sum to +1,509.2, while ternarizing all of those layers together measures +15,877,276.1. Per-layer effects do not add.

## 3. Measured Depth Behaviour

Model: SmolLM-135M (30 layers), first 16,384 corpus tokens, 16,368 scored. Stored weights, no retraining.

**Readout after each layer.** The model's own final norm and output head are applied to the state at that depth. The model was not trained to be read early, so this is a diagnostic of the stored weights, not the behaviour of a model trained with exit heads.

| Layers applied | Perplexity | Top prediction equals full-depth | Top prediction correct | Settled |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.36e+45 | 2.4% | 2.3% | 0.0% |
| 1 | 1,329,823.199 | 3.0% | 1.2% | 0.6% |
| 2 | 829,178.972 | 3.4% | 1.5% | 0.7% |
| 3 | 847,859.174 | 3.7% | 1.5% | 0.8% |
| 4 | 853,296.799 | 3.9% | 1.7% | 0.8% |
| 5 | 891,500.620 | 3.8% | 1.7% | 0.9% |
| 6 | 871,742.310 | 4.1% | 1.8% | 1.0% |
| 7 | 925,407.222 | 4.2% | 2.1% | 1.1% |
| 8 | 600,438.536 | 4.1% | 1.9% | 1.3% |
| 9 | 495,119.777 | 4.8% | 2.5% | 1.6% |
| 10 | 506,038.595 | 5.0% | 2.8% | 2.1% |
| 11 | 427,494.757 | 5.8% | 3.4% | 2.5% |
| 12 | 389,916.745 | 7.1% | 4.3% | 3.3% |
| 13 | 291,757.079 | 8.1% | 5.2% | 4.1% |
| 14 | 336,698.228 | 8.5% | 5.5% | 4.6% |
| 15 | 209,999.661 | 10.8% | 7.0% | 5.6% |
| 16 | 210,503.386 | 11.1% | 7.2% | 6.2% |
| 17 | 169,671.910 | 11.5% | 7.4% | 6.9% |
| 18 | 152,494.776 | 12.0% | 8.1% | 8.1% |
| 19 | 51,169.378 | 17.5% | 12.2% | 12.0% |
| 20 | 36,100.028 | 19.0% | 12.7% | 13.5% |
| 21 | 31,240.492 | 22.4% | 15.0% | 15.8% |
| 22 | 22,707.533 | 24.0% | 15.8% | 17.9% |
| 23 | 9,167.329 | 30.0% | 18.9% | 22.0% |
| 24 | 953.545 | 41.0% | 26.2% | 28.7% |
| 25 | 496.350 | 45.1% | 28.4% | 33.4% |
| 26 | 253.763 | 50.7% | 31.9% | 39.3% |
| 27 | 105.328 | 54.6% | 34.5% | 44.7% |
| 28 | 78.280 | 57.9% | 35.6% | 51.0% |
| 29 | 43.603 | 62.9% | 37.5% | 62.9% |
| 30 | 17.973 | 100.0% | 44.5% | 100.0% |

"Settled" is the share of tokens whose top prediction equals the full-depth top prediction at this depth and at every later depth.

**Layer schedule changes.** Layers repeated or skipped, read out at the end.

| Schedule | Layer applications | Perplexity | Δ PPL |
| :--- | ---: | ---: | ---: |
| Every layer once (reference) | 30 | 17.973 | +0.000 |
| Layers 7..23 applied twice | 46 | 55.311 | +37.338 |
| Layers 7..23 applied three times | 62 | 334.768 | +316.795 |
| Layer 15 applied twice | 31 | 19.217 | +1.244 |
| Layer 15 skipped | 29 | 20.057 | +2.084 |
| Layers 7..23 skipped | 14 | 30,793.018 | +30,775.045 |

## 4. Sizes

### SmolLM-135M

| Item | Size | Basis |
| :--- | ---: | :--- |
| Source tensor payload | 513.1 MiB | measured (bytes on disk) |
| Exported GGUF v3, Q8_0 (uniform, all tensors) | 136.3 MiB | measured (file size, sha256 `b57a94cc9e93112f…`) |
| Recipe `Production-Pareto-Smollm135m` (216 × FP16, 28 × Q8_0, 28 × Ternary (1.58b)) | 205.1 MiB | **estimate** from nominal bits per weight; no artifact of this size is written |

## 5. Method

- Forward pass: THME reference CPU implementation (`src/inference/`), dense f32 arithmetic,
  RMSNorm, rotary position embeddings, grouped-query causal attention, SwiGLU MLP, tied output head.
- The corpus is tokenized once with the model's own tokenizer, without special tokens
  (`scripts/prepare_eval_tokens.py`), then cut into consecutive non-overlapping windows.
- Every token except the first of each window is scored given the earlier tokens of that window.
  Perplexity is exp(mean negative log-likelihood) over all scored tokens.
- This differs from overlapping sliding-window protocols, which give each token more context
  and therefore report lower perplexity. Figures are comparable to each other, not to other protocols.
- "Exported artifact" rows decode the written file back to f32 and run the same forward pass.
- "Recipe" rows round-trip each tensor through its selected format in memory
  (Q8_0: GGML 32-value blocks; Ternary: GGML TQ1_0 256-value blocks with MSE-optimal scale)
  with no retraining or calibration.

## 6. Provenance

### SmolLM-135M

- THME version: 0.1.9; run completed 2026-11-12 14:19 UTC
- Host: 12th Gen Intel(R) Core(TM) i5-12400F (12 logical CPUs), windows/x86_64
- Backend: THME reference CPU forward pass (f32, multithreaded)
- Evaluation wall time: 27.8 min total; baseline scored at 519 tokens/s (reference implementation on a machine in normal use; not a performance benchmark)
- Corpus source: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip (`wikitext-2-raw/wiki.test.raw`)
- Corpus sha256: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`
- Tokenizer: HuggingFaceTB/SmolLM-135M, sha256 `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`
- Token file: 303,415 tokens, sha256 `18e663b61dd00f49796c8a6ae498b7dbe009cac4a2eaf3c67fa7d81baf47d945`

## 7. Not Measured

The following have no measurements and are deliberately absent from this dossier:

- Inference throughput, energy, or memory on any GPU or accelerator.
- Any speedup of THME-processed models over a baseline runtime.
- Models other than those listed in section 1.
- Any effect of retraining, distillation, or calibration on the quantized variants.

## 8. Reproduce

```bash
pip install tokenizers
python scripts/prepare_eval_tokens.py
cargo run --release --example production_verification -- \
    --model-path models/SmolLM-135M \
    --tokens target/eval/wikitext2_test.SmolLM-135M.u32
cargo run --release --example depth_probe -- \
    --model-path models/SmolLM-135M \
    --tokens target/eval/wikitext2_test.SmolLM-135M.u32
cargo run --release --example precision_sweep -- \
    --model-path models/SmolLM-135M \
    --tokens target/eval/wikitext2_test.SmolLM-135M.u32 --layer-scan
python scripts/generate_dossier.py
```
