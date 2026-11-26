# THME Benchmark Dossier

> Measured results for THME (Topological Heterogeneous Manifold Engine).
> Every number below was produced by a real forward pass on the host listed under Provenance.
> Nothing in this document is simulated, projected, or estimated unless the row says so.

> These reports were measured with the code after the FP16 rounding fix (THME-3023, 13 November
> 2026). Reports measured before that fix were removed and re-run, not patched.

## 1. Perplexity Quality Gate

### SmolLM-135M

- Corpus: WikiText-2 (raw) test split, complete split
- Scored tokens: 303,118 in 297 windows of 1024
- Gate tolerance: ΔPPL <= 0.45

| Weights | Perplexity | Δ PPL | Cross-entropy (nats) | Gate |
| :--- | ---: | ---: | ---: | :--- |
| Source weights exactly as stored on disk | 19.7985 |  | 2.9856 | baseline |
| Exported GGUF artifact re-imported and decoded (uniform Q8_0) | 19.8326 | +0.0341 | 2.9873 | PASSED [ACCEPT] |
| Recipe 'Production-Pareto-SmolLM-135M' applied in memory: each tensor round-tripped through its selected format | 48.7059 | +28.9074 | 3.8858 | FAILED [ROLLBACK] |

### pythia-14m

- Corpus: WikiText-2 (raw) test split, complete split
- Scored tokens: 286,878 in 281 windows of 1024
- Gate tolerance: ΔPPL <= 0.45

| Weights | Perplexity | Δ PPL | Cross-entropy (nats) | Gate |
| :--- | ---: | ---: | ---: | :--- |
| Source weights exactly as stored on disk | 114.2862 |  | 4.7387 | baseline |
| Exported GGUF artifact re-imported and decoded (uniform Q8_0) | 115.1197 | +0.8335 | 4.7460 | FAILED [ROLLBACK] |
| Recipe 'Production-Pareto-pythia-14m' applied in memory: each tensor round-tripped through its selected format | 618.5311 | +504.2449 | 6.4273 | FAILED [ROLLBACK] |

## 2. Measured Precision Sensitivity: SmolLM-135M

Model: SmolLM-135M. Each row is a separate real forward pass over the first 16,384 corpus tokens (context 1024), so absolute values differ from the full-split figures in section 1. Tensors a row does not name keep their stored values. No retraining or calibration is applied. Bits per weight are nominal arithmetic, not artifact sizes: format bit-widths averaged over every parameter, with tensors a row does not change counted at their stored width (32 bits for FP32 embeddings), so the figure is a whole-model average, not the bits of the changed tensors.

Baseline perplexity on this subset: 17.9730

**Uniform formats**

| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |
| :--- | ---: | ---: | ---: | ---: |
| Every tensor as FP16 | 17.973 | +0.0000 | 100.0% | 16.00 |
| Every linear tensor as Q8_0 | 17.978 | +0.0046 | 78.9% | 13.06 |
| Every linear tensor as Q4_0 | 21.372 | +3.3992 | 78.9% | 10.30 |
| MLP tensors as Q4_0, attention as Q8_0 | 19.515 | +1.5418 | 78.9% | 10.99 |
| Every linear tensor as ternary (TQ1_0) | 1.59e+07 | +15,866,877.0418 | 78.9% | 7.99 |

**One tensor role ternarized at a time**

| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |
| :--- | ---: | ---: | ---: | ---: |
| Only attention_q tensors ternary, all layers | 56.808 | +38.8354 | 7.4% | 29.75 |
| Only attention_out tensors ternary, all layers | 66.576 | +48.6034 | 7.4% | 29.75 |
| Only attention_k tensors ternary, all layers | 75.165 | +57.1922 | 2.5% | 31.25 |
| Only mlp_up tensors ternary, all layers | 92.724 | +74.7508 | 19.7% | 26.00 |
| Only mlp_gate tensors ternary, all layers | 98.007 | +80.0344 | 19.7% | 26.00 |
| Only mlp_down tensors ternary, all layers | 188.019 | +170.0462 | 19.7% | 26.00 |
| Only attention_v tensors ternary, all layers | 376.869 | +358.8962 | 2.5% | 31.25 |

**Recipe versus random placement with the same tensor counts and shapes**

| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |
| :--- | ---: | ---: | ---: | ---: |
| Production recipe: mlp_down ternary and attention_q Q8_0 in layers 1..=28 | 43.431 | +25.4578 | 25.3% | 24.74 |
| 28 ternary and 28 Q8_0 tensors of the recipe's shapes, placed at random (seed 1) | 162.749 | +144.7756 | 25.3% | 24.74 |
| 28 ternary and 28 Q8_0 tensors of the recipe's shapes, placed at random (seed 2) | 101.774 | +83.8007 | 25.3% | 24.74 |
| 28 ternary and 28 Q8_0 tensors of the recipe's shapes, placed at random (seed 3) | 103.460 | +85.4866 | 25.3% | 24.74 |

Recipe Δ +25.4578; random placements Δ mean +104.6876 (min +83.8007, max +144.7756, n=3). The recipe is better than every random placement tested. With n=3 this is exploratory, not a statistical claim.

**All linear tensors of one layer ternarized at a time**

| Layer | Perplexity | Δ PPL |
| ---: | ---: | ---: |
| 0 | 1,447.444 | +1,429.4713 |
| 1 | 22.625 | +4.6517 |
| 2 | 20.442 | +2.4688 |
| 3 | 20.492 | +2.5191 |
| 4 | 19.196 | +1.2229 |
| 5 | 18.954 | +0.9812 |
| 6 | 18.890 | +0.9175 |
| 7 | 19.223 | +1.2501 |
| 8 | 19.415 | +1.4425 |
| 9 | 19.035 | +1.0620 |
| 10 | 19.277 | +1.3038 |
| 11 | 20.658 | +2.6849 |
| 12 | 20.560 | +2.5866 |
| 13 | 24.793 | +6.8202 |
| 14 | 23.175 | +5.2022 |
| 15 | 19.209 | +1.2357 |
| 16 | 19.896 | +1.9232 |
| 17 | 20.149 | +2.1763 |
| 18 | 21.026 | +3.0533 |
| 19 | 19.352 | +1.3790 |
| 20 | 19.424 | +1.4507 |
| 21 | 19.824 | +1.8512 |
| 22 | 19.716 | +1.7427 |
| 23 | 22.453 | +4.4796 |
| 24 | 19.546 | +1.5729 |
| 25 | 19.755 | +1.7822 |
| 26 | 21.608 | +3.6355 |
| 27 | 19.633 | +1.6597 |
| 28 | 21.671 | +3.6978 |
| 29 | 31.032 | +13.0595 |

Additivity check: the per-layer deltas sum to +1,509.3, while ternarizing all of those layers together measures +15,866,877.0. Per-layer effects do not add.

## 3. Measured Precision Sensitivity: pythia-14m

Model: pythia-14m. Each row is a separate real forward pass over the first 16,384 corpus tokens (context 1024), so absolute values differ from the full-split figures in section 1. Tensors a row does not name keep their stored values. No retraining or calibration is applied. Bits per weight are nominal arithmetic, not artifact sizes: format bit-widths averaged over every parameter, with tensors a row does not change counted at their stored width (32 bits for FP32 embeddings), so the figure is a whole-model average, not the bits of the changed tensors.

Baseline perplexity on this subset: 119.4032

**Uniform formats**

| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |
| :--- | ---: | ---: | ---: | ---: |
| Every tensor as FP16 | 119.403 | +0.0000 | 100.0% | 16.00 |
| Every linear tensor as Q8_0 | 119.943 | +0.5401 | 8.4% | 15.33 |
| Every linear tensor as Q4_0 | 199.227 | +79.8236 | 8.4% | 15.04 |
| MLP tensors as Q4_0, attention as Q8_0 | 152.531 | +33.1279 | 8.4% | 15.13 |
| Every linear tensor as ternary (TQ1_0) | 32,654.142 | +32,534.7392 | 8.4% | 14.79 |

**One tensor role ternarized at a time**

| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |
| :--- | ---: | ---: | ---: | ---: |
| Only attention_out tensors ternary, all layers | 290.185 | +170.7819 | 0.7% | 15.90 |
| Only attention_qkv tensors ternary, all layers | 1,995.635 | +1,876.2321 | 2.1% | 15.70 |
| Only mlp_down tensors ternary, all layers | 2,136.097 | +2,016.6937 | 2.8% | 15.60 |
| Only mlp_up tensors ternary, all layers | 2,219.339 | +2,099.9356 | 2.8% | 15.60 |

**Recipe versus random placement with the same tensor counts and shapes**

| Variant | Perplexity | Δ PPL | Parameters changed | Nominal bits/weight |
| :--- | ---: | ---: | ---: | ---: |
| Production recipe: mlp_down ternary and attention_q Q8_0 in layers 1..=4 | 676.220 | +556.8166 | 1.9% | 15.73 |
| 4 ternary and 0 Q8_0 tensors of the recipe's shapes, placed at random (seed 1) | 926.393 | +806.9899 | 1.9% | 15.73 |
| 4 ternary and 0 Q8_0 tensors of the recipe's shapes, placed at random (seed 2) | 752.233 | +632.8298 | 1.9% | 15.73 |
| 4 ternary and 0 Q8_0 tensors of the recipe's shapes, placed at random (seed 3) | 752.233 | +632.8298 | 1.9% | 15.73 |

Recipe Δ +556.8166; random placements Δ mean +690.8832 (min +632.8298, max +806.9899, n=3). The recipe is better than every random placement tested. With n=3 this is exploratory, not a statistical claim.

**All linear tensors of one layer ternarized at a time**

| Layer | Perplexity | Δ PPL |
| ---: | ---: | ---: |
| 0 | 528.609 | +409.2057 |
| 1 | 375.583 | +256.1802 |
| 2 | 600.131 | +480.7281 |
| 3 | 1,438.293 | +1,318.8897 |
| 4 | 364.611 | +245.2082 |
| 5 | 367.345 | +247.9414 |

Additivity check: the per-layer deltas sum to +2,958.2, while ternarizing all of those layers together measures +32,534.7. Per-layer effects do not add.

## 4. Measured Recipe Search: pythia-14m

Model: pythia-14m. Formats Q8_0, Q4_0, grouped by layer. Decisions used 8,192 calibration tokens; every figure below is measured on 16,384 separate validation tokens. Gate: Δ PPL ≤ 0.45.

On calibration, the sum of per-group costs predicted Δ +0.7303; the whole candidate measured Δ +0.3945 (1 moves reverted by refinement).

| Budget | Achieved size | Searched Δ PPL | Random placement Δ PPL (min / mean / max) | Gate |
| ---: | ---: | ---: | :--- | :--- |
| 90% | 83.3% | -0.0033 | +0.102 / +0.143 / +0.176 | PASSED |
| 70% | 66.7% | +0.1766 | +0.153 / +0.283 / +0.417 | PASSED |
| 50% | 58.3% | +0.3032 | +0.320 / +0.407 / +0.502 | PASSED (BUDGET NOT MET) |
| 30% | 58.3% | +0.3032 | +0.320 / +0.407 / +0.502 | PASSED (BUDGET NOT MET) |

Sizes are nominal (bits per weight of the linear tensors), not measured artifact sizes.

## 5. Measured Recipe Search: SmolLM-135M

Model: SmolLM-135M. Formats Q8_0, Q4_0, grouped by layer. Decisions used 8,192 calibration tokens; every figure below is measured on 16,384 separate validation tokens. Gate: Δ PPL ≤ 0.45.

On calibration, the sum of per-group costs predicted Δ -0.0133; the whole candidate measured Δ -0.0138 (0 moves reverted by refinement).

| Budget | Achieved size | Searched Δ PPL | Random placement Δ PPL (min / mean / max) | Gate |
| ---: | ---: | ---: | :--- | :--- |
| 90% | 90.0% | -0.0028 | -0.013 / -0.002 / +0.009 | PASSED |
| 70% | 70.0% | -0.0094 | -0.012 / -0.001 / +0.013 | PASSED |
| 50% | 50.0% | -0.0179 | -0.018 / -0.007 / +0.007 | PASSED |
| 30% | 30.0% | -0.0039 | -0.017 / -0.009 / -0.003 | PASSED |

Sizes are nominal (bits per weight of the linear tensors), not measured artifact sizes.

## 6. Measured Depth Behaviour: SmolLM-135M

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

## 7. Measured Depth Behaviour: pythia-14m

Model: pythia-14m (6 layers), first 16,384 corpus tokens, 16,368 scored. Stored weights, no retraining.

**Readout after each layer.** The model's own final norm and output head are applied to the state at that depth. The model was not trained to be read early, so this is a diagnostic of the stored weights, not the behaviour of a model trained with exit heads.

| Layers applied | Perplexity | Top prediction equals full-depth | Top prediction correct | Settled |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.16e+07 | 1.0% | 0.8% | 0.0% |
| 1 | 571,963.375 | 3.1% | 0.5% | 0.1% |
| 2 | 515,904.356 | 10.9% | 3.4% | 1.3% |
| 3 | 208,939.560 | 8.2% | 2.5% | 1.6% |
| 4 | 44,046.435 | 15.1% | 5.8% | 5.3% |
| 5 | 1,831.140 | 16.8% | 8.9% | 16.8% |
| 6 | 119.403 | 100.0% | 26.7% | 100.0% |

"Settled" is the share of tokens whose top prediction equals the full-depth top prediction at this depth and at every later depth.

**Layer schedule changes.** Layers repeated or skipped, read out at the end.

| Schedule | Layer applications | Perplexity | Δ PPL |
| :--- | ---: | ---: | ---: |
| Every layer once (reference) | 6 | 119.403 | +0.000 |
| Layers 1..5 applied twice | 10 | 5,012.824 | +4,893.421 |
| Layers 1..5 applied three times | 14 | 15,918.221 | +15,798.818 |
| Layer 3 applied twice | 7 | 3,367.017 | +3,247.613 |
| Layer 3 skipped | 5 | 22,407.933 | +22,288.529 |
| Layers 1..5 skipped | 2 | 105,008.841 | +104,889.438 |

## 8. Activation-Aware Scaling (AWQ-style): pythia-14m

Model: pythia-14m (EleutherAI/pythia-14m). Formats Q4_0, Q4_K. Scored on WikiText-2 test (287,159 tokens, context 1024). Calibration statistics come from WikiText-2 train: 16 windows of 2048 tokens (32,768 tokens) per calibration seed 1234, 1, 2, 3, 4. Calibration never reads the scoring split.

Arms: ceiling (unquantized); round-to-nearest without scaling (floor); AWQ-style scaling; control A (each layer's real per-channel scales permuted across input channels); control B1 (the same train tokens with their order shuffled); control B2 (uniform random token ids, a stress arm and not a baseline). Every result carries its benchmark id and version.

**Q4_0**

| Arm | n | Mean PPL | Sample SD | Min | Max | Δ vs RTN | Δ vs ceiling |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ceiling (unquantized) | 1 | 114.2862 | n/a | 114.2862 | 114.2862 | -78.7166 | +0.0000 |
| Round-to-nearest (floor) | 1 | 193.0028 | n/a | 193.0028 | 193.0028 | +0.0000 | +78.7166 |
| AWQ-style, 5 calibration seeds | 5 | 170.7025 | 2.7541 | 167.5659 | 173.5719 | -22.3003 | +56.4163 |
| Control A, 5 permutation seeds | 5 | 194.6773 | 6.6482 | 186.7303 | 202.5059 | +1.6745 | +80.3911 |
| Control B1, 5 shuffle seeds | 5 | 172.8148 | 5.2192 | 166.0696 | 179.5281 | -20.1880 | +58.5286 |
| Control B2 (stress arm, not a baseline) | 1 | 193.3853 | n/a | 193.3853 | 193.3853 | +0.3825 | +79.0991 |

Alpha (grid 0.0 to 1.0) is chosen per tensor inside the CLI and is not reported per run.

AWQ beats round-to-nearest: yes. Gap RTN minus AWQ mean +22.3003 against 2 x pooled SD 10.2648 (pooled SD 5.1324); every seed below RTN: yes.

Control A: AWQ beats it (mean gap +23.9748). Control B1: not beaten (mean gap +2.1123).

Gate (mean dPPL vs unquantized at most 0.45): fail at +56.4163.

**Q4_K**

| Arm | n | Mean PPL | Sample SD | Min | Max | Δ vs RTN | Δ vs ceiling |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ceiling (unquantized) | 1 | 114.2862 | n/a | 114.2862 | 114.2862 | -54.0088 | +0.0000 |
| Round-to-nearest (floor) | 1 | 168.2951 | n/a | 168.2951 | 168.2951 | +0.0000 | +54.0088 |
| AWQ-style, 5 calibration seeds | 5 | 161.2814 | 3.7014 | 157.9677 | 166.8506 | -7.0137 | +46.9952 |
| Control A, 5 permutation seeds | 5 | 167.4629 | 2.3070 | 165.0078 | 170.5948 | -0.8322 | +53.1767 |
| Control B1, 5 shuffle seeds | 5 | 158.0235 | 2.8963 | 155.2278 | 162.3823 | -10.2716 | +43.7373 |
| Control B2 (stress arm, not a baseline) | 1 | 173.4992 | n/a | 173.4992 | 173.4992 | +5.2041 | +59.2130 |

Alpha (grid 0.0 to 1.0) is chosen per tensor inside the CLI and is not reported per run.

AWQ beats round-to-nearest: yes. Gap RTN minus AWQ mean +7.0137 against 2 x pooled SD 6.0455 (pooled SD 3.0227); every seed below RTN: yes.

Control A: AWQ beats it (mean gap +6.1815). Control B1: not beaten (mean gap -3.2579).

Gate (mean dPPL vs unquantized at most 0.45): fail at +46.9952.

Null and negative results stand as reported. Each arm is measured on the same scoring split as the quantization gate above.

## 9. Activation-Aware Scaling (AWQ-style): SmolLM-135M

Model: SmolLM-135M (HuggingFaceTB/SmolLM-135M). Formats Q4_0, Q4_K. Scored on WikiText-2 test (303,415 tokens, context 1024). Calibration statistics come from WikiText-2 train: 16 windows of 2048 tokens (32,768 tokens) per calibration seed 1234, 1, 2, 3, 4. Calibration never reads the scoring split.

Arms: ceiling (unquantized); round-to-nearest without scaling (floor); AWQ-style scaling; control A (each layer's real per-channel scales permuted across input channels); control B1 (the same train tokens with their order shuffled); control B2 (uniform random token ids, a stress arm and not a baseline). Every result carries its benchmark id and version.

**Q4_0**

| Arm | n | Mean PPL | Sample SD | Min | Max | Δ vs RTN | Δ vs ceiling |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ceiling (unquantized) | 1 | 19.7984 | n/a | 19.7984 | 19.7984 | -3.9000 | +0.0000 |
| Round-to-nearest (floor) | 1 | 23.6984 | n/a | 23.6984 | 23.6984 | +0.0000 | +3.9000 |
| AWQ-style, 5 calibration seeds | 5 | 22.0425 | 0.0648 | 21.9790 | 22.1508 | -1.6560 | +2.2440 |
| Control A, 5 permutation seeds | 5 | 23.6073 | 0.1225 | 23.4075 | 23.7365 | -0.0911 | +3.8089 |
| Control B1, 5 shuffle seeds | 5 | 21.9991 | 0.0500 | 21.9307 | 22.0611 | -1.6993 | +2.2007 |
| Control B2 (stress arm, not a baseline) | 1 | 22.3618 | n/a | 22.3618 | 22.3618 | -1.3367 | +2.5633 |

Alpha (grid 0.0 to 1.0) is chosen per tensor inside the CLI and is not reported per run.

AWQ beats round-to-nearest: yes. Gap RTN minus AWQ mean +1.6560 against 2 x pooled SD 0.1702 (pooled SD 0.0851); every seed below RTN: yes.

Control A: AWQ beats it (mean gap +1.5648). Control B1: not beaten (mean gap -0.0434).

Gate (mean dPPL vs unquantized at most 0.45): fail at +2.2440.

**Q4_K**

| Arm | n | Mean PPL | Sample SD | Min | Max | Δ vs RTN | Δ vs ceiling |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ceiling (unquantized) | 1 | 19.7984 | n/a | 19.7984 | 19.7984 | -4.8173 | +0.0000 |
| Round-to-nearest (floor) | 1 | 24.6158 | n/a | 24.6158 | 24.6158 | +0.0000 | +4.8173 |
| AWQ-style, 5 calibration seeds | 5 | 21.6418 | 0.1310 | 21.4494 | 21.8078 | -2.9740 | +1.8434 |
| Control A, 5 permutation seeds | 5 | 24.3832 | 0.2091 | 24.1234 | 24.5863 | -0.2326 | +4.5847 |
| Control B1, 5 shuffle seeds | 5 | 21.6205 | 0.0548 | 21.5507 | 21.6892 | -2.9952 | +1.8221 |
| Control B2 (stress arm, not a baseline) | 1 | 21.8680 | n/a | 21.8680 | 21.8680 | -2.7477 | +2.0696 |

Alpha (grid 0.0 to 1.0) is chosen per tensor inside the CLI and is not reported per run.

AWQ beats round-to-nearest: yes. Gap RTN minus AWQ mean +2.9740 against 2 x pooled SD 0.2919 (pooled SD 0.1459); every seed below RTN: yes.

Control A: AWQ beats it (mean gap +2.7414). Control B1: not beaten (mean gap -0.0212).

Gate (mean dPPL vs unquantized at most 0.45): fail at +1.8434.

Null and negative results stand as reported. Each arm is measured on the same scoring split as the quantization gate above.

## 10. Sizes

### SmolLM-135M

| Item | Size | Basis |
| :--- | ---: | :--- |
| Source tensor payload | 513.1 MiB | measured (bytes on disk) |
| Exported GGUF v3, Q8_0 (uniform, all tensors) | 136.3 MiB | measured (file size, sha256 `17a50f1f20f6c111…`) |
| Recipe `Production-Pareto-SmolLM-135M` (216 × FP16, 28 × Q8_0, 28 × Ternary (1.58b)) | 205.1 MiB | **estimate** from nominal bits per weight; no artifact of this size is written |

### pythia-14m

| Item | Size | Basis |
| :--- | ---: | :--- |
| Source tensor payload | 26.8 MiB | measured (bytes on disk) |
| Exported GGUF v3, Q8_0 (uniform, all tensors) | 14.3 MiB | measured (file size, sha256 `efcb930e51569a20…`) |
| Recipe `Production-Pareto-pythia-14m` (68 × FP16, 8 × Ternary (1.58b)) | 26.4 MiB | **estimate** from nominal bits per weight; no artifact of this size is written |

## 11. Method

- Forward pass: THME reference CPU implementation (`src/inference/`), dense f32 arithmetic,
  RMSNorm, rotary position embeddings, grouped-query causal attention, SwiGLU MLP, tied output head.
- The corpus is tokenized once with the model's own tokenizer, without special tokens
  (`scripts/prepare_eval_tokens.py`), then cut into consecutive non-overlapping windows.
- Supported architectures: LLaMA family and GPT-NeoX (Pythia). The forward pass is unit-tested
  (causality, kernels, format round trips) and has been cross-checked against the Hugging Face
  fp32 reference on fixed prompts for Pythia-14M and SmolLM-135M (largest logit difference 1.9e-4
  and 2.1e-5; reports `docs/research/hf-cross-check-*.json`, THME-2001).
- Every token except the first of each window is scored given the earlier tokens of that window.
  Perplexity is exp(mean negative log-likelihood) over all scored tokens.
- This differs from overlapping sliding-window protocols, which give each token more context
  and therefore report lower perplexity. Figures are comparable to each other, not to other protocols.
- "Exported artifact" rows decode the written file back to f32 and run the same forward pass.
- "Recipe" rows round-trip each tensor through its selected format in memory
  (Q8_0: GGML 32-value blocks; Ternary: GGML TQ1_0 256-value blocks with MSE-optimal scale)
  with no retraining or calibration.

## 12. Provenance

### SmolLM-135M

- THME version: 0.1.9; run completed 2026-11-20 05:12 UTC
- Host: 12th Gen Intel(R) Core(TM) i5-12400F (12 logical CPUs), windows/x86_64
- Backend: THME reference CPU forward pass (f32, multithreaded)
- Evaluation wall time: 26.0 min total; baseline scored at 516 tokens/s (reference implementation on a machine in normal use; not a performance benchmark)
- Corpus source: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip (`wikitext-2-raw/wiki.test.raw`)
- Corpus sha256: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`
- Tokenizer: HuggingFaceTB/SmolLM-135M, sha256 `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`
- Token file: 303,415 tokens, sha256 `18e663b61dd00f49796c8a6ae498b7dbe009cac4a2eaf3c67fa7d81baf47d945`

### pythia-14m

- THME version: 0.1.9; run completed 2026-11-20 04:45 UTC
- Host: 12th Gen Intel(R) Core(TM) i5-12400F (12 logical CPUs), windows/x86_64
- Backend: THME reference CPU forward pass (f32, multithreaded)
- Evaluation wall time: 2.7 min total; baseline scored at 4708 tokens/s (reference implementation on a machine in normal use; not a performance benchmark)
- Corpus source: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip (`wikitext-2-raw/wiki.test.raw`)
- Corpus sha256: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`
- Tokenizer: EleutherAI/pythia-14m, sha256 `870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2`
- Token file: 287,159 tokens, sha256 `973595d9c7c580360a39eedde7297ec050f362a521940f7527eca4a00ea85278`

## 13. Not Measured

The following have no measurements and are deliberately absent from this dossier:

- Inference throughput, energy, or memory on any GPU or accelerator.
- Any speedup of THME-processed models over a baseline runtime.
- Models other than those listed in section 1.
- Any effect of retraining, distillation, or calibration on the quantized variants.

## 14. Reproduce

```bash
pip install tokenizers
python scripts/prepare_eval_tokens.py
python scripts/prepare_eval_tokens.py --split train --calibration-windows 16 --window-len 2048 --seed 1234
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
python scripts/generate_charts.py --data-dir docs/benchmarks/<model>/data --out-dir docs/benchmarks/<model>/charts
```
