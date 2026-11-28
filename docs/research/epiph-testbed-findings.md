<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# .epiph testbed: findings

One entry per experiment. A formula or parameter suggested by a result is labelled **Proposed** until a second,
separately registered experiment confirms it. Plan and registered settings: `docs/research/epiph-testbed-plan.md`.

## 1. Arm A (scratch), baseline checkpoint series, 28 November 2026

**What was run.** The registered settings of the plan: Pythia-14M shape, random initialisation, seed 1234, WikiText-103
train (118.7M Pythia ids), 12,000 steps of 32 windows of 1,024 tokens, checkpoint every 1,000 steps. Each checkpoint was
packed as `.epiph` (each recording the previous one as its parent) and measured with `thme evaluate` on the committed
WikiText-2 test tokens (sha256 973595d9..., the same file as every earlier Pythia-14M report), GPU path (f16 weights).
Report: `docs/benchmarks/epiph-testbed/data/arm_a_checkpoints.json`. The owner had a lightweight game open on the same
GPU; perplexity does not depend on GPU load and no timings are claimed.

**Result.** Test perplexity of the unmodified checkpoint:

| Step | 0 | 1,000 | 2,000 | 4,000 | 6,000 | 8,000 | 10,000 | 12,000 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WikiText-2 test perplexity | 50,689.5 | 157.4 | 107.2 | 72.2 | 63.0 | 59.0 | 56.6 | **55.2** |

- The random model starts at about the vocabulary size (50,689 against 50,304 tokens), as it should.
- The gain per 1,000 steps shrinks from 50 perplexity points (steps 1,000 to 2,000) to about 1 (steps 11,000 to 12,000);
  the curve is still falling at the end, so 12,000 steps is not a converged model.
- Training-curve loss (WikiText-103 validation text) and test perplexity (WikiText-2 test text) are different corpora and
  are not compared to each other.
- THME on the GPU agrees with Hugging Face on a smoke checkpoint of the same shape (largest logit difference 0.0010 against
  a limit of 0.005); the final checkpoint was not separately cross-checked.

**Not comparable.** The pretrained Pythia-14M has perplexity 114.3 on the same test file, but it was trained on the Pile, a different
corpus, so 55.2 does not mean the testbed model is a better model. It is evidence that in-domain training on a small corpus
gives a lower WikiText-2 perplexity at this size, nothing more.

**Proposed parameters or formulas.** None yet. This series is the input for the gate, allocator and PS-16 experiments.

## 2. Arms B (Pythia start) and C (bytes), checkpoint series, 28 November 2026

**What was run.** The same registered settings as entry 1 (12,000 steps, 32 windows of 1,024 tokens, seed 1234, checkpoint
every 1,000), measured the same way. Arm B starts from the published Pythia-14M weights; arm C is the same shape with a
256-entry vocabulary reading raw UTF-8 bytes (sequences of 2,048 bytes, micro-batch 4, no tokenizer). Arm C was measured
on the bytes of the WikiText-2 test text (the same text as the WikiText-103 test set) with a 2,048-byte window. The GPU
was free during measurement. Data: `docs/benchmarks/epiph-testbed/data/arm_b_checkpoints.json` and `arm_c_checkpoints.json`.

**Result.**

| Arm | Step 0 | Step 4,000 | Step 8,000 | Step 12,000 | WikiText-103 validation loss at 12,000 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A scratch (perplexity per token) | 50,689.5 | 72.2 | 59.0 | 55.2 | 3.716 nats per token |
| B Pythia start (perplexity per token) | 114.3 | 58.6 | 50.8 | **47.9** | 3.593 nats per token |
| C bytes (perplexity per byte) | 268.6 | 3.12 | 2.93 | 2.857 | 1.048 nats per byte |

- **Check of the chain.** B at step 0 gives 114.29, the same number as the pretrained Pythia-14M in every earlier report,
  so packing and measuring change nothing.
- **B against A.** Starting from Pythia's weights ends lower (47.9 against 55.2) and is ahead at every checkpoint, but the
  gap shrinks from 84 points at step 1,000 to about 7 at step 12,000. Both curves are still falling.
- **C against A and B, tokenizer-free.** C's final result is 1.514 bits per byte. Converting the token arms by the byte count
  of the same text (287,159 test tokens, 1,287,656 bytes, so 0.2230 tokens per byte) gives about 1.290 bits per byte for A and
  1.245 for B. This conversion assumes the token file covers exactly the bytes of the test file; treat it as approximate to a
  few percent until the token count is checked against the byte count. On this basis C is worse than A and B, as expected for
  a model with 1.26M parameters (A and B have 14.07M, mostly embeddings) that sees less than half the context (2,048 bytes against about 4,600 bytes in 1,024 tokens).
- The engine measures a tokenizer-less checkpoint without any change: pack, verify and evaluate worked, and the first value
  (2.857 per byte at step 12,000) matches the training run's validation loss (1.048 nats, e^1.048 = 2.85).

**Not claimed.** C is not a fair size match to A and B: its parameter count is 11 times smaller. Differences between arms mix
data history (B), tokenizer (C) and capacity; they do not isolate the effect of the math.

**Proposed parameters or formulas.** None yet.

**Also in this entry's change.** `thme evaluate` reports now carry `window_nll`, a list of (summed loss, scored tokens) per
window, the input for paired per-block losses (gate, block partition, PS-16). Reports from before this change lack it.

## 3. S0: post-hoc fake quantisation of arm A, no training, 28 November 2026

**What was run.** `scripts/testbed_s0.py` on arm A's final checkpoint (step 12,000). For each of the six layers, and then
layers 1 to 4 together, the four weight matrices of the layer (attention QKV and output, MLP in and out; 0.79M weights per
layer) were replaced by a quantised copy, one regime at a time, with no training. Regimes: ternary (dead zone 0.7 E|w|,
scale per 1,024 weights), binary (sign plus a scale per 1,024, with a residual of rank 0, 8 or 32), four phase layouts
(adjacent weight pairs as radius and angle: P1 3-bit angle, group radius; P2 3-bit angle and 1-bit radius; P3 4-bit
angle, group radius; P4 4-bit angle and 2-bit radius) and a plain 2-bit scalar control with the Gaussian-optimal step.
Measured in PyTorch (float32 weights, eager attention) on the WikiText-2 test tokens (281 windows of 1,024). The
unquantised baseline measured 55.183, the same as `thme evaluate` (55.184), so the screening and the engine agree.
Bits per weight are computed from the formulas (scale cost included, embeddings and output head not touched).
Report: `docs/benchmarks/epiph-testbed/data/s0_arm_a.json`. Screening values only: numbers for a report should be
re-measured with `thme evaluate` on exported folders (CPU F32).

**Result** (test perplexity, baseline 55.18):

| Regime | bits per weight | layer 0 | layer 1 | layer 2 | layer 3 | layer 4 | layer 5 | layers 1 to 4 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ternary | 1.60 | 103.2 | 85.6 | 84.5 | 90.6 | 81.5 | 82.4 | 380.7 |
| binary, rank 0 | 1.02 | 523.9 | 108.6 | 105.7 | 120.5 | 117.7 | 129.5 | 797.7 |
| binary, rank 8 | 2.35 | 275.2 | 80.0 | 81.5 | 87.0 | 102.7 | 89.9 | 374.9 |
| binary, rank 32 | 6.35 | 104.8 | 66.3 | 68.4 | 72.2 | 77.7 | 71.1 | 170.5 |
| 2-bit scalar (control) | 2.02 | 79.4 | 75.9 | 72.8 | 76.0 | 69.9 | 70.1 | 190.8 |
| phase P1 (3-bit angle, group radius) | 1.52 | 101.5 | 106.0 | 120.3 | 109.2 | 103.5 | 103.5 | 586.5 |
| phase P2 (3-bit angle, 1-bit radius) | 2.02 | 74.2 | 71.1 | 72.9 | 79.7 | 71.4 | 73.5 | 179.6 |
| phase P3 (4-bit angle, group radius) | 2.02 | 95.6 | 99.2 | 121.2 | 104.0 | 95.1 | 92.6 | 584.9 |
| phase P4 (4-bit angle, 2-bit radius) | 3.02 | 58.5 | 57.6 | 57.9 | 58.6 | 59.5 | 57.6 | 70.9 |

**Readings** (what the numbers show, no more):
- Damage adds up across layers. Ternary costs 26 to 35 points per middle layer alone, but 325 points when layers 1 to 4
  are all ternary. This is the same non-additivity the paper reports for the allocator (section 12.2) seen on a trained
  model.
- At about 2 bits per weight, phase P2 is level with the 2-bit scalar control: better on layers 0 and 1, level on
  layer 2, worse on layers 3, 4 and 5 (by 1.5 to 3.7), and better on layers 1 to 4 together (179.6 against 190.8).
  Not a decisive win.
- Radius precision matters more than angle precision at the same rate: P3 (4-bit angle, group radius) is far worse than
  P2 (3-bit angle, 1-bit radius) at 2.02 bits. A single radius per 1,024 weights throws away the magnitude information.
- P4 at 3.02 bits is close to lossless on single layers (+2 to +4) and +15.7 on layers 1 to 4. There is no 3-bit scalar
  control yet, so P4's advantage over a plain 3-bit quantiser is not established.
- Binary with rank 0 is the worst regime, and layer 0 is by far the most fragile layer for binary (523.9). The last layer
  is also somewhat worse (129.5) than layers 1 to 4 (105.7 to 120.5). This supports pinning the boundary layers (paper
  7.6), most strongly for the first layer; for ternary the last layer is not more fragile (82.4).
- Binary with a residual rank of 32 reaches the control's quality only at 6.35 bits per weight, because at width 128 the
  residual costs far more than the signs.

**Against the advisor's pre-registered prediction** (ternary on all middle layers adds more than 100 points; falsified if
less than 20): 380.7 against 55.2 is +325.5, so the prediction held.

**Not claimed.** One checkpoint, one corpus, post-hoc quantisation with no training and no calibration. Weights outside the
four matrices per layer, and the embeddings and head (about 92% of parameters), were not touched, so none of this is a
size saving. The ordering of regimes may change after quantisation-aware training, which is what the branches test.

**Next, from this result.** Add a 3-bit scalar control and a 1.6-bit control (ternary is the natural one) so each regime is
compared at its own rate; repeat on arm B; then re-measure the reported rows through `thme evaluate`.

**Proposed parameters or formulas.** None yet.

## 4. S0 repeated with matched controls, arms A and B, 28 November 2026

**What was run.** Entry 3's procedure with three more controls, so every regime is compared at its own rate: a 3-bit scalar
control (eight uniform levels, Gaussian-optimal step), a 3-level control at ternary's rate with no dead-zone rule, and the 2-bit
control of entry 3. The script also counts the share of ternary and binary codes that change when the float32 weights are first
rounded to float16 (the effect reported in arXiv 2610.07853). Arm A (baseline 55.18) and arm B (baseline 47.89, Pythia start),
both at step 12,000. Reports: `docs/benchmarks/epiph-testbed/data/s0_arm_a.json` and `s0_arm_b.json`. Screening values in
PyTorch float32, no training.

**Result** (test perplexity; layers 1 to 4 quantised together; single-layer column is the mean of layers 1 to 4 alone):

| Rate | Regime | Arm A, layers 1 to 4 | Arm B, layers 1 to 4 |
| :--- | :--- | ---: | ---: |
| 3.02 bits | scalar, 8 levels (control) | 87.2 | 75.9 |
| 3.02 bits | phase P4 (4-bit angle, 2-bit radius) | **70.9** | **66.3** |
| 2.02 bits | scalar, 4 levels (control) | 190.8 | 228.5 |
| 2.02 bits | phase P2 (3-bit angle, 1-bit radius) | **179.6** | **204.0** |
| 1.60 bits | scalar, 3 levels (control) | 423.9 | 519.9 |
| 1.60 bits | ternary (dead zone 0.7 E\|w\|) | **380.7** | 527.8 |
| 1.02 bits | binary, rank 0 | 797.7 | 1,637.8 |

**Readings.**
- **Phase P4 against the 3-bit scalar control.** Over layers 1 to 4 together, the extra perplexity above baseline is 15.7
  (P4) against 32.0 (control) on arm A, and 18.4 against 28.0 on arm B. On single layers the two are close (arm A: 57.6 to
  59.5 for P4, 57.4 to 62.6 for the control). So the advantage of the polar form shows when several layers are quantised at
  once, which is the case that matters. This answers the open point in entry 3.
- **Phase P2 against the 2-bit control** is better on layers 1 to 4 together on both arms (179.6 against 190.8, and 204.0
  against 228.5), mixed on single layers.
- **Ternary against its matched 3-level control.** Level on single layers; the dead-zone rule is 10% lower on arm A over layers
  1 to 4 (380.7 against 423.9) and level on arm B (527.8 against 519.9). The dead zone is not a clear gain here.
- **f16 cast.** Ternary codes that change after rounding to float16: 0.01% (1 weight in 10,000) for post-hoc ternary, none for
  binary. The 0.8 to 1.8% reported for released ternary checkpoints is for trained latent weights exported through bf16; it
  does not appear for our float32 weights rounded to float16. Reported numbers should still be taken on the CPU float32 path.
- **Arm B is more fragile** than arm A under every regime (for example binary rank 0 on layers 1 to 4: 1,637.8 against 797.7),
  and its layers 3 and 5 are the sensitive ones. It was trained on a different corpus, so this may be about the model's history
  rather than its architecture.

**Not claimed.** One checkpoint per arm, post-hoc, no training, matched controls only at three rates. The phase P4 layout is a
candidate from the advisor's list, not a tuned design; the 3-bit scalar control is a plain uniform quantiser, not the best 3-bit
method (for example one with learned scales). Embeddings and head are untouched.

**Proposed parameters or formulas.** None yet. Candidate for a registered follow-up: phase layouts at 3 bits per weight with
different angle/radius splits (4+2, 3+3, 5+1) against the 3-bit scalar and a learned-scale 3-bit control, repeated on arm B
and after quantisation-aware branches.
