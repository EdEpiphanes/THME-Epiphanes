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
