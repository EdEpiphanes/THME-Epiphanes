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
