# THME Research Update: First Measured Results

**Date:** 12 November 2026
**Applies to:** THME Research Programme v2.2 and the draft v3.0 concepts
**Evidence:** [`docs/benchmark_dossier.md`](../benchmark_dossier.md), generated from real forward passes

## 1. Status

Until this update, no THME quality figure came from running a model. The quality gate
used an analytic proxy and the benchmark dossier held hand-written numbers. Both are
replaced. The repository now contains a reference CPU forward pass (`src/inference/`)
and three experiments that run it on SmolLM-135M against the WikiText-2 test split.

Everything below is one model, one corpus, one metric (perplexity), with no retraining
or calibration. It is enough to test some claims and not others; section 5 says which.

## 2. What was measured

| Experiment | Result |
| :--- | :--- |
| Baseline, full test split (303,118 scored tokens) | Perplexity 19.80 |
| Exported GGUF (uniform Q8_0), read back from disk | 19.83 (Δ +0.03), passes the 0.45 gate |
| Production surgery recipe applied to the weights | 48.71 (Δ +28.91), fails the gate |
| Every linear tensor ternary | About 1.6 × 10⁷ |
| One layer ternary at a time | Δ between +0.9 and +6.8 for layers 1–28; +13.1 for layer 29; +1,429 for layer 0 |
| Sum of those per-layer deltas vs all layers together | +1,509 vs +15,877,276 |
| Recipe vs random placement, same tensor counts and shapes | Recipe Δ +25.5; random Δ +53.3, +82.3, +173.2 |
| Tokens whose top prediction has settled by 30% depth | 1.6% |
| Tokens settled by 80% depth | 28.7% |
| Middle block (layers 7–22) applied twice | Δ +37.3 |
| Middle block skipped | Perplexity about 30,800 |

Rows from "Every linear tensor ternary" down use the first 16,384 corpus tokens
(baseline 17.97 on that subset). "Ternary" is the GGML TQ1_0 block format with an
MSE-optimal scale per 256 weights.

## 3. What this says about the v2.2 hypotheses

| Hypothesis | Status | Basis |
| :--- | :--- | :--- |
| H1 Import and execution | Partly supported | Import, forward pass and a Q8_0 GGUF round trip work and are measured. The exporter cannot yet write per-tensor formats, so a heterogeneous artifact does not exist. |
| H2 Heterogeneous beats uniform at comparable storage | Not yet tested | No pair of candidates at matched measured storage exists. So far uniform Q8_0 is nearly free and every ternary assignment without retraining is expensive. |
| H3 Structured placement beats random | Exploratory support | The recipe beat 3 of 3 random placements. All four are far outside the gate, and n = 3 supports no statistical claim. |
| H4 Global refinement helps | Not tested | The additivity result shows why it would be needed. |
| H5 Cache efficiency | Not tested | No KV-cache measurement exists. |
| H6 Energy efficiency | Not tested | No speed or energy measurement exists. |
| H7 Teacher-assisted recovery | Not tested | This is now the critical path: post-training ternary fails on its own. |
| H8 Task specialization | Not tested | |
| H9 Design-space dependence | Not testable yet | One model, one task. |

## 4. What this says about the v2.0 critique

1. **Kernel and hardware overhead.** Unanswered. Nothing here measures speed.
2. **Routing divergence.** The premise is weaker than assumed. Read through the model's
   own output head, 1.6% of tokens have a settled prediction at 30% depth. This is a
   lower bound: the model has no trained exit heads. It shows that early exit is not
   available from stored weights and must be trained and then measured.
3. **Non-additive interactions.** Confirmed, strongly. Summing per-layer effects
   predicts a perplexity increase of about 1,500; the measured joint effect is about
   16 million. An allocator that adds per-layer costs cannot rank aggressive candidates.
4. **Bounded memory versus attention.** Unanswered.

## 5. The draft v3.0 concepts

The v3.0 material (fluid topology, heterogeneous super-matrix, UHTK, Dory loops, tiny
needles) was drafted without access to the repository and is not implemented anywhere
in it. None of it is tested by the results above. Specific points to correct before it
goes into a paper:

- **The "Dory" reference is a different idea.** The paper in the reference folder is
  Dory1.0-4B, a looped transformer: one shared block of layers applied k times, trained
  on about 1.5 trillion tokens with loop sampling. Its memory result is that KV-cache
  size can be made independent of the *loop count*. It still grows with context length.
  It is not a volatile buffer that holds prompt constraints. Our measurement agrees
  with the paper's premise that looping needs training: applying SmolLM's middle block
  twice, untrained, raises perplexity from 18.0 to 55.3.
- **KV cache is linear, not quadratic.** Cache memory grows linearly with context
  length (the v2.2 paper's own formula). Attention compute is what grows quadratically.
- **"O(1) memory" is a claim, not a property.** Nothing yet defines what state replaces
  the cache or measures recall with it. The v2.2 rule applies: report context length,
  quality loss and any replacement state, and do not relabel a buffer as "no cache".
- **Orthogonal items are not free to store.** Holding n independent d-dimensional
  constraints takes n × d numbers whether or not they are orthogonal. Superposition
  trades capacity for interference noise; that trade has to be measured.
- **Incompressibility does not guarantee stability.** The cited Navier–Stokes work, as
  summarized in the planning notes, concerns blowup that occurs *despite* bounded
  energy, so it cannot be the basis for a no-blowup guarantee. (That paper has not been
  reviewed for this note.)
- **NP-hardness and "proven bounds".** Allocation with additive costs is a
  multiple-choice knapsack, which is NP-hard. No approximation bound has been proved
  for the greedy allocator, and section 4 point 3 shows the additive cost model is
  itself wrong at aggressive precisions.
- **α as hash tables, γ as phase superposition, viscosity as attention.** These need
  operational definitions (inputs, outputs, state size, update rule) and a baseline to
  beat before they can be implemented or falsified.

A v3.0 architecture is a new model that must be trained. It is a separate project from
the v2.2 conversion tool and should be scoped as one.

## 6. Next experiments, in order

1. **Find a passing heterogeneous allocation.** Add 4-bit formats to the measured path
   and search for an assignment smaller than uniform Q8_0 that stays inside the gate,
   using the measured per-layer and per-role sensitivities. This is the first real test
   of H2.
2. **Make the exporter honour per-tensor formats**, so the measured recipe and the
   written artifact are the same thing.
3. **Activation-aware calibration** for low-bit formats. Weight-space MSE is not enough.
4. **Fitted exit readouts.** Fit a linear map from each depth to the final state on
   held-out text and repeat the depth probe. This gives a fair answer on early exit.
5. **KV-cache measurement** on an incremental decode path, as the baseline for H5.
6. **Retire the analytic proxy** from `thme validate`, the MCP server and the
   co-surgery loop, or label its output as simulated wherever it is shown.

## 7. Reproduce

Commands are listed at the end of the dossier. Evaluation inputs are written under
`target/eval/` and are not committed.
