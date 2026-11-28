<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Literature scan for the testbed and arm D, 28 November 2026

Owner request: scan the latest papers (starting from the Meta AI publications page) for self-training and other
improvements usable in the research, the architecture, the building and the testing. Ticket: THME-3110.

**How reliable this is.** The Meta page (`ai.meta.com/results/?content_types[0]=publication`) fills its list with
JavaScript, so a plain fetch returns only navigation; no item could be read from it. The papers below come from web
searches. Status column: **Read** = the abstract page was fetched and summarised from the page; **Snippet** = known only
from a search result or a citation in another paper, not opened. Every number is the authors' own claim, not reproduced
here. Nothing in this note is a measured result of ours.

## 1. Papers that change what we do next

| Paper | Status | Claim (authors') | Use for us |
| :--- | :--- | :--- | :--- |
| Lost in the bf16 Cast: exporting ternary language models (arXiv 2610.07853, Oct 2026) | Read | Exporting ternary models through bf16 flips 0.83 to 1.77% of ternary codes in released checkpoints; fine-tuned GSM8K accuracy falls from 58.79% to 0.78% in one model; two remedies are given | **Direct test for S0 and the branches.** Our GPU evaluate path uses f16 weights and our exports are dequantised floats. Add a check that counts ternary codes that change after the f16 cast, and keep reported numbers on the CPU F32 path |
| CAT-Q: cost-efficient and accurate ternary quantization (arXiv 2606.26650, ICML 2026 oral) | Read | Post-training ternary with 512 calibration samples (learnable modulation and softened ternarization) beats BitNet 1.58-bit v1 and v2 trained on 100B tokens; quantizes 14B to 235B models in 8 to 60 hours on 8 A100 | A no-training route to compare against S0's plain ternary rule and our branch training. Code is said to be at `github.com/IntelChina-AI/BitTern`; licence not checked |
| Agreement rather than normalisation keeps depth reusable in looped language models (arXiv 2610.31943, under review) | Read | Looped models fail when the loop count at inference differs from training; agreement of the two counts matters, not the loop-boundary normalisation; the advantage of looping shrinks at matched overshoot | **Arm D rule:** evaluate D only at the trained loop count first, and treat other counts (the exit-head scenario) as a separate test; log the loop count in every report |
| Recurrent Looped Transformer (arXiv 2610.07591) | Read | A recurrent decoder over an encoder generalises parity and permutation tasks to much longer inputs than a plain Transformer; algorithmic tasks only | Mechanism idea only; a language-modelling result at small scale is not in the abstract (a search snippet says it did not beat a matched Transformer at 140M; not verified) |
| Augment the cache, replace the activation: a 2x2 study of internal-state sharing (arXiv 2610.22026) | Read | Between frozen models, augment the receiver's KV cache (cross-attention plus a small query adapter) rather than swap it; write activations in directly, no bridge | Input for the parked "models talk without text" idea: if it is picked up, start with cache augmentation and direct activation writes before any learned binary code |

## 2. Relevant, not opened (Snippet)

- **Self-training and self-improvement.** Meta-Rewarding Language Models (arXiv 2407.19594; one model as actor, judge and
  meta-judge), Self-Taught Evaluator (synthetic preference data), Language Self-Play (reported by a news site), Dr. Zero
  (proposer and solver, no labelled data; secondary source) and Meta FAIR's Autodata (June 2026; a blog summary says an agent
  generates, inspects and scores training data and a 4B model trained on it beats a roughly 400B baseline on legal reasoning).
  No primary page was found for Autodata or Language Self-Play. Use: design of stage 2 (lessons) as a loop where the teacher
  writes, the student is tested, and the weak topics get new lessons. Still test-only and local for hosted-teacher text.
- **Ternary and low-bit.** Tequila (arXiv 2509.23809, QAT), QTEA (2609.00224, sparse residual salient weights), RoleQ
  (2610.21808, role-aligned scales, relates to the paper's role tags), TWLA (2606.13054), MiniCPM4 (2506.07900) and Continual
  QAT (2502.11895: switch from 16-bit to 1.58-bit part-way; MiniCPM4's authors report that about 40% of tokens spent on QAT
  approaches training ternary from scratch). The last two support the advisor's branch route over quantise-from-step-0.
- **Looped models.** Ouro (arXiv 2510.25741) and its follow-ups: Gated Recurrent Transformers (2608.15062), Attractor
  Models (2605.12466), Inside the Loop (2610.24135), Right Direction, Wrong Step (2609.16665), Looped World Models
  (2606.18208).
- **Adaptive depth and self-speculative decoding.** KnapSpec (arXiv 2602.20217: layer selection for the draft model posed as
  a knapsack, training-free; close in form to our layer allocator), SpecBound (ACL 2026 Findings: confidence-calibrated
  early exit with adaptive speculation length), River-LLM (2604.18396), plus LayerSkip, DEL, SWIFT, DSSD, D3. LayerSkip's
  code is CC-BY-NC and must not be copied.
- **Latent communication between models.** Cache-to-Cache (ICLR 2026), Latent Cache Flow (2605.22863), CacheFuse
  (2610.16165, training-free), StateBridge (2608.13317), Swarm Telepathy (2610.15095), a survey (2606.05711), and safety
  papers LCGuard (2605.22786) and When Latent Agents Lie (2606.28958).
- **Small-model data.** Synthetic pretraining for very small models (blog, April 2026), SYNTH fully synthetic single-stage
  recipe (2609.37891), curriculum-guided layer scaling (2506.11389), active synthetic data generation (2512.00884).

## 3. What we propose to take (each is a scenario, registered before it runs)

1. **f16-cast code flips** (paper 1). Cheap and decisive: count flipped ternary and binary codes between the F32 export and
   the f16 path for every S0 and branch checkpoint, and record both numbers. No training.
2. **Post-training ternary as a control.** Implement a CAT-Q-style softened ternarization from the abstract's description,
   or run the released code if its licence allows (check before copying), and put it next to S0's plain ternary rule at the
   same bits per weight.
3. **Loop count discipline for D** (paper 3). D1 is trained and measured at one loop count; later scenarios train with the
   count the exit heads will use. Reports always state the loop count.
4. **Stage 2 as a self-training loop**, after D1. The teacher writes lessons, the student is measured per topic on held-out
   questions, and new lessons target the weak topics. Test-only, local, as already decided.
5. **Layer allocation as a knapsack** (KnapSpec). Our allocator already is one; compare formulations once S0 per-layer
   costs exist for arm B too.
6. **Keep the no-text idea parked**, with the 2x2 study as the first reading when it is picked up.

## 4. Gaps in this scan

The Meta publications list itself was not read; the date range is whatever the search engine returned; most 2026 entries
are Snippet. Next pass: fetch and read the Snippet papers marked above in order of use (items 2 and 4 first), and ask the
owner for the exact Meta paper links to read if specific ones are wanted.

## 5. Dr. Zero read in full (owner-supplied, arXiv 2601.07055)

**Status: Read** (abstract page and HTML version; a summary of part of the body is marked unverified in the notes below).
Dr. Zero: Self-Evolving Search Agents without Training Data. Yue, Upasani, Yang, Ge, Nie, Mao, Liu, Wang; work done at Meta
Superintelligence Labs and UIUC; submitted 11 January 2026, v2 21 July 2026; COLM 2026. Code: `github.com/facebookresearch/drzero`
(licence not stated on the pages read).

**Method (authors').** A proposer and a solver start from the same base model. The proposer writes questions from seed
Wikipedia documents; the solver answers them with a search tool. The proposer is rewarded for questions of middling
difficulty: with n solver attempts and k correct, reward (n-k)/(n-1) when 0<k<n, so questions the solver always or never
gets right earn nothing. The proposer trains with HRPO (advantage normalised within groups of equal hop count, strictly
on-policy; 6 rollouts per question against 20 for their GRPO setting, and the global baseline was unstable). The solver
trains with GRPO on the proposed questions with an exact-match reward. They alternate for 3 iterations. There is no teacher
model and no distillation in the paper.

**Results (authors', exact match, average over seven QA sets).** 3B: 0.326 against supervised Search-R1 at 0.327. 7B: 0.372
against 0.384. Data-free baselines at 3B: SQLM 0.233, R-Zero 0.256. The 7B model stops improving after iteration 2 (0.366,
0.372, 0.360). Only Qwen2.5-3B and 7B were used. The abstract's "matches or surpasses" is level at 3B and slightly lower at
7B.

**Limits stated by the authors.** Short-form verifiable QA only; the reward does not remove wrong or ambiguous synthetic
answers; the 7B run shows instability. Quantisation, ternary or binary weights, shared looped layers and small models are
not addressed at all.

**How it could pair with a teacher (our reading, not the paper's; untested).** The pass-rate reward needs only the
student's own attempts, so it can pick which of the teacher's lessons to keep. A teacher could write the questions
(proposer role) while the student's measured pass rate selects the ones worth training on; the student's loss could add a
distillation term on the teacher's answers. For a looped student, grouping the baseline by loop count mirrors HRPO's
grouping by hop count. This is the shape of stage 2 in the testbed plan, now with a selection rule. A 14M model cannot
search or reason in the paper's sense, so the first version would be on next-token tasks (cloze and short completion
questions with an exact answer), not search. Registered scenario before any run; test-only and local for hosted-teacher text.
