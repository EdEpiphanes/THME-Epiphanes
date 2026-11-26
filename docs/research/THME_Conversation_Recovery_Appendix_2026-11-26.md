# THME Conversation Recovery Appendix (26 November 2026)

Research content recovered from the Antigravity conversation `6c6bc3e7-defd-4508-913f-bd9a6e27c760` (transcript of 3,285 steps, 25 to 26 November 2026) that was not yet in the repository. Every item below was checked against the transcript; step numbers refer to `transcript_full.jsonl` (`step_index`). Nothing here is a new result.

Evidence labels follow AGENTS.md Rule 2: every figure in this file is **reported** by the run owner, not reproduced here.

## 1. The 140-round duel: what exists

The transcript holds no per-round text for the 140-round adversarial duel. The only round-level content is the grouped table (rounds 1 to 10, 11 to 21, and so on up to 131 to 140) in the final message at step 3284, which matches dossier section 8.2. Earlier drafts (10-, 21- and 100-round tables, steps 1939, 1967 and 1986) are also grouped. Single-round reasoning appears only for rounds 21, 37, 39 and 98 (steps 1993, 2109, 2118, 2297) as rationale. The dossier table is therefore the complete committed record of the duel.

## 2. Superseded draft: coupled Lagrangian-DP knapsack

An early draft of the allocator (step 1888, an edit to the v3.2 working text) stated the allocation as a 0/1 knapsack over groups `g` and formats `f`:

> Solve min over x(g,f) in {0,1} of the sum over g,f of c(g,f)^(t) x(g,f), subject to the sum over f of x(g,f) = 1 for every group and the sum over g,f of s(g,f) x(g,f) <= B_max, updating Gamma(l,m) from measured `thme evaluate` residuals.

Status: **superseded**. Dossier section 4.2 replaces it with the fusion-aware Signed-Rank-R DC-CCCP plus ADMM-DP solver (duel rounds 11 to 21 and 106 to 115). The dossier keeps only "Additive Knapsack DP" as a rejected baseline. This text is kept so the earlier formulation is not lost.

## 3. Decisions missing from the negative-results log

| Step | Date (UTC) | Decision |
| ---: | :--- | :--- |
| 1758 | 2026-11-26 04:46 | Translation layers rejected for THME: ZLUDA (fragile Windows maintenance), HIP/ROCm (multi-GB SDK, weak Windows consumer support), SYCL/oneAPI (C++/DPC++ dependency). CubeCL is pure Rust with Tensor Core support, but its JIT and codegen dependencies risk the 16 MiB CLI size cap. Preferred path: `ash` with `VK_KHR_cooperative_matrix` (16x16x16 f16 to f32) beside native `cuda.rs` and the `wgpu` fallback. |
| 998 | 2026-11-25 16:25 | All 27 external items (agent room #1 to #5, `thme-tools/inbox/`, `thme-tools/tune/patches/`) were audited and rejected for `perf/gpu-throughput`: none carried a GPU throughput patch. |

Three further abandonment claims were reported by the recovery scout (steps 348/352 reverting `conversion.rs`, 1451 abandoning the native Tensor Core path, 1855 dropping parametric models) but the step text could not be confirmed, so they are not recorded here.

## 4. Open questions

- **H15 (which llama.cpp parts to port, and how)**: unanswered in the conversation. Answered by the advisor on 26 November in the agent room (message #7): default to the clean-room route of AGENTS.md Rule 5 for RMSNorm, RoPE, GQA, online softmax and the K-quant decoders; translating MIT code stays unapproved; the copyleft, custom-licence and unclear-copyright list stays banned. Any licence decision beyond that default is the owner's, with counsel.
- **Plan scope and timeout plan (step 135)**: the three questions about splitting `run_referee.py` per format, bundling `EXP-0002`, and live progress were followed at step 136 by the owner approving the plan document.
- Other "pending owner approval" lines seen by the scout (steps 1213, 1215, 1777, 1803) are text inside private research notes that Antigravity read, not questions put to the owner.

## 5. Scratch files not committed

`dossier_v3.3.html` and the two earlier HTML renders are renders of the dossier; `render_dossier.py` and `render_pdf.py` are tooling; `eval_smollm2/` (zip and token file) and `edge_profile/` (a browser profile) are binaries or caches. None is research content.
