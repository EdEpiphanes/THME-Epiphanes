# Review: THME Research Dossier v3.1

**Date:** 13 November 2026
**Reviews:** [`THME_Research_Dossier_v3.1_2026-11-13.pdf`](THME_Research_Dossier_v3.1_2026-11-13.pdf) (147 pages)
**Repository state:** `main` after #47 (commit `8d8c8f5`), about 32,500 lines of Rust
**Evidence:** every repository claim below was checked against the code at that commit. Nothing
was trained, benchmarked or run on a language model for this review.

## 1. Status

The dossier is a theoretical proposal and engineering brief. By its own account it reports no
training run and no hardware measurement; its only measured numbers are the repository's own
perplexity results (Appendix J). Every efficiency figure in it is an **estimate**, and its toy
simulations are **simulated**. Nothing in it changes a committed measured number.

Its claims carry status letters (M measured, T tested, S supported, U unverified, C contradicted).
Treat T, S and U as things to re-derive in the repository, not as facts. Method names and
citations were written from memory and are **unverified** until checked against the sources
(ticket THME-3000).

## 2. Verdict

The dossier is a sound plan for the next stage. Its strongest parts are the correction register,
the stage-gated study (Appendix H), the engine work packages with oracle tests (Appendix M) and
the benchmarking rules (Appendix R). Its weakest parts are the ones it flags itself: the Clifford,
stabilizer, torus and attractor components have no supporting evidence and come last.

It fits the current direction: **foundations first, tools and engine, no further model testing**
(B.1). Of its prerequisites (B.6), steps 1 and 3 are done in this change, and step 2 is proposed
as tickets for the owner to place.

## 3. Discrepancies the dossier asked to resolve first (B.5)

| Item | Dossier says | Repository shows | Done here |
| :--- | :--- | :--- | :--- |
| Gate tolerance | 0.25 in the tolerances file and tests, 0.45 in reports; default untraced | **Confirmed.** `thme evaluate`, `search` and `validate` default to `--max-ppl-delta 0.45`. `configs/quality_gates.yaml` (0.25) is read by no command. `quality::evaluator` tests use 0.25. | The YAML is marked unread and stale; the rule in force is stated. Versioned gate: THME-3003. |
| Allocator | Docs describe a DP knapsack; code uses thresholds | **Confirmed.** `src/allocator/pareto.rs` compares sensitivity with 0.75/0.45 and 0.45/0.25 thresholds. `thme search` is greedy plus refinement. No exact solver exists. | `docs/subsystems.md` and the AGENTS.md ledger corrected. Exact solver: THME-3002. |
| Table J7 row H1 | Per-tensor export not written | Out of date: per-tensor GGUF export shipped in v0.2.2.1. | Nothing to change in the repo; the PDF row is stale. |
| Licence statements | Some documents still say Apache-2.0 | Only `LICENSING.md` and `CHANGELOG.md` mention Apache-2.0, both correctly, as the licence of history up to `56ae520`. The Engineering Playbook is outside the repo. | No change. |
| Nominal bits | 13.06 for Q8_0 does not match about 8.5 stored | The column averages over **every** parameter, counting unchanged tensors (FP32 embeddings) at 32 bits. It was never defined. | The dossier generator now defines it; no number changed. |
| Size baseline | 513 MiB is FP32; about 47% smaller than FP16, not 73% | The README said "513 MiB source" without the precision. No "73%" appears in the repo now. | README now says FP32 and gives the FP16 comparison. |
| Register | Stops at V-32; later IDs scattered | Confirmed in the PDF. | THME-3000. |
| Accuracy gates | MMLU and GSM8K listed, never measured | Confirmed: no accuracy harness exists. | Marked "not yet measured" in the YAML. |

## 4. Where the dossier is behind the repository

The dossier read commit `c83a236`, plus a later check that found the AWQ commit. Since then:

- **No-panic rule is now enforced by the compiler** (#46): the library and binaries deny `unwrap`,
  `expect`, `panic!`, `unreachable!`, `unimplemented!` and `todo!` outside tests.
- **`src/gui/app.rs` is split** into one module per Studio tab (#47, THME-2003). The GUI is still
  about 8,800 lines in total, as the dossier says, but no longer in one file.
- **THME-2006 is closed**: ruleset, required checks and signing key are configured.
- **The Hadamard problem is worse than Finding 2 says.** There are already **two** transforms:
  `transform::MatrixMath::fast_walsh_hadamard_transform` (whole slice, `Result<(), String>`) and
  `transform::byoa::wht_blocks` (blockwise, used by rotated quantization). THME-3004.
- **SPDX headers do not exist.** B.2 lists them as a repository rule; no file has one. THME-3005.
- **The prototype crate (B.4, M.3) is not in the repository** and was not supplied with the PDF.
  WP1 below was written fresh rather than copied from it.

## 5. Architecture review

Terms follow the codebase-design vocabulary: a **module** is deep when much behaviour sits
behind a small **interface**; a **seam** is where that interface lives; one adapter is a
hypothetical seam, two adapters a real one.

1. **The engine is a new deep module, and its first seam is the error type.** `src/engine/` now
   exists with `EngineError` and `ensure_finite` only (WP1, THME-3001). Engine primitives return
   `EngineResult`; callers in the rest of the crate convert to `anyhow` at the seam, so the engine
   never depends on the CLI's error style.

2. **The Hadamard transform should be one module, not two.** Deletion test: delete either copy
   and nothing of value is lost, but the drift risk goes away. `engine::basis` should own a
   blockwise transform, and both callers should use it (THME-3004). The dossier's rule
   "share code between conditions that must stay identical" applies directly: a rotated-quant
   result compared against another path must use the same rotation.

3. **The allocator seam is hypothetical today and becomes real with WP2.** `ParetoAllocator` is
   one adapter with no alternative behind it. Adding `engine::alloc` (exact DP) as a second
   solver behind `thme search --solver` makes the seam real, and its oracle (exhaustive search on
   small instances) tests both adapters through the same interface (THME-3002).

4. **The quality gate leaks across three places.** A threshold in CLI defaults, an unread YAML
   and unit tests with a different value is a shallow arrangement: every caller must know which
   copy is live. One versioned gate type, recorded in each report, gives locality: change it
   once, and every report says which version judged it (THME-3003).

5. **BYOA has a side channel that will grow.** Adding AWQ (#45) showed the cost of the closed
   rule enum the dossier describes in L.3. The rule needed calibration data, which reached it as
   a JSON string in tensor metadata plus a special case in `inference::weights`. Each future
   data-needing rule (GPTQ needs the input covariance) would add another case. The rule
   interface should take an apply context (calibration statistics, row width) at one seam
   (THME-3010). This is a smaller step than the out-of-process plugin protocol of L.4, which
   waits for the licensing decision.

6. **Per-architecture duplication is the case for WP14.** `LlamaModel::smooth` and
   `NeoxModel::smooth`, like the two forward passes, each re-encode which tensors read which
   norm. An architecture-neutral operation graph (WP14) would let migration, calibration and
   symmetry transforms be written once. It is the largest item here and should follow the
   engine foundations.

## 6. Proposed tickets

Added to BACKLOG.md as `triage` with milestone `unplaced`; existing milestones are not renumbered.

| Ticket | Work package | Status |
| :--- | :--- | :--- |
| THME-3000 | Consolidated claim register, citations checked | triage |
| THME-3001 | WP1 engine error module | **done in this change** |
| THME-3002 | WP2 exact DP allocator | triage |
| THME-3003 | WP20 versioned quality gate | triage |
| THME-3004 | WP5 shared basis (one Hadamard) | triage |
| THME-3005 | SPDX headers | triage |
| THME-3006 | WP3 number formats with error bounds | triage |
| THME-3007 | WP9 SVG visual engine | triage |
| THME-3008 | WP19 permission record | triage |
| THME-3009 | Versioned schemas (B.6.4) | triage |
| THME-3010 | BYOA apply context instead of metadata side channel | triage |

Later tiers (benchmarks WP31-38, translation layer WP14-18, routing WP21-25, distillation
WP26-30, advanced maths WP7-13) are not ticketed yet; they depend on the tier-1 decisions below.

## 7. Decisions for the owner (B.10)

The dossier says to ask, not decide:

1. Where the engine milestones sit in ROADMAP.md, and whether the engine is a library only or
   also a `thme engine` subcommand.
2. Geometry view: default dimension range and embedding; Clifford signature for WP11.
3. Translation layer: which architectures first; whether to load PyTorch pickles and ONNX.
4. Needles: which decision points first; how the permission statement is collected.
5. Distillation: teacher tokenizer and serving, student size, compute, autonomy budget caps.
6. Benchmarks: which families first, which tool to bridge, dataset licences, composite score.
7. Legal: the open-core line, plugin and layer-linking terms, reverse-engineering terms.

## 8. Open items outside the code

- **Licence wording differs.** The dossier's legal notice says commercial use "is not available
  at this time"; `LICENSING.md` offers a commercial licence. One of them should change, and the
  dossier itself asks for a lawyer's review first.
- **Brand freeze.** B.1 puts name, brand and visual design on hold. Brand work stops until the
  owner says otherwise.
- **Publication.** Committing this PDF makes it public and permanent in git history, including
  the quoted outside critique (Appendix J.6). L.2 warns to decide what is proprietary before
  committing; the owner chose to publish it.

## 9. Not done in this review

- The 135-claim register was not consolidated (THME-3000).
- No citation was checked against its source.
- No dossier simulation was re-run, and no model was run.
- The prototype crate was not available to compare against.
