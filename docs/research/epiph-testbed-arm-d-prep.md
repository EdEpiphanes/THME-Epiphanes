<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Arm D preparation: the envisioned Epiphanes architecture

28 November 2026. Ticket: THME-3110. Status: preparation only; nothing trained. Owner decision: finish arms B and C
first, then build arm D (plan D). Arms A, B, C are plain GPT-NeoX with 6 distinct layers, so none of them has the
architecture the paper describes. This note lists what D must contain, what the engine already supports, and what is
missing, so that D is built in one pass after B and C.

## What the paper asks for

| Item | Paper | Meaning for D |
| :--- | :--- | :--- |
| Looped shared-weight layers | 7.10 item 3, Fig. 2 | A few unique layers run k times |
| Dory last-loop KV rule | 7 and Fig. 2 (dM_KV/dk = 0) | KV cache comes from the last loop only, so cache size does not grow with k |
| Mandatory distillation | 7.10, B13 | Untrained looping raised SmolLM-135M perplexity 18.0 to 55.3; D must be trained or distilled, never just re-wired |
| Trained exit heads | 7 and B12 | Untrained probes settled 1.6% of tokens at 50% depth on Pythia-14M; D trains a head per loop |
| RoPE-commutative KV, Hadamard/Sinkhorn preconditioning | 6, 7 | Applied to D after training, as for any model; D only has to expose standard RoPE attention |
| GQA (MHA to GQA conversion) | 7.10 item 1 | Optional second variant; not in the first D |

## Mixed binary, ternary and phase layers (owner requirement, 28 November 2026)

Arm D must carry the full math of the paper (v5, v5.1), not only looping. Section 7.5 puts every layer on a simplex
mixture alpha = (bin, tern, phase), and the layers differ from one another. D therefore has three kinds of weight
slice inside one model, trained so that each regime holds up in use rather than converted afterwards:

| Slice | Paper | Training form for D (to confirm) |
| :--- | :--- | :--- |
| Binary | 7.5.1, 6.1: Sinkhorn S, Kronecker P, LittleBit dual-sign residual SVD, scale s_j | Sign weights with a learned scale and a low-rank residual; straight-through gradient |
| Ternary (BITCOS) | 7.5.1: dead zone 0.7 E[abs w], 1,024-weight scaling, post-SiLU moment match on W_gate | Ternary weights {-1,0,+1} with the dead-zone rule, scale per 1,024 weights |
| Phase / polar | 7.5.1, eq. 29: pairs (r, theta), circular loss 1 - cos(dtheta), radial-energy match | Weights stored as magnitude and angle pairs; cosine-table decode as in the engine design |

What D must also contain from the same section: row-block partition in multiples of 1,024 weights, row-coherent
gate/up pairing, boundary layers (embedding, layer 0, last layer, head) pinned to at least Q8_0 or E8P, per-layer
alpha chosen by the solver (eq. 31), never typed in, the dual-activation staging, and the three sub-gates
(tau_bin, tau_tern, tau_phase), whose defaults the paper lists as open (section 12). Finding those defaults is what D
is for.

Consequence for the plan: D is two things layered. **D-shape** (looped, exit heads) and **D-regimes** (the three
slices). They are separable, so D is built in stages: D1 looped only, D2 adds ternary, D3 adds binary, D4 adds phase,
D5 the full solver-chosen mixture. Each stage is compared with the one before, so any loss is tied to one cause.

Open points to settle by reading sections 6, 7.5, 7.6 and 12 line by line:
- The bits-per-weight of the binary format is an open parameter (Table 16).
- Whether training should be quantisation-aware from the first step or start in full precision and switch on each
  regime (the paper's BitDistill route); this changes the learning-rate and warm-up settings.
- Which of the three slices the engine can already run end to end. BITCOS has code and a GPU kernel
  (`src/epiph/bitcos.rs`, `gemv_bitcos.wgsl`) and there is a regime router (`src/convert/regime_router.rs`); the
  phase format and binary-with-residual path were not checked and are not assumed to exist.

## What the engine already has

- `forward_scheduled(tokens, layer_schedule, exits)` for the GPT-NeoX path (`src/inference/neox.rs:278`): runs a
  chosen sequence of layers (repeats allowed) and reads logits at chosen depths. The test at `src/inference/llama.rs`
  (about line 1380) shows a repeated layer changes the result. This is the looped forward pass for the CPU path.
- Conversion planning types `DoryLooped` and `DeepToDoryLoop` with the BitDistill-required flag
  (`src/convert/arch_graph.rs`).
- `thme epiph pack/verify`, `thme evaluate`, and the checkpoint-series script: unchanged for D if D is exported as a
  GPT-NeoX folder.

## What is missing

1. **Trained looped model.** `scripts/train/pretrain.py` has arms scratch, pythia, byte only. D needs a `loop` arm:
   U unique layers applied k times in PyTorch with tied weights.
2. **Export of tied weights.** Simplest, no engine change: write the looped model as a 6-layer GPT-NeoX folder with
   the unique layers copied into every repeat. It then runs in the engine unchanged and every existing measurement
   applies. The 2-unique-layer file form needs `.epiph` to store a layer schedule; check whether the container allows
   it before choosing.
3. **Exit heads.** The engine reads intermediate depths through the shared final norm and output matrix. Trained
   per-loop heads (small adapters before the shared unembedding) need a place in the checkpoint and in
   `forward_scheduled`. Decide: train heads as separate tensors and add an engine path, or first test the shared
   head only and add trained heads in a second step.
4. **Dory KV rule.** The CPU forward pass has no looped KV cache. For perplexity (the gate) it is not needed; it is
   needed only for decode memory tests. Defer, record as a later row.
5. **Distillation.** D-scratch (trained on the data) and D-distilled (from arm A as teacher, KL on logits plus the
   data loss). Needs a distillation step in `pretrain.py`; `src/trainer` has a Rust path but the testbed is Python.
6. **GPU path.** Not checked whether the GPU evaluate path honours a layer schedule. If not, D is measured on the
   F32 CPU path (slower but exact); the series script must pass `--device cpu` and record that.

## Proposed registered shape (to confirm before training)

- Width 128, 4 heads, 2 unique layers, k = 3 loops, so compute depth 6 like arms A and B. About 5M parameters.
- Same data, steps, batch, learning rate, seed and checkpoints as arm A, so D is comparable to A.
- Exit readout after each loop, with the shared head first, trained heads second.
- Variants: D1 scratch; D2 distilled from arm A at step 12,000; later k = 2 and k = 4.

## Questions the paper leaves open that D should answer

- Does a per-block drift estimate stay valid when the model reuses its own layers (blocks are less independent)?
- How far does looping cost perplexity against A at equal compute, and does distillation close the gap?
- How much of the budget does each exit settle, with and without trained heads (replaces the 1.6% untrained figure)?
- Is the allocator's layer-sensitivity assumption (additive across layers) worse for tied layers? Tied layers
  share weights, so one quantisation choice hits all repeats.

## Order of work

1. Finish B and C, measure them (GPU one job at a time).
2. Read the engine for the GPU schedule question (item 6) and the `.epiph` schedule question (item 2).
3. Add the `loop` arm and its tests (tied weights, same seed same loss, exported folder reloads and matches the
   in-memory logits), then the distillation option.
4. Register D in `epiph-testbed-plan.md`, train D1, measure with the series script, then D2.

## Owner decisions, 28 November 2026

Take the advisor's recommendations on all six open items, and try every alternative as its own scenario: both shapes
(prelude + one shared layer x4 + coda, and 2 unique x3 without boundary pinning), branch quantisation from full
precision and quantisation from step 0, several phase formats, the binary rank sweep. Main goal: beat other inference
engines, so favour what helps speed and accuracy per bit.

**Standing R&D rule.** For anything out of scope: (1) test without building (fake-quantise in PyTorch, export
dequantised weights, use existing engine paths); (2) copy, bridge or translate open-source code (check the licence;
the repo is AGPL-3.0-only); (3) build only last. The advisor runs feasibility checks up front.

## Advisor findings (Opus, read-only; V = checked in files, J = judgement, U = not verified)

- Boundary pinning (paper 7.6, `src/convert/regime_router.rs:80`) removes every regime from a 2x3 loop, hence the
  prelude + shared x4 + coda shape. GPT-NeoX accepts only GELU (`neox.rs:87`), so SwiGLU terms are untestable on NeoX.
- **SwiGLU needs no engine work (V):** a Llama-shaped D runs through `llama.rs` and `thme evaluate` unchanged
  (tensor names and config fields in the advisor's report). Cost: a Llama-shaped full-precision, non-looped control
  run (about 1.5 h), because arm A stops being a like-for-like baseline.
- **Layer schedule (V):** the GPU path takes a schedule internally; `thme evaluate` has no schedule flag; `.epiph`
  stores no schedule. A 6-layer folder with the shared layer copied into each repeat is enough for quality numbers;
  file sizes must be computed, not read.
- **No paper gate in the engine (V):** `thme evaluate` uses the simple maximum-perplexity gate. Smallest fix (J,
  about 10 lines): add a per-window NLL list to `PerplexityReport` (`#[serde(default)]`), emitted in the variant
  JSON; block grouping stays in a script.
- **Solver (V):** `solve_dynamic_regime_mix` lacks the transition term of eq. 31 and nothing calls it. First try a
  Python solver on fake-quantise distortions.
- **Phase candidates (J):** P1 3-bit angle + block radius (1.52 bpw); P2 3-bit angle + 1-bit radius (2.03); P3 4-bit
  angle + block radius (2.02); P4 4-bit angle + 2-bit radius (3.03); control: plain 2-bit scalar at the same bpw. If
  phase does not beat the control it has no case. `quantize_phase` (`numfmt.rs:110`) rounds the angle only, no radius.
- **Training route (J):** full precision first, then each regime as a short branch (about 3,000 steps, lr 3e-4 to
  5e-4, 100 warm-up, blend over about 500 steps, weight decay 0 on quantised latent weights), plus a full-precision
  continuation control. Quantise-from-step-0 stays as a comparison scenario.
- **Binary (J):** rank sweep r in {0, 8, 32}; scale overhead dominates at width 128; report where PPL against bpw
  bends. Do not carry a single bpw number to large models.
- **Open source (licences U unless noted):** BitNet BitLinear (transformers, Apache-2.0 / MIT), larq STE
  (Apache-2.0), quip-sharp E8 (GPL-3.0; compatible with AGPL via GPLv3 s13), llama.cpp IQ2 grids (MIT), recurrent
  depth (arXiv 2502.05171), Universal Transformer (Apache-2.0), CALM early exit (Apache-2.0). Skip LayerSkip
  (CC-BY-NC). Verify each licence before copying code.
- **Scenarios S0 to S6** (ranked by information per GPU hour): S0 fake-quantise arm A after training, no training
  (about 0.1 h); S1 D1 full precision (1.5 h); S2 ternary branch + control (0.8 h); S3 binary rank sweep (1.2 h);
  S4 frozen-trunk exit heads (0.2 h); S5 ternary from step 0 (1.9 h); S6 solver mixture vs best single regime.
- **Cannot claim from 14M (J):** embeddings and head are about 92% of parameters, so no memory or bpw saving claim;
  no transfer to large models; tau values apply to this model only; the gate will mostly fail at this tolerance.

### Pre-flight checklist (before any D run)

1. Same seed twice for 50 steps gives identical loss. 2. Llama smoke run: pack, verify, logits within 0.005 of
Hugging Face. 3. Tied export reloads with logits equal to the in-memory model. 4. Fake-quantise is idempotent;
report final numbers on the CPU F32 path (GPU casts to f16). 5. Extra tensors (exit heads) are ignored or stripped on
export (U). 6. Calibration windows from WikiText-103 validation, hash-disjoint from WikiText-2 test. 7. Hash every
checkpoint; the report records seed, commit, config and data hashes. 8. Time 50 steps with fake-quantise on. 9. Stop
on NaN or loss spike, keep the last good checkpoint. 10. Use a no-op recipe for `thme evaluate`.

### Order of work (revised)

1. Per-window NLL field. 2. S0 on arm A. 3. Llama full-precision control. 4. D1 in both shapes. 5. Ternary and binary
branches, plus quantise-from-step-0. 6. Phase branches P1 to P4. 7. Frozen-trunk exit heads. 8. Python solver for D5.
Figures (next section) are drawn alongside, not after.

## Later idea (owner, 28 November 2026, not planned in detail)

Assign the three regimes (binary, ternary, phase) movement rules, like pieces in 4D chess: rules for how each kind of
slice may move between layers or positions. Held back until the first steps (S0, D1, branches) are verified. Nothing
is specified yet; when it is picked up, write the rules down first and say what each one would change in the
mixture alpha or in the transition penalty of eq. 31, so it can be tested as a scenario like any other.

## Figures: the model and the diagrams (owner requirement, 28 November 2026)

D, and the way it differs from plain Pythia, is shown with the same isometric `anatomy` figures used for the other
THME models (`src/report/figure.rs` is the existing generator; check it and the README figures before drawing).
Numbers shown come from a real model or a committed report; otherwise they are labelled illustrative.

| Figure | Shows | Data source |
| :--- | :--- | :--- |
| D-1 Shape | Prelude, shared layer looped k times, coda; the last-loop KV rule; exit heads per loop | Registered D config |
| D-2 Regimes | Per-layer binary / ternary / phase mixture on the simplex, boundary layers pinned | S0 and solver output (real) |
| D-3 Slices | One weight matrix split into the three slices with their decoders | Fake-quantise recipes |
| D-4 Drift | Per-block drift and the gate verdict across checkpoints | Series JSON files |
| D-5 Comparison | A, B, C and D checkpoint curves side by side | `docs/benchmarks/epiph-testbed/data/` |

Order: D-1 and D-3 once the shape and phase formats are registered; D-2, D-4, D-5 once measured. Pass through
`/impeccable polish` on the page around them, then `code-review` of any generator code.
