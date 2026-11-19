# Studio heuristic values: audit and labelling plan

Status: THME-3035, step A (docs only). Checked against the code on this branch.
Source claims: `docs/research/THME_AUTHENTICITY_AUDIT_AND_ROADMAP.md`, section 3.

Vocabulary (CONTEXT.md): **Estimate** = a figure from a stated formula, never from a run.
**Simulated** = a figure from a toy or synthetic simulation. No model was loaded or run.

Verdicts: `confirmed` (the claim still holds at the cited place; lines may have moved),
`stale` (the cited code has changed and the claim no longer describes it), `not found` (the cited
code does not exist).

## 1. Audit claims (section 3)

| ID | Where (file:line) | Value shown | Formula in code | Audit claim | Verdict | Label | In slice 1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A | `src/profiling/token_flow.rs:296-327` | Generated words in the Live Token Flow tab | With a caller vocabulary: index = `(seed + 19 * generated) % vocab.len()`. Without one: prompt words cycled by `step % prompt_len`. | A fixed 20-word array indexed by modulo | stale: no 20-word array exists. The vocabulary comes from the caller (`with_vocabulary`, `src/gui/app/benchmarks.rs`). Modulo cycling remains in the fallback path. | Simulated | no |
| B | `src/gui/app.rs:1137-1200` (`auto_scan_and_benchmark`) | Baseline tok/s, TTFT, KV cache, energy | The `42.0 + 1.8L ms` and `28.0 + 1.1L ms` forms are gone. Now a micro-benchmark timing gives `base_ns_per_param`; `est_hw_bw_gbps` (l.1169), `ms_per_tok` (l.1170), `ttft_ms` (l.1171), `tokens_per_sec` (l.1173), `kv_cache_mb` (l.1175), `energy_mj` (l.1177). | Linear telemetry formulas in `app.rs` | stale: the linear form is not in the code. The values are still formulas, not measurements: they depend on a timer over sample weights, then fixed scale factors. | Estimate | yes (group `baseline-*`) |
| C | `src/trainer/mod.rs:147-200` | Micro-trainer loss and weights | Forward input uses `sin`/`cos` positional features of token ids (l.163). Gradients are analytic: `dW = dLogits * X^T` (l.249-254), then AdamW-style moments (l.276-280). | Sine wave used as a surrogate gradient | stale: the sine is an input feature, not the gradient. The gradient is analytic. | none (backend, not in slice 1) | no |
| D | `src/speculative/engine.rs:166-186` | Acceptance rate, speedup | Accept test uses `u = ((step+1)*73 + (token_pos+1)*31) % 100 / 100` (l.171, 179, 186) against the acceptance ratio. | Deterministic pseudo-random hash instead of joint forward verification | confirmed (lines moved from 160-175). The draw is still a hash, not a verification. | Simulated (backend, not in slice 1) | no |
| E1 | `src/gui/app.rs:177` (in `create_synthetic`, l.147) | `benchmarked_latency_ms` of the demo model | constant `2.8` | GUI demo constant 2.8 | confirmed | Estimate ("Est. Latency" tooltip exists, `src/gui/theme.rs:542`) | no |
| E2 | `src/gui/app.rs:392` (in `from_model_ir`, l.361) | `benchmarked_latency_ms` of a loaded model | constant `3.2` | GUI demo constant 3.2 | confirmed | none beyond the theme tooltip | no |

## 2. Other Studio values from a formula or constant

| ID | Where (file:line) | Value shown | Formula in code | Audit claim | Verdict | Label today | In slice 1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| F1 | `src/gui/app.rs:1291-1298` (`step_optimization_loop`) | Tuner candidate latency and distortion (log lines) | `cand_latency = 3.2 * (bytes_pw / 0.5) * (0.75 or 0.95)`; `cand_distortion = 0.04a + 0.012b + 0.018g` | not in audit | not in audit (new) | none | yes (`tuner-candidate`) |
| F2 | `src/gui/app.rs:1342-1343` | Locked winner latency and distortion (`WINNER LOCKED`) | `opt_lat = 1.65 + 0.4 * depth`; `opt_dist = 0.008 + 0.004 * depth` | not in audit | not in audit (new) | none | yes (`tuner-winner`) |
| F3 | `src/gui/app.rs:1404-1442` (`finalize_auto_tuning`) | Scorecard speedup, storage, KV, energy | `base_tok` fallback 22.0; `base_store = total_stored * 2.3`; `base_kv` fallback 1.2; `base_e` fallback 36.0; `speedup = 1/(f_bar*0.80 + 0.15)`; `opt_kv = base_kv * 0.30`; `opt_e = base_e * 0.28`; `cache_savings_percent: 70.0`; `energy_savings_percent: 72.0` | not in audit | not in audit (new) | none | yes (`scorecard`) |
| F4 | `src/gui/app.rs:472-500` (`compute_costs`) | Stats ribbon: VRAM, traffic/token, depth, energy/token, throughput | `joules = traffic_MB/1024 * 8 * 15`; `throughput = 1000 / (f_bar*38 + 10)` | not in audit | not in audit (new) | Estimate (slice 2; PARAMETERS is a count, left unlabelled) | In slice 1 = yes (slice 2): `ribbon-vram`, `ribbon-traffic`, `ribbon-depth`, `ribbon-energy`, `ribbon-throughput` |
| F5 | `src/gui/app/benchmarks.rs:96-102` | Live simulator stats (tok/s, TTFT, KV, bandwidth) | from the `LiveFlowSimulator` snapshot | Simulated source | inherits A | none | no |
| F6 | `src/gui/app/benchmarks.rs:188-191` | Depth, traffic, energy, est. throughput | same formulas as F4 | not in audit | not in audit | "EST." on one box only | no |
| F7 | `src/gui/app/benchmarks.rs:255-258` | Speculative speedup, acceptance, tokens/pass, traffic saved | from `SpeculativeEngine` telemetry | Simulated (D) | inherits D | none | no |
| F8 | `src/gui/app/benchmarks.rs:444-447`, `:533` | Roofline ceilings, bandwidth, knee, bound split, est. latency | hardware DB constants and roofline formulas (`src/profiling/roofline.rs`, `hardware.rs`) | not in audit | not checked in depth | none | no |
| F9 | `src/gui/app/matrix_engine.rs:320-321` | BYOA compression ratio and hardware speedup | `16.0 / bits` | not in audit | not in audit | none | no |
| F10 | `src/gui/app/dashboard.rs:100` | KV cache (dashboard) | from the `LiveFlowSimulator` snapshot | Simulated source | inherits A | none | no |
| F11 | `src/gui/app/surgery_export.rs:254` | "EST. SPEEDUP" box | `diff.estimated_speedup` | not in audit | not in audit | partial ("EST." in the title) | no |
| F12 | `src/gui/app/co_surgery.rs:25` | Co-Surgery perplexity figures | analytic proxy | not in audit | already labelled | "SIMULATED" | no |
| F13 | `src/gui/app/dev_lab.rs:712-750` | Training live telemetry (tok/s, mJ) | from live training stats | not in audit | not checked in depth | none | no |
| F14 | `src/gui/theme.rs:542` | Layer tooltip "Est. Latency" | reads `benchmarked_latency_ms` (E1, E2) | not in audit | confirmed; the values are demo constants | "Est. Latency" | no |

## 3. Slice 1 (labelled in THME-3035 step C)

1. Baseline benchmark, from `auto_scan_and_benchmark`: `baseline-throughput` (tok/s in the status
   message and the generated text), `baseline-ttft` and `baseline-kv-cache` (in the generated
   text). `baseline-energy` and the per-token time have no screen string today, so this slice
   gives them no label. The per-layer latency vector is not shown either.
2. Auto-tuner log lines: `tuner-candidate` (candidate latency and distortion) and `tuner-winner`
   (the `WINNER LOCKED` line).
3. Every number shown from the `Scorecard`: `scorecard` (speedup and storage in the status
   message, the `[SUCCESS]` line and the checkpoint details). The `Scorecard` fields
   (`tuned_kv_cache_mb`, `cache_savings_percent`, ...) are stored but not displayed, so they are
   not labelled here.

Evidence: baseline values and the scorecard are `Estimate` (formulas). Tuner candidate and winner
are `Simulated` (fixed candidate list and fixed latency model).

## 4. Follow-up tickets (proposed; BACKLOG.md not edited here)

- Label the stats ribbon (F4) and the Benchmarks tab boxes (F6, F8) as Estimate or Simulated.
- Label the speculative stat boxes (F7) and live simulator values (F5, F10) as Simulated.
- Remove the two demo latency constants (E1, E2) or label them as demo values.
- Replace the speculative pseudo-random accept draw (D) with a joint forward verification.
- Confirm the trainer's sine encoding (C) is documented as an input feature map, not a gradient.
- Label or remove the unused `Scorecard` fields and the baseline energy and per-token values.
