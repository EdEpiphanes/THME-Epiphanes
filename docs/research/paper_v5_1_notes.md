# Paper v5.1 notes: where the text and the code differ

Working notes for the next revision of the research paper. The paper itself is not in this
repository; each entry gives the code reference and a replacement sentence.

## Loop breaker (paper section 7.9, THME-3099)

- **Paper says:** a detected loop raises the temperature and resets DoLa.
- **Code does:** generation stops. `NgramLoopBreaker::push_and_check_loop`
  (`src/inference/grounded_decode.rs`) returns `true` when the trailing 4-gram has now
  appeared three times within the last `max_history` 4-grams (256 in `src/chat.rs`; the
  constructor never goes below 16). `chat.rs` then ends the reply with `Finish::LoopBroken`.
  Temperature is not raised and DoLa is not reset.
- **Pinned by tests:** a repeating `1,2,3,4,...` stream fires at the 12th token and not before; a
  stream with no repeated 4-gram never fires
  (`loop_breaker_fires_on_the_third_repeat_of_a_four_gram_and_not_before`,
  `loop_breaker_ignores_a_sequence_with_no_repeated_four_gram`).
- **Suggested sentence:** "When the trailing 4-gram of the generated text has appeared three times in
  the last 256 positions, generation stops and the reply is marked as loop-broken."

## Measured code state, 27 November 2026

- **Paper says:** Check the paper text against this.
- **Code and measurement:** On branch `feat/solidify`, a full `cargo test` run passed 856 tests with 0 failures and 1 ignored. The release `thme.exe` is 14.90 MiB (project cap 16 MiB for `thme` and 25 MiB for `thme-gui`; the GUI binary was not re-measured on this branch). The crate version is 0.2.0 (the v0.2.0 release itself is deferred).
- **Suggested sentence:** "On branch `feat/solidify` (crate version 0.2.0, with the v0.2.0 release deferred), `cargo test` passes 856 tests (0 failures, 1 ignored) and the release `thme.exe` binary is 14.90 MiB within the 16 MiB cap (`thme-gui`, capped at 25 MiB, was not re-measured on this branch)."

## Quality gate: declared range and v2

- **Paper says:** Check the paper text against this.
- **Code and measurement:** The gate now follows the declared-range specification: the upper e-process uses a declared range `M` of 0.25, a block above `M` fails, and the identical-candidate case matches the paper's Table 4 values as an independent check. A v2 gate also exists with two regimes by baseline perplexity: an absolute margin of 0.45 below a baseline of 20, and a relative margin of 2.25 percent at or above 20 (v1 is still the default; ADR `docs/adr/0010-quality-gate-v2-two-regimes.md`).
- **Suggested sentence:** "The quality gate implements the declared-range specification with upper e-process declared range `M = 0.25` (failing any block above `M` and matching the paper's Table 4 values on identical candidates as an independent check); v1 remains the default alongside a v2 gate that uses an absolute perplexity margin of 0.45 below a baseline of 20 and a relative margin of 2.25 percent at or above it."

## DC-CCCP sign split

- **Paper says:** Check the paper text against this.
- **Code and measurement:** The allocator now implements the sign split of the coupling matrix (positive coupling terms bounded above by AM-GM around the expansion point, non-positive terms kept exact) and takes monotone majorize-minimize steps inside the byte budget before the tabu polish. A property test over 200 random instances checks that the surrogate majorizes the objective, and a second test checks that the descent is monotone and budget-feasible. The solver matches or beats an independent dynamic-programming choice on the tested instances; it was not measured on a real model.
- **Suggested sentence:** "The allocator implements the coupling-matrix sign split (bounding positive terms above by AM-GM around the expansion point while keeping non-positive terms exact) and takes monotone, budget-feasible majorize-minimize steps before tabu polish, matching or beating an independent dynamic-programming choice on 200-instance property tests (not yet measured on a real model)."

## Hardware-independent export check

- **Paper says:** Check the paper text against this.
- **Code and measurement:** A GGUF written with `thme export --llama-cpp-layout` (SmolLM-135M, F16) loaded in `llama.cpp` build `b12584` on the CPU and produced the same 20 greedy tokens as THME's own forward pass (report `docs/benchmarks/smollm-135m/data/llama_cpp_load.json`). The same export without the q/k row permutation did not match. Scope is one model and one prompt, with no pre-tokenizer type written yet (`THME-3102`).
- **Suggested sentence:** "A SmolLM-135M F16 GGUF written with `thme export --llama-cpp-layout` (with q/k row permutation; no pre-tokenizer type written yet, `THME-3102`) loaded in `llama.cpp` build `b12584` on CPU and matched THME's forward pass across 20 greedy tokens on one prompt."

## KV cache precision

- **Paper says:** Check the paper text against this.
- **Code and measurement:** Measured on SmolLM-135M, WikiText-2, CPU, token by token through the cache: 4 windows of 256 tokens give perplexity 30.392 (`f32`), 30.397 (`int8`), and 32.153 (`int4`); 2 windows of 1,024 tokens give 16.094 (`f32`), 16.087 (`int8`), and 17.185 (`int4`); cache bytes at 256 tokens are 11.75 MB (`f32`), 3.30 MB (`int8`), and 1.84 MB (`int4`). Not measured: 4,096 context (the model is trained for 2,048), other models, and decode speed (report `docs/benchmarks/smollm-135m/data/kv_precision_ppl.json`; ADR `docs/adr/0012-kv-cache-precision.md`).
- **Suggested sentence:** "Evaluated token by token through the cache on SmolLM-135M (WikiText-2, CPU), `f32`, `int8`, and `int4` KV cache precisions give perplexities of 30.392, 30.397, and 32.153 across 4 windows of 256 tokens (cache bytes 11.75 MB, 3.30 MB, and 1.84 MB) and 16.094, 16.087, and 17.185 across 2 windows of 1,024 tokens; 4,096 context (the model is trained for 2,048), other models, and decode speed were not measured."

## Softmax precision (not verified)

- **Paper says:** Check the paper text against this.
- **Code and measurement:** The owner pasted a summary of an arXiv paper on softmax collapse and cancellation (`arXiv 2610.31642`, not read or verified here). THME evaluation computes the loss from the maximum-subtracted form with the sum in `f64`, so perplexity figures are not affected by the claimed effect; training and distillation gradients (not built yet, `THME-3036`) should be computed as direct differences.
- **Suggested sentence:** "Because THME evaluation computes the loss from the maximum-subtracted form with the sum in `f64`, reported perplexity figures are unaffected by softmax cancellation; training and distillation gradients (`THME-3036`, not built yet) should be computed as direct differences."
