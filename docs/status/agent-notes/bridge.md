<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Agent notes: PyTorch training bridge, quantization-aware distillation with LoRA recovery (branch agent/v02-bridge)

Scope: first slice of ROADMAP v0.2.3 with PyTorch, PEFT and transformers as an external trainer.
Nothing here was run on SmolLM-135M and nothing used the GPU (`CUDA_VISIBLE_DEVICES=-1` in every
command). Every test uses a tiny random LLaMA (2 layers, hidden 64, vocab 128, tied head) built in
`scripts/train/tiny.py`. The tiny-model results are tests of the machinery, not evidence about
SmolLM-135M. Environment used: torch 2.11.0+cu128, transformers 5.24.1, peft 0.22.0 (CPU).

## What was built

- `examples/dump_q4_0.rs`: pseudo-random 4 x 128 matrix (xorshift64*), 16 blocks of 32 with
  edge cases (all zeros, a +max/-max tie, a -3.0 outlier, FP16-subnormal scales), round-tripped
  through THME's own `fake_quantize(Q4_0)`. Writes input and output as f32 bit patterns.
  Fixture: `tests/fixtures/q4_0_roundtrip.json` (10 KB).
- `scripts/train/quant.py`: `q4_0_roundtrip` (torch, float32), `q4_0_param_names`, `quantize_model_`.
- `scripts/train/eval_ppl.py`: `evaluate_ppl`, `read_token_file`, a small CLI.
- `scripts/train/qad_lora.py`: the trainer (`run`, CLI with the requested arguments), plus helpers
  `build_teacher`, `build_student`, `add_lora`, `merge`, `kl_loss`, `batch_indices`.
- `scripts/train/tiny.py`: test helper (the tiny model). Tests: `test_quant.py`, `test_eval_ppl.py`,
  `test_qad_lora.py`. Run: `python -m unittest discover -s scripts/train -p "test_*.py"` with the
  venv python.

## Which tensors THME quantizes (read from the code)

A `SetPrecision` recipe with format Q4_0, all layers, role null (`src/surgery/recipe.rs`) selects a
tensor only if it has a layer index (`extract_layer_index`: the name segment after `layers`, `h`,
`blocks`, ... must be an integer). A tensor without a layer index is reached only by naming its
role, so with role null the token embedding, final norm and `lm_head` are not selected. Then
`load_tensor` (`src/inference/llama.rs`) applies the block format only when the tensor has at
least two dimensions ("1-D tensors (norms, biases) stay as stored"). For a HF LLaMA this is
q, k, v, o, gate, up, down in every layer (7 x 30 = 210 for SmolLM-135M), and nothing else. The
tied embedding/head stays at its stored precision. Not run on the real model: the count of 210 is
from the architecture, not from a listing of SmolLM-135M's state dict.

## THME's Q4_0 definition (`encode_q4_0_block`, `unpack_q4_0`)

Blocks are 32 consecutive values of the row-major flattened tensor. `extreme` = the value of the
largest magnitude (first one on a tie, folding from +0.0); `scale = extreme / -8` in f32; the
stored scale is FP16 (round to nearest even); the code is `clamp(trunc(v * (1/scale) + 8.5), 0,
15)` where `1/scale` uses the unrounded f32 scale; decode is `f32(stored) * (code - 8)`. Errors:
FP16 scale not finite, or a non-zero scale that underflows to zero in FP16. quant.py raises
`ValueError` in the same cases. A subtle point the fixture covers: an all-zero block decodes to
-0.0 (stored scale is -0.0), which the torch version reproduces bit for bit.

## THME's windowing (`src/inference/perplexity.rs`)

Consecutive non-overlapping windows of `context_len` tokens (default 1024); a final chunk is kept
if it has at least 2 tokens and dropped if it has 1; one forward pass per window with no context
carried over; the first token of each window is not scored; NLL per token in f32, summed in f64;
perplexity = exp(total NLL / scored tokens). `--max-tokens` truncates the corpus before windowing.
eval_ppl.py does the same (log-softmax in float32, sum in float64).

## Training design (decisions made)

- Student base is float32, frozen, with the Q4_0 tensors rounded; teacher is frozen, any of
  float32 (default), bfloat16, float16 via `--teacher-dtype`.
- LoRA via PEFT on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`, dropout 0, default
  PEFT init (B = 0, so the first step is exactly the quantized model).
- Loss: KL(teacher || student), mean over all positions of all windows (including the last
  position of each window, whose next token is outside the window; the distribution still exists).
  Eval scores per THME's rule, so train and eval positions differ by that one position.
- AdamW, weight decay 0, no schedule, no clipping. Batches: `batch_size` windows drawn with
  replacement per step from a generator seeded by `--seed`.
- `train_log.json` records the loss before each step's update, so entry 0 is the zero-adapter loss.
- Outputs: `adapter/` (PEFT), `merged/` (fp16 safetensors), `result.json` (adapter bytes,
  trainable params, wall time, `eval_ppl` of the adapter model after training, plus the teacher's
  and the quantized student's perplexity under the same rule).

## Tests, and which failed first

- `test_quant.py` (7): `test_round_trip_equals_the_rust_fixture_bit_for_bit`,
  `test_hand_worked_block` (arithmetic in the comment), `test_all_zero_block_gives_zeros`,
  `test_rejects_wrong_size_and_scale_out_of_fp16_range`, `test_keeps_shape_and_does_not_touch_the_input`,
  `test_smollm_style_names`, `test_tiny_llama_has_seven_matrices_per_layer`.
- `test_eval_ppl.py` (6): `test_final_window_of_two_is_scored`,
  `test_final_window_of_one_token_is_dropped`, `test_max_tokens_truncates_first`,
  `test_zero_weights_give_perplexity_of_the_vocabulary`, `test_reads_thme_token_file`,
  `test_too_few_tokens_and_bad_window_raise`. The brute force is plain Python floats.
- `test_qad_lora.py` (7): `test_student_has_exactly_the_thme_q4_0_tensors_rounded`,
  `test_zero_adapter_loss_equals_plain_kl_of_the_quantized_model`,
  `test_kl_matches_a_hand_computed_value`, `test_twenty_steps_lower_the_loss` (batch of all 24
  windows, last-5 mean below half the first loss), `test_outputs_and_result_fields`,
  `test_merged_logits_equal_adapter_logits` (max difference below 1e-5),
  `test_run_is_bit_reproducible_for_a_fixed_seed` (log and adapter file byte-identical; a
  different seed gives a different log).

Seen to fail first: before quant.py and eval_ppl.py existed, test_quant and test_eval_ppl failed
with `ModuleNotFoundError` (this proves only that they ran, not that they test anything). A
mutation check showed the fixture test is not vacuous: changing the rounding offset 8.5 to 8.4 in
a copy of quant.py made it fail. qad tests, first run with a tiny model at HF default init: the
loss test failed (the loss rose; the model's KL was about 1e-4 and batches of 4 are noisy) and
the merged-logits test failed because my test compared against a module the adapter had already
been injected into (a test bug). I then raised the tiny model's init scale (0.1), compared
against logits taken before injection, lowered the learning rate to 3e-3 and used all windows per
step. With init scale 0.2 the merge difference was 1.3e-5, above 1e-5, so 1e-5 depends on logit
magnitude; at 0.1 it passes. For fp32 SmolLM-135M logits this tolerance may need to be relative.

## NOT verified

- Anything on SmolLM-135M: the 210 tensor count, that no SmolLM tensor makes the Q4_0 encoder error
  (FP16 range, underflow), the tied-head behavior in `merge_and_unload`/`save_pretrained`, memory
  and speed, whether the loss helps real perplexity, the merged-vs-adapter 1e-5 bound on real logits.
- Anything on CUDA: `--device cuda` is written and untested; fp16 and bf16 teacher paths untested.
- That `thme evaluate` loads `merged/` (not run; HF names and fp16 safetensors should load, check it).
- Equality of eval_ppl.py with the Rust `thme evaluate` on the same file and model: the rule is
  copied and tested against a brute force, not against a Rust run. Small float differences remain
  (the Rust forward pass is its own implementation).
- Bit reproducibility on CUDA, or on the CPU across thread counts or machines.

## For the coordinator

- The merged model is Q4_0 base plus a dense LoRA delta, so it is no longer on the Q4_0 grid.
  `thme evaluate` with a Q4_0 recipe would round it again and undo much of the recovery; measure
  it with an FP16 (no quantization) recipe, and say that the deployed artifact is then FP16
  weights, or decide how a recovered model is meant to ship (adapter kept separate, or the merged
  weights re-quantized and re-measured).
- The merged weights are cast to fp16 before saving, which adds its own rounding; the reported
  `eval_ppl` is of the fp32 adapter model, not of the saved fp16 file.
- Training uses all positions, eval drops each window's first token; decide if that matters.
- No learning-rate schedule, clipping or warm-up yet; hyper-parameter defaults in the CLI
  (rank 16, alpha 32, lr 1e-4, 200 steps) are untested guesses.
