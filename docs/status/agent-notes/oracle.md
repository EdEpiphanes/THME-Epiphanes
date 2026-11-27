# Agent notes: PyTorch oracle for the hand-written backward pass (THME-3036 follow-up)

Branch: agent/v02-oracle, from feat/finish-v0.2. Scope: an independent check of
`src/trainer/backward.rs` against PyTorch (float64, CPU). No model was downloaded or run, no
dependency was added, nothing was installed. The GPU was not used (`CUDA_VISIBLE_DEVICES` is
emptied in the script). Every number below comes from the tiny model built by
`tests/common/tiny_oracle.rs`, a fixed-seed generator, not from a real model.

## What was done

1. `examples/dump_tiny_gradients.rs` builds the tiny model (2 layers, hidden 16, 4 query heads,
   2 key/value heads, intermediate 32, vocab 40, RMS epsilon 1e-5 as f32 widened to f64,
   rope theta 10000, tied head) and the fixed token sequence `[3, 17, 5, 9, 30, 2, 11]`. It runs
   `loss_and_grads` in f64 and writes one JSON file: config, tokens, Rust loss, and every weight
   with its Hugging Face name, shape, values and Rust gradient.
   `cargo test` never runs this example; it is a generator.
2. `scripts/make_torch_gradients.py` loads that JSON into `LlamaForCausalLM` (transformers
   5.24.1, torch 2.11.0, float64, CPU), computes the mean cross-entropy of `logits[:-1]` against
   `tokens[1:]` (the same positions and the same mean as `backward::loss`), runs autograd, and
   compares. Losses first: above 1e-9 relative, it stops and writes nothing. Then per tensor:
   max relative error with a 1e-9 absolute floor. The fixture is
   `tests/fixtures/torch_gradients_tiny.json`: the PyTorch loss, and per tensor the norm and the
   first eight values.
3. `tests/torch_oracle.rs` rebuilds the same weights from `tests/common/tiny_oracle.rs`, checks
   the config and tokens against the fixture, and checks the Rust loss and each tensor's gradient
   norm and first eight values against the fixture within 1e-6 relative. It skips with a message
   only if the fixture file is absent.

## The first STOP and what it turned out to be

The first run of the script stopped: forward loss rust `3.53090817430865`, torch
`3.53090819766947`, relative difference **6.6e-9**, above the 1e-9 limit. Nothing was compared
and nothing was written, as the task asks. I did not change the Rust side.

I then measured each suspect in a scratch script (outside the repo) with transformers' stock
ops, one at a time. Loss relative difference to Rust:

| PyTorch side | rel diff to Rust loss |
| :--- | ---: |
| stock transformers (rotary and RMSNorm cast to float32) | 6.6e-9 |
| float64 rotary only | 1.0e-8 |
| float64 RMSNorm only | 1.3e-9 |
| float64 rotary and RMSNorm (attention already f64 under SDPA) | 0 to printed precision (about 1e-16) |

The cause is on the PyTorch side, not the Rust side. In the installed `modeling_llama.py`, the
rotary `inv_freq` and angles are built in float32 (lines 106 and 111), and `LlamaRMSNorm` casts its
input to float32 (line 64), even for a float64 model. The task asks for float64 throughout, so
the script now runs those two ops in float64 with the same formulas (`use_float64_ops`). The
stock behaviour is still reachable with `--hf-precision` and still stops with the 6.6e-9 result.
Nothing in `src/trainer/backward.rs` changed. No bug was found.

Two points for the coordinator. First, this changed the oracle's numerics after a STOP. I made
the change because the stock ops are float32 and the spec says float64, and I kept the failing
configuration reproducible behind a flag. Second, the agreement is evidence that the Rust
forward and backward match PyTorch's autograd on these ops in float64. It is not evidence about
float32 training, which the oracle does not test.

## Table: per-tensor errors (float64, float64 rotary and RMSNorm)

Loss: rust `3.53090817430865`, torch `3.53090817430865`, relative difference 0 to printed
precision. Limit for each tensor: 1e-6.

| tensor (Hugging Face name) | shape | torch grad norm | max rel err |
| :--- | :--- | ---: | ---: |
| model.embed_tokens.weight | [40, 16] | 2.546920e+00 | 1.770e-13 |
| model.layers.0.input_layernorm.weight | [16] | 2.087841e-01 | 7.266e-15 |
| model.layers.0.self_attn.q_proj.weight | [16, 16] | 6.025070e-01 | 3.347e-13 |
| model.layers.0.self_attn.k_proj.weight | [8, 16] | 6.094988e-01 | 3.086e-14 |
| model.layers.0.self_attn.v_proj.weight | [8, 16] | 1.325998e+00 | 1.322e-14 |
| model.layers.0.self_attn.o_proj.weight | [16, 16] | 2.381135e+00 | 1.973e-14 |
| model.layers.0.post_attention_layernorm.weight | [16] | 1.434991e-01 | 2.901e-15 |
| model.layers.0.mlp.gate_proj.weight | [32, 16] | 1.291700e+00 | 4.757e-13 |
| model.layers.0.mlp.up_proj.weight | [32, 16] | 8.711817e-01 | 6.888e-14 |
| model.layers.0.mlp.down_proj.weight | [16, 32] | 1.088720e+00 | 3.671e-13 |
| model.layers.1.input_layernorm.weight | [16] | 3.000972e-01 | 9.176e-15 |
| model.layers.1.self_attn.q_proj.weight | [16, 16] | 4.224894e-02 | 7.099e-13 |
| model.layers.1.self_attn.k_proj.weight | [8, 16] | 7.124338e-02 | 2.219e-13 |
| model.layers.1.self_attn.v_proj.weight | [8, 16] | 1.267967e+00 | 3.385e-14 |
| model.layers.1.self_attn.o_proj.weight | [16, 16] | 8.239300e-01 | 4.447e-12 |
| model.layers.1.post_attention_layernorm.weight | [16] | 1.256924e-01 | 3.370e-13 |
| model.layers.1.mlp.gate_proj.weight | [32, 16] | 7.674116e-01 | 1.315e-12 |
| model.layers.1.mlp.up_proj.weight | [32, 16] | 8.304788e-01 | 9.830e-14 |
| model.layers.1.mlp.down_proj.weight | [16, 32] | 4.957955e-01 | 6.903e-13 |
| model.norm.weight | [16] | 2.044391e-01 | 1.544e-15 |

Worst per-tensor max relative error: **4.4e-12** (layers.1 o_proj). The layer-1 q and k
gradients have norms of about 0.04 and 0.07 here, and their errors are 7e-13 and 2e-13. This is
a different model from `tiny_model` (see the shortcut in `tests/common/tiny_oracle.rs`), so the
small-gradient caveat in `train.md` (layer-1 q and k near 5e-7) is not tested by this oracle.

## Verdict

**Met** for the tiny model in float64 on the CPU: loss agreement to double precision, and every
one of the 20 tensors within 4.4e-12 relative, against the 1e-6 limit. No bug in
`src/trainer/backward.rs` was found, so the file is unchanged.

Not covered by this oracle:

- The untied head (`lm_head.weight`). The generator only builds the tied head, so the
  `dhead` path in `backward` is not checked here. The earlier finite-difference test covers it.
- LoRA adapters, quantized weights, Gemma, batching and dropout.
- Real checkpoints. The tiny model is random, with scale 0.25 and norm gains near 1.

## Red observations

- Layout test (`tiny_weights_follow_the_hf_layout`): written before `tests/common/tiny_oracle.rs`
  existed, but the module was created before the first compile, so no compile failure was seen.
  **Red was not observed.**
- Fixture test: while the fixture was absent it printed the skip line and passed, which is
  vacuous. After the fixture was generated, I checked the test can fail: a loss of 3.5409 in the
  fixture gave "relative error 2.8e-3" and FAILED; one tensor norm of 9.0 gave "relative error
  7.2e-1" and FAILED. The fixture was restored byte for byte (`cmp`) each time.

## Gate

- Commit `dcc4892` (oracle model, dump example, layout test and fixture test): `cargo fmt
  --check` ok, `cargo clippy --all-targets -- -D warnings` ok, `cargo test -- --skip medgemma`
  exit 0 (the full suite, before the fixture existed).
- Commit `8e5f3f1` (script and fixture): fmt and clippy re-run ok, `torch_oracle` tests 2 passed
  with the fixture present, `py scripts/spdx.py --check` exit 0. The full suite was not re-run
  after this commit, because only a Python script and a JSON file were added.

## Decisions for the coordinator

1. Accept the float64 rotary and RMSNorm in the oracle, with the stock float32 path kept behind
   `--hf-precision`, or require a different reference for the forward.
2. Add an untied-head case (a second dump with `lm_head`) if the `lm_head` gradient needs its own
   oracle. This is one extra argument to the example and a second fixture.
3. The fixture is small and holds summaries only. Regenerating it needs the dump from the example
   and the venv, which is not in CI.
