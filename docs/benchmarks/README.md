# Measured decode speed (THME-3070)

`<model>/data/decode_speed.json` is written by `examples/decode_speed.rs` (release build). It is
**measured on the machine that ran it**, not a published benchmark. Figures here come only from
these files.

## What is compared

- **whole_sequence**: `forward_logits` over the growing sequence on every step (the loop the Studio
  chat tab used before THME-3070).
- **kv_cache**: `forward_extend`, the prompt prefilled once, then one token per step.

Both decode the same greedy reply from the same fixed prompt (8 token ids, built by a formula stored
in the JSON). The tokens/s figure is generated tokens over the wall time of the whole reply,
prompt prefill included. Each length was timed 5 times after 1 warmup run. Paired runs are
interleaved: each run times the two paths back to back, and which goes first alternates.

## Machine and conditions

- Intel64 Family 6 Model 151 (12 logical threads reported), Windows, release build, commit `d23f84c`
  (the JSON's `git_tracked_files_dirty` flag is `true` only because `BACKLOG.md` was edited after
  that commit; no code changed).
- **The machine was shared**: another CPU-heavy job (an AWQ evaluation, and a GPU builder) ran during
  these measurements. Timing is noisy; the spread (min to max) is in each file. Rerun on an idle
  machine before quoting any figure as a benchmark.
- CPU reference path only. The GPU path is not measured here.
- Pythia-14M is GPT-NeoX; SmolLM-135M is LLaMA. Both are small, so fixed overhead matters.

## Results (median, min to max over 5 runs)

Pythia-14M, tokens/s (generated tokens / wall time):

| generated | whole sequence | KV cache | paired speedup |
| ---: | :--- | :--- | :--- |
| 16 | 87.4 (85.6 to 90.6) | 523.4 (501.3 to 547.4) | 6.0x (5.5 to 6.4) |
| 64 | 74.6 (73.3 to 76.0) | 698.7 (674.5 to 700.7) | 9.2x (9.1 to 9.4) |
| 256 | 57.2 (56.0 to 58.0) | 711.3 (686.2 to 721.5) | 12.3x (11.9 to 12.9) |

SmolLM-135M, tokens/s:

| generated | whole sequence | KV cache | paired speedup |
| ---: | :--- | :--- | :--- |
| 16 | 8.1 (5.4 to 8.4) | 33.2 (24.4 to 34.6) | 4.1x (3.6 to 4.5) |
| 64 | 5.8 (4.6 to 6.1) | 36.2 (34.9 to 38.1) | 6.4x (5.8 to 7.6) |
| 256 | 3.6 (1.8 to 3.6) | 37.5 (30.9 to 38.2) | 10.6x (10.4 to 17.1) |

Checks in the same files: greedy tokens identical between the two paths in every run; KV-cache bytes
at the end of each run equal `KvCacheEstimator::calculate_bytes` (f32, model KV heads and head
dimension) exactly (Pythia-14M 256 generated: 1,615,872 bytes at 263 positions; SmolLM-135M: 12,119,040
bytes at 263 positions).

## What is not claimed

- No GPU figure, no energy figure, no peak-RSS figure, no comparison with other engines.
- Threads and matmul kernels are unchanged (separate tickets). The cached path still runs the same
  per-row matmul kernels, so these numbers measure the algorithmic change, not kernel speed.
- Machine shared, so no speed number here is a benchmark until rerun on an idle machine.

## Regenerate

```text
cargo build --release --example decode_speed
target/release/examples/decode_speed --model-path models/pythia-14m --model-name pythia-14m \
  --lengths 16,64,256 --runs 5 --warmup 1 --shared-machine --out docs/benchmarks/pythia-14m/data/decode_speed.json
```
