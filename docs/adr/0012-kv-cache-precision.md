# 0012. KV cache precision is chosen at cache creation

Date: 2026-11-27

## Status

Accepted

## Context

The KV cache grows with context (THME-3082).

## Decision

- `KvPrecision` provides `F32`, `Int8`, and `Int4`, chosen when the cache is made (`thme chat --kv-precision`).
- `Int8` and `Int4` store symmetric groups of 32 values with one `f32` scale per group (`Int8` scale `max / 127`, `Int4` scale `max / 7`, two 4-bit values per byte).
- Values are quantised on write and decoded on read.
- The CUDA path declines a non-f32 cache and falls back.

## Consequences

- Evidence level: measured on SmolLM-135M, WikiText-2, CPU:
  - `Int8` within noise of `F32` (30.392 vs 30.397 at 4x256 tokens; 16.094 vs 16.087 at 2x1024).
  - Pythia-14M is very different: perplexity 190.8 (`F32`), 245.2 (`Int8`), 5,805 (`Int4`) at 4x256 tokens and 75.5, 101.5, 10,350 at one 2,048-token window (`docs/benchmarks/pythia-14m/data/kv_precision_ppl.json`). The default therefore stays `F32`. A local experiment narrowed the cause to the keys: quantizing only the values leaves Pythia-14M unchanged (190.9 at `Int8`, 190.7 at `Int4`), quantizing only the keys produces all of the loss. A separate key and value precision is a possible follow-up, not built.
  - `Int4` about 6 to 7 percent higher perplexity (32.153; 17.185).
  - Cache bytes 3.6 and 6.4 times smaller.
- Open gaps:
  - One model, no 4k context (models trained for 2,048).
  - No decode-speed figure, no GPU cache.
  - Placement budget does not count KV bytes yet.
