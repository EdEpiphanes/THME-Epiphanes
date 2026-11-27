# 0011. llama.cpp layout is an opt-in export mode

Date: 2026-11-27

## Status

Accepted

## Context

GGUF files from `thme export` passed THME's own checker but did not load in llama.cpp (no tokenizer, Hugging Face tensor names, BF16 norms, unpermuted q/k rows, no rope dimension key).

## Decision

`thme export --llama-cpp-layout` writes GGML tensor names, F32 norms, q and k rows permuted for llama.cpp's interleaved rotary pairs (the permutation is derived from the rotary pair layout, not copied from llama.cpp source), `rope.dimension_count`, and the tokenizer from `tokenizer.json`.
- It refuses a file that fails the new llama.cpp requirement checks.
- The default export is unchanged and byte-identical.
- THME's own reader undoes the permutation for GGUF files whose architecture is `llama`.

## Consequences

- Evidence level: measured once. SmolLM-135M F16 loaded in llama.cpp b12584 on the CPU and gave the same 20 greedy tokens as THME's forward pass; a control without the permutation did not match.
- Open gaps:
  - One model, one prompt.
  - No `tokenizer.ggml.pre` (ticket THME-3102).
  - Quantised types and other architectures not checked.
