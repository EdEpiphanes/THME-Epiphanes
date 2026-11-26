# 0002. Prioritize Incremental KV Cache, Packed `Q4_0` Decode, and Task/BPB Benchmarks

Date: 2026-11-26

## Status

Accepted

## Context

Prefill evaluation (`thme evaluate`, 1024-token windows) now runs at **7,038 tok/s** (exact) to **9,506 tok/s** (`F16_ACC`). However:
1. `chat::generate` currently re-runs the entire prompt on every generated token (no runtime incremental KV cache).
2. `Q4_0` weights are expanded to `F16` in the 10 GiB VRAM cache, which is optimal for compute-bound prefill but incurs ~3.5× extra memory bandwidth during memory-bound single-token decode.
3. Relying solely on WikiText-2 perplexity cannot compare across tokenizers or measure multi-step reasoning, math, or coding fidelity.

## Decision

1. **Prioritize Decode Architecture (`v0.2.4`):** Implement an incremental KV cache (supporting INT8/INT4 group-wise KV quantization and attention sinks + sliding window) and packed-domain `Q4_0` matrix-vector decode kernels across CUDA and `wgpu` before further prefill micro-tuning.
2. **Expand Quality Metrics (`v0.2.8`):** Add tokenizer-invariant **Bits-Per-Byte (BPB)** to `thme evaluate` and wire the existing `src/bench/` runner (`MultipleChoice` and `CheckableAnswer`) for MMLU, ARC-Challenge, HellaSwag, and GSM8K.

## Consequences

- Real-time chat/generation latency improves quadratically with sequence length across all hardware targets (NVIDIA, AMD, Intel, Apple Silicon, and CPU).
- Evaluation gates gain both cross-tokenizer comparability (BPB) and functional reasoning verification alongside WikiText-2 perplexity.
