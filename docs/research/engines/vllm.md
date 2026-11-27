# vLLM survey (for THME)

Read-only survey of the shallow clone at `C:\Users\TechTroniX\engines\vllm`. Paths below are relative to that folder. Every claim names a file that was opened; anything else is marked unverified. Effort sizes (S/M/L) are estimates, not measurements.

THME today (opened: `README.md`, `docs/subsystems.md`, `src/`): single-sequence `KvCache` with f32/int8/int4 per-group-32 storage (`src/inference/kv.rs`), a CPU reference forward pass, GPU backends (CUDA PTX in `src/inference/gpu/kernels/decoder.ptx`, wgpu WGSL shaders in `src/inference/gpu/shaders/`), a speculative-decoding module (`src/speculative/`), a token sampler (`src/inference/sampler.rs`), and allocators (`src/allocator/`, `src/engine/alloc.rs`). A grep of `src/` found no paged KV, batching scheduler, prefix cache or grammar constraint.

## 1. Licence
- `LICENSE` first lines: "Apache License / Version 2.0, January 2004". Only the unfilled appendix template ("Copyright [yyyy] [name of copyright owner]") is present, so the LICENSE file names no copyright holder.
- Source headers opened (`vllm/v1/worker/gpu/spec_decode/ngram/speculator.py`, `vllm/v1/structured_output/backend_types.py`) say `SPDX-License-Identifier: Apache-2.0` and "Copyright contributors to the vLLM project".
- NOTICE file: none at repo root (a find for LICENSE/NOTICE/COPYING found only `LICENSE` and the nested file below).
- Different licence: `vllm/third_party/flash_linear_attention/LICENSE` is MIT, "Copyright (c) 2023-2025, Songlin Yang, Yu Zhang".
- Obligations (Apache-2.0 section 4 in `LICENSE`): ship a copy of the licence, keep copyright/patent/attribution notices in copied source, mark changed files, carry the NOTICE text if the original has one (none here). Apache-2.0 code may go into AGPL-3.0-only. The nested MIT folder needs its own notice if copied. Rewriting an algorithm from scratch in Rust copies no code, but cite the source in the module doc.

## 2. What it is
vLLM is a Python serving engine for LLMs built around a paged KV cache, a token-budget scheduler and many GPU kernels (CUDA, Triton, ROCm). The `vllm/v1/` tree holds the engine core (scheduler, KV manager, worker, sampler, structured output); `csrc/` holds C++/CUDA kernels. A small Rust OpenAI-compatible frontend lives in `rust/` but still talks to the Python engine over ZMQ (`rust/README.md`).

## 3. Components
| Component | What it does | Main paths opened |
| :--- | :--- | :--- |
| Scheduler | One loop assigns tokens to requests per step under a token budget; no separate prefill/decode phase (comment in `schedule`); chunked prefill, spec tokens, preemption | `vllm/v1/core/sched/scheduler.py` (4118 lines; `schedule`, `_preempt_request`, `update_from_output`) |
| KV manager and block pool | Fixed-size blocks, free queue in eviction order, hash-to-block map for prefix hits, ref counts | `vllm/v1/core/kv_cache_manager.py`, `block_pool.py`, `kv_cache_utils.py` |
| Block table (device side) | Maps request to block ids for attention | `vllm/v1/worker/gpu/block_table.py` |
| Attention backends and ops | Many backends; Triton unified attention; INT4 per-token-head KV mode | `vllm/v1/attention/backends/`, `vllm/v1/attention/ops/triton_unified_attention.py`, `int4_per_token_head.py` |
| Speculative decoding | ngram scan kernel, EAGLE, MTP, suffix and more; rejection sampler in Triton | `vllm/v1/worker/gpu/spec_decode/` (`ngram/speculator.py`, `rejection_sampler.py`, `rejection_sampler_utils.py`), `vllm/config/speculative.py` |
| Structured output | JSON/regex/grammar/choice, pluggable backends, bitmask fill, rollback | `vllm/v1/structured_output/` (`backend_types.py`, `backend_guidance.py`, `backend_outlines.py`, `backend_xgrammar.py`) |
| Quantization | Registry of methods plus online requantizers | `vllm/model_executor/layers/quantization/__init__.py`, `online/`, `turboquant/`, `compressed_tensors/` |
| Kernels | Marlin, GPTQ, AWQ, Hadamard, W8A8, FP4 CUDA | `csrc/quantization/{marlin,gptq,awq,hadamard,fp4,w8a8}`, `csrc/attention/` |
| Rust frontend | HTTP, chat template, tokenizer layers | `rust/README.md`, crate names under `rust/src/` |

## 4. What THME lacks that vLLM has, ranked
Ranking is value to a small, quantization-focused, measurement-driven Rust engine.

### 4.1 Prefix caching by block hash (rank 1)
- Why: reuses KV for shared prompt prefixes; exact, and easy to measure (hit rate, tokens skipped).
- Paths: `vllm/v1/core/block_pool.py` (`cache_full_blocks`, `touch`, `free_blocks`, `BlockHashToBlockMap`); `vllm/v1/core/kv_cache_utils.py` (`hash_block_tokens`: hash over parent block hash, block token ids and extra keys; `FreeKVCacheBlockQueue`).
- Translate: plain Python bookkeeping, so an algorithm that can be rewritten in Rust (hash map, free list, ref counts). No kernel beyond the block-table gather that comes with 4.2.
- Dependencies: block-structured KV cache (4.2); a hash function (unverified whether THME already has one).
- Effort: M. Risk: low for logic, medium with int4/int8 KV (cached blocks must keep their scales; how vLLM handles it is unverified).
- Test oracle: a run with the cache off must give identical greedy tokens and logits; property tests on ref counts and eviction order.

### 4.2 PagedAttention and the block manager (rank 2)
- Why: removes KV fragmentation and lets many sequences share memory; prerequisite for 4.1 and 4.3. THME's `KvCache` (`src/inference/kv.rs`) is one cache per sequence.
- Paths: `vllm/v1/core/kv_cache_manager.py` (`get_computed_blocks`, `allocate_slots`, `free`), `single_type_kv_cache_manager.py`, `vllm/v1/worker/gpu/block_table.py`; kernels `vllm/v1/attention/ops/triton_unified_attention.py`, `csrc/attention/` (the classic paged CUDA kernel was not located; unverified).
- Translate: the manager is a rewritable algorithm. Attention over blocks is a kernel: THME's `decoder.cu` and `attention_tiled.wgsl` / `attention_coop.wgsl` need a block-table indirection written fresh. Do not port the Triton.
- Dependencies: none external. Effort: L (the GPU attention change on two backends dominates). Risk: medium (blocked softmax must match the dense path numerically).
- Test oracle: THME's dense reference forward (CPU) on the same prompts, logits within a stated tolerance across block sizes.

### 4.3 Continuous batching and scheduler (rank 3)
- Why: token-budget scheduling raises throughput with several concurrent requests. Value is moderate: THME is a workbench, not yet a server.
- Paths: `vllm/v1/core/sched/scheduler.py` (`schedule`, `token_budget`, `long_prefill_token_threshold`, `_preempt_request`), `request_queue.py`.
- Translate: the policy is rewritable. The file also carries pipeline/multimodal/KV-connector/DP code; take only the core loop (num_computed_tokens catching up to total tokens).
- Dependencies: 4.2 and a batched forward pass (unverified whether THME's decoder batches). Effort: L. Risk: medium-high (scope creep).
- Test oracle: sequential single-request runs; batched greedy output must match token for token. Throughput via THME's bench module.

### 4.4 Speculative decoding: ngram and rejection sampler (rank 4)
- Why: THME's speculative module is formulas and a tree structure, with no timing measured (`README.md` table, `docs/subsystems.md` 5.5). vLLM's ngram speculator needs no draft model.
- Paths: `vllm/v1/worker/gpu/spec_decode/ngram/speculator.py` (`_ngram_scan_kernel`, Triton), `ngram/config.py` (prompt_lookup_min/max, default 5), `rejection_sampler.py`, `rejection_sampler_utils.py` (Triton logsumexp and residual-mass kernels). EAGLE/MTP need trained heads.
- Translate: suffix match is an algorithm (CPU Rust, S). Rejection-sampling math is an algorithm already in THME docs. The Triton kernels are kernels and need not be ported.
- Dependencies: a multi-token verify pass. Effort: M. Risk: low-medium.
- Test oracle: plain greedy decode must give identical text; a statistical test that sampled output matches target-only sampling.

### 4.5 Quantization coverage (rank 5, cherry-pick)
- Paths: `vllm/model_executor/layers/quantization/__init__.py` lists awq, gptq, gptq_marlin, awq_marlin, fp8, mxfp4, modelopt, compressed-tensors, torchao and online variants (fp8, int4, int8, nvfp4, mxfp8). `vllm/v1/attention/ops/int4_per_token_head.py` (docstring): INT4 KV packs two values per byte, pre-rotates with one random Hadamard transform and hides a 4-bit zero point in the scale. That maps onto THME's int4 KV and its shared Hadamard code in `src/engine/`.
- Translate: format specs and the rotation idea are algorithms. Marlin/GPTQ/AWQ kernels (`csrc/quantization/marlin`, `gptq`, `awq`) are CUDA: do not port.
- Dependencies: none. Effort: S to study, M to add rotated int4 KV. Risk: low; quality cost is measured by THME's gate.
- Test oracle: WikiText-2 perplexity versus f32 KV, plus the explicit Hadamard-matrix oracle already used in `src/engine`.

### 4.6 Structured output (rank 6; same topic as sglang.md)
- Paths: `vllm/v1/structured_output/backend_types.py` (interface: `accept_tokens`, `rollback`, `fill_bitmask`, `is_terminated` in the backends); backends depend on xgrammar, llguidance and outlines_core (`requirements/common.txt`).
- Translate: the interface is small; the grammar compilers are external native libraries. A Rust-native JSON-schema or regex token mask avoids them. Effort: M. Risk: medium.
- Test oracle: a JSON-schema validator on every output; unconstrained runs unchanged.

## 5. What NOT to port
- Pipeline/data/context parallelism, KV connectors and offload (`vllm/v1/kv_offload`), multimodal encoder cache, LoRA serving: out of scope for a single-box engine.
- All Triton and CUTLASS kernels (`vllm/v1/attention/ops/*`, `csrc/`): tied to PyTorch tensors and vendor toolchains; THME's PTX and WGSL path stays.
- Per-model fused kernels (`csrc/fused_*_kernel.cu`) and `vllm/models/`: many model-specific files.
- EAGLE/MTP draft heads: need trained weights THME does not ship.
- The Rust frontend (`rust/`): a proxy to the Python engine, no engine logic.
- Marlin/GPTQ/AWQ kernels: fixed formats fit poorly with THME's per-tensor allocator.
