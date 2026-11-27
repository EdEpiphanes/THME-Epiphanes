# SGLang survey (for THME)

Read-only survey of the shallow clone at `<engines>\sglang`. Paths are relative to that folder. Every claim names a file that was opened; anything else is marked unverified. Effort sizes (S/M/L) are estimates, not measurements.

THME today (opened: `README.md`, `docs/subsystems.md`, `src/`): single-sequence `KvCache` with f32/int8/int4 storage (`src/inference/kv.rs`), CPU reference forward, CUDA PTX and wgpu WGSL GPU paths (`src/inference/gpu/`), a sampler chain derived from llama.cpp (`src/inference/sampler.rs`), a speculative module (`src/speculative/`). A grep of `src/` found no prefix cache, radix tree or grammar constraint.

## 1. Licence
- `LICENSE` first lines: "Apache License / Version 2.0, January 2004". Only the unfilled appendix template is present; no copyright holder is named in `LICENSE`.
- Source headers opened name the holder: `python/sglang/srt/mem_cache/radix_cache.py` and `python/sglang/srt/constrained/base_grammar_backend.py` say "Copyright 2023-2024 SGLang Team" under the Apache-2.0 text.
- NOTICE file: none found at repo root (a find for LICENSE/NOTICE/COPYING found only LICENSE files).
- Different licences (each opened): `python/sglang/kernels/jit/csrc/deep_select/vendor/LICENSE` is MIT, "Copyright (c) 2025 DeepSeek"; `python/sglang/kernels/ops/attention/flash_attn/cute/LICENSE` is BSD 3-Clause, "Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file"; `docs/LICENSE` and `python/sglang/kernels/aot/LICENSE` are Apache-2.0 texts. The `fast-hadamard-transform` directory under `python/sglang/kernels/jit/csrc/` has no LICENSE file of its own (its upstream licence is unverified).
- Obligations (Apache-2.0 section 4, `LICENSE`): include the licence, keep copyright and attribution notices in any copied source, mark changes, and carry NOTICE text if there is one (none). Apache-2.0 into AGPL-3.0-only is allowed. BSD-3 and MIT folders need their own notices only if copied. A clean Rust rewrite of an algorithm copies no code; cite the source in the module doc.

## 2. What it is
SGLang is a Python serving framework with a frontend language for LLM programs (`python/sglang/lang/`) and a runtime ("srt") built on a radix-tree prefix cache and constrained decoding backends. It runs on PyTorch with a large JIT/AOT kernel tree (`python/sglang/kernels/`). Unlike vLLM, it already ships a Rust crate for the radix tree core (`rust/sglang-radix-tree/`), though that crate is built as a Python extension tied to libtorch.

## 3. Components
| Component | What it does | Main paths opened |
| :--- | :--- | :--- |
| Radix cache | Token-keyed radix tree over KV slots; `match_prefix`, `insert`, `evict`, lock refs, node split | `python/sglang/srt/mem_cache/radix_cache.py` (839 lines), `evict_policy.py` (listed only) |
| Rust tree core | Same idea in Rust: nodes, LRU list, full/SWA/Mamba components | `rust/sglang-radix-tree/src/node.rs`, `unified_tree_core.rs`, `unified_lru_list.rs`, `components/full.rs`, `Cargo.toml`, `README.md` |
| Memory layers | Allocator, device pool, host pool, storage tiers | `python/sglang/srt/mem_cache/README.md` |
| Schedule policy | Cache-aware ordering of the waiting queue (`lpm`, `dfs-weight`) plus cache-agnostic ones | `python/sglang/srt/managers/schedule_policy.py` (`CacheAwarePolicy`, `BOUNDED_LPM_WINDOW_SIZE`) |
| Constrained generation | Grammar object per request, vocab mask fill and apply, rollback; backends xgrammar, llguidance, outlines | `python/sglang/srt/constrained/base_grammar_backend.py`, `xgrammar_backend.py`, `llguidance_backend.py`, `outlines_backend.py`, `json_schema_validation.py` |
| Speculative | EAGLE and DFlash workers, plus a C++ ngram corpus and suffix automaton | `python/sglang/srt/speculative/`, `python/sglang/kernels/jit/csrc/ngram_corpus/` (`trie.cpp`, `suffix_automaton.cpp`) |
| Frontend language | Program IR and interpreter for multi-call LLM programs | `python/sglang/lang/ir.py`, `interpreter.py` (names only) |
| Kernels | JIT CUDA/CuTe: attention, gemm, moe, sampling, Hadamard | `python/sglang/kernels/jit/csrc/` listing |
| Router | Rust request router | `sgl-router/ARCHITECTURE.md` (header only) |

## 4. What THME lacks that SGLang has, ranked

### 4.1 RadixAttention prefix cache (rank 1)
- Why: a compressed-trie index of token sequences to KV slots gives automatic prefix reuse across requests with partial-edge splits and LRU eviction. For a measurement-driven engine it is attractive: hit rate and saved prefill tokens are directly measurable. Compared with vLLM's block-hash map (see vllm.md 4.1) a radix tree matches at token granularity and shares the structure for ordering the queue (`lpm`).
- Paths: `python/sglang/srt/mem_cache/radix_cache.py` (`RadixKey`, `TreeNode` with `lock_ref`, `hit_count`, `last_access_time`; `match_prefix`, `insert`, `evict`, `inc_lock_ref`, `dec_lock_ref`, `_split_node`); Rust version `rust/sglang-radix-tree/src/node.rs` (`Node`, `KeyNamespace` for extra-key and cache-salt isolation), `unified_lru_list.rs`.
- Translate: pure data-structure algorithm, so it can be rewritten in Rust directly (the crate already shows the shape). Nothing here is a kernel. Do not take the crate: its `Cargo.toml` pulls `tch =0.24.0` (libtorch), `pyo3`, `sha2`, `hashbrown`, edition 2024, and it is a `cdylib` for Python. THME would write a small std-only tree.
- Dependencies: a KV cache whose slots can be shared and indexed (same prerequisite as vLLM paging; either contiguous-per-slot or paged). Effort: M (estimate). Risk: medium (lock/evict bugs, sharing of int8/int4 scales between sequences, unverified how SGLang handles quantised KV).
- Test oracle: property test against a naive longest-common-prefix scan over all stored sequences (the tree must return the same match length); end-to-end, greedy decode with the cache on must equal the cache off.

### 4.2 Constrained (structured) generation (rank 2)
- Why: forcing output to match JSON schema, regex or grammar by masking logits. Small to integrate with THME's sampler chain (`src/inference/sampler.rs`), valuable for agents and for the MCP server, and cheap to verify.
- Paths: `python/sglang/srt/constrained/base_grammar_backend.py` (`BaseGrammarObject`: `accept_token`, `rollback`, `try_accept_tokens`, `is_terminated`, `allocate_vocab_mask`, `fill_vocab_mask`, `apply_vocab_mask`), `xgrammar_backend.py`, `llguidance_backend.py`, `outlines_backend.py`, `json_schema_validation.py`. Compare vllm.md 4.6.
- Translate: the per-token interface and mask application are trivial algorithms. The expensive part, compiling a grammar or schema to a token-level automaton, is done by external native libraries (xgrammar, llguidance, outlines; imports seen in `xgrammar_backend.py`). Mask application on GPU uses Triton/torch ops (`python/sglang/kernels/ops/grammar/token_filter_ops`, name from imports; unverified contents); in THME a CPU mask over the logit vector before sampling is enough at small vocab.
- Dependencies: choose one: write a Rust regex/JSON-schema to byte-DFA plus a tokenizer-trie walker (M to L), or bind an external Rust crate (dependency and licence audit; unverified which are suitable). Effort: M for regex/JSON subset, L for full grammars. Risk: medium (token boundaries versus grammar characters is the classic bug).
- Test oracle: every constrained output must validate against the schema with a standard validator, and for a one-token-legal grammar the output must be forced; with an all-allow mask output must equal unconstrained sampling.

### 4.3 Cache-aware scheduling policy (rank 3)
- Paths: `python/sglang/srt/managers/schedule_policy.py` (`CacheAwarePolicy` with `lpm` and `dfs-weight`, bounded LPM window of 64). Only meaningful after 4.1 and some batching.
- Translate: ordering heuristic, algorithm. Effort: S after 4.1. Risk: low. Oracle: same outputs, measured hit rate and prefill tokens.

### 4.4 ngram speculative corpus (rank 4)
- Paths: `python/sglang/kernels/jit/csrc/ngram_corpus/` (`trie.cpp`, `suffix_automaton.cpp`, `ngram_corpus.py` in `srt/speculative/cpp_ngram/`). Draft-free speculation from a trie/suffix automaton over earlier text. Algorithm, rewritable in Rust (S to M). Oracle: greedy output identical to plain decode. EAGLE/DFlash workers need trained draft heads and are not worth it here.

### 4.5 Frontend language (rank 5, low)
- `python/sglang/lang/ir.py` and `interpreter.py` express multi-call programs with shared prefixes. Not a THME need; the MCP server (`thme mcp`) already exposes the engine. Skip unless agents drive THME heavily.

## 5. What NOT to port
- The Python server stack: tokenizer manager, detokenizer, scheduler mixins for pipeline, data and expert parallelism, disaggregation (`python/sglang/srt/managers/`, `disaggregation/`, `eplb/`).
- Hybrid caches for Mamba, sliding-window, DeepSeek-specific pools (`python/sglang/srt/mem_cache/hybrid_cache/`, `deepseek_v4_*`, `mamba_*`): model-family specific.
- Hierarchical host/disk cache tiers (`pool_host`, `storage`, HiCache files): needs a server fleet story THME lacks.
- JIT kernel tree (`python/sglang/kernels/`): CUDA/CuTe/Triton bound to PyTorch.
- The radix crate itself (libtorch and pyo3 dependencies); port the idea only.
- The router (`sgl-router/`): multi-node routing, out of scope.
- EAGLE/DFlash draft machinery and the frontend DSL (see 4.4, 4.5).
