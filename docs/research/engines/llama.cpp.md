# Survey: llama.cpp (clone at <engines>\llama.cpp)

Read-only survey, 2026-11-27. Only paths I opened or listed are cited. "Unverified" = not opened.
Effort S/M/L is my estimate, not measured. No source is copied here.

## 1. Licence
- `LICENSE` opens with "MIT License" then "Copyright (c) 2023-2026 The ggml authors" (read, lines 1-3).
- No `NOTICE*` file in the repo root (ls returned nothing). `licenses/LICENSE-jsonhpp` exists for
  vendored nlohmann json; `vendor/` holds cpp-httplib, miniaudio, stb, sheredom, hash, nlohmann
  (listed; their licences unverified).
- Obligation (standard MIT text): keep the copyright line and permission notice in copies or
  substantial portions. THME already does this: `LICENSES/llama.cpp-MIT.txt` exists and
  `src/inference/sampler.rs` carries a "Derived from llama.cpp" header with SPDX `AGPL-3.0-only AND MIT`.
  Repeat that header per ported file. MIT code inside an AGPL-3.0-only work is a one-way fit; not legal advice.
- Do not port from `vendor/` or from `ggml-*` backends without checking each file's own header.

## 2. What it is
llama.cpp is a C/C++ inference engine for GGUF models on CPU and many GPU backends (CUDA, Metal,
Vulkan, SYCL, HIP, CANN, OpenCL, WebGPU, RPC and more under `ggml/src/`). The tensor library `ggml/`
holds the quantization codecs, compute graph and kernels; `src/` holds model loading, per-architecture
graph builders, KV memory, vocab, sampler and grammar; `tools/` holds quantize, imatrix, perplexity,
server and bench. It is the de-facto reference for GGUF, so it is the oracle for THME's exporter.

## 3. Components
| Component | What it does | Paths opened |
| :-- | :-- | :-- |
| ggml quantization | block codecs; encoders take optional importance weights | `ggml/src/ggml-quants.c` (5772 lines), `ggml/src/ggml-quants.h`, `ggml/include/ggml.h` (type enum) |
| Quantize driver | per-tensor type policy by ftype | `src/llama-quant.cpp` (3885 lines), `tools/quantize/` (listed) |
| imatrix | collects activation stats over calibration text | `tools/imatrix/imatrix.cpp`, `tools/imatrix/README.md`; `common/imatrix-loader.cpp` (listed) |
| Perplexity and KL | ppl and KL divergence vs saved base logits | `tools/perplexity/perplexity.cpp` |
| Grammar | GBNF parser and stack matcher | `src/llama-grammar.cpp`, `grammars/README.md`, `grammars/*.gbnf` (listed); `common/json-schema-to-grammar.cpp` (listed) |
| Sampler chain | 14+ samplers | `src/llama-sampler.cpp` (4381 lines), `include/llama.h`, `common/sampling.cpp` |
| Speculative | draft, EAGLE3, MTP, n-gram self-speculation | `common/speculative.cpp`, `common/common.h`, `docs/speculative.md` |
| Attention and RoPE | flash attn op, YaRN, NEOX, MROPE | `ggml/include/ggml.h`, `ggml/src/ggml-cpu/ops.cpp`; `ggml/src/ggml-cuda/fattn*` (listed) |
| KV memory | unified, SWA, recurrent, hybrid caches | `src/llama-kv-cache*.cpp` (listed), `common/arg.cpp` (cache types) |
| Chat templates | hard-coded matcher plus a Jinja engine | `src/llama-chat.h`, `common/jinja/` (listed), `src/llama-chat.cpp` (957 lines) |
| Architectures | one graph file per model | `src/models/` (190 files listed); `src/llama-arch.cpp` (listed) |
| Server | HTTP, slots, queue, router | `tools/server/README.md`, `tools/server/` listing (server-context.cpp 7298 lines) |
| Tests | the oracles | `tests/` listing: `test-quantize-fns`, `test-sampling`, `test-rope`, `test-grammar-*`, `test-chat-template`, `test-backend-ops`, `test-tokenizer-*` |

## What THME already has (opened)
- GGUF parse of the ggml type list incl. IQ family: `src/parser/gguf.rs`. Encoders Q4_K, Q5_K, Q6_K and decoder IQ4_XS: `src/exporter/kquant.rs`. Q2_K fake-quantize is refused (`src/inference/weights.rs`, test near line 433).
- Sampler: temperature, top-k, top-p, min-p, Mirostat v1/v2 only, derived from llama.cpp: `src/inference/sampler.rs`.
- KV cache f32/int8/int4 with group scales: `src/inference/kv.rs`; also `src/inference/polar_kv.rs`.
- RoPE: forward reads only `rope_theta` (`src/inference/llama.rs`); scaling keys are only written into GGUF metadata (`src/exporter/metadata.rs` near line 160). No YaRN, linear or NTK at run time (grep of `rope_scaling|yarn` in `src/` found nothing else relevant).
- Chat templates via minijinja (`Cargo.toml` lines 110-113, `src/cli/chat_template.rs`).
- Speculative decoding is a model/simulation only (`src/speculative/engine.rs`: `simulate_speculative_run`).
- Architectures: `ModelArchitecture` = Llama, Mistral, Qwen, DeepSeek, Gemma, Unknown (`src/ir/model.rs`); forwards in `llama.rs`, `neox.rs`, `gemma.rs`.
- GPU: wgpu shaders incl. `gemv_q4_k.wgsl`, `gemv_iq4_xs.wgsl`, `attention_tiled.wgsl`, plus CUDA PTX (`src/inference/gpu/`).
- imatrix: no code in `src/`, only `docs/research/calibration-best-practice.md`. No GBNF code found. No HTTP server module in the `src/` listing.

## 4. What THME lacks, ranked (value for a small, quantization-focused, measurement-driven engine)
**1. imatrix collection and use in encoders. Effort S-M.**
- Why: importance weighting is what makes 2-3 bit formats usable; the file format is the interchange standard. `ggml_quantize_requires_imatrix` lists IQ2_XXS, IQ2_XS, IQ1_S as mandatory (`ggml/src/ggml.c` near 8235).
- Paths: `tools/imatrix/imatrix.cpp`, README (GGUF and legacy `.dat` output, `--in-file` merge); `quant_weights` argument of encoders, e.g. `quantize_row_q4_K_impl` (`ggml-quants.c` line 1563).
- Translate: per-column sum-of-squares accumulator (overlap with `src/inference/calibration.rs` unverified), imatrix reader/writer, weights into `kquant.rs` encoders.
- Drags in: nothing new. Risk: low.
- Test: `llama-imatrix -m M.gguf -f text -o ref.gguf`; compare THME values per tensor to it.
**2. Missing K-quant and legacy encoders: Q2_K, Q3_K, Q8_K, Q4_1, Q5_0, Q5_1. Effort S each.**
- Why: closes the format ladder. Paths: `quantize_row_q2_K_impl` (1159), `quantize_row_q3_K_impl` (1365), `q4_1` (2153), `q5_0` (2198), `q5_1` (2252) in `ggml-quants.c`; type mix policy in `src/llama-quant.cpp` (`llama_tensor_get_type_impl` line 517, `use_more_bits` line 522), a good baseline recipe to compare THME search against.
- Translate: scale-search loops and block layout. Drags in: none. Risk: low (bit-exact target).
- Test: `llama-quantize` with and without `--imatrix`, byte-compare blocks; thresholds in `tests/test-quantize-fns.cpp`.
**3. I-quant encoders: IQ4_NL, IQ4_XS first, then IQ3_*, IQ2_*, IQ1_*. Effort M (IQ4) to L (grid types).**
- Why: best quality under 4 bits; THME decodes IQ4_XS only. Paths: `quantize_row_iq4_nl_impl` (5050), `iq3xs_init_impl` (3816), `iq2xs_init_impl` (2869), `quantize_row_iq1_s_impl` (4621), `iq2_s` (5227).
- Translate: neighbour-search encoders and the grid tables (tables live in `ggml/src/ggml-common.h`, not opened).
- Drags in: large constant tables, runtime grid init. Risk: medium (tie-break details).
- Test: `llama-quantize --imatrix` byte comparison on the same F16 GGUF.
**4. RoPE scaling at run time: linear, YaRN, longrope. Effort S-M.**
- Why: models with `rope_scaling` run wrong past base context today; cheap correctness fix. Paths: `rope_yarn`, `rope_yarn_ramp` (`ggml/src/ggml-cpu/ops.cpp` 6311-6340), `LLAMA_ROPE_SCALING_TYPE_*` and `yarn_*` fields (`include/llama.h` 170-175, 397-401).
- Translate: frequency table, ramp mix, attention magnitude factor into `llama.rs` and the GPU kernels. Drags in: none. Risk: low-medium (CPU/GPU parity).
- Test: `tests/test-rope.cpp`; logits from `llama-perplexity` or `llama-cli` past the original context.
**5. Sampler gaps: typical, XTC, top-n-sigma, DRY, repetition penalties, logit bias, temp_ext, adaptive-p. Effort S each.**
- Why: same file, same method as the existing port. Paths: `include/llama.h` 1516-1633; `src/llama-sampler.cpp`; chain order `common/sampling.cpp`. Risk: low.
- Test: `tests/test-sampling.cpp` fixed inputs as reference.
**6. GBNF grammar-constrained decoding. Effort M.**
- Why: structured output for agents/tools. Paths: `src/llama-grammar.cpp`, `grammars/README.md`, `grammars/json.gbnf`; JSON-schema front end `common/json-schema-to-grammar.cpp` is a separate, later add-on.
- Translate: GBNF parser, rule stacks, per-token candidate rejection (needs vocab piece strings, UTF-8 partials). Drags in: tokenizer piece access. Risk: medium (speed on large vocab).
- Test: `tests/test-grammar-parser.cpp`, `test-llama-grammar.cpp`, `test-grammar-integration.cpp`.
**7. Real speculative decoding. Effort M-L.**
- Why: THME only models it. Cheapest real variant needs no second model: n-gram self-speculation (`COMMON_SPECULATIVE_TYPE_NGRAM_*`, `common/common.h` 177-190). Draft-model variant needs two loaded models. EAGLE3/MTP/DFlash need trained heads (`docs/speculative.md`).
- Translate: propose, batched verify, KV rollback, acceptance rule. Prereq: batched multi-token forward and KV truncate (unverified in `src/inference/kv.rs`). Risk: medium.
- Test: output must equal plain greedy decode; sampling case via `tests/test-speculative-rejection-sampling.cpp`.
**8. KV cache types: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, BF16, F8_E4M3. Effort S-M.**
- Why: the baseline for THME's KV measurement story (`common/arg.cpp` lines 281-292). Reuse THME codecs where formats match. Risk: low.
- Test: `llama-perplexity` with `--cache-type-k/--cache-type-v` vs `src/quality/kv_perplexity.rs`; its KL mode (`tools/perplexity/perplexity.cpp` line 175).
**9. Flash attention variants. Effort L, low priority.** `GGML_OP_FLASH_ATTN_EXT` (`ggml.h` 400), CUDA `fattn-*` files (listed). THME already has tiled WGSL attention; take ideas, not kernels. Test: `tests/test-backend-ops.cpp`.
**10. More architectures. Effort M each.** 190 graph files in `src/models/`. Pick by demand (Phi, Gemma2/3, Qwen3, MoE, Mamba, RWKV); each needs tensor map and graph; MoE adds routing. Test: logits vs `llama-perplexity` on the same GGUF.
**11. Chat template fallback list. Effort S.** minijinja covers HF templates; `src/llama-chat.cpp` has a hard-coded matcher for GGUFs without a usable template (57 `LLM_CHAT_TEMPLATE_` lines in `src/llama-chat.h`). Port only if needed. Test: `tests/test-chat-template.cpp`.
**12. Server and batching. Effort L.** `tools/server/` slots, continuous batching, prompt cache, router. See section 5.

## 5. What NOT to port
- ggml graph runtime, backend registry and the 10+ GPU backends (`ggml/src/ggml-*`): scale and C++ build; THME has wgpu and CUDA paths.
- `tools/server/` whole (HTTP, MCP, realtime, tools): large, drags in cpp-httplib and nlohmann. A thin local endpoint is a separate decision.
- Multimodal (`tools/mtmd`), TTS, speech, vision models in `src/models/`: outside a text quantization workbench.
- `common/llguidance.cpp` (external lib hook) and `common/jinja/` (THME has minijinja).
- `convert_hf_to_gguf.py`: THME has `src/exporter/` and `src/convert/`; use the script as an oracle only.
- Anything under `vendor/`: separate licences, unverified.
- EAGLE3, MTP, DFlash drafters: no trained heads in THME to measure with.
