# ExLlamaV2 survey (for THME)

Read-only survey of the shallow clone at `<engines>\exllamav2`. Paths are relative to that folder. Every claim names a file that was opened; anything else is marked unverified. Effort sizes (S/M/L) are estimates, not measurements.

THME today (opened: `README.md`, `docs/subsystems.md`, `src/allocator/`, `src/engine/alloc.rs`, `src/inference/kv.rs`): per-tensor format choice by an exact multiple-choice DP over measured costs (`src/engine/alloc.rs`, "dynamic programming in O(items x options x capacity)"), a greedy solver and a sensitivity-threshold `ParetoAllocator` (`docs/subsystems.md` 5.3), a coupled DC-CCCP solver (`src/allocator/dc_cccp.rs`), GPTQ and AWQ transforms (`src/transform/gptq.rs`, `awq.rs`), per-group-32 int8/int4 KV (`src/inference/kv.rs`), Hadamard code in `src/engine/`.

## 1. Licence
- `LICENSE` first lines: "MIT License / Permission is hereby granted, free of charge, to any person obtaining a copy of this software ...".
- Copyright holder: the file contains no "Copyright (c)" line (grep for "copyright" matched only the standard "above copyright notice" and "AUTHORS OR COPYRIGHT HOLDERS" sentences). The holder is therefore not named in the file; `README.md` was grepped for "license" and had no match. Treat the holder as unverified.
- NOTICE file: none (only `LICENSE` found).
- Different licence in any directory: none found (the find for LICENSE/NOTICE/COPYING returned one file). Vendored code headers inside `exllamav2/exllamav2_ext/cuda/` were not individually checked (unverified).
- Obligations (MIT, `LICENSE`): keep the copyright and permission notice in copies or substantial portions. MIT into AGPL-3.0-only is allowed. Rewriting the algorithm in Rust and citing the source in the module doc is enough for ideas; copied code or data tables need the notice. Because no holder is named, ask upstream or credit "ExLlamaV2 authors" and flag it.

## 2. What it is
ExLlamaV2 is a Python plus CUDA inference library for 4-bit-and-below GPTQ-style quantized LLMs, centred on its own EXL2 format. EXL2 mixes bit widths inside a single weight matrix and across layers, hitting a target average bits per weight. The runtime adds a Q4/Q6/Q8 KV cache with Hadamard rotation, a dynamic generator with paged attention and deduplication, and a few samplers and filters.

## 3. Components
| Component | What it does | Main paths opened |
| :--- | :--- | :--- |
| Converter | Two passes: measure per-module error under many quant options, then pick options for a target bpw and quantize | `convert.py` (listing), `exllamav2/conversion/convert_exl2.py`, `measure.py`, `optimize.py`, `quantize.py`, `doc/convert.md` |
| Quant options | Per-matrix group sizes, bit widths and the fraction of rows at each width | `exllamav2/conversion/qparams.py` (`QParams`, `qparams_attn`, `qparams_mlp`) |
| GPTQ core | Hessian from calibration, act-order permutation by Hessian diagonal, damping 0.12 | `exllamav2/conversion/adaptivegptq.py` (`percdamp`, `argsort(diagonal, descending)`) |
| Annealer | Simulated annealing over per-module options under a bit budget | `exllamav2/exllamav2_ext/ext_quant.cpp` (`sim_anneal`) |
| Quantized matrix | Device layout: packed weights, permutation and inverse, scales, per-group bit info | `exllamav2/exllamav2_ext/cuda/q_matrix.cuh`, `q_gemm.cu`, `q_gemm_kernel.cuh` |
| KV cache | FP16, 8-bit, Q4/Q6/Q8 caches | `exllamav2/cache.py` (`ExLlamaV2Cache_Q4`, `_Q6`, `_Q8`), `exllamav2_ext/cuda/cache_q.cuh`, `doc/qcache_eval.md` |
| Dynamic generator | Paged attention via Flash Attention, continuous batching, prompt deduplication, ngram draft | `doc/dynamic.md`, `exllamav2/generator/dynamic.py`, `ngram.py` |
| Filters | Constrain tokens (prefix, select) | `exllamav2/generator/filters/` (`prefix.py`, `select.py`, `base.py`) |
| Hadamard tables | Precomputed Hadamard matrices of odd sizes | `exllamav2/hadamard/` |

## 4. What THME lacks that ExLlamaV2 has, ranked

### 4.1 Mixed precision inside one tensor (EXL2 row-group allocation) (rank 1)
- What EXL2 does (from `qparams.py` and `adaptivegptq.py`): each matrix is cut along its input dimension into row groups; a `QParams` lists bit widths and the proportion of rows at each (for example `[6, 3, 2]` with `[0.05, 0.2, 0.75]` for an MLP down-projection, group sizes `[32, 64, 64]`, scale bits 4). Rows are sorted by the GPTQ Hessian diagonal (act-order, `argsort(diagonal, descending=True)`), so the most activation-heavy rows get the most bits. The kernel reads a permutation and per-group bit counts (`q_perm`, `q_invperm`, `q_groups`, `q_scale`, `q_scale_max` in `q_matrix.cuh`).
- Why it matters for THME: THME's allocator chooses a format per tensor (README: "every tensor can have its own representation"). Finer granularity inside a tensor is the next resolution step, and the cost is measurable with THME's own perplexity gate. Whether THME already has any intra-tensor mixing: README table lists none; the `.epiph` placement code (`src/epiph/placement.rs`) was not opened for this, so unverified.
- Translate: format spec, row permutation and bit packing are algorithms (rewrite in Rust, S to M). The decode GEMV is a kernel: THME has its own per-format GEMV shaders (`gemv_q4_k.wgsl` etc. in `src/inference/gpu/shaders/`) and PTX decoder; a new variable-bit GEMV is a new kernel on two backends (L).
- Dependencies: none external. Effort: L overall (M for a CPU-only reference and exporter, L with CUDA and wgpu kernels; estimate). Risk: medium-high (kernel work, GGUF cannot carry it, so it fits `.epiph` only; unverified whether `.epiph` can hold it).
- Test oracle: dequantize-then-matmul in f32 (THME CPU reference) must match the new GEMV within a stated tolerance; round-trip of pack/unpack; perplexity against the uniform format at equal bpw.

### 4.2 Measurement-driven option selection: ideas to compare with THME's solvers (rank 2)
- EXL2 method (opened): `measure.py` quantizes each attention and MLP module under several option sets and records, per option, `total_bits` and `accuracy`, where accuracy = 1 minus the mean relative Frobenius error of the module output against the unquantized module on calibration hidden states (`test_error`). `optimize.py` builds one slot per module (cost = bits, energy = (1 - accuracy)), then runs `sim_anneal` for several sample rounds with an error exponent `norm` swept in 1.5 to 3.5, refines the exponent around the best, and finally greedily spends leftover budget. `measurement.json` can be reused for many target bitrates (`doc/convert.md`).
- Comparison: THME's DP is exact for its gridded problem (`src/engine/alloc.rs`; test `equals_exhaustive_search`), so EXL2's annealer is weaker as an optimiser. The annealer is stochastic (std::random_device seed in `ext_quant.cpp`), so results are not reproducible, which conflicts with THME's measurement-driven stance. In `ext_quant.cpp` the check `delta_cost < 0` is applied to an unsigned `uint64_t`; this looks like a latent bug but its effect was not tested (unverified).
- What to take: (a) the per-module output relative error as a cheap local cost proxy that needs no full perplexity run, to feed THME's DP; (b) the reusable measurement file idea (THME's recipes and `--report` already point that way); (c) the error-norm exponent as an objective shape (sum of err^p) if THME wants to punish outlier layers, which the DP could take as costs. All algorithms, rewritable in Rust (S). Effort S to M. Risk low.
- Test oracle: THME's exhaustive-search oracle on small instances; compare chosen plans and measured perplexity against the existing `thme search --solver dp`.

### 4.3 Hadamard-rotated Q4 KV cache (rank 3)
- Paths: `exllamav2/cache.py` (`ExLlamaV2Cache_Q4`), `doc/qcache_eval.md` (says the new method uses Hadamard rotations on keys and values, and shows Q4 close to FP16 on the Pile for the listed models; these are upstream's numbers, not THME's). THME already has int4 per-group-32 KV (`src/inference/kv.rs`) and a shared Walsh-Hadamard transform (`src/engine`). The gap is rotating before quantizing, which vLLM also does (see vllm.md 4.5).
- Translate: algorithm only (S). The GPU cache kernels (`cache_q.cuh`) are CUDA and not needed. Effort S to M. Risk low. Oracle: perplexity on WikiText-2 with int4 KV, rotated versus not, on THME's gate; round-trip error bound test.

### 4.4 Dynamic generator: paged attention plus deduplication (rank 4)
- Paths: `doc/dynamic.md` (sections on continuous batching, deduplication of shared prefixes, page reuse), `exllamav2/generator/dynamic.py`. It relies on Flash Attention paged kernels (`doc/dynamic.md`), a CUDA dependency. Same algorithm class as vLLM and SGLang (see their notes); lower priority because those two have fuller code. Effort L. Risk medium.

### 4.5 ngram draft and token filters (rank 5)
- `exllamav2/generator/ngram.py` (count-based next-token table with a threshold) and `generator/filters/` (`prefix.py`, `select.py`). Both are small Python algorithms, rewritable in Rust (S each). Oracles: greedy output unchanged for ngram; allowed-token set equality for filters.

## 5. What NOT to port
- CUDA GEMM kernels (`exllamav2_ext/cuda/q_gemm*.cuh`, `q_gemm_autotune.cuh`): written for EXL2's exact layout and NVIDIA only; THME needs wgpu and PTX parity, so write new kernels from THME's own spec.
- Simulated annealing as an optimiser: THME's exact DP is better and reproducible.
- GPTQ adaptive code as is (`adaptivegptq.py`): THME already has `src/transform/gptq.rs`; only borrow the act-order grouping idea if 4.1 is built.
- Tensor-parallel, LoRA, VLM, server code (`exllamav2/tensor_p.py`, `lora.py`, `vlm/`, `server/`): out of scope.
- The EXL2 file format as an export target: it is tied to the upstream library; use `.epiph` for any new layout, and keep GGUF export as is.
