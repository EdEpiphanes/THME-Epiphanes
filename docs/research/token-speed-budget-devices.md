# Raising tokens per second on budget devices: what a format plus an engine can do

Status: research note, 2026-11-20. Question: for people on CPU-only, small-GPU or integrated-GPU
machines (for example a Mixture-of-Experts model at about 5 tokens/s), what raises decode and prefill
speed, and which part can the `.epiph` format plus THME's engine do?

Rules for this note:

- Every external claim has a URL. "Not verified" means only a secondary source or an abstract was read.
- Most papers were read as abstracts through a summarising fetcher. Hardware is often missing from the
  abstract; where it is, the table says "hardware not stated in abstract".
- All external numbers are **reported, not measured by THME**. THME has measured no speed (see
  `THME_measured_findings_2026-11.md`, H6: "No speed or energy measurement exists"). The only THME
  timing mentioned is the one-run GPU vs CPU figure from the task brief; it is a single run, not a result.
- "Arithmetic" lines are computed from stated inputs here, not measured.
- Licence column is the licence of the reference implementation. THME policy
  (`third-party-code-policy`): permissive only, ideas not code, never read GPL/AGPL/unlicensed code.

## 1. Why decode is memory-bandwidth bound

Decode produces one token at a time and uses each weight once, so speed is set by bytes moved, not by
arithmetic. Arm's roofline guide states it: `decode tokens/s = effective memory bandwidth / bytes read
per token`, and gives a worked example: Raspberry Pi 5 at 13.8 GB/s sustained and a 346 MB model gives a
ceiling of about 40 tokens/s. It also says halving bytes per token roughly doubles decode speed, and that
extra threads past memory saturation do not help and can hurt, while prefill is compute-bound and keeps
scaling with threads
([Arm learning path](https://learn.arm.com/learning-paths/cross-platform/llm-decode-roofline/how-decode-uses-memory/)).

Arithmetic for THME's own models (illustrative; bandwidth is an assumed round number, not a measured
machine):

| Model | Weights read per token | Ceiling at 40 GB/s (assumed) |
| :--- | :--- | :--- |
| SmolLM-135M, F32 (what THME's forward pass reads today, about 135M x 4 B) | about 540 MB | about 74 tokens/s |
| same, 4.5 bits/weight (Q4_0 class) | about 76 MB | about 520 tokens/s |

For MoE, bytes per token are the **active** experts plus attention and embeddings, not the whole file. A
5 tokens/s MoE on a budget box is usually not limited by the active-parameter arithmetic above but by
where the experts live: if they do not fit RAM or VRAM they are fetched from a slower tier every token
(PCIe or SSD), and that tier's bandwidth becomes the denominator. Expert reuse across tokens is the lever
(section 2, MoE rows).

Other bandwidth facts: dual-channel RAM gave about 55 percent faster generation than single-channel at 4
threads on one i7-6700T (one blog, secondary, not verified)
([InsiderLLM](https://insiderllm.com/guides/dual-channel-ram-local-llm-cpu-inference/)); a thread sweep
on an i7-8700 peaked near the physical core count and 12 threads ran slower than 8 (one blog, secondary)
([InventiveHQ](https://inventivehq.com/blog/cpu-thread-scaling-llm-inference)). Treat both as direction,
not numbers.

## 2. Technique table

Gains are as the sources report them, on the stated hardware. "Format" = needs support in the `.epiph`
file. "Engine" = needs support in `src/inference/`.

### 2a. Dense and general

| Technique | Typical reported gain (reported, not measured by THME) | Format? | Engine? | Reference licence |
| :--- | :--- | :--- | :--- | :--- |
| Keep weights quantized in memory, multiply in the packed domain (no F32 dequant) | Follows from section 1: bytes per token fall about 4x for 4-bit vs F32. No single source number. | Already stores quant types | Yes, new kernels | ggml/llama.cpp: MIT ([LICENSE](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE)) |
| Dot-product / i8mm integer kernels (ARM `sdot`, `smmla`; x86 AVX2/AVX-512 VNNI) | Arm i8mm Q6_K/Q4_K kernels: "big performance uplift" on Neoverse-N2, numbers only in a chart, not in text. Layout detail: `smmla` computes a 2x2 tile from two interleaved rows ([Arm blog](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction)) | Helps if rows are pre-interleaved | Yes | MIT |
| Runtime repack / interleaved layouts (Q4_0_4_4 family, "online repack") | PR author: Graviton 3, phi-2 Q4_0, 4 threads: prefill 38.0 to 110.8 t/s (2.9x), decode 16.95 to 39.4 t/s (2.3x) ([PR 9921](https://github.com/ggml-org/llama.cpp/pull/9921)). **Mixed on x86 in the same thread**: one 13900K run, 7B Q4_0: prefill 50.6 to 64.8 t/s but decode 20.8 to 15.8 t/s. Q4_0_N_M files are deprecated in favour of repacking plain Q4_0 at load ([HF post, secondary](https://huggingface.co/posts/bartowski/807894839859408)) | **Yes, best fit**: store the packed layout, skip load-time repack | Yes, kernel must match layout | MIT |
| ik_llama.cpp interleaved quants (`_R4/_R8/_R16`), IQK kernels | README gives relative claims only (for example much faster CPU prompt processing for non-interleaved quants; MoE decode up to about 5x on CPU in one case), no CPU model named; wiki has comparisons, not read ([README](https://github.com/ikawrakow/ik_llama.cpp)) | Yes (layout) | Yes | MIT |
| LUT-based low-bit kernels (T-MAC) | Up to 4x throughput vs llama.cpp, 70 percent less energy. BitNet-3B: 30 t/s on 1 M2-Ultra core, 71 t/s on 8 cores, 11 t/s on Raspberry Pi 5 ([arXiv 2407.00088](https://arxiv.org/abs/2407.00088)). Repo: Surface Laptop 7 3B BitNet about 20 t/s 1 core, 48 t/s 4 cores ([T-MAC](https://github.com/microsoft/T-MAC)) | Yes: weights permuted offline for the LUT kernel (offline step implied by the pipeline, not explicit on the page) | Yes, large | MIT |
| Ternary/1.58-bit kernels (bitnet.cpp: I2_S, TL1, TL2) | 2.37x-6.17x on x86, 1.37x-5.07x on ARM vs llama.cpp, 55-82 percent less energy ([arXiv 2410.16144](https://arxiv.org/abs/2410.16144), [repo](https://github.com/microsoft/BitNet)). Only for models trained as BitNet. 100B model "5-7 tokens/s on a single CPU" is the repo's claim; hardware not stated (not verified) | Yes | Yes | MIT |
| Tuned CPU GEMM (llamafile) | Page fetched had no numbers; claim not verified. ([repo](https://github.com/Mozilla-Ocho/llamafile)) | No | Yes | Apache-2.0 (llamafile), MIT (llama.cpp changes) |
| Thread count = physical cores; separate counts for prefill and decode | Direction only (see section 1 blogs). | Per-device hint possible | Yes, easy | n/a |
| Prefill batching / chunking (`-b` 2048, `-ub` 512 defaults) | No gain number found. Prefill is compute-bound so batching raises arithmetic intensity ([llama-bench README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md), [Arm](https://learn.arm.com/learning-paths/cross-platform/llm-decode-roofline/how-decode-uses-memory/)) | No | Yes | MIT |
| KV cache itself (incremental decode) | Not an "optimisation" in any source; every engine above assumes it. Without it each token reruns the whole sequence. | No | **Yes, critical for THME** | n/a |
| KV-cache quantization (KIVI 2-bit) | 2.6x less peak memory incl. weights, up to 4x batch, 2.35x-3.47x throughput; GPU serving setting ([arXiv 2402.02750](https://arxiv.org/abs/2402.02750)). On CPU at batch 1 the win is memory, not speed, except at long context. | No | Yes | Repo exists (jy-yuan/KIVI); licence not verified |
| mmap with page-aligned tensors | Mechanism: first token pays page faults; later tokens reuse the page cache. Cold vs warm runs differ a lot, and mmap vs no-mmap results vary by platform (blogs, secondary, not verified: [omr.it.com](https://omr.it.com/blog/memory-mapped-model-weights-ios-llm-loading/), [NVIDIA forum](https://forums.developer.nvidia.com/t/apparently-mmap-is-still-slow-on-dgx-spark-on-linux-6-17/360538)) | **Yes**: aligned, contiguous, kernel-ready payloads (THME `ALIGN` is 64 B; page alignment is a possible change) | Yes (no copy into F32) | n/a |
| NUMA-aware placement | Mentioned by KTransformers as part of its CPU MoE design; no isolated number ([repo](https://github.com/kvcache-ai/ktransformers)). Relevant only on multi-socket servers, not budget boxes. | No | Yes | n/a |

### 2b. Speculative decoding

| Technique | Typical reported gain | Format? | Engine? | Reference licence |
| :--- | :--- | :--- | :--- | :--- |
| Draft-model speculative decoding | 2X-3X on T5-XXL, output distribution unchanged; hardware not in abstract ([arXiv 2211.17192](https://arxiv.org/abs/2211.17192)) | A draft model could ship inside the file | Yes | paper only |
| Prompt-lookup / n-gram (draftless) | 2x-4x claimed on input-grounded tasks; about 2.4x on summarization and context QA; small on role-play; Mistral-7B, one A100 40GB ([repo](https://github.com/apoorvumang/prompt-lookup-decoding)). llama.cpp ships several n-gram types and says they suit code rewriting, summarization and repeated reasoning ([docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)) | No | Yes, small | repo licence not stated on page; llama.cpp MIT |
| EAGLE / Medusa (trained heads) | EAGLE 2.7x-3.5x latency on LLaMA2-Chat 70B ([arXiv 2401.15077](https://arxiv.org/abs/2401.15077)); Medusa over 2.2x (Medusa-1), 2.3x-3.6x (Medusa-2) ([arXiv 2401.10774](https://arxiv.org/abs/2401.10774)). llama.cpp lists `draft-eagle3` and `draft-mtp` types ([docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)) | **Yes**: heads stored in the file | Yes | licences not verified |

When speculation hurts (reasoning, partly not verified): verification runs several tokens through the
target in one pass. That is nearly free only while the pass stays memory-bound. On a CPU the batched pass
becomes compute-bound sooner, and for MoE a batch of draft tokens usually touches **more distinct
experts**, so the weights read per step grow instead of staying flat. llama.cpp's own page says MoE
n-gram drafting needs long drafts and does not discuss slowdowns
([docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)). No primary measurement
of "speculation slower than plain decode on CPU" was found: not verified. Test before enabling.

### 2c. MoE specific

| Technique | Typical reported gain | Format? | Engine? | Reference licence |
| :--- | :--- | :--- | :--- | :--- |
| Dense/attention on GPU, experts in CPU RAM (`--cpu-moe`, `--n-cpu-moe N`, `--override-tensor`) | PR adds the flags only; one user report about 108 t/s for gpt-oss-120b on two 5090 with `--n-cpu-moe 3` (prompt or generation not stated) ([PR 15077](https://github.com/ggml-org/llama.cpp/pull/15077)) | Placement hints help | Yes | MIT |
| CPU computes cold experts, GPU computes hot (Fiddler) | 1.26x single-batch, 1.30x long prefill, 11.57x beam search vs baselines; hardware not in abstract ([arXiv 2402.07033](https://arxiv.org/abs/2402.07033)) | Hot-expert list | Yes | not verified |
| Mixed quantization + LRU expert cache + speculative prefetch (Mixtral-offloading) | Targets Mixtral-8x7B on desktop and free Colab; no numbers in abstract ([arXiv 2312.17238](https://arxiv.org/abs/2312.17238)) | Per-expert precision | Yes | not verified |
| Activation-aware expert cache and prefetch (MoE-Infinity) | 3.1-16.7x per-token latency vs vLLM, Ollama, DeepSpeed, BrainStorm; single-user batch 1; hardware not in abstract ([arXiv 2401.14361](https://arxiv.org/abs/2401.14361)) | **Yes**: popularity statistics can ship in the file | Yes | not verified |
| Predictive prefetch (ProMoE) | Prefill 2.20x avg (up to 3.21x), decode 2.07x avg (up to 5.02x) vs offloading baselines; RTX 4090 24 GB per a search summary (secondary) ([arXiv 2410.22134](https://arxiv.org/abs/2410.22134)) | Predictor weights | Yes | not verified |
| Mixed-precision on cache miss (HOBBIT) | Up to 9.93x decode vs MoE offloading systems; built on llama.cpp; "several edge devices" not named ([arXiv 2411.01433](https://arxiv.org/abs/2411.01433)) | **Yes**: low-precision copy of each expert | Yes | not verified |
| CPU experts compute while GPU cache fills (2025) | Up to 4.8 t/s Mixtral-8x7B, 10.4 t/s Phi-3.5-MoE; about 1.6x over Fiddler, up to 4.4x over Pre-gated MoE; Threadripper 7960X + RTX 4090 over PCIe 4.0 x16 ([arXiv 2512.16473](https://arxiv.org/html/2512.16473v1)). Its profile says PCIe transfer costs about 28 ms per expert, so CPU compute can beat fetching. | Hot-expert list | Yes | not verified |
| Substitute a missing expert with a cached similar one (2025) | 48 percent lower decode latency, over 60 percent cache hit, "nearly lossless" (author claim) ([arXiv 2508.18983](https://arxiv.org/html/2508.18983v1)) | Similarity table | Yes, changes outputs | not verified |
| Hot neurons on GPU, cold on CPU (PowerInfer, dense models) | Up to 11.69x vs llama.cpp on one RTX 4090 ([arXiv 2312.12456](https://arxiv.org/abs/2312.12456)). Needs ReLU-style sparsity; weak on SwiGLU models (my reading, not verified) | Predictors | Yes, large | MIT (not verified this session) |
| Expert cache in mainline llama.cpp | 14.95 to 19.26 t/s (1.29x) with `--moe-cache-mib 8000`, one 4090 benchmark, one secondary article (not verified) ([source](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)) | No | Yes | MIT |
| Stream experts from SSD | Flash-MoE: Qwen3.5-397B at about 4.4-5.5 t/s on a 48 GB Mac, custom Metal engine (secondary, not verified) ([InsiderLLM](https://insiderllm.com/guides/ssd-streaming-moe-engines/)). Needs fixed expert records laid out for sequential reads. | **Yes** | Yes | not verified |
| Whole-stack CPU+GPU MoE (KTransformers) | Up to 16 t/s DeepSeek-R1/V3 on one 24 GB GPU plus 382 GB DRAM (Feb 2025 note); AMX/AVX512 INT4/INT8 kernels ([repo](https://github.com/kvcache-ai/ktransformers)) | Yes (pre-packed experts) | Yes | Apache-2.0 |
| ik_llama.cpp MoE work | CPU-resident experts cut CPU-to-GPU copies "about 9x"; dynamic expert cache. README relative claims only ([repo](https://github.com/ikawrakow/ik_llama.cpp)) | Yes | Yes | MIT |

Not read or not verified this session: DeepSpeed-MoE inference, newer 2026 work beyond the items above.

## 3. What a file format can precompute

Ranked by how directly each removes work at load or run time:

1. **Pre-repacked tensors in the exact kernel layout** (interleaved rows, LUT-permuted weights). Removes
   load-time repack (which costs RAM and time) and makes the layout the engine's contract. Cost: the file
   is tied to a kernel family, so store the layout id and keep a canonical copy path.
2. **Alignment and contiguity for mmap**: each tensor starts on a boundary the kernel wants (THME is
   already 64 B aligned, `src/epiph/mod.rs` `ALIGN`), and all experts of a layer are contiguous with a
   fixed record size so SSD reads are sequential.
3. **Expert placement hints and per-expert popularity** collected on calibration text (hot list per layer,
   and co-activation counts). `.epiph` already carries calibration statistics as an attachment, so the
   slot exists. MoE-Infinity-style caches use exactly this kind of trace (see table).
4. **Per-expert precision tiers** (a 2/3-bit copy for HOBBIT-style miss handling, a higher-bit copy for
   hot experts).
5. **Embedded draft**: n-gram table, EAGLE/MTP head, or a small draft model.
6. **Tokenizer and chat template** (already planned in v0.3 work).
7. **Per-device tuning record** (thread counts, ubatch) written by `thme plan-run`
   (see `engine-autotuning.md`).

Caveat: anything in 1-5 is only worth storing after the engine has a kernel that reads it, and after a
measurement shows a gain on the target device.

## 4. What THME has today (read from source)

Read: `src/inference/llama.rs`, `src/inference/weights.rs`, `src/inference/gpu/backend.rs`,
`src/chat.rs`, `src/epiph/mod.rs`, `src/epiph/placement.rs`, `src/profiling/{roofline,kv_cache}.rs`.

- **Weights are F32 in the hot path.** `read_tensor_f32` / `decode_payload` decode any stored dtype to
  `Vec<f32>` at load. Decode therefore reads 4 bytes per weight whatever the file stores, and RAM use is
  4 B x parameters. This is the largest gap against section 1.
- **CPU matmul**: AVX2+FMA kernel with a token block of 8 (`BLOCK = 8`), portable fallback; rows split by
  `std::thread::scope` with `available_parallelism()` threads (logical, not physical cores) and threads
  are spawned per call. The 8-token block helps prefill; a single-row decode falls back to one dot product
  per output row. No integer or packed-domain kernel, no NEON/AVX-512 path.
- **No KV cache in generation.** `chat::generate` calls `model.forward_logits(&ids)` with the whole
  growing sequence for every new token (`src/chat.rs`). Every decoded token therefore costs a full
  prefill over all tokens so far, so per-token cost grows with length. This is likely the biggest single
  loss for decode and is independent of every other technique. (`profiling/kv_cache.rs` only estimates KV
  size; it is not a runtime cache.)
- **GPU path (wgpu)**: `dispatch_gemv_single` handles one row at a time and loops over rows for batches
  (`backend.rs`, `compute_linear`), shaders: F32, Q4_K, Q5_K, IQ4_XS. Per the brief, one run showed
  SmolLM-135M, 128 tokens: 4.3 s GPU vs 1.1 s CPU (single run, not a result). Plausible reasons (not yet
  diagnosed): per-row dispatch and readback overhead, tiny matrices, only linear layers on GPU.
- **`.epiph`**: manifest plus aligned payloads (64 B), optional DEFLATE per tensor, SHA-256 per tensor,
  attachments (calibration), signatures. `placement.rs` already plans GPU / CPU / stream tiers from
  byte sizes with an exact allocator, but nothing consumes it at run time. No expert metadata, no kernel
  layout field, no draft, no MoE-aware inference exists (`arch.rs` has no expert routing).
- **Roofline** (`profiling/roofline.rs`): modelled memory-bound vs compute-bound classification per layer
  from a hardware database; analytic, not measured.
- GGUF/SafeTensors parsers use `memmap2`; the forward pass does not run from the mapping.

## 5. Ranked proposal: smallest steps first

All tickets assume a **release build**, fixed prompt and seed, greedy decoding, 3 warmup runs discarded,
median of N=11 timed runs, fixed thread count recorded, machine idle, same machine before and after.
Report tokens/s for prefill (prompt tokens / time) and decode (new tokens / time) separately, plus peak RSS.
Correctness gate for every ticket: greedy token output identical to the current path (or logits within a
stated tolerance for quantized kernels, checked against the existing F32 reference).
Test models: Pythia-14M and SmolLM-135M. These are **dense, not MoE**.

| # | Ticket | Why (source) | Measured how | Risk |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **KV cache and incremental decode** in `forward_logits`/`chat::generate` | Section 4: today each token reruns all tokens | Decode tokens/s at 16, 64, 128 generated tokens; expect the gap to widen with length. Also check the slope of time vs length | Medium: touches the reference forward pass used by the measured gate; keep the old path as the reference |
| 2 | **Persistent thread pool, physical-core default, `--threads` flag** | Per-call spawn and logical-core count (section 4); threads past saturation hurt decode (Arm) | Sweep 1..N threads for prefill and decode; record both optima | Low |
| 3 | **Decode-shaped matvec** (1 row x many weights): multiple accumulators, no BLOCK overhead | Decode is the common case; bandwidth ceiling in section 1 | Decode tokens/s, plus achieved GB/s = weight bytes / time against a measured STREAM-style copy on the same box | Low |
| 4 | **Packed-domain Q8_0 then Q4_0 matvec** with F32 activations quantized to Q8 per block (scalar+AVX2, then NEON as a separate ticket) | Bytes per token fall about 4x vs F32; the central claim of section 1 | Decode tokens/s and peak RSS vs F32 path; logits max-abs-diff vs F32 reference on a fixed prompt; perplexity delta through the existing measured gate | Medium: new unsafe SIMD; keep scalar reference and a property test |
| 5 | **Load from mmap, no F32 copy** for quantized `.epiph` payloads (needs 4) | RSS and load time; only valid once kernels read packed bytes | Cold and warm load time (state which; drop page cache or reboot for cold), RSS, decode tokens/s | Medium: Windows page-cache behaviour; document platform |
| 6 | **Interleaved row layout** written by `thme pack` and read by the kernel (a `layout` field in `TensorEntry`) | llama.cpp PR 9921 shows gains on ARM but a decode regression on one x86 case | A/B on the same file: plain vs interleaved, prefill and decode separately, on at least one x86 and one ARM box. Ship only if it wins on the target | Medium: format change (`epiph-v1.md` revision and signature impact) |
| 7 | **Prompt-lookup (n-gram) speculation**, greedy only, off by default | Draftless, no format change; strong only on input-grounded text (PLD repo, llama.cpp docs) | Tokens/s with and without, on a summarisation prompt and on a chat prompt, plus acceptance rate. Report a loss if there is one | Low; tiny models may show no gain |
| 8 | **GPU path: batch rows into one dispatch, keep activations on device** | Per-row dispatch is the suspected cause of 4.3 s vs 1.1 s | Re-run the same 128-token benchmark; ship only if GPU beats CPU, else leave GPU off by default | Medium; first diagnose with a timing split (upload, dispatch, readback) |

Honest note on MoE: none of the eight steps can show an MoE gain on Pythia-14M or SmolLM-135M, because
neither is MoE. MoE steps below need a **synthetic MoE test** (a mock model from
`examples/generate_mock_model.rs` with 8-16 experts, top-2 routing and random weights, labelled
synthetic) or a small real MoE that the owner chooses later (policy: no model downloads or runs unless
the owner asks). Synthetic tests can verify correctness and cache logic, and give relative timing only.

MoE tickets (after 1-5):

| # | Ticket | Measured how |
| :-: | :--- | :--- |
| M1 | Expert-granular tensors and routing in the engine (top-k, per-expert matvec, active bytes per token counter) | Active vs total bytes counted; tokens/s on the synthetic model |
| M2 | Expert LRU cache over `placement.rs` tiers with a fixed RAM budget; log hit rate | Hit rate and tokens/s vs budget on a synthetic routing trace with skewed popularity (state the skew, since it decides the result) |
| M3 | Popularity statistics attachment in `.epiph` (hot list per layer) and use it to pre-warm the cache | Cold-start tokens/s with and without the hot list on the same trace |
| M4 | Contiguous fixed-size expert records for sequential SSD reads | Read throughput MB/s vs expert count; cache-miss latency |
| M5 | Per-expert two-precision copies, low-bit on cache miss (HOBBIT idea) | Quality gate delta and tokens/s; quality must pass the measured gate, no claim otherwise |

Not proposed now: EAGLE/Medusa heads (need training), LUT kernels (large, and only a clear win for
low-bit or BitNet-style models), PowerInfer-style predictors (need ReLU sparsity), speculative
prefetch predictors (ProMoE-style; need a real MoE trace).

## 6. Caveats

- Almost every number above comes from an abstract or a README, on different hardware, baselines and
  models. They cannot be added up or compared across rows. Gains rarely stack.
- Several papers (HOBBIT, ProMoE, MoE-Infinity, Fiddler) report speedups against **weak offloading
  baselines** on GPU servers; budget CPU boxes may see far less. Licences for most research systems were
  not checked here: "not verified" means do not copy code, learn the idea only.
- Mixed results exist even for the best-known trick: repacking helped decode on Graviton 3 and hurt it on
  one x86 run in the same PR thread. Every layout and kernel ticket must be A/B tested on the target CPU.
- Tiny models (14M, 135M parameters) can be dominated by overhead (thread spawn, allocation, cache
  effects) and fit in cache, so they may not follow the DRAM-bandwidth roofline at all. A gain on them
  is not evidence for a 7B or MoE model, and a loss may be noise. Use them to check correctness and
  relative effects, and say so in any report.
- Speculative decoding output equality holds only for greedy decoding or exact rejection sampling.
- Speed work must not change quality silently: every quantized or approximate step goes through the
  existing measured quality gate (`AGENTS.md`, no unverified empirical claims).
- Published THME speed numbers need a committed report under `docs/benchmarks/` and regeneration, per
  `AGENTS.md` rule 2. Nothing in this note qualifies.
- Not covered: AMX, Apple AMX/Metal, Vulkan on integrated GPUs, NPU paths, DeepSpeed-MoE inference,
  KV eviction/compression methods, and 2026 papers beyond those cited.
