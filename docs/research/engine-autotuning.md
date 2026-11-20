# How inference engines auto-tune model loading

Status: research note, 2026-11-20. Question: THME must pick the best settings to run each LLM on the
user's machine instead of one fixed loading path. This note studies open engines, distils a recipe and
proposes `thme plan-run`.

Rules for this note:

- Every engine fact has a URL. "Not verified" means I could not confirm it in a primary source.
- Several pages were read through a summarising fetcher, so wording is paraphrased, not quoted.
- THME policy: permissive licences only; learn ideas, never copy code; never read GPL/AGPL/unlicensed
  code to write ours. All THME numbers below are labelled; none are measured here (no models were run).
- Licences were read from the GitHub licence API unless stated.

## 1. Comparison table

| Engine | Decides automatically | How | Licence |
| :--- | :--- | :--- | :--- |
| llama.cpp | GPU layer count, context size (down to a floor), per-tensor CPU/GPU overrides (MoE experts first), multi-GPU split | `--fit` (default on): virtual test allocations per device, iterate until projected use fits free memory minus a margin (default 1024 MiB). Order: keep as is, shrink ctx, move tensors to RAM, dense tensors stay on GPU. Anything the user sets explicitly (`-ngl`, `-ts`, `-ot`, `-c`) is never touched. [discussion 18049](https://github.com/ggml-org/llama.cpp/discussions/18049), [fit-params README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/fit-params/README.md) | MIT ([LICENSE](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE)) |
| llama.cpp (manual knobs) | Nothing for these: threads (default "system dependent"), batch 2048, ubatch 512, flash-attn `auto`, KV K/V type f16, load mode `auto` (mmap/mlock/dio) | `llama-bench` is the sweep tool: pp/tg tokens per second over every combination of listed values, 5 repeats, mean and stddev. It does not support the `-fit` flags. [llama-bench README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/llama-bench/README.md), [18049](https://github.com/ggml-org/llama.cpp/discussions/18049) | MIT |
| Ollama | Context length by VRAM tier, layer offload, flash attention on when supported, keep-alive, parallel requests | Context default: under 24 GiB VRAM 4k, 24-48 GiB 32k, 48 GiB+ 256k ([docs](https://docs.ollama.com/context-length.md)). Keep-alive 5 min, parallel 1, max loaded models 3 per GPU (3 on CPU), flash attention automatic when supported, KV type f16 default and global ([FAQ](https://docs.ollama.com/faq)). New engine "measures" memory before load instead of estimating; claims fewer OOMs and more GPU layers, on a supported-model list only ([blog](https://ollama.com/blog/new-model-scheduling)). Independent test on 16 GB found it worse for some models (one blog, [secondary](https://glukhov.org/llm-performance/ollama/memory-allocation-in-ollama-new-version/)) | MIT ([API](https://api.github.com/repos/ollama/ollama/license)) |
| LM Studio | GPU offload amount, load refusal ("guardrails") | `lms load --estimate-only` prints a memory estimate that includes chosen context and offload; without `--gpu` it picks offload itself; user may override the guardrail ([CLI docs](https://lmstudio.ai/docs/cli/load), [v0.3.27](https://lmstudio.ai/blog/lmstudio-v0.3.27)). "Limit offload to dedicated GPU memory" mode, CUDA only ([v0.3.14](https://lmstudio.ai/blog/lmstudio-v0.3.14)). Exact algorithm: not verified | Closed source app (licence not verified; do not learn from code) |
| vLLM | KV cache size (block count), chunked prefill, batch token budget | Startup profiling: dummy forward pass of one `max_num_batched_tokens` batch to measure peak activation memory; weights + peak + overhead subtracted from the `gpu_memory_utilization` budget; rest becomes fixed-size KV blocks ([arXiv cold-start paper](https://arxiv.org/pdf/2606.07362), [forum](https://discuss.vllm.ai/t/question-about-profile-run/1194); upstream source not read, secondary). Chunked prefill on by default; `max_num_batched_tokens` small favours latency, large favours throughput; default utilization 1.0 on CUDA/ROCm, 0.92 elsewhere; preemption by recompute when KV is full ([docs](https://docs.vllm.ai/en/latest/configuration/optimization/)) | Apache-2.0 ([API](https://api.github.com/repos/vllm-project/vllm/license)) |
| mistral.rs | Device map (layers per device incl. CPU), quantization suggestion | You declare a worst case: `max_seq_len` (default 4096) and `max_batch_size` (default 1); auto device mapping plans memory from it; manual mapping deprecated ([docs](https://ericlbuehler.github.io/mistral.rs/DEVICE_MAPPING.html), [tune](https://ericlbuehler.github.io/mistral.rs/reference/cli/tune/)). `mistralrs tune` recommends a quantization and mapping from a profile (`quality` starts at 8-bit, `balanced` 6, `fast` 4) and can emit a TOML config; whether it runs the model or only estimates: not verified | MIT ([API](https://api.github.com/repos/EricLBuehler/mistral.rs/license)) |
| ExLlamaV2/V3 | GPU split across devices | `--gpu_split auto` / autosplit loader fills devices in turn; V2 code allocates the (lazy) cache first, then spreads weights (inferred from third-party code, [example](https://huggingface.co/spaces/WhiteD2125/text-generation-webui/blame/main/modules/exllamav2_hf.py)). V3 details: not verified. [V2 README](https://raw.githubusercontent.com/turboderp-org/exllamav2/master/README.md) | V3: MIT per [GitHub API](https://api.github.com/repos/turboderp-org/exllamav3/license). V2: README states none; not verified |
| MLC-LLM / TVM | Context and prefill chunk fall back to fixed defaults from model config (for example 4096 and 2048 in one log) | Compile-time config, not hardware measurement; serve has `gpu-memory-utilization` and `max-total-seq-length` flags, behaviour not verified. TVM tuning is offline search, not read here ([compile docs](https://llm.mlc.ai/docs/compilation/compile_models.html), [gen_config log](https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_2-MLC/raw/199b4ad956a26a0c444f6c7647e6de73ad320673/logs.txt)) | Apache-2.0 ([API](https://api.github.com/repos/mlc-ai/mlc-llm/license)) |
| candle | Not verified; I found no auto-tuning docs. Treat as "caller chooses device and dtype" | n/a | Apache-2.0 ([API](https://api.github.com/repos/huggingface/candle/license)) |
| MLX | Memory ceiling: the system GPU working-set limit; no model-level tuning | `max_recommended_working_set_size` and `memory_size` are queryable via `device_info()`; `set_wired_limit` above the system limit is an error; the limit is raised by sysctl `iogpu.wired_limit_mb` (older macOS: `debug.iogpu.wired_limit`), resets on reboot. Default is about 75 percent of RAM in several sources, 84 percent measured on one 128 GB machine, so it varies ([search summary](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html), [Apple forum](https://developer.apple.com/forums/thread/732035)) | MIT ([API](https://api.github.com/repos/ml-explore/mlx/license)) |
| TensorRT-LLM | Nothing automatic for batch sizes: defaults `max_batch_size` 2048 and `max_num_tokens` 8192; you sweep powers of two (case study: 64 batch bottlenecked, 512 gave about 20 percent more throughput) | Manual grid search ([tuning guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html)). Underlying TensorRT builder autotunes kernels per layer (tactic timing) and stores a device/version specific timing cache ([builder docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/builder-performance.html)) | Apache-2.0 text, but repo is flagged NOASSERTION: it bundles other licences and one community-licence directory ([API](https://api.github.com/repos/NVIDIA/TensorRT-LLM/license)). Ideas only |
| ONNX Runtime GenAI | Execution provider is chosen at model build/deploy time, not by a runtime probe (models are tied to the EP they were built for) | Not verified beyond secondary pages ([NVIDIA ModelOpt page](https://nvidia.github.io/Model-Optimizer/pr-preview/pr-2334/deployment/2_onnxruntime.html)) | MIT, expected; the API check returned 404, so not verified |

Newer (2026) items found:

- llama.cpp `--fit` is the main shift: formula-free, allocation-based, on by default
  ([18049](https://github.com/ggml-org/llama.cpp/discussions/18049)). I found no official release note
  for the date.
- Ollama scheduler rewrite (blog dated 2025; one guide says it shipped in 0.17, February 2026, dates
  conflict, not verified).
- `llmfit` (AlexsJones) is a third-party "will it fit" estimator; not read, licence not verified.
  [intro](https://www.mintlify.com/AlexsJones/llmfit/introduction).

What is NOT automatic anywhere I looked: threads count, batch/ubatch, KV precision (all manual with
fixed defaults), and speculative decoding (no engine above turns it on from a probe; not verified for
each). Quantization choice is automatic only in mistral.rs `tune`. This is the gap THME can fill.

## 2. The common recipe

1. Read facts: free VRAM and RAM now (not total), device list, backend limits. Add a safety margin
   (llama.cpp 1024 MiB default; vLLM utilization fraction).
2. Read the model header: layer count, per-tensor bytes, attention shape (KV bytes per token), context
   the model supports.
3. Fix what the user fixed; tune only the rest (llama.cpp's all-or-nothing rule is the simplest
   contract).
4. Decide context first, because KV cache is the knob with the cheapest quality cost: pick a target,
   shrink to a floor if memory is short (llama.cpp, Ollama VRAM tiers).
5. Decide weight placement: dense/attention tensors on the fastest tier first, whole layers where
   possible, spill the rest to RAM, then stream (llama.cpp MoE experts last on GPU).
6. Reserve activations and compute scratch from a measured or modelled peak (vLLM profile run;
   llama.cpp virtual allocations), not a guess.
7. Give the leftover to KV cache and batch budget (vLLM), or lower KV precision when still short
   (Ollama q8_0/q4_0, needs flash attention).
8. Validate by a real allocation or a very short run, then loop down on failure.
9. Optionally sweep a small grid (llama-bench, TensorRT-LLM guide) and cache the winner per
   machine + model + engine version (TensorRT timing cache is the model of a safe cache key).
10. Print the plan and the equivalent explicit flags, so it is inspectable and overridable
    (`llama-fit-params` prints CLI args; `mistralrs tune --emit-config`).

## 3. Failure modes and lessons

- OOM from estimate error. vLLM's profile run models one batch and can underestimate with chunked
  prefill and long prefixes ([forum](https://discuss.vllm.ai/t/question-about-profile-run/1194));
  llama.cpp misses context memory for one model family (Qwen 3 Next) ([18049](https://github.com/ggml-org/llama.cpp/discussions/18049)).
  Lesson: estimates need a margin and a real allocation check; keep a retry path that lowers
  context, not just layers.
- Waste is unavoidable: fit lands near 85-90 percent VRAM because single tensors are GBs
  ([18049](https://github.com/ggml-org/llama.cpp/discussions/18049)).
- Shared GPU memory on Windows. The NVIDIA driver can silently spill VRAM to system RAM
  ("System Memory Fallback") and cause severe slowdowns; llama.cpp discussion reports disabling the
  fallback fixed one case ([18049](https://github.com/ggml-org/llama.cpp/discussions/18049)). Vulkan can
  use shared memory on discrete GPUs but ggml's memory reporting may not count it
  ([issue 16092](https://github.com/ggml-org/llama.cpp/issues/16092), title only, not fully read).
  Lesson: plan with dedicated VRAM only; treat shared memory as RAM.
- Windows iGPU shared budget is driver-capped (about 3.5 GB on one Radeon 890M, one blog, cause
  disputed: [dev.to](https://dev.to/lanternproton/windows-vs-linux-for-local-ai-my-radeon-890m-has-96gb-of-ram-but-windows-only-lets-me-use-35gb-39ln)).
  Not verified generally. Lesson: query the adapter's budget, do not use RAM size.
- Failure style differs by API: CUDA reports "no device" and falls back, Vulkan says "unable to
  allocate" and the load fails ([LLamaSharp issue 1290](https://github.com/SciSharp/LLamaSharp/issues/1290),
  secondary). Lesson: wrap init and allocation, always have a CPU fallback.
- iGPU plus dGPU: LM Studio's Vulkan path treats Strix Halo as an iGPU and prefers shared memory
  (reports only, [search result](https://lilting.ch/en/articles/strix-halo-vram-memory-optimization)).
  Lesson: rank devices by measured speed and dedicated memory, not by enumeration order.
- Unified memory (Apple): usable GPU memory is the OS working-set limit, not RAM; MLX exposes it and
  it can be raised only by an OS setting (see table). Lesson: read the limit from the API.
- Thermal throttling: not found in any engine's auto-tuner (not verified). A 1-2 second probe
  measures the cool clock; sustained speed can be lower. THME already has a thermal model
  (`src/profiling/thermal.rs`); mark probe results as burst speed.
- More GPUs is not monotonically faster ([18049](https://github.com/ggml-org/llama.cpp/discussions/18049)).
  Lesson: single-device wins must be considered, not only split.
- Explicit user flags disabling auto-fit made one Windows user slower. Lesson: when the user
  overrides one thing, still show what the rest would have been.
- Ollama's "exact measurement" claim is one vendor blog with two benchmarks; an independent test
  disagreed on 16 GB. Lesson: THME must publish its own measured plan-vs-actual error.

## 4. What THME already has

Read briefly from the worktree (accuracy checked against source, not run):

- `src/doctor.rs`: `HostFacts` with `Fact<T>` = Checked / Assumed / Unknown. Fields today: OS, arch,
  total and available RAM, free disk, one `gpu_adapter` name (no VRAM figure), CPU threads, models.
  Gap: no VRAM free/total, no list of adapters, no dedicated-vs-shared flag, no physical core count.
- `src/discover.rs`: finds models and their format (`DiscoveredModel`, `Discovery`); it is the entry
  for "which header to read".
- `src/epiph/placement.rs`: `plan(tensors, Budget{vram_bytes, host_bytes})` returns tier per tensor
  (`Gpu`, `Cpu`, `Stream`) by two exact multiple-choice selections (`engine::alloc`), with a reason
  string per tensor. Pure functions, sizes in, placement out. Gap: weights only; no KV, activations
  or margin; no speed model; maximises bytes on the fastest tier, which is llama.cpp's goal too but
  ignores that GPU can be slower (below).
- `src/profiling/hardware.rs` + `roofline.rs`: `HardwareDevice` (peak TFLOPS per precision, bandwidth,
  `machine_balance`, `from_pci_recipe`) with predefined entries (RTX 4060 Ti, 4090, H100, Apple M3 Max),
  `RooflineEngine`, `profile_model`; `kv_cache.rs`, `thermal.rs`, `cache_sim.rs` also exist in
  `src/profiling/`. These give predicted tokens/s from bandwidth; the DB is small and the numbers
  are datasheet figures, so predictions are "predicted", not measured.
- `src/inference/gpu` (wgpu): `DeviceArg {Auto, Gpu, Cpu}`; `Auto` picks the GPU whenever
  `GpuContext::init()` succeeds. Only linear layers run on GPU (dispatch functions `dispatch_linear*`,
  `dispatch_score_head*`), shaders for Q4_K, Q5_K, IQ4_XS, F32. Owner-reported figure: 4x slower than
  CPU on SmolLM-135M with one dispatch per token row. I found no committed report for it under
  `docs/` (grep for "4x" and "slower than CPU" found nothing), so it is **not a committed
  measurement**; claim register V-30 is about a different claim. Consequence: `Auto` choosing GPU
  just because init works is wrong today; a speed probe must be able to choose CPU.
- Studio chat device selector (CPU, GPU, CPU+GPU, experimental): it is in an open pull request
  (#99), not yet on `main`, which is why `src/gui` does not show it. The plan
  should feed this selector a recommended default and a reason line.

## 5. Proposal: `thme plan-run` (name open; alternative `thme tune`)

### 5.1 Contract

Inputs:

- Model header only (tensor names and bytes, layer count, hidden size, heads, KV heads, vocab,
  max context, quant types). No weights read.
- `HostFacts` extended (see tickets): per-device dedicated VRAM free/total, shared budget,
  RAM available, physical cores and logical threads, unified-memory flag.
- Optional probe results (cached or fresh, section 5.4).
- User overrides (section 5.6) and a goal: `--goal interactive|throughput|fit-largest` (default
  interactive: first-token latency and decode speed for one chat).

Output: `RunPlan` as JSON plus a text table (every field carries `basis`, reusing `Fact` idea).

```text
RunPlan {
  schema, model_id, plan_version,
  device: Cpu | Gpu(adapter) | Hybrid,
  placement: per-layer tier (reuse epiph Placement) + bytes per tier,
  threads, batch, ubatch,
  context_limit, kv_precision: f16 | q8 | q4,
  expected: { weights_bytes, kv_bytes, scratch_bytes, total_bytes,
              decode_tok_s, prefill_tok_s },     // each {value, basis}
  basis per field: predicted | measured(probe_id) | user,
  reasons: [string]  // one per decision, same style as placement reasons
  explicit_flags: "equivalent thme chat/serve command line"
}
```

`basis` rule (AGENTS.md rule 2): `predicted` comes from formulas and the hardware DB; `measured`
only from a probe run on this machine, with the probe id and timestamp; numbers copied into docs
must come from a committed report under `docs/benchmarks/`.

### 5.2 Order of decisions

1. Facts and margin: free (not total) dedicated VRAM minus margin (start: max(512 MiB, 5 percent),
   tunable; llama.cpp uses 1024 MiB); RAM available minus OS reserve; unified memory uses the OS
   working-set limit.
2. Context target: min(model max, tier by free memory like Ollama's 4k/32k/256k idea, user request);
   floor 2048 (name `--min-context`).
3. KV bytes per token from header (2 x layers x kv_heads x head_dim x bytes); weights + KV + scratch
   model = candidate memory. Scratch from a per-model formula, replaced by a measured dry allocation
   once a probe exists.
4. Candidate devices: CPU always; each GPU adapter whose backend initialises. Predict decode tok/s
   per candidate from the roofline (bytes read per token / bandwidth) and, for GPU, a dispatch
   overhead term. Predicted only.
5. Pick device by predicted speed, then correct by probe (5.3). Rule: a GPU must beat CPU by a margin
   (say 1.25x) or CPU wins; ties go to CPU (no driver risk).
6. Placement: call `epiph::placement::plan` with budget = memory left after KV, scratch and margin.
   Hybrid only if the probe shows the partial offload beats CPU-only; otherwise all-CPU.
7. Shrink loop if the total does not fit: lower context to the floor, then KV q8 then q4 (flag
   quality cost), then stream tier, then fail with an actionable message. Never touch fixed flags.
8. Threads: start at physical cores; batch and ubatch from the goal (interactive: small ubatch;
   prefill speed grid in probe).
9. Emit plan, reasons, explicit flags.

### 5.3 The 1-2 second probe

Purpose: replace the two least trustworthy predictions (device speed, scratch memory).

- Build the first K layers' worth of work (K=1 transformer block, real shapes from the header) with
  synthetic weights of the right quant type; no model file read and no download (fits the repo's
  "no LLM runs" rule: this runs a synthetic kernel, not a model).
- CPU: time N=16 single-token matvec passes at 1, physical-cores/2 and physical-cores threads; keep
  the median. GPU: same shapes on each adapter, including the real dispatch pattern THME uses
  (one dispatch per row), so the 4x slowdown shows up in the probe instead of in user chat.
- Allocation check: allocate and free the planned scratch buffers on the GPU to confirm free memory
  is real (catches shared-memory spill and driver caps).
- Extrapolate: tok/s ~ 1 / (layers x per-block time + head time). Label `measured (probe, extrapolated)`
  and keep the raw numbers.
- Budget: hard 2 s wall clock, abort on first error; run again only if the cache key changed.
- Thermal: record that the probe is a burst; if a longer `--probe-seconds 15` was run, store the
  sustained ratio.
- Validation: `plan-run --verify` loads the real model, generates a few tokens and writes
  predicted vs measured into a report; that report is the only thing docs may quote.

### 5.4 Cache

- File `plan-cache.json` in the user cache dir (use the OS cache path API; never print absolute
  paths in docs, per AGENTS.md rule 1).
- Key: (adapter name + driver version string, backend, CPU model, physical cores, THME version,
  quant type, model shape hash). Free memory is NOT in the key (it is read fresh every run).
- Entry: raw probe numbers, timestamp, schema version. TTL 30 days, invalidated by any key change
  (TensorRT's timing cache is similarly tied to device and versions, see section 1).
- `thme plan-run --no-cache` and `--reprobe`.

### 5.5 Where it plugs in

- `thme chat`, `thme serve`, Studio call `plan_run()` when device is `Auto` (replaces the "GPU if init
  works" rule in `DeviceArg::resolve`).
- Studio device selector keeps the four choices; the plan supplies the default and a one-line reason.
  "CPU+GPU" is offered only when the plan has a Hybrid candidate.
- `thme doctor` prints a short "recommended device" line from the cached plan.

### 5.6 Overrides

- Any explicit flag wins and removes that decision from tuning (llama.cpp's rule): `--device`,
  `--threads`, `--ctx`, `--kv-type`, `--batch`, `--gpu-layers`.
- `--budget VRAM_MIB:HOST_MIB` reuses the existing `Budget::parse_mib`.
- `--plan FILE` loads a saved RunPlan; `--explain` prints reasons; `thme plan-run --json` for tools.
- A user override is shown with what the auto choice would have been.

### 5.7 Ticket list (numbers are placeholders; assign real THME ids when merging into BACKLOG.md)

Style follows BACKLOG.md: milestone, priority, area, problem, done means. Milestone v0.3.0 (plug and
play workspace) is the natural home.

1. **[THME-PLAN-1] VRAM and adapter facts in `HostFacts`** - p1, doctor. Problem: only an adapter name
   is reported. Done: `Fact<u64>` dedicated VRAM total/free per adapter, device type (discrete,
   integrated, CPU), shared-budget field, physical core count; unit tests with fake data; JSON schema
   updated; unknown stays `Unknown`.
2. **[THME-PLAN-2] Memory model for a run** - p1, inference. Problem: placement counts weights only.
   Done: pure function from header + context + KV type to weights, KV and scratch bytes; table test
   against hand-computed LLaMA and GPT-NeoX shapes; no model read.
3. **[THME-PLAN-3] `RunPlan` type and decision order, CPU-only** - p1, cli. Done: `thme plan-run
   --model X` prints the plan for a header, with basis labels, using only `predicted`; golden-file
   test for a mock model (`examples/generate_mock_model.rs`).
4. **[THME-PLAN-4] Shrink loop** - p1. Done: property test: for any budget the result fits or errors;
   context never below the floor except by explicit user value; fixed flags unchanged.
5. **[THME-PLAN-5] Roofline decode estimate per device** - p2, profiling. Done: predicted decode tok/s
   from bandwidth and bytes per token for CPU and GPU, GPU dispatch-overhead term; unit tests; output
   labelled predicted; no claim in docs.
6. **[THME-PLAN-6] CPU probe** - p1. Done: threads sweep on synthetic matvec under 1 s; deterministic
   test using an injected clock; result struct with raw samples.
7. **[THME-PLAN-7] GPU probe and CPU-wins rule** - p1, gpu. Done: probe uses the real dispatch
   pattern, 2 s timeout, returns an error instead of hanging; with a fake backend the plan picks CPU
   when the GPU is slower; on a real machine, a report under `docs/benchmarks/` records probe vs real
   tokens/s (pending hardware run; mark pending, no LLM run by agents).
8. **[THME-PLAN-8] Hybrid placement through `epiph::placement`** - p2. Done: plan feeds budget
   after KV, scratch and margin; partial offload accepted only when the probe says it beats CPU.
9. **[THME-PLAN-9] Plan cache** - p2. Done: key and TTL tests, corrupt file ignored, `--no-cache`,
   `--reprobe`.
10. **[THME-PLAN-10] Wire `DeviceArg::Auto` and Studio selector to the plan** - p1. Done: Auto no
    longer means "GPU if init works"; Studio shows the recommended device and reason; test with a
    stubbed plan.
11. **[THME-PLAN-11] `--verify` predicted vs measured report** - p2. Done: JSON report with error
    percentages per field; schema in `docs/schemas/`; becomes the source for any published number.
12. **[THME-PLAN-12] Windows shared-memory guard** - p2. Done: plan counts dedicated VRAM only; warns
    when the driver reports shared memory spill; documented in the manual.

## 6. Honest limits of this note

- No engine source was read for implementation; ideas only, consistent with the permissive-licence
  policy. The repo's own licence tag in the files I read is AGPL-3.0-only; copying from MIT/Apache
  code would still need the third-party register, so write original Rust.
- Several vLLM, ExLlama, MLC and MLX details come from secondary pages and are marked as such.
- Not verified: LM Studio algorithm, ExLlamaV3 internals, MLC runtime estimator, ONNX Runtime GenAI
  licence and auto-selection, any engine's thermal handling, threads default logic in llama.cpp.
- No THME speed figure here is measured. The "4x slower on SmolLM-135M" figure is a local one-run
  timing from 20 November 2026 (128 tokens: 4.3 s GPU, 1.1 s CPU, RTX 4060 Ti, release build, includes
  adapter start-up and weight upload); it is not a committed report.
