# Multi-vendor GPU compute backends for LLM inference

> Naming (owner, 20 November 2026): the GPU and CPU backends described here are called **Compute Bridge**; the run planner is `thme pathfinder`; detection stays `thme doctor`. Every performance number below is reported by other projects, not measured by THME, unless it cites a committed THME report.

Status: research note, 2026-11-20. Question: how should THME build a fast, portable GPU backend with
vendor detection and driver checks (NVIDIA, AMD, Intel, Apple Silicon, integrated GPUs)?

Rules for this note:

- Scope is ordinary GPU compute. Nothing here covers unlocking, firmware or bypassing vendor limits.
- Every outside fact has a URL. "Not verified" means I found it only in a secondary source, or not at all.
- Several pages were read through a summarising fetcher, so wording is paraphrased.
- Benchmark numbers are reported by others, on the hardware named. No number in this note is
  THME-measured, except the one local timing in section 5, which is not from a committed report.
- No models were run, nothing was downloaded or built.
- Licences come from the GitHub licence API or the repo's own licence file, unless stated.
- THME policy: AGPL-3.0-only, permissive third-party code only, pure Rust where feasible
  (AGENTS.md section 3). Ideas are learned from other engines, code is not copied.

## 1. API options

### 1.1 Summary table

| Option | Vendors | Needs at build | Needs at run | Licence | Verdict for THME |
| :--- | :--- | :--- | :--- | :--- | :--- |
| wgpu (WGSL; Vulkan, DX12, Metal) | all | nothing native | system Vulkan/DX12/Metal driver | Apache-2.0 per [GitHub API](https://api.github.com/repos/gfx-rs/wgpu/license) (the repo states dual MIT or Apache-2.0: not verified) | base path; fix first |
| Raw Vulkan via `ash` | NVIDIA, AMD, Intel; Apple through MoltenVK only | nothing native (loader found at run time) | Vulkan loader + driver | ash Apache-2.0 per [API](https://api.github.com/repos/ash-rs/ash/license); Vulkan-Loader Apache-2.0 per [LICENSE.txt](https://raw.githubusercontent.com/KhronosGroup/Vulkan-Loader/main/LICENSE.txt) | second step, only for features wgpu lacks |
| CUDA driver API, loaded with dlopen | NVIDIA | no CUDA toolkit | `libcuda.so` / `nvcuda.dll` from the driver | `cudarc` is MIT OR Apache-2.0 and has a `fallback-dynamic-loading` feature ([Cargo.toml](https://raw.githubusercontent.com/coreylowman/cudarc/main/Cargo.toml)); libloading ISC ([Cargo.toml](https://raw.githubusercontent.com/nagisa/rust_libloading/master/Cargo.toml)) | native backend candidate |
| ROCm / HIP | AMD | ROCm SDK (llama.cpp needs it to build) | ROCm runtime, narrow GPU list | HIP MIT, rocm_smi_lib MIT, amdsmi MIT ([HIP](https://api.github.com/repos/ROCm/HIP/license), [smi](https://api.github.com/repos/ROCm/rocm_smi_lib/license)) | detect only; Vulkan covers AMD |
| Metal (direct) | Apple | Xcode SDK to compile MSL offline, or runtime compile | macOS | Apple SDK terms: not verified | native backend candidate on macOS only |
| MLX | Apple | C++ build | macOS | MIT ([API](https://api.github.com/repos/ml-explore/mlx/license)) | ideas only |
| Intel SYCL / oneAPI + Level Zero | Intel | oneAPI toolchain | Level Zero runtime from the GPU driver | Level Zero MIT ([API](https://api.github.com/repos/oneapi-src/level-zero/license)) | detect only; heavy toolchain |
| OpenCL | all, patchy | headers + ICD loader | vendor ICD | not checked | skip; llama.cpp lists it for Adreno only ([build.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md)) |
| DirectML (DX12) | Windows, all vendors | Windows SDK | Windows | Microsoft | skip: "in sustained engineering", new work moved to Windows ML ([Microsoft Learn](https://learn.microsoft.com/en-us/windows/ai/directml/dml-intro)) |

### 1.2 wgpu / WebGPU: what it can and cannot do for LLM kernels

All from the wgpu docs for the 30.x line unless stated.

- Subgroups: `Features::SUBGROUP`, Vulkan, DX12, Metal ([Features](https://docs.rs/wgpu/latest/wgpu/struct.Features.html)).
  `AdapterInfo` reports `subgroup_min_size` and `subgroup_max_size` (typical: NVIDIA 32, AMD RDNA 32 to 64,
  Intel 8 to 32) ([AdapterInfo](https://docs.rs/wgpu/latest/wgpu/struct.AdapterInfo.html)).
  Good: the fused dequant-matvec reduction can use subgroup adds instead of the 32-float shared array
  the current Q4_K shader uses.
- f16: `Features::SHADER_F16`, Vulkan, Metal, DX12, WebGPU; WGSL needs `enable f16;` ([Features](https://docs.rs/wgpu/latest/wgpu/struct.Features.html)).
- Cooperative matrix (the tensor-core-like path): `EXPERIMENTAL_COOPERATIVE_MATRIX`, Vulkan via
  `VK_KHR_cooperative_matrix` and Metal. The Features page says only 8x8 f32 is in the baseline; the
  adapter reports its real tile list at run time through `cooperative_matrix_properties()`, and that
  list is the source of truth ([docs module](https://wgpu.rs/doc/wgpu/documentation/extensions/cooperative_matrices/index.html), via search summary).
  Changelog: cooperative load/store in v29.0.0, mixed precision `f16 x f16 -> f32` on Metal in v30.0.0
  ([CHANGELOG](https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/CHANGELOG.md)).
  Whether DX12 exposes it: not verified. Treat as optional, behind a run-time check.
- int8 dot products (`dot4I8Packed` style): not verified in wgpu; I found no page confirming it.
  Do not plan on it.
- Pipeline cache: `Features::PIPELINE_CACHE`, Vulkan only; DX12 and Metal unimplemented
  ([Features](https://docs.rs/wgpu/latest/wgpu/struct.Features.html)). Cold shader compile cost on
  DX12/Metal cannot be cached through wgpu.
- Buffer limits: defaults are `max_storage_buffer_binding_size` 128 MiB, `max_buffer_size` 256 MiB,
  8 storage buffers per stage ([Limits](https://docs.rs/wgpu/latest/wgpu/struct.Limits.html)). THME
  already asks for `adapter.limits()`, which lifts them to what the device offers. On Metal the binding size
  is capped near `u32::MAX` again in the unreleased changelog. A 4 GiB-plus weight tensor
  therefore needs splitting into several bindings, and Metal has a shared argument table limit
  (`max_buffers_and_acceleration_structures_per_shader_stage`, 29 on Metal) ([CHANGELOG](https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/CHANGELOG.md)).
- Readback: `map_async` plus `poll` is the only route; the cost is one submit and one wait per sync, so the
  design rule is one readback per token (section 3).
- Evidence that WebGPU-class kernels can be fast: WebLLM reports "up to 80% native performance on the same
  device" using MLC-LLM and TVM generated kernels; hardware not named on the abstract page
  ([arXiv 2412.15803](https://arxiv.org/abs/2412.15803)).
- Not exposed by wgpu: Vulkan `VK_NV_cooperative_matrix2`, vendor intrinsics, CUDA graphs. Conclusion:
  wgpu can reach a good fraction of native, not all of it on NVIDIA prefill.

### 1.3 Vulkan cooperative_matrix and cooperative_matrix2

- `VK_NV_cooperative_matrix2` is an NVIDIA extension, built on the KHR cooperative matrix types, not
  ratified ([Vulkan docs](https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_matrix2.html)).
- llama.cpp added it for `mul_mat` and FlashAttention in PR 10206. The author wrote it "helps
  significantly with prompt processing", needs a driver that ships the extension, and flash attention
  gives "a few percent" on token generation ([PR 10206](https://github.com/ggml-org/llama.cpp/pull/10206)).
- A forum thread reports driver crashes when querying cooperative-matrix properties on a Blackwell laptop
  GPU; so run the query behind a guard ([NVIDIA forum](https://forums.developer.nvidia.com/t/blackwell-rtx-5050-laptop-sm-120-vulkan-driver-crashes-in-cooperative-matrix-property-queries/369162), via search summary, not verified).

### 1.4 CUDA driver API via dlopen

- The driver API lives in the driver library (`libcuda.so` on Linux), not in the toolkit. Device queries
  `cuDeviceGetAttribute` and `cuDeviceTotalMem` are in the driver API reference
  ([docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html)). Exact names for the
  compute-capability attributes, `cuMemGetInfo` and `cuDriverGetVersion` were not on that page: not verified.
- Kernels would be PTX text that THME ships (generated or hand-written) and the driver JIT-compiles.
  Driver must be new enough for the PTX ISA version used; table not found: not verified.
- Redistribution: the CUDA EULA lists `libcuda.so` (Linux only) among distributable driver libraries,
  under conditions: the app must add material functionality, only the listed parts are used, and the
  Linux files stay unmodified. No driver libraries are listed for Windows or macOS
  ([EULA](https://docs.nvidia.com/cuda/eula/index.html)). Reading: we do not need to ship it at all;
  we load what the user's driver installed. This is not legal advice; owner reviews before any bundling.
- NVML (`libnvidia-ml.so`, or `nvml.dll` under NVSMI or System32) ships with the driver and is not on PATH
  on Windows ([NVML docs](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html)).
  The EULA text I read does not mention NVML: licence of the headers not verified.
- The CUDA compatibility guide describes minor-version compatibility (CUDA 11 on) and `cuda-compat`
  forward-compat packages; the landing page had no driver table ([guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)).

### 1.5 How other engines structure backends, and what wins

| Engine | Structure | Licence |
| :--- | :--- | :--- |
| llama.cpp | One `ggml` backend per API: CUDA, HIP, Metal, SYCL, Vulkan, OpenCL (Adreno), WebGPU via Dawn, and others. Several can be built together; `GGML_BACKEND_DL` builds backends as loadable libraries so one binary runs on different GPUs; `--list-devices`, `--device` select ([build.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md)) | MIT ([LICENSE](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE)) |
| mistral.rs | Built on candle; automatic device mapping ([docs](https://ericlbuehler.github.io/mistral.rs/DEVICE_MAPPING.html)); backend details not read | MIT ([API](https://api.github.com/repos/EricLBuehler/mistral.rs/license)) |
| candle | Caller picks device and dtype; no auto-tuning found | Apache-2.0 ([API](https://api.github.com/repos/huggingface/candle/license)) |
| burn / CubeCL | One kernel language in Rust, compiled to PTX (CUDA runtime), AMDGPU (HIP), MSL (Metal), SPIR-V and WGSL (wgpu). Cargo features per runtime; autotune cache per device; tensor cores unavailable with the WGSL compiler; self-described alpha ([README](https://github.com/tracel-ai/cubecl)) | burn and CubeCL Apache-2.0 ([burn](https://api.github.com/repos/tracel-ai/burn/license), [cubecl](https://api.github.com/repos/tracel-ai/cubecl/license)); the README says MIT or Apache-2.0 |
| MLC / TVM | Compile models ahead of time to each target's kernels (WebGPU, Vulkan, Metal, CUDA) | Apache-2.0 ([API](https://api.github.com/repos/apache/tvm/license)) |
| tinygrad | Runtime list in docs: NV, AMD, QCOM, METAL, CUDA ([docs](https://docs.tinygrad.org/runtime/)); other details secondary | MIT ([API](https://api.github.com/repos/tinygrad/tinygrad/license)) |
| ONNX Runtime | Execution providers (CUDA, TensorRT, DirectML, OpenVINO, CoreML, WebGPU, MIGraphX; ROCm marked deprecated); each node goes to the first provider that can run it, else CPU ([EP list](https://onnxruntime.ai/docs/execution-providers/)) | MIT expected; licence API returned nothing: not verified |

Pattern: every serious engine has one backend per vendor API behind one device interface, and the
portable (Vulkan or WGSL) path is the baseline.

Reported numbers, llama-bench, Llama 2 7B Q4_0, from community scoreboards (not controlled, builds and drivers vary):

| Hardware | Backend | pp512 t/s | tg128 t/s | Source |
| :--- | :--- | ---: | ---: | :--- |
| RTX 4060 Ti 8 GB | CUDA | 3394.63 | 63.86 | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| RTX 4090 | CUDA | 11992.70 | 186.21 | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| RTX 4090 | Vulkan | 9452.03 | 187.97 | [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879) |
| RTX 3080 | CUDA (Dec 2024) | 4499.47 | 131.01 | [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879) comment |
| RTX 3080 | Vulkan (same build, Dec 2024) | 1706.07 | 62.16 | same comment |
| RX 7900 XTX | Vulkan | 3726.99 | 182.63 | [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879) |
| Radeon Pro VII | ROCm vs Vulkan (Jan 2025) | 1064.99 vs 329.86 | 87.45 vs 75.22 | same discussion |
| RX 5700 XT | HIP vs Vulkan | 354.17 vs 439.42 | 67.55 vs 70.13 | same discussion |
| Arc A750 | SYCL vs Vulkan (Jan 2025) | 1616.11 vs 88.86 | 36.64 vs 27.57 | same discussion |
| Arc B580 | Vulkan | 620.94 | 70.14 | [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879) |
| Ryzen AI Max+ 300 iGPU | Vulkan | 1487.08 | 51.81 | [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879) |
| Core Ultra 300 iGPU | Vulkan | 1380.07 | 24.45 | same |
| Apple M4 Max (40 cores) | Metal | 885.68 | 83.06 | [#4167](https://github.com/ggml-org/llama.cpp/discussions/4167) |
| Apple M4 Max | Vulkan | 724.77 | 75.02 | [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879) |

Reading, with limits: the CUDA and Vulkan rows for the 4090 come from different builds and dates, so
they are not a controlled pair. The same-build 3080 pair is. Reported pattern: native wins prompt
processing by the largest margin on NVIDIA (3080: about 2.6x pp, 2.1x tg in the Dec 2024 build); Vulkan is
close on token generation on newer builds; on AMD the answer flips by card (RDNA1 faster on Vulkan,
older GCN faster on ROCm); on Intel, SYCL beat Vulkan by a wide margin in that Jan 2025 pair. A Mesa
developer attributes CUDA's token-generation lead to CUDA graphs and op fusion, and RADV beating ROCm
on tg but trailing on pp ([blog](https://airlied.blogspot.com/2025/07/ramalamamesa-benchmarks-on-my-hardware.html);
numbers are in images there, so only the direction is cited).

## 2. Detection and driver requirements

### 2.1 What to ask, and how, with no heavy dependency

| Fact | NVIDIA | AMD | Intel | Apple | Any |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Name, vendor id, device id, type, backend, driver text | wgpu `AdapterInfo` (vendor is the PCI id on Vulkan; has `driver`, `driver_info`) ([docs](https://docs.rs/wgpu/latest/wgpu/struct.AdapterInfo.html)) | same | same | same | PCI ids 0x10DE, 0x1002, 0x8086, Apple none: id values are common knowledge, not verified here |
| Dedicated VRAM, free | NVML `nvmlDeviceGetMemoryInfo` via dlopen of the driver's NVML library ([NVML](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html); function names not on that page, not verified) | Linux sysfs `mem_info_vram_total` and `mem_info_vram_used` ([kernel docs](https://docs.kernel.org/gpu/amdgpu/driver-misc.html)) | Level Zero Sysman `zesMemoryGetState` returns free and total per module ([spec](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/sysman/PROG.html)) | Metal `recommendedMaxWorkingSetSize`, `hasUnifiedMemory`, `maxBufferLength` ([Apple](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize); page text did not load, wording from model memory: not verified) | Windows DXGI `IDXGIAdapter3::QueryVideoMemoryInfo` gives a per-process budget and usage; staying inside it avoids paging stalls ([Microsoft](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo)) |
| Compute capability | CUDA driver attribute query | n/a (gfx id from ROCm tools) | n/a | Apple GPU family | n/a |
| Subgroup size range | wgpu `subgroup_min_size` / `subgroup_max_size` | same | same | same | |
| Features (f16, subgroup, cooperative matrix, pipeline cache) | `adapter.features()`, `adapter.cooperative_matrix_properties()` | same | same | same | |
| Integrated vs discrete | `device_type` from wgpu | same | same | integrated by design (unified memory) | |

Rules: every query may fail; each failure becomes `Unknown`, never a panic (AGENTS.md rule 4). Windows
Level Zero and sysfs probes are optional extras behind `libloading`; wgpu alone gives name, type, limits and
features on every vendor. ADL (AMD Windows) was not researched: not verified, and DXGI covers the need.

### 2.2 Driver and hardware requirements

- Vulkan 1.3 on NVIDIA: first conformant drivers are Windows 473.11 and Linux 470.62.22 (Jan 2022);
  Vulkan 1.4: Windows 553.45, Linux 550.40.81 (Dec 2024) ([NVIDIA](https://developer.nvidia.com/vulkan-driver)).
- Cooperative matrix 2 on NVIDIA: needs a recent driver (575 series reported by one source, not verified)
  ([Phoronix via mirror](https://hn.jakechampion.name/reader/https://www.phoronix.com/news/NVIDIA-Vulkan-AI-ML-Success)).
- CUDA minimum driver per toolkit: not verified. We ship PTX, so the real requirement is the PTX version.
- AMD ROCm 7.14 supported list includes Instinct MI100 and newer, Radeon RX 9000 (gfx1200, gfx1201), RX 7000
  (gfx1100, gfx1101) and some Radeon PRO; Radeon VII (gfx906), MI50 and MI25 are not supported. Radeon
  cards are limited to Ubuntu 24.04.4, 22.04.5, RHEL 10.1, 9.7 on that page
  ([ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)).
  The Windows HIP SDK list was not found: not verified. Consequence: millions of AMD cards (RDNA1, RDNA2
  consumer, iGPUs) are outside ROCm; Vulkan is the only common path for them.
- Intel: llama.cpp SYCL verifies Arc A and B, Meteor, Arrow and Lunar Lake iGPUs, 11th Gen Core and newer;
  needs the Intel GPU driver's Level Zero runtime; no minimum driver stated; Linux users must be in the
  `render` and `video` groups; iGPUs under 80 EUs "will likely be too slow" ([SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)).
- Apple: Metal on macOS; the working-set limit is a recommendation, not a hard cap. MLX sources report about
  75 to 84 percent of RAM as the default and a sysctl to raise it ([search summary in engine-autotuning.md](engine-autotuning.md)).

### 2.3 What to tell the user (fix-it advice)

| Situation | Message (short, actionable) |
| :--- | :--- |
| No Vulkan loader or no adapter | "No GPU API found. Install or update your graphics driver from your GPU maker, then run `thme doctor` again. CPU mode works now." |
| NVIDIA, driver older than 470.62 (Linux) or 473.11 (Windows) | "Your NVIDIA driver is older than the first Vulkan 1.3 release. Update to a current driver." |
| AMD, Vulkan only, ROCm absent | "Vulkan is used for your Radeon. ROCm is not needed." (never push ROCm) |
| AMD Linux, user not allowed on `/dev/dri` | "Add your user to the `render` group." |
| Intel Linux | "Add your user to `render` and `video`; install the Intel compute runtime (Level Zero)." |
| Intel iGPU, shared memory small | "Integrated GPU shares system memory; GPU mode may not beat CPU. `thme plan-run` will tell." |
| Apple | "Using Metal unified memory; limit is X GiB." |
| Software adapter (llvmpipe, WARP) | "Only a software renderer was found; GPU mode disabled." |
| Cooperative matrix query fails | "Skipping matrix-core path on this driver; using plain kernels." |

## 3. What makes LLM inference fast on GPUs

Reported by others; THME has no measurement for any of it.

- Prefill is a GEMM, decode is a GEMV. In the CUDA scoreboard, one RTX 4060 Ti shows pp512 3394.63 t/s
  against tg128 63.86 t/s on Llama 2 7B Q4_0 ([#15013](https://github.com/ggml-org/llama.cpp/discussions/15013)).
  That is about 53x in tokens per second, because prefill batches tokens against one read of the weights. Per-row
  dispatch, one token at a time, throws that away.
- Decode is memory-bandwidth bound: tokens per second is roughly bandwidth divided by bytes of weights read
  (roofline). Hence fused dequant-matvec, which reads the 4-bit data once and never writes dequantised weights.
- Matrix units matter for prefill: cooperative matrix "helps significantly with prompt processing", not decode
  ([PR 10206](https://github.com/ggml-org/llama.cpp/pull/10206)).
- Launch overhead matters for decode: CUDA's tg lead is attributed to CUDA graphs and fusion
  ([airlied](https://airlied.blogspot.com/2025/07/ramalamamesa-benchmarks-on-my-hardware.html)).
  wgpu analogue: record a whole layer (or whole token) in one command buffer, reuse bind groups.
- Keep weights and KV cache resident on the device; upload once.
- One readback per token: logits or the argmax only, not each matmul.
- FlashAttention-style tiling: gain "a few percent" on token generation, mostly at large context
  ([PR 10206](https://github.com/ggml-org/llama.cpp/pull/10206)).
- Subgroup reductions replace shared-memory trees (wgpu `SUBGROUP`).
- Pipeline cache: Vulkan only in wgpu; elsewhere pay compile once per process.
- Typical gain over naive per-row dispatch: no primary source found: not verified. The only honest number
  will come from THME's own `--verify` report (THME-3065) after the fixes below.

## 4. Distribution

| Component | Can it ship in the repo or archive? | Notes |
| :--- | :--- | :--- |
| wgpu, naga, ash, libloading | Yes, as Rust crates | Apache-2.0 / MIT / ISC; list in the third-party register; naga licence file reads NOASSERTION in the API, so check it by hand (not verified) |
| Vulkan loader (`vulkan-1.dll`, `libvulkan.so.1`) | No need: found on the user's machine; wgpu loads it at run time | Loader is Apache-2.0 ([LICENSE.txt](https://raw.githubusercontent.com/KhronosGroup/Vulkan-Loader/main/LICENSE.txt)) |
| `libcuda`, NVML | Do not ship; dlopen what the driver installed | EULA permits `libcuda.so` redistribution on Linux only, under conditions ([EULA](https://docs.nvidia.com/cuda/eula/index.html)); no gain from shipping |
| CUDA toolkit pieces (runtime, cuBLAS) | Not needed; we do not link them | Listed as distributable in Attachment A, but weight and size cost; avoid |
| ROCm, HIP, SYCL, oneAPI | Do not ship, do not link | Detect only |
| Metal | System framework; no redistribution | Apple SDK terms not verified |

- Feature flags: `gpu-wgpu` (default), `gpu-cuda` (dlopen, no link-time dependency), `gpu-metal`
  (macOS target only). A default build must still pass `cargo clippy -- -D warnings` with no vendor SDK installed.
- Binary size: CLI limit is 16 MiB (AGENTS.md). Cost of each backend is unmeasured here; the first ticket
  below records the release size per feature as a committed number. Dynamic loading keeps native code
  out of the binary, matching `GGML_BACKEND_DL` in llama.cpp.
- Do not publish a bundled vendor binary in releases. Fits AGENTS.md rule 3 (no committed binaries).

## 5. Where THME is today

Read in this worktree: `src/doctor.rs`, `src/inference/gpu/`.

- `GpuContext::init` asks wgpu for the **Vulkan backend only** (`Backends::VULKAN`), high-performance
  adapter, no features (`Features::empty()`), limits = adapter limits. So: no Metal on macOS, no DX12, no
  f16, no subgroups, no pipeline cache.
- Shaders: `gemv_f32`, `gemv_q4_k`, `gemv_q5_k`, `gemv_iq4_xs` (WGSL). Q4_K uses a 32-float shared array
  for the reduction.
- Per call, `dispatch_gemv_single` writes params and input, builds a bind group, submits, maps a staging
  buffer and waits. That is one GPU round trip per matrix-vector product, and is the most likely reason
  for the local observation below. The trait `GpuBackend` has only `compute_linear`, `compute_score_head`,
  `is_format_supported`, `device_name`.
- Weight buffers are cached per tensor (`get_or_create_gpu_buffer`, a `OnceLock`), so weights are
  already resident after first use; activations and KV are not.
- Local observation, not a committed report: one run, SmolLM-135M, 128 tokens, RTX 4060 Ti: 4.3 s on GPU
  against 1.1 s on CPU. One sample on one machine; no claim is made from it.
- `doctor` reports only `gpu_adapter: Option<String>` (the name of the first Vulkan adapter), and says no
  speed is claimed. No VRAM, driver version, device type.
- `docs/research/engine-autotuning.md` (on main; read headings and its table) proposes `thme plan-run`: facts,
  memory model, probe, CPU-wins rule. This note supplies the device layer that plan consumes
  (THME-3055 to THME-3066 in BACKLOG.md).

## 6. Proposal

### 6.1 Order of work

1. Fix the wgpu path first. Honest hypothesis, to be tested: the slowdown comes from per-row sync, so
   (a) batched prefill GEMM, (b) all activations and KV resident, (c) one command buffer per layer, one readback
   per token, (d) subgroup reduction and f16 where reported. Measure before and after with a committed report.
2. Enable DX12 and Metal in the wgpu instance (`Backends::PRIMARY`), so Windows and macOS users with
   non-Vulkan drivers get a GPU. Keep Vulkan preferred on Linux.
3. Detection layer and `HardwareReport` (below), with fix-it advice.
4. Only if the committed report shows wgpu well short of reported native numbers on a vendor, add a native
   backend for that vendor: CUDA driver API via dlopen first (NVIDIA prefill is where the reported gap is
   largest), Metal second. Not before.
5. Never ship ROCm or SYCL backends; Vulkan covers AMD and Intel and has no GPU-list wall.

### 6.2 Trait and capability struct (sketch)

```rust
pub struct GpuCaps {
    pub backend: ApiKind,            // Vulkan | Dx12 | Metal | Cuda
    pub vendor: Vendor,              // Nvidia | Amd | Intel | Apple | Other
    pub device_type: DeviceKind,     // Discrete | Integrated | Software | Unknown
    pub name: String,
    pub driver: Fact<String>,
    pub vram_total: Fact<u64>,       // dedicated; Unknown if not queryable
    pub vram_free: Fact<u64>,
    pub shared_budget: Fact<u64>,    // unified or shared memory the OS offers
    pub max_buffer: u64,
    pub max_binding: u64,
    pub subgroup: Option<(u32, u32)>,
    pub f16: bool,
    pub coop_matrix: Vec<CoopTile>,  // empty when absent
}

pub trait GpuBackend: Send + Sync {
    fn caps(&self) -> &GpuCaps;
    fn supports(&self, d: QuantDtype) -> bool;
    fn upload(&self, t: &QuantizedTensor) -> Result<WeightHandle, ThmeError>;
    fn linear(&self, w: &WeightHandle, x: &[f32], rows: usize) -> Result<Vec<f32>, ThmeError>; // rows>1 = prefill
    fn layer(&self, l: &LayerHandle, state: &mut DeviceState) -> Result<(), ThmeError>;       // whole layer, no readback
    fn logits(&self, state: &DeviceState) -> Result<Vec<f32>, ThmeError>;                      // the one readback
}
```

Keep `compute_linear` as a thin shim so existing callers compile. `Fact<T>` is the existing doctor type
(checked / unknown).

### 6.3 Detection and `HardwareReport`

`probe_hardware() -> HardwareReport { adapters: Vec<GpuCaps>, advice: Vec<FixIt>, cpu: ... }`:

1. Enumerate all wgpu adapters across backends (not only the first Vulkan one).
2. Add memory facts per vendor through optional dlopen probes (NVML, sysfs, Level Zero, DXGI, Metal), each
   behind `libloading`, each failing to `Unknown`.
3. Map facts to `FixIt` using the table in section 2.3; advice text lives in one table with a test per row.
4. `doctor` prints it; `thme doctor --json` gets a schema entry. No speed claim.

### 6.4 Probe feeds the planner

`HardwareReport` becomes the input of THME-3055 (facts). THME-3059 (roofline) takes bandwidth from a one-second
copy kernel. THME-3061 (GPU probe, CPU-wins rule) runs the real dispatch pattern, so a per-row-dispatch GPU path
loses the probe and the planner picks CPU: today's behaviour made explicit. When the fixed path wins, the same
rule picks GPU with no code change. THME-3066 (Windows shared-memory guard) uses DXGI budget.

### 6.5 Test strategy without a GPU in CI

- `FakeBackend` implementing `GpuBackend` on the CPU reference code: tests the trait, planner and routing.
- Golden outputs: mock model from `examples/generate_mock_model.rs`; CPU result committed as the golden
  file; every backend must match within a stated tolerance (for example max abs error 1e-3 in f32, looser for f16).
- `GpuCaps` fixtures per vendor (JSON) drive `FixIt` and planner tests; no hardware required.
- Real-GPU tests are `#[ignore]` and run on the owner's machine; their results go to `docs/benchmarks/` as a
  report (pending until run; agents do not run models).
- Software adapters (llvmpipe, lavapipe) on CI: optional job, not blocking; not verified that the
  current runners expose one.

### 6.6 Tickets (BACKLOG.md style)

### [THME-30xx] Enumerate all adapters and add vendor, device type, features to `HostFacts`

- **Milestone**: v0.3.0 · **Priority**: p1 · **Area**: doctor · **Type**: feature · **Status**: triage
- **Done means**: `GpuCaps` struct filled from wgpu for every adapter on all backends; unknown stays `Unknown`.
- **Measured how**: unit tests with fixture adapters; `thme doctor --json` validates against schema.

### [THME-30xx] Fix-it advice table

- **Done means**: `FixIt` table from section 2.3, one message per situation, no absolute paths.
- **Measured how**: one test per row from a `GpuCaps` fixture; markdown lint on the manual page.

### [THME-30xx] Optional dedicated-VRAM probes behind dlopen (NVML, sysfs, Level Zero, DXGI)

- **Done means**: each probe in its own module, `libloading`, any error becomes `Unknown`, no link-time dependency.
- **Measured how**: `ldd`/`dumpbin` check in CI shows no vendor library linked; unit tests parse canned probe outputs.

### [THME-30xx] Enable Metal and DX12 in the wgpu instance

- **Done means**: `Backends::PRIMARY` with Vulkan preferred on Linux; `DeviceArg` unchanged.
- **Measured how**: `cargo test` on the three-OS CI matrix; adapter list printed in `doctor`.

### [THME-30xx] `FakeBackend` and backend parity test harness

- **Done means**: CPU-backed `GpuBackend` impl plus a test that runs mock-model layers on every backend.
- **Measured how**: parity within a stated tolerance on the mock model; runs in CI without a GPU.

### [THME-30xx] Batched prefill linear (`rows > 1`) in WGSL for F32 and Q4_K

- **Done means**: one dispatch for N tokens against one weight read; CPU parity test.
- **Measured how**: parity test in CI; prefill tokens per second before and after in a report under
  `docs/benchmarks/` (pending hardware run).

### [THME-30xx] Resident activations and KV cache on device

- **Done means**: no per-matmul upload or readback inside a layer.
- **Measured how**: counter of `queue.submit` and `map_async` calls per token, asserted in a test with a call-counting wrapper.

### [THME-30xx] One command buffer per layer, one readback per token

- **Done means**: `layer()` and `logits()` in the trait; bind groups built once at load.
- **Measured how**: submit count per token equals the layer count or less, one readback; test with the counting wrapper.

### [THME-30xx] Subgroup reduction variant of the Q4_K, Q5_K, IQ4_XS shaders

- **Done means**: chosen only when `Features::SUBGROUP` is present; fallback to current shaders.
- **Measured how**: CPU parity on mock tensors; per-kernel timing in a pending-hardware report.

### [THME-30xx] f16 activations path behind `SHADER_F16`

- **Done means**: f16 variant for decode matvec; off by default until parity is shown.
- **Measured how**: parity test with a looser tolerance, documented; timing report pending.

### [THME-30xx] Cooperative-matrix prefill kernel (optional, guarded)

- **Done means**: used only when `cooperative_matrix_properties()` lists a usable tile; query wrapped so a driver fault cannot abort `doctor`.
- **Measured how**: parity on mock tensors on a machine with the feature (ignored test); timing report pending.

### [THME-30xx] Large-tensor binding split

- **Done means**: tensors above `max_binding` are split across bindings; error with advice if still too big.
- **Measured how**: unit test with a fake limit of a few KiB; parity against the unsplit result.

### [THME-30xx] Binary-size record per feature flag

- **Done means**: release size of CLI and GUI with and without `gpu-cuda` and `gpu-metal`, committed as a report.
- **Measured how**: CI step prints `ls -l` of the stripped binaries; fails above 16 MiB (CLI) or 15 MiB (GUI).

### [THME-30xx] CUDA driver-API backend via dlopen (only if the report justifies it)

- **Before building**: THME-3061 report shows fixed wgpu on NVIDIA well below reported native prefill.
- **Done means**: `gpu-cuda` feature, `libloading`, PTX kernels for F32 and Q4_K, falls back with advice when `libcuda` is missing.
- **Measured how**: parity test (ignored, needs an NVIDIA GPU); `ldd` shows no `libcuda` link; before/after report pending.

### [THME-30xx] GPU probe uses the real dispatch pattern (links THME-3061)

- **Done means**: probe runs one layer of synthetic work through the same trait calls as inference, 2 s timeout.
- **Measured how**: with `FakeBackend` slow, plan picks CPU; with fast, picks GPU; real-machine report pending.

## 7. Honest limits

- No benchmark here was run by THME. The scoreboards are community posts; builds, drivers and flags differ.
- Several vendor facts could not be read from a primary page and are marked "not verified" (CUDA driver per
  toolkit table, NVML function names, Windows HIP list, wgpu int8 dot product, DX12 cooperative matrix, Apple
  page text, naga licence).
- Licence readings are engineering notes, not legal advice. The owner reviews anything bundled.
- The claim "per-row sync causes the 4.3 s versus 1.1 s result" is a hypothesis from reading the code, not a
  measurement.
