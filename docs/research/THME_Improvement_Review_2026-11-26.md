> **Status:** review notes written by Antigravity on 26 November 2026 for the advisor. Perf figures are as reported in the cited handoffs; the web-sourced patterns in sections 5 to 8 are a literature synthesis that THME has **not measured or verified** (AGENTS.md Rule 2). Treat every number as reported, and the proposals as unreviewed.

# THME Technical Review, Math Audit & Improvement Suggestions for Claude (`2026-11-26`)

*Companion to `claude-handoff-2026-11-26.md` (read-only audit of `feat/kv-cache-gpu-decode` at `cc8bd8f`).*

---

## 1. Why `Q4_0` Decode Is `16.88 tok/s` (vs `F16` `62.87 tok/s`) & How to 5–10× It

### Finding 1A: `forward_decoder_extend` Has No CUDA Graph Capture (~337 Host Launches/Token)
- **Where:** `src/inference/gpu/cuda.rs:1978-2308`
- **Issue:** Unlike `forward_decoder`, `forward_decoder_extend` launches ~14 kernels/GEMMs per layer × 24 layers + head GEMM (~337 driver calls) on every single generated token (`t == 1`). On Windows WDDM, host launch overhead alone is ~3–4 ms/token.
- **Fix:** Capture `forward_decoder_extend` for `t == 1` as a `SendCudaGraph`.
  - **Key trick:** `offset` (and `total_kv`) changes by `+1` each token. Pass `offset` via a 1-element `CudaSlice<u32>` (`ws.kv_offset`) updated with `memcpy_htod` *before* `graph.launch()` (same pattern as `ws.cos`/`ws.sin`). Then `kv_append_f16` and `kv_attention_f16` read `*offset_ptr` from device memory, keeping the graph topology and all device pointers 100% invariant across all decode tokens!

### Finding 1B: `gemv_q4_0_aligned` Shared-Memory & Arithmetic Bottlenecks
- **Where:** `src/inference/gpu/kernels/decoder.cu:832-891` and `cuda.rs:1057-1087`
- **Issues:**
  1. **50% SM Occupancy on `wd` (`in_dim = 8192`):** `extern __shared__ float sx[]` allocates `in_dim * 4 = 32 KB` per block (`cuda.rs:1068`), capping Ada (RTX 4060 Ti, 100 KB SMEM/SM) to 3 blocks/SM (24 of 48 warps).
  2. **Scalar inner loop:** `for (w = 0..4) for (k = 0..8)` issues 32 scalar `sx[base + k]` loads and 32 `(float)q` int-to-float casts per block.
- **Fixes (in order of impact):**
  1. **Store `sx` as `half` (`u16`, 16 KB) or `Q8_1` (9 KB) in SMEM:** Immediately halves SMEM footprint and doubles occupancy on `inter = 8192`.
  2. **Use `Q8_1` activation + `__dp4a` (CUDA `dp4a.s32.s32`):** Quantize `x16` once per layer into `Q8_1` (`int8` + `half` scale per 32 elements), then compute 4× `int8 × int8` dot products per `__dp4a` instruction (8 `__dp4a` ops per 32-element block instead of 32 scalar `int->float` + `fma`s).
  3. **Vectorize `sx` loads:** Even in FP16/FP32, load `float4` / `half2` from `sx` and unpack 2 nibbles at a time with `__hmul2` / `fma`.

### Finding 1C: `kv_attention_f16` Wastes 87.5% of Block Threads During `t == 1` Decode
- **Where:** `src/inference/gpu/kernels/decoder.cu:714-830`, `cuda.rs:1032-1037`
- **Issue:** `launch_kv_attention_f16` always launches `blockDim = (32, 8, 1)` (8 warps = 256 threads). When `t_len == 1` (decode), `t = blockIdx.y * 8 + warp`, so `warp 0` has `t = 0 < 1` and **warps 1..7 (`224/256` threads) sit idle** at `if (t < t_len)` (line 788) while `warp 0` sequentially loops `j = 0..total_kv`.
- **Fix:** Add a `t_len == 1` FlashDecoding / warp-parallel path in `kv_attention_f16` (or a dedicated `kv_decode_f16` kernel): all 8 warps cooperate on `kb = 0..total_kv` (each warp handles `kb + warp` or splits `total_kv` into 8 slices), then merge `(m_i, l_i, acc[r])` across the 8 warps in shared memory using the standard online-softmax log-sum-exp reduction:
  $$m = \max_w m_w, \quad l = \sum_w l_w e^{m_w - m}, \quad \text{acc} = \frac{1}{l} \sum_w e^{m_w - m} \text{acc}_w.$$

### Finding 1D: `t == 1` Head Projection Uses `gemm_f16_preconverted` Instead of `gemm_f16_to_f16` / `gemv`
- **Where:** `cuda.rs:2273-2275`
- **Issue:** `self.gemm_f16_preconverted(&ws.x16, &head, &mut ws.logits, 1, d, plan.vocab)` runs the generic tiled `fn_gemm_f16_tiled` kernel on a `1 × 2048 × 49152` matrix (`201 MB` weight read per token).
- **Fix:** Route `t == 1` `head` through a dedicated `gemv_f16` (or `gemv_q4_0_aligned` when `head` is quantized) to saturate memory bandwidth without 16×16 tile waste.

---

## 2. Math & Numerical Precision Audit

1. **`gemv_q4_0_aligned` Nibble Ordering (`decoder.cu:876`):**
   - `(int)((word >> (k * 4u)) & 0xfu) - 8` with `k = 0..7` matches GGML `Q4_0` nibble packing (low nibble first, zero-point `-8`). Verified consistent with `resident_q4_0` and CPU parity tests.
2. **Double-Quantization Risk on `.epiph` / GGUF Models:**
   - Currently `resident_q4_0` (`cuda.rs`) re-quantizes CPU `f32` weights into `Q4_0` on the fly. For models already stored in `.epiph` `Q4_0` (or `Q8_0`), expanding to `f32` on CPU and re-quantizing wastes ~1.8 s at startup and introduces double-quantization error.
   - **Recommendation:** Add a direct packed-slice path (`resident_q4_0_raw(scales, qs)` / `resident_f16_raw(u16_slice)`) from `.epiph` mmap directly to `CudaSlice` without intermediate `Vec<f32>`.
3. **`kv_attention_f16` Online Softmax (`decoder.cu:767-828`):**
   - Uses `float` accumulators (`m_i`, `l_i`, `acc[8]`, `q_reg[8]`) and `cvt.f32.f16` on K/V loads, converting to `f16` only on final writeback (`cvt.rn.f16.f32`). Numerically sound; `m_i` initialized to `-FLT_MAX` (`-3.402823466e+38f`), safe against `expf(m_i - m_new)` underflow on the first element (`expf(-inf) == 0`).

---

## 3. Pushing Prefill Toward `15k+ tok/s` (`9,268 tok/s` Baseline)

1. **Zero-Extra-VRAM Weight Concatenation for `QKV` and `Gate+Up`:**
   - Earlier `resident_f16_pair` regressed because it kept *duplicate* weights in VRAM.
   - Instead, concatenate `[Wq; Wk; Wv]` (and `[Wgate; Wup]`) **once** during `resident_f16` upload (freeing the individual slices) so `QKV` is 1 cuBLAS GEMM (`t × d -> t × (d + 2*kv_dim)`) and `Gate+Up` is 1 cuBLAS GEMM (`t × d -> t × 2*inter`). That eliminates 72 cuBLAS calls per forward pass on 24 layers with zero extra VRAM.
2. **Fused `swiglu_f16_in` on Strided `[Gate | Up]` Output:**
   - Read `gate` from `col 0..inter` and `up` from `col inter..2*inter` of a single `gate_up16` buffer (or split-column stride) to pair with the fused `Gate+Up` GEMM.

---

## 4. Prioritized Checklist for Next Session

1. **[High ROI, Low Risk]** Add `SendCudaGraph` + `ws.kv_offset` device buffer to `forward_decoder_extend` (`t == 1`) -> eliminates ~337 launches/token.
2. **[High ROI]** Halve `gemv_q4_0_aligned` SMEM (`half` or `Q8_1` + `__dp4a`) and add warp-split `kv_decode_f16` for `t == 1` attention.
3. **[Architecture]** Direct `.epiph` packed `Q4_0` / `F16` upload (skip CPU `Vec<f32>` expansion).
4. **[Cross-Vendor + Eval]** `wgpu` `forward_decoder_extend` + `gemv_q4_0`, `bits_per_byte` in `thme evaluate`, `src/bench/` runner, and `thme-gui` release size check (`<= 25 MiB`).

---

## 5. External Research & Industry Patterns (Web Synthesis, `2026-11-26`)

1. **`llama.cpp` `mmvq` / `vec_dot_q4_0_q8_1` + `__dp4a` Pattern:**
   - Memory-bound `B=1` decode in `ggml-cuda` pairs `Q4_0` weights with on-the-fly `Q8_1` (int8 + scale + block sum) activations, then executes `__dp4a(srcA, srcB, c)` (4× int8 MACs in 1 cycle on integer ALUs) while pre-loading weight blocks into registers and pairing with CUDA Graphs (`GGML_CUDA_GRAPH_OPT=1`) to eliminate CPU launch overhead.
   - **Direct takeaway for THME:** Adding a tiny `quantize_x16_to_q8_1` kernel (or warp-inline conversion into SMEM `Q8_1`) and switching `gemv_q4_0_aligned` to `__dp4a` aligns THME directly with `llama.cpp`'s fastest Pascal-through-Ada decode path.

2. **Dynamic KV-Cache Offsets inside CUDA Graphs (vLLM / FlashInfer Pattern):**
   - Production engines avoid `cuGraphExecKernelNodeSetParams` overhead on per-token scalar tweaks by keeping **all device pointers constant** at capture time and mutating a pre-allocated static metadata buffer (`ws.kv_offset`, `indptr`/`indices`) via `cudaMemcpyAsync` on the active stream *immediately before* `cuGraphLaunch`.
   - **Direct takeaway for THME:** Use a 1-element `CudaSlice<u32>` (`ws.kv_offset`) in `CudaWorkspace`, update it with `memcpy_htod` before `graph.launch()`, and capture the entire `t == 1` `forward_decoder_extend` loop once per model/workspace.

3. **Cross-Vendor Pure-Rust Vulkan (`ash` vs `wgpu`):**
   - `wgpu` exposes `Features::EXPERIMENTAL_COOPERATIVE_MATRIX`, but WGSL lags behind raw SPIR-V/GLSL for `VK_KHR_cooperative_matrix` (`16×16×16` `f16 -> f32`) and subgroup-scoped `coopMatMulAdd`.
   - Production Rust Vulkan engines use **`ash`** directly with `VkPhysicalDeviceCooperativeMatrixFeaturesKHR` + `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` and pre-compiled SPIR-V (`GL_KHR_cooperative_matrix`), keeping subgroup control flow uniform when unpacking 4-bit weights.

---

## 6. Deep Research: Cross-Vendor GPU Translation Layers & Pure-Rust Backends

| Approach | Vendor Coverage | Cooperative / Matrix Core Support | Binary & Runtime Fit for THME (`<=16 MiB` CLI / `<=25 MiB` GUI) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Native `cuda.rs` + `wgpu` (Current)** | NVIDIA (CUDA) + AMD/Intel/Apple (`wgpu`) | NVIDIA Tensor Cores (`cuBLAS`); `wgpu` lacks stable `VK_KHR_cooperative_matrix` | Embedded `decoder.ptx` (`111 KB`) + Pure-Rust `wgpu`; zero external DLLs required | **Keep as base** |
| **`ash` + `VK_KHR_cooperative_matrix` (SPIR-V)** | AMD RDNA3/4, Intel Arc/Xe2, NVIDIA, Mobile | Native `coopMatMulAdd` (`16×16×16` `f16 -> f32`) at subgroup scope | Pure-Rust `ash` (~thin C-struct bindings) + precompiled SPIR-V compute shaders | **Best Pure-Rust upgrade** to close AMD/Intel prefill gap |
| **`CubeCL` (Tracel / Burn)** | CUDA, HIP/ROCm, Vulkan, Metal, CPU | Native Tensor Core / TMA + portable `#[cube]` Rust kernels | Heavy JIT compiler & LLVM/codegen dependencies — risk to `16 MiB` CLI footprint | **Evaluate footprint** before adopting |
| **ZLUDA** | AMD Radeon (CUDA binary translation) | Translates CUDA WMMA to HIP | External runtime/driver hook, fluctuating maintenance & fragile on Windows | **Reject** (ADR-0001) |
| **HIP / ROCm (`hip-sys`)** | AMD (primary), NVIDIA | Native MFMA / WMMA via `hipBLASLt` | Requires multi-GB ROCm/HIP runtime SDK; poor Windows consumer support | **Reject** for core distribution |
| **SYCL / oneAPI** | Intel Arc/Max, AMD, NVIDIA | `sycl::ext::oneapi::experimental::matrix` | C++ runtime / DPC++ toolchain dependency; breaks Pure-Rust goal | **Reject** |

---

## 7. Deep Research: Converting `SmolLM2-1.7B` to 1.58-Bit Ternary (`{-1, 0, +1}`)

Converting a dense FP16/BF16 model like `SmolLM2-1.7B` to ternary (`log2(3) ≈ 1.585 bits/weight`) **cannot** rely on naive AbsMean RTN alone (which causes perplexity collapse on 1.7B models). There are two viable paths depending on compute budget:

### Path A: Zero-Training / Fast PTQ Pipeline (Minutes on 1× RTX 4060 Ti 16 GB)
To avoid perplexity collapse in pure PTQ without 10B-token retraining:
1. **Walsh-Hadamard / Randomized Hadamard Rotation (RHT, QuaRot/SpinQuant/HARP style):**
   - Multiply activations & weights by an orthogonal Hadamard matrix $H$ ($H^\top H = I$) before quantization to smear outlier channels into a near-Gaussian distribution without changing exact FP math.
   - *(Note: THME already has `v0.2.2` Hadamard rotation & `BitNetPacker` planned/scaffolded in `src/exporter/bitnet_pack.rs`.)*
2. **Iterative Grid Fitting (HQQ-style) + Second-Order Compensation (Ternary-GPTQ):**
   - Replace naive `AbsMean` threshold with alternating $L_2$ minimization (`HQQ`) for optimal per-group/per-row scale $\gamma$, followed by damped Hessian error compensation (`GPTQ`) so rounding errors don't compound across all 24 layers.
3. **Selective Mixed-Precision Safety Net (THME's Core Strength):**
   - Keep `embed_tokens`, `lm_head`, `attn_norm`, `mlp_norm`, and the most sensitive attention/first-last layers at `Q4_0`/`Q8_0`/`F16` via THME's DP Pareto allocator, while pushing insensitive MLP/middle layers to `1.58-bit` ternary — preventing collapse while retaining most of the memory savings.

### Path B: Full Quality Conversion via `BitDistill` + QAT (`microsoft/BitNet`, Wu et al. 2025)
For near-lossless 1.58-bit conversion of dense 0.6B–4B models:
1. **Replace `nn.Linear` with `BitLinear` + `SubLN`:**
   - Insert **SubLN (Sub-Layer Normalization)** before the output projections of MHSA (`o_proj`) and FFN (`down_proj`) to stabilize extreme low-bit activation variance.
   - Forward: quantize activations to `INT8` (`absmax`, `[-128, 127]`) and weights to ternary `{-1, 0, +1}` via `round(clip(W / mean(|W|), -1, 1))`, using **Straight-Through Estimator (STE)** (`w_ternary = w_quant.detach() - w.detach() + w`) to update FP32 shadow weights.
2. **Continual Pre-Training Warm-up + Dual-Signal Distillation:**
   - Freeze FP16 `SmolLM2-1.7B` as Teacher; train 1.58-bit Student with:
     - **CE Loss** + **MiniLM Multi-Head Attention Relation Distillation** + **Logits KL Distillation** ($\tau = 5$).
     - Reference `BitDistill` hyperparameters: `Adam` ($\beta=(0.9, 0.98)$, $\epsilon=10^{-6}$), max LR `1e-4` (linear decay), ~10B tokens for full continual pre-training warm-up (smaller LoRA/QAT recovery runs can still improve PTQ checkpoints on 1× 4060 Ti).

### Inference Packing & Kernels (`bitnet.cpp` vs THME `.epiph`)
- **Packing Formats:**
  - `I2_S`: 2 bits/weight (`-1 -> 00`, `0 -> 01`, `+1 -> 10`, `4 weights/byte`) + FP16/FP32 block scale — matches THME's existing `BitNetPacker` (`src/exporter/bitnet_pack.rs`).
  - `5-trits/byte` (`3^5 = 243 < 256`) / `TL1`/`TL2`: Ultra-dense 1.6-bit packing with Lookup Table (LUT) dot-product acceleration (replacing multiplications with precomputed partial-sum table lookups).
- **GPU Kernel Opportunity for THME:**
  - Add a CUDA `gemv_i2_s_q8_1` kernel (2-bit ternary weights × `Q8_1` activations using bitwise sign/mask or `dp4a` / LUT in shared memory) and store `BitNet1_58` weights directly in `.epiph` so `SmolLM2-1.7B` ternary/mixed-precision models load zero-copy to GPU without CPU `f32` expansion.

---

## 8. Cross-Disciplinary Mathematical Foundations & Novel Packing Techniques for THME / `.epiph`

Moving beyond standard scalar rounding (`Q4_0`, `AbsMean`) unlocks 1.48–2.0 bpw compression and faster memory-bound GPU decode by importing exact theorems from **lattice theory, information theory, Riemannian geometry, optimal transport, and compressed sensing**:

### 8.1 Optimal Sphere Packing & Channel Coding: $E_8$ Lattice (`QuIP#`) & Bitshift Trellis (`QTIP`)
- **Mathematical Formulation ($E_8$ Root Lattice):**
  In $\mathbb{R}^8$, the $E_8$ lattice achieves the provably densest sphere packing (Viazovska):
  $$E_8 = \left\{ x \in \mathbb{Z}^8 \;\cup\; \left(\mathbb{Z} + \tfrac{1}{2}\right)^8 \;:\; \sum_{i=1}^8 x_i \equiv 0 \pmod{2} \right\}.$$
  Paired with **Randomized Hadamard Transform (RHT)** incoherence ($W' = H_L \text{diag}(s_L)\, W\, \text{diag}(s_R) H_R^\top$), weight blocks become isotropic sub-Gaussian vectors in $\mathbb{R}^8$.
- **Zero-Codebook Decoding:** Instead of storing an explicit $O(2^{bd})$ lookup table, the **Conway-Sloane algorithm** projects any $v \in \mathbb{R}^8$ to its nearest $E_8$ point in $O(1)$ arithmetic (round to $\mathbb{Z}^8$ and $(\mathbb{Z}+\frac{1}{2})^8$, flip the worst coordinate if parity is odd, pick closer) — giving **~30% lower MSE distortion than 2-bit scalar quantization** with zero codebook memory.
- **Scaling to $d > 100$ (`QTIP`):** Replace 8D lattices with **Trellis Coded Quantization (TCQ)** — a finite-state convolutional trellis solved via **Viterbi** at encode time, and decoded on GPU with a **lookup-free bitshift trellis** (`1MAD` / `3INST`) in $\le 4$ GPU instructions/weight.

### 8.2 Skewed-Entropy Ternary Packing: `BITCOS` Bitmap+Sign vs 5-Trit Base-3 & Warp-Chunked tANS
- **Why 5-Trit Packing ($3^5 = 243 \le 256$) Leaves Bandwidth on the Table:**
  Equipartition assumes $p_{-1}=p_0=p_{+1}=\frac{1}{3}$, giving $H = \log_2 3 \approx 1.585$ bits/trit (`1.60–1.625` bpw aligned). Real trained/distilled ternary LLMs exhibit **38%–51.5% zeros** ($p_0 \in [0.38, 0.515]$), lowering true Shannon entropy to:
  $$H(p) = -p_0 \log_2 p_0 - (1 - p_0)\log_2\!\left(\frac{1 - p_0}{2}\right) \approx 1.46\text{–}1.54\text{ bits/trit}.$$
- **GPU-Parallel Solution (Warp-Synchronous Bitmap + Compacted Sign Vector):**
  Sequential ANS (`tANS`) stalls GPU warps due to serial state dependencies. Instead, pack each 32-weight warp tile as:
  1. **32-bit Non-Zero Presence Bitmap** (`nz_mask`, 1 bit/weight): `1` if $w_i \neq 0$, `0` if $w_i = 0$.
  2. **Compacted Sign Bits** (`sign_bits`, `popc(nz_mask)` bits): `0` for $-1$, `1` for $+1$.
  At $p_0 = 0.515$, average size drops to $1 + (1 - 0.515) = \mathbf{1.485\text{ bpw}}$ (beating 5-trit packing!), and each CUDA lane decodes its weight in parallel using hardware `__popc(nz_mask & ((1u << lane) - 1))` + bit-extract — **zero serial loops, zero LUT bank conflicts**.

### 8.3 Riemannian Information Geometry (`Fisher-Rao`) + Entropic Optimal Transport (`Sinkhorn`)
- **Fisher-Rao Sensitivity for Mixed-Precision DP:**
  Treat the model parameter space as a Riemannian manifold with Fisher metric $g_{ij}(\theta) = \mathbb{E}_{x,y}[\partial_i \log p_\theta(y|x)\,\partial_j \log p_\theta(y|x)]$. The KL divergence induced by quantization perturbation $\Delta\theta = \tilde{\theta} - \theta$ is locally:
  $$D_{\mathrm{KL}}(p_\theta \,\|\, p_{\tilde\theta}) \approx \tfrac{1}{2} \Delta\theta^\top F(\theta)\, \Delta\theta.$$
  Using diagonal/K-FAC Fisher blocks in THME's DP allocator (`v0.2.5.2`) bounds output distribution drift rather than naive weight-space $L_2$ error.
- **Sinkhorn Optimal Transport for Outlier Migration:**
  Given activation channel magnitudes $\mu$ and weight column capacities $\nu$, solve the entropically regularized Wasserstein problem:
  $$\min_{P \in \Pi(\mu,\nu)} \langle C, P \rangle - \varepsilon H(P) \quad\text{via Sinkhorn scaling } u \leftarrow \mu \oslash (Kv),\; v \leftarrow \nu \oslash (K^\top u)$$
  to compute smooth diagonal rescaling $S$ ($W' = W S$, $X' = X S^{-1}$) that equalizes dynamic range before 1.58-bit/2-bit quantization.

### 8.4 Compressed Sensing (`RIP`) & Fast Walsh-Hadamard Transforms for KV Cache & Weights
- **RIP Inner-Product Preservation:**
  If projection $\Phi \in \mathbb{R}^{m \times d}$ ($m \ll d$) satisfies the **Restricted Isometry Property** $(1-\delta)\|x\|_2^2 \le \|\Phi x\|_2^2 \le (1+\delta)\|x\|_2^2$, then attention dot products are preserved within $\pm \delta \|q\|_2 \|k\|_2$ by polarization:
  $$\left|\langle \Phi q, \Phi k \rangle - \langle q, k \rangle\right| \le \delta \|q\|_2 \|k\|_2.$$
- **Fast Walsh-Hadamard ($O(d \log d)$, Zero Parameters):**
  Applying $H_d$ in shared memory / warp shuffles (`__shfl_xor_sync` butterfly across 32 lanes!) before INT4 KV-cache quantization or `Q4_0`/`E8` weight quantization eliminates axis-aligned outliers with **zero stored projection matrix** and exact invertibility ($H_d^{-1} = H_d^\top$).

### 8.5 Concrete `.epiph` Implementation Roadmap
1. **CUDA Warp-Shuffle Walsh-Hadamard (`hadamard32_f16`):** 5 `__shfl_xor_sync` stages per 32-element block in registers — unlocks `QuaRot`-style outlier smoothing for both `Q4_0` weights and `INT4` KV cache.
2. **Sparse Ternary `BITCOS` Tensor Variant in `BitNetPacker` (`src/exporter/bitnet_pack.rs` + `.epiph`):** `[u32 nz_mask | packed_sign_bits | f16 scale]` decoded on GPU with `__popc` + `__dp4a(x_q8_1)` when $p_0 \ge 0.35$, falling back to 5-trit or `I2_S` when dense.
3. **$E_8$ Lattice `Q2` Block (`E8P`) in `.epiph`:** Pair `hadamard32` + Conway-Sloane $E_8$ encoding for 2.0 bpw layers that collapse under `Q4_0`/`BitNet1_58`.
