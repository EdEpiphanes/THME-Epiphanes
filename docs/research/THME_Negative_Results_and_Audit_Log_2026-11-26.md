# THME Negative Results, Dropped Claims & Audit Log (`2026-11-26`)

> **Purpose:** Separate ledger of all contradicted v3.1 claims, falsified hypotheses, regressed experiments, and quarantined literature. Keeping negative results here preserves full scientific honesty while keeping the main research specification ([THME_Research_Dossier_v3.3_2026-11-26.md](THME_Research_Dossier_v3.3_2026-11-26.md)) focused on validated mathematics and working systems.

---

## 1. Dropped Claims from Dossier v3.1 (`claim-register.md` & `REVIEW_dossier_v3.1.md`)

| ID / Topic | Original v3.1 Claim | Reason Dropped / Falsified |
| :--- | :--- | :--- |
| **V-04** | Multipliers consume 70–80% of AI processor energy | Contradicted for batch-1 LLM decode; VRAM/DRAM weight movement dominates energy and latency. |
| **V-08, V-13, V-14, V-15** | Calabi–Yau cycles store memory-free info; CY Laplacian $m^2 \propto n(n+d)/R^2$; 11D natural limit; `(3,2)` causal cone | Contradicted physics analogies (sphere formula misapplied; no ML/information-theoretic basis). |
| **Navier–Stokes stability** | Bounded energy guarantees no blowup | Misreads PDE regularity literature: blowup can occur despite bounded energy. |
| **"O(1) memory" without replacement state** | Attention + recurrence yields $O(1)$ cache | Contradicted: any retained attention layer keeps a $\Theta(N)$ KV cache in context length $N$. |
| **Clifford / stabilizer / torus / attractor core** | Core engine built on `Cl(p,q)` rotors and stabilizer codes | Zero empirical support; rotor norm preservation holds only for same-sign bivectors. |
| **4-bit phase LUT removes all FP logic** | Phase-only inference needs no floating-point adders | Contradicted: accumulation still requires Cartesian adders. |

---

## 2. Empirical Negative Results & Falsified Assumptions (`Measured, THME`)

1. **Additive Per-Layer Cost Assumption (`SmolLM-135M`, WikiText-2 16,384 tokens, baseline `17.97`):**
   - Quantizing one layer at a time to `TQ1_0` ternary gave per-layer PPL deltas summing to **`+1,509`** (`+0.9` to `+6.8` for L1–L28; `+13.1` for L29; `+1,429` for L0).
   - Quantizing all layers jointly gave **`+15,877,276`** (`PPL ≈ 1.6 × 10⁷`). Additive cost ranking fails under aggressive quantization.
2. **Untrained Early Exit (`SmolLM-135M`):**
   - Reading hidden states through the stored output head showed only **`1.6%`** of tokens settled by 30% depth and **`28.7%`** by 80% depth. Early exit requires trained exit heads.
3. **Untrained Block Looping (`SmolLM-135M`):**
   - Applying layers 7–22 twice without training raised PPL from `18.0` to `55.3` (`Δ +37.3`); skipping them raised PPL to `~30,800`.
4. **Absolute Gate Thresholds Across Scales (`Pythia-14M`):**
   - Uniform `Q8_0` passed the `0.45` absolute gate on `SmolLM-135M` (`19.80 → 19.83`, `Δ +0.03`) but **failed** on `Pythia-14M` (`114.29 → 115.12`, `Δ +0.83`).
5. **Data-Free `Q4_0` Clipping & `HQQ Lp=0.7` (`SmolLM2-1.7B`, 37-window VAL-A):**
   - Baseline `Q4_0`: `10.1298` PPL. `Quantile Clip` (`10.2326`) and `HQQ Lp=0.7` (`10.6262`) both **regressed** vs baseline, whereas `AWQ Q4_0` (`9.7125`) and `GPTQ-Diag Q4_0` (`9.9077`) improved.

---

## 3. Abandoned GPU Kernel Experiments (`SmolLM2-1.7B`, RTX 4060 Ti)

1. **`CUBLAS_COMPUTE_16F` on `PV` in `launch_attention_qkv16`:** Regressed throughput (`8,466` vs `9,506 tok/s`) due to cuBLAS strided batched GEMM fallback/precision path. Kept `COMPUTE_32F` on `PV`.
2. **`Gate + Up` Fusion via `resident_f16_pair`:** Regressed throughput (`3,271 / 4,732 tok/s`) because keeping paired copies increased VRAM cache pressure and host copy overhead. Replaced with zero-extra-VRAM concatenation-at-upload design.
3. **Raw `wmma` Register Fragments without Shared-Memory Store:** Produced `Non-finite output` at `t=1024` due to hardware fragment lane-mapping mismatch.

---

## 4. Quarantined / Out-of-Scope Literature & Licence Blocks

- **`MIRIX` (`arXiv:2507.07957`):** Agent-layer SQLite/PostgreSQL external memory store; zero model-weight or KV-cache applicability.
- **`Grassmann Flows` (`arXiv:2512.19428v1`):** LM validation PPL is **11% worse** than `TransformerLM` (`275.7 vs 248.4` at 6L; `261.1 vs 235.2` at 12L) with an unresolved causality issue (`t` paired with future `t + Δ`).
- **External Critique (`halawah-2026-thme-critique.md`):** Raised 4 structural risks against v2.0 (kernel unpacking overhead, 80/20 routing cascade, non-additive allocator, $\Theta(N)$ KV cache); all 4 are now directly resolved mathematically and empirically in v3.2.
- **Permissive-Only Licence Blocks:** `QTIP` (`GPL-3.0`), `QuIP#` (`GPL-3.0`), `LittleBit` code (`CC BY-NC 4.0`), `SpinQuant` (`CC-BY-NC 4.0` header), `FastFlowLM` (closed NPU binaries), `LFM2.5` (revenue-capped licence). Only clean-room equations from papers are used; zero code copied.
