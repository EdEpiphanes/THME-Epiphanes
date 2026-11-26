# ADR-0003: BITCOS 1,024-Weight Hierarchical Dual-Scale Super-Block

- **Status:** Accepted
- **Date:** 2026-11-26
- **Context:** Naive `5-trit` packing (`1.625` nominal bits/weight) ignores the $38\text{–}51.5\%$ zero fraction in ternary weights. A 256-weight super-block with per-tile headers adds `0.625` bpw overhead (`2.125` bpw total, losing to `5-trit`), while a single block-level `f16` scale causes cross-tile clipping and fixed `u4` steps saturate above $181\times$ dynamic range.
- **Decision:** Standardize `.epiph` sparse ternary storage on a **1,024-weight super-block** (8 warp-groups of 128 weights) with a **128-bit GPU-aligned header**: `[u16 base_off | u8 grp_nz_cnt[7] | u4 grp_lscale[8] | f8_e4m3 d_log | f16 block_min | u32 nz_mask[32] | packed_signs]`. Group prefix sums execute via a 3-cycle warp shuffle (`__shfl_up_sync`).
- **Consequences:** Header overhead drops to `0.1250` bpw, yielding `1.610` nominal bits/weight at $p_0 = 0.515$ (beating `5-trit`) with zero saturation and $O(1)$ warp-group random access.
