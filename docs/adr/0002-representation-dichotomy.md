# ADR-0002: Representation Dichotomy (Sinkhorn + Hadamard vs. Sinkhorn-Only)

- **Status:** Accepted
- **Date:** 2026-11-26
- **Context:** Orthogonal Walsh–Hadamard rotation $H_n$ suppresses channel outliers for dense/lattice formats (`Q4_0`, `E8P`, `INT4` KV), but applying $H_n$ to sparse ternary weights ($p_0 \in [0.38, 0.515]$ zeros) destroys zero sparsity and forces dense storage.
- **Decision:** Enforce a strict **Representation Dichotomy** in `src/engine/basis.rs` and `src/engine/transport.rs`:
  1. **Dense / Lattice Branch:** Column-side Sinkhorn $S$ followed by orthogonal $H_n$ ($X' = (XS^{-1})H_n^\top$, $W' = WSH_n^\top$).
  2. **Sparse Ternary Branch (`BITCOS`):** Column-side Sinkhorn $S$ only ($X' = XS^{-1}$, $W' = WS$), preserving $w_{ij}=0$ identically.
  3. **KV-Cache RoPE Commutativity:** Keep high-kurtosis RoPE pairs in pre-RoPE `KIVI` `INT8`/`F16`, and apply $H_{d_h}$ + `RIP` $\Phi$ *post-RoPE* on both $q_m$ and $k_n$.
- **Consequences:** Preserves exact linear equivalence $X'(W')^\top = XW^\top$ and RoPE positional phase while maximizing compression efficiency per codec family.
