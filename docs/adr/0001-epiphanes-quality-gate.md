# ADR-0001: Distribution-Free Three-State Epiphanes Quality Gate

- **Status:** Accepted
- **Date:** 2026-11-26
- **Context:** Classical Student-$t$ confidence intervals, Pearson correlation, and fixed-horizon Empirical Bernstein bounds either assume Gaussian i.i.d. residuals or require a prior upper bound $M$ (introducing data-snooping bias). Furthermore, [CONTEXT.md](../../CONTEXT.md) requires a three-state `Decision` (`pass`, `fail`, `inconclusive`) rather than a binary pass/fail.
- **Decision:** Adopt a three-part distribution-free `Quality gate` in `src/quality/gate.rs`:
  1. **Mixture-GKL E-Process** $E_t^{\mathrm{mix}}$ with hard lower bound $\tilde{\delta}_b \ge -1$ (anytime-valid two-sided bounds $[L_t, U_t]$).
  2. **Kusuoka-Coherent $\mathrm{WCVaR}$** $\rho_\phi(\tilde{\delta})$ ($\phi(\beta)=2\beta$) for subadditive tail-risk composition across stages.
  3. **Asymmetry-Adaptive Jeffreys–Rényi** $J_\alpha^{\mathrm{AA}}$ on FP32 Top-$K$+Tail ($K=256$) with data-driven MGF certificate.
- **Consequences:** Eliminates normality assumptions, prevents snooping bias, composes cleanly across repair stages, and natively outputs `pass` ($U_t \le \tau$), `fail` ($L_t > \tau$), or `inconclusive` ($L_t \le \tau < U_t$).
