# Paper v5 against the code on main: review findings, 27 November 2026

Scope: paper "Epiphanes Math" version 5 read against the code on main at commit a31ac66. The paper's own check scripts were not re-run for this review; numbers in section 5 were measured on main.

1. The quality gate code did not match the paper. `upper_bound_eprocess` in `src/quality/epiphanes.rs` was the version 3.3 exponential process betting on a drift below the null. Paper Appendix A1.15 and B1 show it is not a supermartingale when the drift is unbounded above. The code had no declared range M, no range-violation rule and no nine-bet mixture, and it cut the two sides separately at 1/eta (up to 2 eta error). Status: fixed by the change that adds `DEFAULT_DECLARED_RANGE` and `hedged_epiphanes_bounds(drifts, eta, tau_ra, declared_range)`; it reproduces paper Table 4 (0.0360, 0.0187, 0.0126, 0.0095 at 37, 74, 111, 148 blocks) within 6e-5.
2. The allocator in `src/allocator/dc_cccp.rs` uses the AM-GM majorizer on the whole coupling matrix, not the paper's sign split (paper section 4.2). Open.
3. The branch labels in paper sections 11.1 to 11.3, 12.3 and Table 18 say the CUDA, KV-cache and decode code is unmerged. It is merged into main through pull request 128 as a squash commit, so the commit hashes aa8e949, a97b07e and cc8bd8f are not on main (this explains Appendix D22). The figures were still not re-run on the merged tree. Correct label: branch run, code merged in #128, not re-run on the merged tree.
4. Paper section 7.9 item 4 says the loop breaker raises temperature and resets DoLa. The code (`NgramLoopBreaker` in `src/inference/grounded_decode.rs`) only reports a loop and `chat::generate` stops with `Finish::LoopBroken`.
5. Paper section 7.8.3 says `ptx_bitcos_gemv` indices are clamped. The WGSL BITCOS kernel on main only accumulates ternary signs on pre-expanded 2-bit data; its clamp was removed as a no-op and it ignores block_min, group scales and the nz mask of paper section 5.2.
6. Paper section 8.7 says code is permissive-only. The project policy (AGENTS.md rule 5) now allows translating code under permissive licences with notices kept; `src/inference/sampler.rs` is derived from llama.cpp (MIT).
7. Paper Table 9 cites `src/cli/doctor.rs`; the file is `src/doctor.rs`.
8. The paper cites `checks/verify_all.py`, `checks/make_figures.py` and `figures/`; the repository has no `checks/` folder and the paper is not committed.

Evidence now available (main at a31ac66): fmt and clippy clean, 780 Rust tests pass, 11 of 11 VS Code extension tests pass, release thme.exe 14.81 MiB and thme-gui.exe 15.56 MiB (caps 16 MiB and 25 MiB).
