# THME claim register

**Source:** [`THME_Research_Dossier_v3.1_2026-11-13.pdf`](THME_Research_Dossier_v3.1_2026-11-13.pdf), consolidated 2026-11-13 (THME-3000)
**Evidence:** status labels are the dossier author's own classification, copied as written. Nothing here was re-measured, and no literature citation in the dossier has been checked against its source: every citation is **unverified**.
**Generated** from [`claim-register.json`](claim-register.json) by `scripts/claim_register.py`; edit the JSON, not this file.

## Status labels

| Label | Meaning | Claims |
| :--- | :--- | ---: |
| S | Standard: established in the literature or by elementary derivation (dossier Table 1) | 45 |
| M | Modified: appeared in earlier drafts and has been corrected (Table 1) | 17 |
| U | Unverified: plausible design hypothesis without supporting evidence (Table 1) | 29 |
| C | Contradicted: cannot hold as originally stated (Table 1) | 33 |
| T | Used in Appendices M to R for results checked in a prototype or test; not defined in Table 1 | 9 |
| Design decision | Used once (V-101); not defined in Table 1 | 1 |
| not stated | No status appears for this claim in the dossier text | 1 |

*Qualified* marks a status the dossier limits to a case, for example "S for independent noise; C for state error"; the qualifier opens the note. *Page* is the PDF page where the claim's register table starts. *Section* is the paper section for Appendix A rows and the appendix for the others.

## Register

| ID | Claim | Status | Section | Page | Note |
| :--- | :--- | :--- | :--- | ---: | :--- |
| V-01 | Attention compute O(N²); KV memory grows as O(N²) | M | 2.1 | 29 | Compute O(N²) per sequence; cache O(N) |
| V-02 | 32 GB KV cache at 32k tokens for 70B | M | 2.1 | 29 | 10.7 GB (GQA) or 85.9 GB (MHA); recompute per config |
| V-03 | FP arithmetic is non-associative | S | 2.2 | 29 | None |
| V-04 | Multipliers use 70–80% of AI processor energy | C | 2.2, 5.3 | 29 | Drop; weight movement dominates at batch 1 |
| V-05 | Backward locking and weight transport | S | 2.3 | 30 | Frame as plausibility problem, not GPU accuracy problem |
| V-06 | dim Cl(p,q) = 2ⁿ; grade dimensions are binomial | S | 4.2 | 30 | Unit test |
| V-07 | Rotors preserve norm and prevent explosion | M | 4.2 | 30 | True only for same-sign bivectors; test |
| V-08 | Signature (3,2) enforces a causal cone | C | 4.3 | 30 | Use (m+1, 1) |
| V-09 | Light cone replaces masking; topological invariance | M | 4.3 | 30 | Learned extra restriction on stored events |
| V-10 | Stabilizer correction Ψ − Σσ _kS_k Ψ | M | 4.4 | 30 | Use Equations 11/13; test idempotence |
| V-11 | Topological protection against rounding noise | U | 4.4 | 30 | Test H-S |
| V-12 | 4-bit phase LUT removes all floating-point logic | M | 4.5 | 30 | Accumulation needs Cartesian adders |
| V-13 | Calabi–Yau cycles store information without using memory | C | 4.5 | 30 | Drop; test H-K surrogate |
| V-14 | CY Laplacian spectrum m² ∝ n(n+d)/R² | C | 4.5 | 30 | Sphere formula; drop |
| V-15 | Eleven dimensions is the natural limit for the manifold | C | 4.2 | 30 | No ML basis; n is a hyperparameter |
| V-16 | Schedule yields 80/15/5, 20/60/20, 5/15/80 | M | 4.6 | 30 | Approximate at λ ≈ 0.15, 0.5, 0.85 only |
| V-17 | 80% of tokens take a free fast path | U | 4.6 | 30 | Design target; f̄ = 0.44, not 0.2 |
| V-18 | Skew-Hermitian coupling gives a Hopfield energy | C | 4.7 | 30 | Replace with Equation 23 |
| V-19 | Langevin noise scale √(2T dt) | S | 4.7 | 30 | Unit test |
| V-20 | Predictive coding removes weight transport | C | 4.8 | 30 | Errors still use W ᵀ |
| V-21 | Energy table for 7/5 nm with MAC entries | M | 5.3 | 31 | 45 nm multiply/add figures; relabeled |
| V-22 | 0.42 mJ/token; 80× energy reduction | C | 5.3 | 31 | Off-chip DRAM at batch 1: ≈ 0.25–1.2 J; weight-byte gain 8– 18× |
| V-23 | 17.5 GB total inference memory | M | 5.2 | 31 | Packed body only; add embeddings, state, scales |
| V-24 | 720 Wh pack gives 28.8 h at 25 W | S | 5.4 | 31 | Arithmetic correct; earlier “18+ h” inconsistent |
| V-25 | 25 W runs full 70B-grade reasoning | U | 5.4 | 31 | Measure |
| V-26 | 5–15B-token distillation in days on one GPU | C | 6.4 | 31 | About 85 days at assumed 10 TFLOP/s |
| V-27 | Training fits under 9.5 GB VRAM | U | 6.4 | 31 | About 16 B/param without 8-bit states |
| V-28 | SVD maps top singular vectors to ternary, rotations to skew | U | 6.2 | 31 | Use absmean and polar decomposition |
| V-29 | Ternary packing; 1B model in 250 MB | M | 5.2, 6.5 | 31 | Packing is standard; add embeddings (about 0.8 GB) |
| V-30 | CPU ternary inference 2–4× faster than GPU at 1/10 power | M | 6.5 | 31 | Gains were relative to CPU baselines |
| V-31 | State-space models have O(1) per-token inference | S | 3 | 31 | None |
| V-32 | 2,048 components per channel with memory savings | C | 4.2 | 31 | Truncate grades or restrict to even subalgebra |
| V-33 | Averaging K readings reduces noise as 1/K | S (qualified) | Appendix E | 48 | for independent noise; C for state error Table E2: floor from shared error |
| V-34 | Keeping the best half of readings improves accuracy | U | Appendix E | 48 | Better only for heavy-tailed noise (Table E1) |
| V-35 | Multiple stochastic readings are cheaper than extra precision | C (qualified) | Appendix E | 48 | on digital hardware Table E3 |
| V-36 | Dispersion of dithered readings signals token difficulty | C | Appendix E | 48 | Only head disagreement can; H-E |
| V-37 | Loop-back as adaptive re-reading saves computation | U | Appendix E | 48 | H-C3 |
| V-38 | Gated delta state is bounded | S | Appendix E | 48 | Derivation after Equation E1 |
| V-39 | LP gives the best mix of α , β , γ per layer | M | Appendix F | 54 | Linear objective gives at most one mixed layer; use channel-level allocation (F.1, F.2) |
| V-40 | Optimal bit allocation has a closed form | S | Appendix F | 54 | Reverse water-filling (Equation F2), high-resolution approximation |
| V-41 | Northwest-corner method solves the allocation | C | Appendix F | 54 | Starting solution only; 48% above optimum in toy case |
| V-42 | Optimal transport / Sinkhorn can enforce routing capacity | S | Appendix F | 54 | Used for balanced routing; to be tested here (H-A3) |
| V-43 | TSP improves the architecture | U | Appendix F | 54 | Narrow hardware-layout use only |
| V-44 | Self-tuning loop converges and can be frozen | U | Appendix F | 54 | Requires external evaluator and three splits (F.5) |
| V-45 | Accuracy approaches 1 as tuning continues | C | Appendix F | 54 | Ceiling a_∞ < 1 in general; achievable only for verifiable tasks (F.6) |
| V-46 | More complex formulas improve precision, speed, and energy | C | Appendix G | 60 | Only mechanisms passing Equation G1 do; complexity has a cost |
| V-47 | Tropical (max-plus) layers are multiplier-free and useful | U | Appendix G | 60 | No energy advantage over ternary; quality at scale untested |
| V-48 | Adaptive solver step size gives adaptive depth | U | Appendix G | 60 | H-D1 |
| V-49 | Unrolled optimization layers reduce depth | S (qualified) | Appendix G | 60 | for sparse recovery (LISTA); U for language H-D2 |
| V-50 | MIP/LP/QP solvers inside the forward pass | C (qualified) | Appendix G | 60 | for MIP; U for LP/QP Not differentiable or O(n³); use offline |
| V-51 | OR tools applied to deployment (placement, partitioning, tiling) lower cost | S | Appendix G | 60 | Established practice; H-D4 to measure |
| V-52 | Attractor layer with large FP16 memory is cheap | C | Appendix G | 60 | About 24 times a ternary layer under stated assumptions |
| V-53 | Trial and error can reach a breakthrough without a formal protocol | C | Appendix H | 68 | Multiple comparisons inflate false positives (Equation H1) |
| V-54 | Component gains are real if they appear in one run | C | Appendix H | 68 | Needs at least 5 seeds and matched budgets |
| V-55 | AI-generated critique counts as validation | C | Appendix H | 68 | Useful for ideas; measurements decide |
| V-56 | A stage-gated program with kill rules can establish or refute THME | S | Appendix H | 68 | Standard experimental practice |
| V-57 | A looped shared block with elastic depth improves accuracy | S (qualified) | Appendix I | 81 | at 4B scale (reported); U for ternary nano Dory: 35.2 to 39.7 to 40.6 at k = 1, 2, 4 |
| V-58 | Last-loop KV caching keeps KV memory at one-loop size | S (qualified) | Appendix I | 81 | (reported) Score change −2.1 to +0.4 at k = 4 |
| V-59 | Self-speculation with k = 1 draft and k = 4 verify speeds decoding | S (qualified) | Appendix I | 81 | (reported) 1.37–1.71×; Equation D1 consistent with acceptance 0.81–0.92 |
| V-60 | Early exit saves 56% of compute at nano scale | M | Appendix I | 81 | 38% once the output head is counted |
| V-61 | Integrate-and-fire time-to-threshold gives adaptive halting | U | Appendix I | 81 | H-N2 |
| V-62 | The delta memory is Hopfield’s synapse rule with decay and error correction | S | Appendix I | 81 | Algebra, Equation I3 |
| V-63 | Majority error of K independent readings is at most exp(−2K(½ − p)²) | S | Appendix I | 81 | Requires independence and p < ½ |
| V-64 | Quantum readings reduce noise beyond classical readings | C | Appendix I | 81 | Shot noise obeys the same 1/√K law |
| V-65 | Hybrid quantum models beat classical baselines | U | Appendix I | 81 | Classical GP baseline worse than persistence |
| V-66 | Fly-brain sparsity supports sparse language-model layers | U | Appendix I | 81 | Structural analogy; H-N7 |
| V-67 | Effective H100 training throughput of 2.3 × 10¹⁴ FLOP/s | M | Appendix I | 81 | Derived from the reported 9 hours; extrapolated to other sizes |
| V-68 | Low-rank plus ternary residual is a well-posed transmutation | S (qualified) | Appendix I | 81 | for rank-r optimality; U for downstream quality Eckart–Young–Mirsky; H-N9 |
| V-69 | The ARE metric applies directly to activations | M | Appendix I | 81 | Scaled form needed near zero |
| V-70 | Quantitative evidence for the GBM/LSTM/Transformer ranking in the Acta paper | C (qualified) | Appendix I | 82 | in the sections read Qualitative ratings only |
| V-71 | Per-layer perplexity increases add across layers | C (qualified) | Appendix J | 91 | in perplexity; M in cross-entropy 1,509 against 15.9 million; 8.36 against 13.69 nats (1.64×) |
| V-72 | All-linear ternary without retraining gives a usable model | C | Appendix J | 91 | Perplexity 1.59 × 10⁷, worse than uniform guessing |
| V-73 | Q8_0 export is near-lossless | S (qualified) | Appendix J | 91 | on SmolLM-135M; M on Pythia-14M Δ +0.03 and +0.83; verdict depends on gate (J.4b) |
| V-74 | Sensitivity-guided placement beats random placement | U | Appendix J | 92 | 3 of 3; smallest attainable p is 0.25 |
| V-75 | 73% size reduction | M | Appendix J | 92 | Relative to FP32; 47% relative to FP16 |
| V-76 | Early exit is available without training | C | Appendix J | 92 | 1.6% settled at 30% depth |
| V-77 | Looping an untrained block is harmless | C | Appendix J | 92 | Δ +37.3 |
| V-78 | Distribution shift in token difficulty cannot change output quality under speculation | S | Appendix J | 92 | Speed floor 0.40×; break-even acceptance 0.64 (c = 0.375, γ = 4) |
| V-79 | Deep-layer read weight under speculation is (1 − a)/(1 − a^( γ +1)) | S | Appendix J | 92 | Derivation, J.5 |
| V-80 | Attention V is the most fragile role per weight | S (qualified) | Appendix J | 92 | for this model and regime 143 Δ perplexity per percent of weights; may change after QAT |
| V-81 | A fixed absolute gate of 0.45 is scale-free | C | Appendix J | 92 | 2.3% of SmolLM perplexity, 0.39% of Pythia |
| V-82 | The critique’s “guaranteed catastrophic failure” from additive screening | C | Appendix J | 92 | Interaction is 1.64× in cross-entropy energy |
| V-83 | Equal-coupling matrix has eigenvalues 1 - r (L - 1 times) and 1 + (L - 1) r | S | Appendix K | 97 | Symbolic check, L = 6; general form is standard |
| V-84 | Simultaneous unit-step updates diverge when r > 1/(L - 1) | S | Appendix K | 97 | Toy: converged at 0.03, diverged at 0.05 for L = 30 |
| V-85 | Lock-one (Gauss-Seidel) converges for every symmetric positive definite coupling | S | Appendix K | 97 | Standard result; toy agrees |
| V-86 | Lock-one is cheaper than simultaneous updates when gradients cost L + 1 evaluations | M | Appendix K | 97 | Toy only, one seed; depends on the 31-evaluation assumption |
| V-87 | A language-model proposer beats random search | U | Appendix K | 97 | No test yet (H-K1) |
| V-88 | The toy threshold applies to real layers | U | Appendix K | 97 | Real coupling not measured pairwise |
| V-89 | The repository already has dual-licence files, a permissive-only dependency policy, and contributor terms | S | Appendix L | 103 | Observed at commit c83a236; legal sufficiency not assessed |
| V-90 | BYOA is declarative, with seven rule variants and no foreign code | S | Appendix L | 103 | Observed in the BYOA source file |
| V-91 | No search report is committed | S | Appendix L | 103 | Benchmark folders list three report types per model |
| V-92 | Copies before commit 56ae520 stay available under Apache-2.0 | S | Appendix L | 103 | As stated in LICENSING.md; legal effect to confirm |
| V-93 | Table J7 entry for H1 (per-tensor export not yet written) | C | Appendix L | 103 | Superseded by v0.2.2.1 |
| V-94 | The existing allocator is a dynamic-programming knapsack solver | C (qualified) | Appendix M | 112 | for the part read First 70 lines use sensitivity thresholds; rest unread; documentation says DP |
| V-95 | Exact DP allocation equals exhaustive search | T | Appendix M | 112 | 300 random instances, prototype |
| V-96 | Greedy upgrade allocation can be strictly worse than exact | T | Appendix M | 113 | 110 against 100 on a constructed instance; never better on 200 random |
| V-97 | Absmax integer quantization error is at most scale / 2 | T | Appendix M | 113 | Bit widths 2, 4, 8, 12; 20 seeds each |
| V-98 | Mean absolute value is the MSE-optimal scale for sign quantization | T | Appendix M | 113 | Compared with 39 alternative scales |
| V-99 | A Walsh-Hadamard transform already exists in the repository | S | Appendix M | 113 | Observed in the BYOA transform code; must be shared, not copied |
| V-100 | The prototype is verified for the repository toolchain | U | Appendix M | 113 | Compiled on rustc 1.97 only; pinned 1.99.1, MSRV 1.89, fmt and clippy not run |
| V-101 | THME's contribution is an integration layer plus new tools | Design decision | Appendix N | 116 | Stated by the owner; guides claims in all appendices |
| V-102 | The candidates in Table N3 are new | U | Appendix N | 116 | No prior-art search done |
| V-103 | A native reimplementation matches its reference | U | Appendix N | 116 | Required differential test per technique |
| V-104 | Permuting feed-forward units or heads, and folding normalization weights, leave the output unchanged | T | Appendix O | 123 | Tiny random model, 64-bit, one seed; differences about 4e-15 |
| V-105 | Rotating the residual stream is exact only after the normalization weights are folded | T | Appendix O | 123 | Folded: 1.2e-14; unfolded: 4.7 |
| V-106 | Layer swap, skip, and repeat change the output | T | Appendix O | 123 | Random weights; says nothing about trained models |
| V-107 | The attention head count cannot be read from tensor shapes | S | Appendix O | 123 | Any divisor of the hidden width is consistent with the shapes |
| V-108 | Permutations and rotations are inexact after block quantization | U | Appendix O | 123 | Reasoned, not measured |
| V-109 | The repository's forward pass covers LLaMA-family and GPT-NeoX models only | S | Appendix O | 124 | Roadmap item v0.2.0.1 |
| V-110 | Symmetries hold with rotary embeddings, grouped-query attention, biases, and tied embeddings | U | Appendix O | 124 | Not in the toy; each needs its own test |
| V-111 | The repository's gate tolerances are consistent | C | Appendix P | 130 | 0.25 in the tolerances file and tests; 0.45 in measured reports; default not traced |
| V-112 | Tiny needles exist in the repository | C | Appendix P | 130 | The repository's findings note says the concept is not implemented |
| V-113 | A linear needle costs K / 12d of a block | S | Appendix P | 130 | Arithmetic; order of magnitude |
| V-114 | A needle saves compute when p (1 - c) exceeds rho | S | Appendix P | 130 | Derivation, Equation P1 |
| V-115 | Identity-initialised injection preserves the model | U | Appendix P | 130 | True by construction; not yet tested |
| V-116 | Teacher-label needles need no backward pass through the model | S | Appendix P | 130 | Reasoning; to confirm in H-P5 |
| V-117 | Educational purpose removes the permission requirement | C | Appendix P | 130 | Policy is the opposite (P.1); legal effect varies |
| V-118 | Two-stage scoring has minimum cost near 2 times the square root of K | S | Appendix P | 132 | Derivation; K = 255 gives about 32 |
| V-119 | Threshold routing needs calibrated confidence | S | Appendix P | 132 | Reasoning; the early-exit operation uses a confidence threshold |
| V-120 | Calibration measured on the tune split transfers to the final split | U | Appendix P | 132 | Part of H-P6; to test |
| V-121 | KL(q p) equals minus the log of the top-k mass for renormalised top-k | not stated | Appendix Q | 137 |  |
| V-122 | A top-64 cache with 2-byte index and probability costs 256 bytes per position | S | Appendix Q | 137 | Arithmetic; 256 GB per billion positions |
| V-123 | The soft-target loss has the form of Equation Q1 | U | Appendix Q | 137 | From memory; check the original definition |
| V-124 | The repository already has a distillation loss type and a feature-distillation category | S | Appendix Q | 137 | Observed in the trainer and BYOA code; not run |
| V-125 | A teacher-led recipe search beats random search | U | Appendix Q | 137 | No test |
| V-126 | A student reaches a stated fraction of teacher quality | U | Appendix Q | 138 | No student has been trained |
| V-127 | The Wilson interval for 7 of 10 is 0.397 to 0.892 | S | Appendix R | 146 | Computed |
| V-128 | The simple interval undercovers at small n: 80.1% at 30 items, accuracy 0.9; Wilson 97.0%; bootstrap 95.0% | T | Appendix R | 146 | 4,000 repeats, one seed |
| V-129 | Half-widths at accuracy 0.70 are 7.0, 2.8, and 0.76 points for 164, 1,000, and 14,000 items | S | Appendix R | 146 | Arithmetic |
| V-130 | Resolving one point needs about 16,100 items unpaired and 1,900 paired at 5% discordance | S | Appendix R | 147 | Under the stated assumptions |
| V-131 | The pass@k estimator equals enumeration | T | Appendix R | 147 | n = 6, c = 2 only; formula from memory |
| V-132 | Exact McNemar for 12 and 3 gives p = 0.035 | S | Appendix R | 147 | Computed |
| V-133 | The repository's task score carries no item count or interval | S | Appendix R | 147 | Observed in the gate structure |
| V-134 | Benchmark names and metrics in Table R1 | U | Appendix R | 147 | From memory; verify |
| V-135 | The repository's benchmark script reports only measured quantities | U | Appendix R | 147 | Its header mentions simulated token generation; contents not audited |
