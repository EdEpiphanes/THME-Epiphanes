# Math engine: v0.2.5 sub-milestone notes (agent branch agent/v02-engine)

Written by the engine agent for the coordinator. ROADMAP.md, BACKLOG.md and CHANGELOG.md are not
edited here; the coordinator merges the verdicts below. All tests are oracle tests: expected values
come from a hand-worked case, a naive or brute-force reference, or an identity, never from the code
under test. Nothing in this note is a measured result; no model was run.

Gate observed on the final tree (run in this worktree, `target-nodbg`, debug info off): `cargo fmt
--check` passed; `cargo clippy --all-targets -- -D warnings` passed; `cargo test -- --skip medgemma`
passed with 686 library tests and every integration target at 0 failed; `py scripts/spdx.py --check`
passed. The engine filter alone ran 130 tests, all passing.

## v0.2.5.2 Exact dynamic-programming allocator (THME-3002)

- Already in the tree before this branch: `src/engine/alloc.rs` (`min_cost_choice`), exact DP over
  integer weights; `src/cli/search.rs` `exact` and `--solver dp`, default stays `greedy`.
- Tests: `engine::alloc::tests::equals_exhaustive_search` (300 random instances against exhaustive
  enumeration), `beats_ratio_greedy_on_a_constructed_instance` (hand-worked: optimum 11, ratio greedy
  11.5), `errors_are_reported`; `cli::search::tests::exact_is_never_worse_than_greedy` (200 random
  cost tables, 6 groups, checked against greedy at equal budget).
- Done means (BACKLOG): matches exhaustive search on small instances, and never worse than greedy at
  equal budget; default solver unchanged until compared on committed cost tables.
- Verdict: **met** for the done-means text. The engine test covers the exhaustive oracle; the
  greedy comparison lives in the CLI module, not in `engine::alloc`.
- Still open, not part of the done means: the comparison on real measured cost tables (needs model
  runs, paused). The roadmap row's "comparison on real cost tables pending" stays true.

## v0.2.5.5 Bases and rotations (THME-3004)

- Added to `src/engine/basis.rs`:
  - `fft(re, im)`: in-place radix-2 complex FFT, forward sign, unnormalized. Power-of-two lengths
    only.
  - `givens(a, b)` and `apply_givens(x, i, j, (c, s))`.
  - `householder_vector(x)` (unit vector, sign chosen to avoid cancellation) and
    `apply_householder(v, y)` (refuses a non-unit `v`).
- Tests added: `fft_matches_the_naive_dft` (naive O(n^2) DFT as oracle, n = 1 to 64),
  `fft_hand_worked_cases` (impulse, constant, [1, 0, -1, 0] gives [0, 2, 0, 2]),
  `fft_rejects_bad_shapes_and_non_finite_values`, `givens_zeroes_the_second_component` ((3, 4) gives
  (0.6, 0.8)), `givens_rotation_in_a_plane_preserves_the_norm`,
  `householder_maps_a_vector_onto_the_first_axis` ((3, 4) gives (-5, 0)),
  `householder_matches_the_explicit_matrix` (explicit I - 2vv^T in the test),
  `householder_rejects_zero_mismatch_and_non_unit_input`.
- Existing tests kept: `matches_the_explicit_hadamard_matrix`, `preserves_norm_and_is_its_own_inverse`,
  `bit_identical_to_the_replaced_copies` (the WHT callers in `src/transform/byoa.rs` and
  `src/transform/mod.rs` are unchanged).
- Done means of the roadmap row: one WHT, FFT against a naive DFT, Givens and Householder.
- Verdict: **met** for the four named pieces. Limits: the FFT has no real-input or non-power-of-two
  path; Givens and Householder are primitives with no QR or basis consumer in the crate yet.

## v0.2.5.7 Projections that report retained structure (WP8)

- New file `src/engine/projection.rs`. One cyclic Jacobi eigen-solver (`symmetric_eigen`, n up to
  256, `NoConvergence` on the sweep limit) shared by PCA and MDS.
  - `pca(data, rows, cols, k)`: returns scores, components, all explained variances, the retained
    variance share and the participation ratio. Errors on zero variance.
  - `participation_ratio(values)`: (sum l)^2 / sum l^2.
  - `classical_mds(distances, n, k)`: Torgerson MDS, returns coordinates and the goodness of fit
    (sum of the k largest eigenvalues over the sum of all |eigenvalues|).
  - `stereographic_from_north(point)` and `stereographic_to_sphere(coords)`: the point comes back
    with its `distance_scale` = 1 / (1 - x_last). The map is conformal, so angles are kept.
- Tests: `eigen_pairs_satisfy_the_definition` (A v = l v from the definition, orthonormal V),
  `symmetric_two_by_two_by_hand` ([[2, 1], [1, 2]] gives 3 and 1),
  `pca_of_a_line_keeps_everything_in_one_component` (retained 1.0),
  `pca_of_a_cross_keeps_half_the_variance` (hand-worked covariance diag(1/2, 1/2): retained 0.5,
  participation ratio 2), `participation_ratio_by_hand` (4, 1, 25/17),
  `mds_reproduces_distances_of_points_on_a_line` (points 0, 1, 3: distances reproduced, fit 1),
  `mds_of_the_unit_square_in_two_dimensions` (fit 1 in 2D, below 1 in 1D),
  `stereographic_by_hand` (south pole, equator, (0.6, 0, 0.8) gives (3, 0), scale 5),
  `stereographic_round_trips_on_random_sphere_points`.
- Done means of the roadmap row: PCA, participation ratio, MDS and stereographic, each reporting the
  structure it keeps.
- Verdict: **met** for the four named methods. Limits: classical MDS only (shortcut comment in the
  code: no SMACOF stress refinement); PCA is dense and exact, with no randomized solver; the
  stereographic scale is the only retained-structure figure for that map; no caller in the crate
  uses these projections yet.

## v0.2.5.9 Typed conversion registry and engine benchmarks (WP10, WP39)

- New file `src/engine/registry.rs`. `Dtype` (F32, F16, Bf16, Fp8E4M3, Fp8E5M2), `Conversion`,
  `ConversionRegistry` (`standard()`, `register`, `get`, `convert`, `convert_path`), `Converted`
  with `error_bound`.
- Every conversion widens exactly to F32 or rounds once through `engine::minifloat` (the one rounding
  routine). The input must already be a value of its source type; an overflow to infinity or NaN is
  an `Infeasible` error.
- Accumulated bound: E_0 = 0 and E_(i+1) = E_i + b_i(|x| + E_i), with b_i the monotone bound of step
  i. This is an upper bound on |final - x|, checked by test.
- Tests: `chain_bounds_accumulate_by_hand` (F32 to BF16 to E4M3 on 1.0 gives 0.066650390625, the
  single rounding gives 0.0625), `one_rounding_step_by_hand` (FP16 bound 2^-11 at 1.0),
  `chain_bound_holds_on_random_inputs` (2,000 log-uniform inputs, path with widening in between),
  `widening_to_f32_is_exact_with_zero_bound`, `input_must_already_be_a_value_of_its_source_type`,
  `overflow_and_non_finite_inputs_are_errors`, `register_refuses_duplicates_and_identity`,
  `standard_registry_has_every_ordered_pair_of_distinct_types`,
  `paths_need_registered_steps_and_at_least_two_types`.
- Verdict for the conversion registry: **partly met**. Limits: float formats only. Integer, ternary
  and binary types need a scale and are not registered; the exporter and BYOA code do not call the
  registry yet.
- Verdict for "engine benchmarks run on every release": **not attempted**. There is no engine
  benchmark harness in the crate, and a release CI step plus a committed report under
  `docs/benchmarks/` is a coordinator decision. No timing was run and no figure is claimed.
- Overall verdict for v0.2.5.9: **partly met: engine benchmarks missing; registry covers float
  formats only**.

## Coordinator decisions

1. Whether the roadmap rows for v0.2.5.5 and v0.2.5.7 move from "Implemented in part" or blank to
   "Done", given the limits above.
2. Whether v0.2.5.9 stays open until a benchmark job exists, and where its report lives.
3. Whether the registry should be wired into `src/exporter` and the BYOA rules now or in a later
   sub-milestone.
4. THME-3002 is still `blocked` in BACKLOG.md on the real cost-table comparison. This branch does not
   change that status.
5. `.gitignore` only ignores `/target`. A worktree that builds into `target-nodbg` shows it as
   untracked. This branch does not change `.gitignore`.
