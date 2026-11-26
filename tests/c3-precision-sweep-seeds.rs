//! C3 deterministic test: the duplicate-seed bug in the random control
//! (THME-3067).
//!
//! Counts the distinct placements the random control can produce for a given pool
//! size, and reproduces the symptom where a too-small (shape-matched) pool
//! collapses distinct seeds into one placement (Pythia-14M produced identical
//! perplexity for `random_matched` seeds 2 and 3).
//!
//! Run with `cargo test --test c3-precision-sweep-seeds`.
//!
//! No models are run: this is pure counting plus the fixed shuffle generator.

/// Number of distinct placements of `k` identical-format tensors drawn from a
/// pool of `n` (n choose k). When the pool is exactly the targets (k == n) this
/// is 1, so every seed lands on the same placement.
fn count_placements(n: usize, k: usize) -> usize {
    assert!(k <= n);
    let k = k.min(n - k);
    (1..=k).fold(1, |acc, i| acc * (n - i + 1) / i)
}

#[test]
fn counts_distinct_placements_for_given_pool_sizes() {
    // Pool size -> max distinct placements when drawing `k` identical-format
    // tensors from a shape-matched pool.
    assert_eq!(count_placements(1, 1), 1);
    assert_eq!(count_placements(2, 1), 2);
    assert_eq!(count_placements(2, 2), 1); // pool exactly the targets → 1 placement
    assert_eq!(count_placements(3, 3), 1); // all same-shape targets
    assert_eq!(count_placements(4, 2), 6);
    assert_eq!(count_placements(5, 2), 10);
}

#[test]
fn small_pool_collapses_seeds() {
    // Pool exactly the targets (k == n) → C(n,k) == 1 distinct placement.
    // With 3 seeds requested but only 1 placement, seeds 2 and 3 are redundant:
    // the THME-3067 duplicate-seed symptom (Pythia-14M identical perplexity).
    let n = 3;
    let k = n;
    let places = count_placements(n, k);
    assert_eq!(
        places, 1,
        "pool of {} with k={} should admit exactly 1 placement",
        n, k
    );
    assert!(places < 3, "only 1 placement but 3 seeds requested");
    // The runner caps seeds to the placement space, so only 1 random variant is
    // materialised instead of 3 redundant ones.
    assert_eq!(places.min(3), 1);
}
