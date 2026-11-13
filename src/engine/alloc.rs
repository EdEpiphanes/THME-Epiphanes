// SPDX-License-Identifier: AGPL-3.0-only
//! Exact multiple-choice allocation (dossier WP2).
//!
//! Each item (a tensor group) must take exactly one of its options; an option has an
//! integer weight (size units) and a cost (measured quality loss). The solver returns the
//! choice of minimum total cost whose total weight fits the capacity, by dynamic
//! programming in O(items x options x capacity) time and O(items x capacity) memory.
//!
//! The problem is NP-hard in general but pseudo-polynomial: with weights on a grid of a few
//! thousand units it is solved exactly in milliseconds. Callers that start from real-valued
//! sizes round each weight **up** to the grid, so every returned choice also meets the real
//! budget; optimality is then exact for the gridded problem.

use super::error::{ensure_finite, EngineError, EngineResult};

/// One option of an item: its size in integer units and its cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Choice {
    pub weight: u64,
    pub cost: f64,
}

/// The optimal choice per item, with its total cost and weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Allocation {
    pub choice: Vec<usize>,
    pub cost: f64,
    pub weight: u64,
}

/// Largest capacity accepted, to keep the table in memory (items x capacity entries).
pub const MAX_CAPACITY: u64 = 10_000_000;

/// Minimum-cost selection of one option per item with total weight at most `capacity`.
/// Ties in cost go to the lower total weight, then to the lower option index.
pub fn min_cost_choice(items: &[Vec<Choice>], capacity: u64) -> EngineResult<Allocation> {
    if capacity > MAX_CAPACITY {
        return Err(EngineError::InvalidInput(format!(
            "capacity {capacity} exceeds {MAX_CAPACITY}; use a coarser grid"
        )));
    }
    for (i, options) in items.iter().enumerate() {
        if options.is_empty() {
            return Err(EngineError::InvalidInput(format!(
                "item {i} has no options"
            )));
        }
        let costs: Vec<f64> = options.iter().map(|c| c.cost).collect();
        ensure_finite(&format!("cost of item {i}"), &costs)?;
    }
    let cap = capacity as usize;
    // best[w] = minimum cost using exactly weight w over the items processed so far.
    let mut best = vec![f64::INFINITY; cap + 1];
    best[0] = 0.0;
    // pick[i][w] = option chosen for item i when the running weight after item i is w.
    let mut pick: Vec<Vec<u32>> = Vec::with_capacity(items.len());
    for options in items {
        let mut next = vec![f64::INFINITY; cap + 1];
        let mut chosen = vec![u32::MAX; cap + 1];
        for (w, &base) in best.iter().enumerate() {
            if !base.is_finite() {
                continue;
            }
            for (o, option) in options.iter().enumerate() {
                let Some(total) = (w as u64).checked_add(option.weight) else {
                    continue;
                };
                if total > capacity {
                    continue;
                }
                let t = total as usize;
                let cost = base + option.cost;
                if cost < next[t] {
                    next[t] = cost;
                    chosen[t] = o as u32;
                }
            }
        }
        pick.push(chosen);
        best = next;
    }
    let mut end: Option<(f64, usize)> = None;
    for (w, &cost) in best.iter().enumerate() {
        if cost.is_finite() && end.is_none_or(|(c, _)| cost < c) {
            end = Some((cost, w));
        }
    }
    let Some((cost, weight)) = end else {
        let minimum: u64 = items
            .iter()
            .map(|o| o.iter().map(|c| c.weight).min().unwrap_or(0))
            .sum();
        return Err(EngineError::Infeasible(format!(
            "the lightest choice needs {minimum} units, capacity is {capacity}"
        )));
    };
    // Walk back through the pick tables.
    let mut choice = vec![0usize; items.len()];
    let mut w = weight;
    for i in (0..items.len()).rev() {
        let o = pick[i][w];
        if o == u32::MAX {
            return Err(EngineError::InvalidInput(
                "internal: broken back-pointer table".to_string(),
            ));
        }
        choice[i] = o as usize;
        w -= items[i][o as usize].weight as usize;
    }
    Ok(Allocation {
        choice,
        cost,
        weight: weight as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    fn exhaustive(items: &[Vec<Choice>], capacity: u64) -> Option<f64> {
        let mut best: Option<f64> = None;
        let mut idx = vec![0usize; items.len()];
        loop {
            let (w, c) = idx.iter().enumerate().fold((0u64, 0.0), |(w, c), (i, &o)| {
                (w + items[i][o].weight, c + items[i][o].cost)
            });
            if w <= capacity && best.is_none_or(|b| c < b) {
                best = Some(c);
            }
            let mut k = 0;
            loop {
                if k == items.len() {
                    return best;
                }
                idx[k] += 1;
                if idx[k] < items[k].len() {
                    break;
                }
                idx[k] = 0;
                k += 1;
            }
        }
    }

    /// Oracle: exhaustive search over every combination, on 300 random instances.
    #[test]
    fn equals_exhaustive_search() {
        let mut rng = Lcg(2026);
        for _ in 0..300 {
            let n = 1 + (rng.next() % 5) as usize;
            let items: Vec<Vec<Choice>> = (0..n)
                .map(|_| {
                    (0..1 + rng.next() % 4)
                        .map(|_| Choice {
                            weight: rng.next() % 9,
                            cost: (rng.next() % 1000) as f64 / 10.0,
                        })
                        .collect()
                })
                .collect();
            let capacity = rng.next() % 25;
            match (
                min_cost_choice(&items, capacity),
                exhaustive(&items, capacity),
            ) {
                (Ok(a), Some(best)) => {
                    assert!((a.cost - best).abs() < 1e-9, "{} vs {best}", a.cost);
                    // Reconstruction matches the reported totals.
                    let (w, c) = a
                        .choice
                        .iter()
                        .enumerate()
                        .fold((0, 0.0), |(w, c), (i, &o)| {
                            (w + items[i][o].weight, c + items[i][o].cost)
                        });
                    assert_eq!(w, a.weight);
                    assert!(w <= capacity);
                    assert!((c - a.cost).abs() < 1e-9);
                }
                (Err(EngineError::Infeasible(_)), None) => {}
                (got, want) => panic!("solver {got:?}, exhaustive {want:?}"),
            }
        }
    }

    /// A greedy-by-ratio choice is not optimal: items A (save 6 for cost 6, ratio 1) and
    /// B, C (save 5 each for cost 5.5, ratio 0.91). Saving 10 is needed. Greedy takes A
    /// first, then must add B: cost 11.5. The optimum is B + C: cost 11.
    #[test]
    fn beats_ratio_greedy_on_a_constructed_instance() {
        let keep = |w| Choice {
            weight: w,
            cost: 0.0,
        };
        let items = vec![
            vec![
                keep(10),
                Choice {
                    weight: 4,
                    cost: 6.0,
                },
            ],
            vec![
                keep(10),
                Choice {
                    weight: 5,
                    cost: 5.5,
                },
            ],
            vec![
                keep(10),
                Choice {
                    weight: 5,
                    cost: 5.5,
                },
            ],
        ];
        let a = min_cost_choice(&items, 20).unwrap();
        assert_eq!(a.choice, vec![0, 1, 1]);
        assert!((a.cost - 11.0).abs() < 1e-12);
    }

    #[test]
    fn errors_are_reported() {
        let one = vec![vec![Choice {
            weight: 5,
            cost: 1.0,
        }]];
        assert!(matches!(
            min_cost_choice(&one, 4),
            Err(EngineError::Infeasible(_))
        ));
        assert!(matches!(
            min_cost_choice(&[vec![]], 4),
            Err(EngineError::InvalidInput(_))
        ));
        let nan = vec![vec![Choice {
            weight: 1,
            cost: f64::NAN,
        }]];
        assert!(matches!(
            min_cost_choice(&nan, 4),
            Err(EngineError::NonFinite { .. })
        ));
        assert!(min_cost_choice(&one, MAX_CAPACITY + 1).is_err());
        assert_eq!(min_cost_choice(&[], 0).unwrap().choice, Vec::<usize>::new());
    }
}
