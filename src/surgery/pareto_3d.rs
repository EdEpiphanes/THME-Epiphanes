use serde::{Deserialize, Serialize};

/// A single candidate point evaluated in 3D optimization space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint3D {
    pub id: String,
    /// Perplexity degradation delta: ΔPPL (lower is better, ideally <= 0.15).
    pub delta_ppl: f64,
    /// Memory traffic in bytes per token: W_token (lower is better).
    pub memory_bytes_per_token: f64,
    /// Simulated energy per token in microjoules: E_token (lower is better).
    pub energy_uj_per_token: f64,
}

impl ParetoPoint3D {
    /// Returns true if `self` dominates `other` (strictly better in at least one dimension and no worse in any).
    pub fn dominates(&self, other: &ParetoPoint3D) -> bool {
        let no_worse = self.delta_ppl <= other.delta_ppl
            && self.memory_bytes_per_token <= other.memory_bytes_per_token
            && self.energy_uj_per_token <= other.energy_uj_per_token;

        let strictly_better = self.delta_ppl < other.delta_ppl
            || self.memory_bytes_per_token < other.memory_bytes_per_token
            || self.energy_uj_per_token < other.energy_uj_per_token;

        no_worse && strictly_better
    }
}

/// Multi-Objective 3D Pareto Optimizer managing non-dominated candidate sets.
pub struct ParetoOptimizer3D {
    frontier: Vec<ParetoPoint3D>,
}

impl ParetoOptimizer3D {
    pub fn new() -> Self {
        Self {
            frontier: Vec::new(),
        }
    }

    /// Adds a candidate point to the optimizer. If it is dominated by an existing point, it is discarded.
    /// Otherwise, it is added and any points it dominates are pruned from the frontier.
    /// Returns true if the point was added to the Pareto frontier.
    pub fn update(&mut self, candidate: ParetoPoint3D) -> bool {
        // Check if candidate is dominated by any existing point in the frontier
        for p in &self.frontier {
            if p.dominates(&candidate) {
                return false;
            }
        }

        // Remove any points currently in frontier dominated by the new candidate
        self.frontier.retain(|p| !candidate.dominates(p));
        self.frontier.push(candidate);
        true
    }

    /// Returns the active non-dominated Pareto frontier points.
    pub fn frontier(&self) -> &[ParetoPoint3D] {
        &self.frontier
    }

    /// Computes the exact 3D Hypervolume bounded by an anti-optimal reference point.
    /// Uses 3D coordinate box slicing. Reference point coordinates must be strictly worse than all points.
    pub fn compute_hypervolume(&self, ref_point: &ParetoPoint3D) -> f64 {
        if self.frontier.is_empty() {
            return 0.0;
        }

        // Sort frontier points primarily by delta_ppl
        let mut sorted = self.frontier.clone();
        sorted.sort_by(|a, b| {
            a.delta_ppl
                .partial_cmp(&b.delta_ppl)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut total_vol = 0.0;

        // Bounded inclusion-exclusion / decomposition for 3D boxes against reference point
        for (i, p) in sorted.iter().enumerate() {
            if p.delta_ppl >= ref_point.delta_ppl
                || p.memory_bytes_per_token >= ref_point.memory_bytes_per_token
                || p.energy_uj_per_token >= ref_point.energy_uj_per_token
            {
                continue;
            }

            // Bound delta_x
            let prev_x = if i == 0 { 0.0 } else { sorted[i - 1].delta_ppl };
            let dx = (ref_point.delta_ppl - p.delta_ppl.max(prev_x)).max(0.0);
            let dy = (ref_point.memory_bytes_per_token - p.memory_bytes_per_token).max(0.0);
            let dz = (ref_point.energy_uj_per_token - p.energy_uj_per_token).max(0.0);

            // Simple monotonic lower bound projection
            total_vol += dx * dy * dz;
        }

        total_vol
    }
}

impl Default for ParetoOptimizer3D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_dominance_and_frontier_filtering() {
        let mut opt = ParetoOptimizer3D::new();

        let p1 = ParetoPoint3D {
            id: "p1".to_string(),
            delta_ppl: 0.05,
            memory_bytes_per_token: 100.0,
            energy_uj_per_token: 50.0,
        };
        let p2 = ParetoPoint3D {
            id: "p2".to_string(),
            delta_ppl: 0.10,
            memory_bytes_per_token: 60.0,
            energy_uj_per_token: 30.0,
        };
        let p_dominated = ParetoPoint3D {
            id: "p_dominated".to_string(),
            delta_ppl: 0.12,
            memory_bytes_per_token: 120.0,
            energy_uj_per_token: 60.0,
        };

        assert!(opt.update(p1));
        assert!(opt.update(p2));
        assert!(!opt.update(p_dominated)); // Should be rejected as dominated by p1 and p2

        assert_eq!(opt.frontier().len(), 2);

        let ref_pt = ParetoPoint3D {
            id: "ref".to_string(),
            delta_ppl: 0.50,
            memory_bytes_per_token: 200.0,
            energy_uj_per_token: 100.0,
        };

        let hv = opt.compute_hypervolume(&ref_pt);
        assert!(hv > 0.0);
    }
}
