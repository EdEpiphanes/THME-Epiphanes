// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark statistics (roadmap v0.2.8.5, dossier WP34 and R.5).
//!
//! Intervals for a score, paired comparisons of two models on the same items, the number of
//! items a comparison needs, and a multiple-comparison correction. Wilson (one proportion) and
//! the percentile bootstrap (any score) are the defaults; the simple normal interval is kept
//! only to show why it is not used (it under-covers on small sets).
//!
//! Bootstrap resampling uses a seeded SplitMix64 generator, so every interval is reproducible.

use anyhow::{bail, Result};

use super::{Interval, IntervalMethod};

/// Standard normal quantile (inverse CDF): Acklam's rational approximation, relative error below
/// 1.15e-9 on (0, 1).
pub fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.38357751867269e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    let low = 0.02425;
    if p < low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

fn z_for(level: f64) -> Result<f64> {
    if !(level > 0.0 && level < 1.0) {
        bail!("confidence level {level} must be between 0 and 1");
    }
    Ok(normal_quantile(1.0 - (1.0 - level) / 2.0))
}

/// Wilson score interval for `correct` successes out of `n`.
pub fn wilson(correct: u64, n: u64, level: f64) -> Result<(f64, f64)> {
    if n == 0 || correct > n {
        bail!("Wilson interval needs 0 <= correct <= n and n > 0 (correct = {correct}, n = {n})");
    }
    let z = z_for(level)?;
    let (n, p) = (n as f64, correct as f64 / n as f64);
    let z2 = z * z;
    let centre = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / (1.0 + z2 / n);
    let low = if correct == 0 {
        0.0
    } else {
        (centre - half).max(0.0)
    };
    let high = if p == 1.0 {
        1.0
    } else {
        (centre + half).min(1.0)
    };
    Ok((low, high))
}

/// The simple normal-approximation (Wald) interval. Kept for comparison only: it covers the
/// truth far less often than its nominal level on small sets (dossier R.5).
pub fn wald(correct: u64, n: u64, level: f64) -> Result<(f64, f64)> {
    if n == 0 || correct > n {
        bail!("Wald interval needs 0 <= correct <= n and n > 0");
    }
    let z = z_for(level)?;
    let p = correct as f64 / n as f64;
    let half = z * (p * (1.0 - p) / n as f64).sqrt();
    Ok(((p - half).max(0.0), (p + half).min(1.0)))
}

/// SplitMix64: a small, fast, seedable generator for resampling.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `0..n` (Lemire's multiply-shift; `n` > 0).
    pub fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn percentile_interval(mut stats: Vec<f64>, level: f64) -> (f64, f64) {
    stats.sort_by(f64::total_cmp);
    let alpha = (1.0 - level) / 2.0;
    let pick = |q: f64| {
        let i = ((q * stats.len() as f64).ceil() as usize).clamp(1, stats.len()) - 1;
        stats[i]
    };
    (pick(alpha), pick(1.0 - alpha))
}

/// Percentile bootstrap interval for the mean of `scores`, from `resamples` resamples.
pub fn bootstrap_mean(
    scores: &[f64],
    level: f64,
    resamples: usize,
    seed: u64,
) -> Result<(f64, f64)> {
    z_for(level)?;
    if scores.is_empty() || resamples == 0 {
        bail!("bootstrap needs at least one score and one resample");
    }
    let mut rng = SplitMix64::new(seed);
    let n = scores.len();
    let means = (0..resamples)
        .map(|_| (0..n).map(|_| scores[rng.below(n)]).sum::<f64>() / n as f64)
        .collect();
    Ok(percentile_interval(means, level))
}

/// Paired bootstrap interval for mean(`a`) - mean(`b`) over the same items: items are resampled
/// together, so the shared difficulty of each item cancels.
pub fn paired_bootstrap_difference(
    a: &[f64],
    b: &[f64],
    level: f64,
    resamples: usize,
    seed: u64,
) -> Result<(f64, f64)> {
    if a.len() != b.len() {
        bail!(
            "paired scores need the same items ({} vs {})",
            a.len(),
            b.len()
        );
    }
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    bootstrap_mean(&diffs, level, resamples, seed)
}

/// Exact two-sided McNemar test: `improved` items correct only for the candidate, `regressed`
/// items correct only for the baseline. The p-value is the two-sided binomial tail with p = 1/2.
pub fn mcnemar_exact(improved: u64, regressed: u64) -> f64 {
    let n = improved + regressed;
    if n == 0 {
        return 1.0;
    }
    let k = improved.min(regressed);
    // P(X <= k) for X ~ Binomial(n, 1/2), summed in log space so large n stays finite.
    let ln_half_n = -(n as f64) * std::f64::consts::LN_2;
    let mut ln_c = 0.0; // ln C(n, 0)
    let mut tail = 0.0;
    for i in 0..=k {
        if i > 0 {
            ln_c += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        tail += (ln_c + ln_half_n).exp();
    }
    (2.0 * tail).min(1.0)
}

/// Items needed so that a `delta` difference between two independently scored models at
/// accuracy `p` is resolved at confidence `level`: z^2 * 2p(1-p) / delta^2.
pub fn required_items_independent(p: f64, delta: f64, level: f64) -> Result<u64> {
    if !(0.0..=1.0).contains(&p) || delta <= 0.0 {
        bail!("needs 0 <= p <= 1 and delta > 0");
    }
    let z = z_for(level)?;
    Ok((z * z * 2.0 * p * (1.0 - p) / (delta * delta)).ceil() as u64)
}

/// Items needed for a paired comparison on the same items, where a fraction `discordant` of
/// items change between the two models: z^2 * discordant / delta^2.
pub fn required_items_paired(discordant: f64, delta: f64, level: f64) -> Result<u64> {
    if !(0.0..=1.0).contains(&discordant) || delta <= 0.0 {
        bail!("needs 0 <= discordant <= 1 and delta > 0");
    }
    let z = z_for(level)?;
    Ok((z * z * discordant / (delta * delta)).ceil() as u64)
}

/// Holm-Bonferroni adjusted p-values, in the input order. Reject hypothesis i at level alpha
/// when its adjusted p-value is at most alpha.
pub fn holm(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| p_values[i].total_cmp(&p_values[j]));
    let mut adjusted = vec![0.0; m];
    let mut running = 0.0f64;
    for (rank, &i) in order.iter().enumerate() {
        running = running.max(((m - rank) as f64 * p_values[i]).min(1.0));
        adjusted[i] = running;
    }
    adjusted
}

/// The interval a result record carries for its item scores. Wilson needs 0/1 scores; the
/// bootstrap works for any score in [0, 1].
pub fn interval_for(
    scores: &[f64],
    method: IntervalMethod,
    level: f64,
    seed: u64,
) -> Result<Interval> {
    let (low, high) = match method {
        IntervalMethod::Wilson => {
            if scores.iter().any(|&s| s != 0.0 && s != 1.0) {
                bail!("Wilson intervals need 0/1 item scores; use the bootstrap");
            }
            let correct = scores.iter().filter(|&&s| s == 1.0).count() as u64;
            wilson(correct, scores.len() as u64, level)?
        }
        IntervalMethod::Bootstrap => bootstrap_mean(scores, level, 2000, seed)?,
    };
    Ok(Interval {
        low,
        high,
        method,
        level,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn normal_quantile_known_values() {
        for (p, z) in [
            (0.975, 1.959963984540054),
            (0.5, 0.0),
            (0.995, 2.5758293035489),
            (0.001, -3.090232306167813),
        ] {
            assert!(
                close(normal_quantile(p), z, 1e-8),
                "p={p}: {}",
                normal_quantile(p)
            );
        }
    }

    /// Dossier Table R3: Wilson interval for 7 correct of 10 is 0.397 to 0.892.
    #[test]
    fn wilson_reproduces_table_r3() {
        let (lo, hi) = wilson(7, 10, 0.95).unwrap();
        assert!(
            close(lo, 0.397, 5e-4) && close(hi, 0.892, 5e-4),
            "{lo} {hi}"
        );
        assert_eq!(wilson(0, 5, 0.95).unwrap().0, 0.0);
        assert!(wilson(3, 2, 0.95).is_err() && wilson(1, 2, 1.5).is_err());
    }

    /// Dossier Table R3: exact McNemar with 12 improved and 3 regressed gives p = 0.035.
    #[test]
    fn mcnemar_reproduces_table_r3() {
        let p = mcnemar_exact(12, 3);
        assert!(close(p, 576.0 * 2.0 / 32768.0, 1e-12), "{p}");
        assert!(close(p, 0.035, 5e-4));
        assert_eq!(mcnemar_exact(0, 0), 1.0);
        assert_eq!(mcnemar_exact(5, 5), 1.0);
        assert!(mcnemar_exact(5000, 4000) < 1e-20, "large n stays finite");
    }

    /// Dossier Table R3: the half-width at accuracy 0.70 on 164 items is 7.0 points.
    #[test]
    fn half_width_reproduces_table_r3() {
        let (lo, hi) = wald(115, 164, 0.95).unwrap(); // 115/164 = 0.701
        assert!(close((hi - lo) / 2.0, 0.070, 5e-4), "{}", (hi - lo) / 2.0);
    }

    /// Dossier R.5: about 16,100 items for independent runs and about 1,900 for a paired
    /// comparison with 5% discordant items, to resolve one point at accuracy 0.70.
    #[test]
    fn required_items_reproduce_r5() {
        let ind = required_items_independent(0.70, 0.01, 0.95).unwrap();
        let pair = required_items_paired(0.05, 0.01, 0.95).unwrap();
        assert!((16_000..=16_200).contains(&ind), "{ind}");
        assert!((1_850..=1_950).contains(&pair), "{pair}");
        assert!(close(ind as f64 / pair as f64, 8.4, 0.05));
    }

    /// Dossier Table R3: percentile bootstrap coverage on 200 items at accuracy 0.70 is about
    /// 0.95. Simulated with fixed seeds, so the test is deterministic.
    #[test]
    fn bootstrap_coverage_near_nominal() {
        let mut rng = SplitMix64::new(42);
        let (repeats, n, truth) = (300, 200, 0.70);
        let mut covered = 0;
        for r in 0..repeats {
            let scores: Vec<f64> = (0..n)
                .map(|_| if rng.unit() < truth { 1.0 } else { 0.0 })
                .collect();
            let (lo, hi) = bootstrap_mean(&scores, 0.95, 400, r as u64).unwrap();
            if lo <= truth && truth <= hi {
                covered += 1;
            }
        }
        let cov = covered as f64 / repeats as f64;
        assert!((0.91..=0.98).contains(&cov), "coverage {cov}");
    }

    /// Dossier R.5: at 30 items and accuracy 0.9 the simple interval under-covers and Wilson
    /// does not.
    #[test]
    fn wilson_covers_where_wald_does_not() {
        let mut rng = SplitMix64::new(7);
        let (repeats, n, truth) = (4000, 30u64, 0.9);
        let (mut w, mut s) = (0, 0);
        for _ in 0..repeats {
            let c = (0..n).filter(|_| rng.unit() < truth).count() as u64;
            let (lo, hi) = wilson(c, n, 0.95).unwrap();
            w += usize::from(lo <= truth && truth <= hi);
            let (lo, hi) = wald(c, n, 0.95).unwrap();
            s += usize::from(lo <= truth && truth <= hi);
        }
        let (w, s) = (w as f64 / repeats as f64, s as f64 / repeats as f64);
        assert!(w > 0.93 && s < 0.86, "Wilson {w}, simple {s}");
    }

    #[test]
    fn paired_bootstrap_sees_what_independent_intervals_miss() {
        // Candidate fixes 12 items and breaks 3 out of 400; most items agree.
        let base: Vec<f64> = (0..400)
            .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 })
            .collect();
        let mut cand = base.clone();
        let mut fixed = 0;
        for (i, c) in cand.iter_mut().enumerate() {
            if base[i] == 0.0 && fixed < 12 {
                *c = 1.0;
                fixed += 1;
            }
        }
        for c in cand.iter_mut().skip(1).step_by(3).take(3) {
            *c = 0.0;
        }
        let (lo, hi) = paired_bootstrap_difference(&cand, &base, 0.95, 2000, 1).unwrap();
        assert!(
            lo > 0.0 && hi < 0.06,
            "paired interval {lo}..{hi} excludes zero"
        );
        let a = interval_for(&cand, IntervalMethod::Wilson, 0.95, 0).unwrap();
        let b = interval_for(&base, IntervalMethod::Wilson, 0.95, 0).unwrap();
        assert!(a.low < b.high, "independent intervals overlap");
        assert!(paired_bootstrap_difference(&cand, &base[1..], 0.95, 10, 1).is_err());
    }

    #[test]
    fn holm_adjustment() {
        let adj = holm(&[0.01, 0.04, 0.03, 0.005]);
        // Sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06.
        let want = [0.03, 0.06, 0.06, 0.02];
        for (a, b) in adj.iter().zip(want) {
            assert!(close(*a, b, 1e-12), "{adj:?}");
        }
        assert!(holm(&[]).is_empty());
    }

    #[test]
    fn record_intervals() {
        let scores = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let w = interval_for(&scores, IntervalMethod::Wilson, 0.95, 0).unwrap();
        assert!(close(w.low, 0.397, 5e-4));
        let b1 = interval_for(&scores, IntervalMethod::Bootstrap, 0.95, 9).unwrap();
        let b2 = interval_for(&scores, IntervalMethod::Bootstrap, 0.95, 9).unwrap();
        assert_eq!(b1, b2, "same seed, same interval");
        assert!(b1.low <= 0.7 && 0.7 <= b1.high);
        assert!(interval_for(&[0.5], IntervalMethod::Wilson, 0.95, 0).is_err());
    }
}
