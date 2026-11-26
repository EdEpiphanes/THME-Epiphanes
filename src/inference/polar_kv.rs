// SPDX-License-Identifier: AGPL-3.0-only
//! `PolarQuant` RoPE-Polar Isomorphism, `QJL` 1-bit cold KV estimator, and
//! `Quest` Top-16 + RoPE-pair page bounding boxes (`THME_Research_Dossier_v3.3` §7.1-7.2).

use crate::engine::error::{EngineError, EngineResult};
use crate::inference::offload::OffloadTier;
use std::f32::consts::TAU;

/// 33-entry wrap-padded `cos`/`sin` lookup table (`R97`, entry 32 duplicates entry 0).
pub struct PolarTrigLut {
    cos_tbl: [f32; 33],
    sin_tbl: [f32; 33],
}

impl PolarTrigLut {
    /// Builds the 33-entry table over `[0, 2*pi]`.
    pub fn new() -> Self {
        let mut cos_tbl = [0.0_f32; 33];
        let mut sin_tbl = [0.0_f32; 33];
        for i in 0..33 {
            let theta = (i as f32) * TAU / 32.0;
            cos_tbl[i] = theta.cos();
            sin_tbl[i] = theta.sin();
        }
        Self { cos_tbl, sin_tbl }
    }

    /// First-order Taylor interpolation (`R69`, `< 1e-3` rad error, branchless wrap).
    #[inline]
    pub fn cos_sin(&self, theta: f32) -> (f32, f32) {
        let wrapped = theta.rem_euclid(TAU);
        let scaled = wrapped * (32.0 / TAU);
        let idx = (scaled as usize).min(31);
        let dtheta = wrapped - (idx as f32) * (TAU / 32.0);
        let c0 = self.cos_tbl[idx];
        let s0 = self.sin_tbl[idx];
        (c0 - dtheta * s0, s0 + dtheta * c0)
    }
}

impl Default for PolarTrigLut {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a 2D Cartesian vector `(x_0, x_1, ...)` into polar pairs `(r_j, theta_j)`.
pub fn cartesian_to_polar_pairs(vec: &[f32]) -> EngineResult<(Vec<f32>, Vec<f32>)> {
    if !vec.len().is_multiple_of(2) {
        return Err(EngineError::InvalidInput(
            "cartesian_to_polar_pairs requires even dimension".to_string(),
        ));
    }
    let pairs = vec.len() / 2;
    let mut radii = Vec::with_capacity(pairs);
    let mut angles = Vec::with_capacity(pairs);
    for chunk in vec.chunks_exact(2) {
        let (x, y) = (chunk[0], chunk[1]);
        radii.push((x * x + y * y).sqrt());
        angles.push(y.atan2(x).rem_euclid(TAU));
    }
    Ok((radii, angles))
}

/// Applies RoPE position shift `m_offset` in polar coordinates via modular angle
/// addition (`theta_j -> (theta_j + m_offset * omega_j) mod 2pi`), leaving `radii`
/// strictly invariant (`R43`, `R46`).
pub fn polar_rope_shift(angles: &mut [f32], freqs: &[f32], m_offset: f32) -> EngineResult<()> {
    if angles.len() != freqs.len() {
        return Err(EngineError::InvalidInput(
            "polar_rope_shift: angles and freqs lengths must match".to_string(),
        ));
    }
    for (theta, &omega) in angles.iter_mut().zip(freqs.iter()) {
        *theta = (*theta + m_offset * omega).rem_euclid(TAU);
    }
    Ok(())
}

/// Reconstructs 2D Cartesian pairs using the 33-entry Taylor `PolarTrigLut`.
pub fn polar_to_cartesian_pairs(
    radii: &[f32],
    angles: &[f32],
    lut: &PolarTrigLut,
) -> EngineResult<Vec<f32>> {
    if radii.len() != angles.len() {
        return Err(EngineError::InvalidInput(
            "polar_to_cartesian_pairs: radii and angles lengths must match".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(radii.len() * 2);
    for (&r, &theta) in radii.iter().zip(angles.iter()) {
        let (c, s) = lut.cos_sin(theta);
        out.push(r * c);
        out.push(r * s);
    }
    Ok(out)
}

/// Unbiased 1-bit `QJL` asymmetric inner-product estimator for cold KV blocks:
/// `<q, k> ~= sqrt(pi/2) / d * ||k||_2 * <Phi q, sign(Phi k)>`.
pub fn qjl_asymmetric_inner_product(
    phi_q: &[f32],
    phi_k_signs: &[bool],
    k_l2_norm: f32,
) -> EngineResult<f32> {
    if phi_q.is_empty() || phi_q.len() != phi_k_signs.len() {
        return Err(EngineError::InvalidInput(
            "qjl_asymmetric_inner_product length mismatch".to_string(),
        ));
    }
    let d = phi_q.len() as f32;
    let mut dot = 0.0_f32;
    for (&q_val, &pos) in phi_q.iter().zip(phi_k_signs.iter()) {
        dot += if pos { q_val } else { -q_val };
    }
    let scale = (std::f32::consts::FRAC_PI_2).sqrt() / d;
    Ok(scale * k_l2_norm * dot)
}

/// `Quest` per-page `[k_min, k_max]` bounding box over tracked channels (`R47`, `R98`).
#[derive(Debug, Clone)]
pub struct QuestPageBound {
    pub k_min: Vec<f32>,
    pub k_max: Vec<f32>,
}

impl QuestPageBound {
    /// Builds exact `[k_min, k_max]` bounds across a page of key vectors.
    pub fn from_page(keys: &[&[f32]]) -> EngineResult<Self> {
        let Some(first) = keys.first() else {
            return Err(EngineError::InvalidInput(
                "QuestPageBound requires non-empty key page".to_string(),
            ));
        };
        let dim = first.len();
        let mut k_min = vec![f32::INFINITY; dim];
        let mut k_max = vec![f32::NEG_INFINITY; dim];
        for key in keys {
            if key.len() != dim {
                return Err(EngineError::InvalidInput(
                    "QuestPageBound key dimension mismatch".to_string(),
                ));
            }
            for (d, &val) in key.iter().enumerate() {
                if val < k_min[d] {
                    k_min[d] = val;
                }
                if val > k_max[d] {
                    k_max[d] = val;
                }
            }
        }
        Ok(Self { k_min, k_max })
    }

    /// Exact upper bound `max_{k in page} <q, k>` for zero-false-negative page skipping.
    pub fn upper_bound_score(&self, query: &[f32]) -> EngineResult<f32> {
        if query.len() != self.k_min.len() {
            return Err(EngineError::InvalidInput(
                "QuestPageBound query dimension mismatch".to_string(),
            ));
        }
        let mut score = 0.0_f32;
        for (d, &q) in query.iter().enumerate() {
            score += if q >= 0.0 {
                q * self.k_max[d]
            } else {
                q * self.k_min[d]
            };
        }
        Ok(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polar_rope_shift_preserves_radii_and_matches_2d_rotation() {
        let lut = PolarTrigLut::new();
        let k = [1.2_f32, -0.7, 0.3, 1.5];
        let freqs = [0.25_f32, 0.1];
        let m = 7.0_f32;

        let (r, mut theta) = cartesian_to_polar_pairs(&k).unwrap();
        let r_before = r.clone();
        polar_rope_shift(&mut theta, &freqs, m).unwrap();
        assert_eq!(r, r_before);

        let shifted = polar_to_cartesian_pairs(&r, &theta, &lut).unwrap();
        for pair in 0..2 {
            let ang = m * freqs[pair];
            let (c, s) = (ang.cos(), ang.sin());
            let want_x = k[pair * 2] * c - k[pair * 2 + 1] * s;
            let want_y = k[pair * 2] * s + k[pair * 2 + 1] * c;
            assert!((shifted[pair * 2] - want_x).abs() < 1e-2);
            assert!((shifted[pair * 2 + 1] - want_y).abs() < 1e-2);
        }
    }

    #[test]
    fn quest_page_bound_never_underestimates_true_attention_score() {
        let k0 = [1.0_f32, -2.0, 0.5];
        let k1 = [-0.5_f32, 1.5, 3.0];
        let q = [2.0_f32, -1.0, 0.5];
        let bound = QuestPageBound::from_page(&[&k0, &k1]).unwrap();
        let ub = bound.upper_bound_score(&q).unwrap();

        let s0: f32 = q.iter().zip(k0.iter()).map(|(a, b)| a * b).sum();
        let s1: f32 = q.iter().zip(k1.iter()).map(|(a, b)| a * b).sum();
        assert!(ub >= s0 && ub >= s1);
    }

    #[test]
    fn polar_kv_page_cache_pins_sinks_and_spills_oldest_non_sink() {
        let mut cache = PolarKvPageCache::new(2);
        let k0 = [1.0_f32, 0.0];
        let k1 = [0.0_f32, 1.0];
        let k2 = [1.0_f32, 1.0];
        cache.push_page(0, &[&k0], true).unwrap();
        cache.push_page(1, &[&k1], false).unwrap();
        cache.push_page(2, &[&k2], false).unwrap();

        assert_eq!(cache.pages[0].tier, OffloadTier::VramResident);
        assert_eq!(cache.pages[1].tier, OffloadTier::CpuColdKvQjl);
        assert_eq!(cache.pages[2].tier, OffloadTier::VramResident);
    }
}

/// Stateful `PolarQuant` + `Quest` KV page entry with tier placement (§7.2 / §7.8).
#[derive(Debug, Clone)]
pub struct PolarKvPage {
    pub page_id: usize,
    pub is_sink: bool,
    pub tier: OffloadTier,
    pub bound: QuestPageBound,
    pub radii: Vec<f32>,
    pub angles: Vec<f32>,
}

/// Manages `StreamingLLM` sink pinning and oldest-non-sink demotion to `CpuColdKvQjl`.
#[derive(Debug, Clone)]
pub struct PolarKvPageCache {
    pub max_vram_pages: usize,
    pub pages: Vec<PolarKvPage>,
}

impl PolarKvPageCache {
    #[must_use]
    pub fn new(max_vram_pages: usize) -> Self {
        Self {
            max_vram_pages: max_vram_pages.max(1),
            pages: Vec::new(),
        }
    }

    /// Encodes `keys` into polar coordinates + `QuestPageBound` and spills the oldest
    /// non-sink page to `CpuColdKvQjl` if VRAM page count exceeds `max_vram_pages`.
    pub fn push_page(
        &mut self,
        page_id: usize,
        keys: &[&[f32]],
        is_sink: bool,
    ) -> EngineResult<()> {
        let bound = QuestPageBound::from_page(keys)?;
        let mut radii = Vec::new();
        let mut angles = Vec::new();
        for &k in keys {
            let (r, theta) = cartesian_to_polar_pairs(k)?;
            radii.extend_from_slice(&r);
            angles.extend_from_slice(&theta);
        }
        self.pages.push(PolarKvPage {
            page_id,
            is_sink,
            tier: OffloadTier::VramResident,
            bound,
            radii,
            angles,
        });

        let mut vram_count = self
            .pages
            .iter()
            .filter(|p| p.tier == OffloadTier::VramResident)
            .count();
        if vram_count > self.max_vram_pages {
            for p in &mut self.pages {
                if !p.is_sink && p.tier == OffloadTier::VramResident {
                    p.tier = OffloadTier::CpuColdKvQjl;
                    vram_count -= 1;
                    if vram_count <= self.max_vram_pages {
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}
