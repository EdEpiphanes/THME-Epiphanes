// SPDX-License-Identifier: AGPL-3.0-only
//! `BITCOS` 1,024-weight hierarchical dual-scale super-block & `BITCOS-2:4` codec (`ADR-0003`).
//!
//! Layout for `SuperBlock1024` (128-bit GPU-aligned header = 16 bytes + 128B mask + signs):
//! - `base_off: u16` (2B)
//! - `grp_nz_cnt: [u8; 7]` (7B, counts for groups 0..7; each <= 128)
//! - `grp_lscale: [u8; 4]` (4B, packed 4-bit log-uniform scales for 8 groups)
//! - `d_log_uq44: u8` (1B, unsigned Q4.4 fixed-point log2 dynamic range)
//! - `block_min_bits: u16` (2B, f16 bits of minimum non-zero group scale)
//! - `nz_mask: [u32; 32]` (128B, 1.0 bpw)
//! - `packed_signs: Vec<u8>` (`ceil(total_nz / 8)` bytes)

use crate::engine::error::{EngineError, EngineResult};

pub const SUPERBLOCK_LEN: usize = 1024;
pub const NUM_GROUPS: usize = 8;
pub const GROUP_LEN: usize = 128;

/// A 1,024-weight sparse ternary `BITCOS` super-block (`ADR-0003`).
#[derive(Debug, Clone, PartialEq)]
pub struct BitcosSuperBlock1024 {
    pub base_off: u16,
    pub grp_nz_cnt: [u8; 7],
    pub grp_lscale: [u8; 4],
    pub d_log_uq44: u8,
    pub block_min: f32,
    pub nz_mask: [u32; 32],
    pub packed_signs: Vec<u8>,
    pub total_nz: usize,
}

impl BitcosSuperBlock1024 {
    /// Encodes 1,024 weights into a `BITCOS` super-block using dead-zone threshold `tau`.
    pub fn encode(weights: &[f32], tau: f32) -> EngineResult<Self> {
        if weights.len() != SUPERBLOCK_LEN {
            return Err(EngineError::InvalidInput(format!(
                "BitcosSuperBlock1024 requires {SUPERBLOCK_LEN} weights, got {}",
                weights.len()
            )));
        }
        if !tau.is_finite() || tau < 0.0 {
            return Err(EngineError::InvalidInput(format!(
                "dead-zone threshold tau must be finite and >= 0, got {tau}"
            )));
        }

        let mut grp_scales = [0.0_f32; NUM_GROUPS];
        let mut grp_has_nz = [false; NUM_GROUPS];
        let mut grp_nz_cnt = [0_u8; 7];
        let mut nz_mask = [0_u32; 32];
        let mut signs = Vec::new();

        for g in 0..NUM_GROUPS {
            let g_slice = &weights[g * GROUP_LEN..(g + 1) * GROUP_LEN];
            let mut sum_abs = 0.0_f32;
            let mut cnt = 0_usize;

            for (local_idx, &w) in g_slice.iter().enumerate() {
                if w.abs() > tau {
                    let global_idx = g * GROUP_LEN + local_idx;
                    nz_mask[global_idx / 32] |= 1_u32 << (global_idx % 32);
                    signs.push(w >= 0.0);
                    sum_abs += w.abs();
                    cnt += 1;
                }
            }
            if g < 7 {
                grp_nz_cnt[g] = cnt as u8;
            }
            if cnt > 0 {
                grp_has_nz[g] = true;
                grp_scales[g] = (sum_abs / (cnt as f32)).max(1e-6);
            }
        }

        let block_min_raw = grp_scales
            .iter()
            .zip(grp_has_nz.iter())
            .filter(|&(_, &has)| has)
            .map(|(&s, _)| s)
            .fold(f32::INFINITY, f32::min);
        let block_min = if block_min_raw.is_finite() {
            block_min_raw
        } else {
            1e-6
        };
        let block_max = grp_scales
            .iter()
            .zip(grp_has_nz.iter())
            .filter(|&(_, &has)| has)
            .map(|(&s, _)| s)
            .fold(block_min, f32::max);
        let log_range = (block_max / block_min).log2().max(0.0);
        let d_log_f = (log_range / 15.0).clamp(0.0, 15.9375);
        let d_log_uq44 = (d_log_f * 16.0).round().clamp(0.0, 255.0) as u8;
        let d_log_actual = (f32::from(d_log_uq44) * 0.0625).max(1e-6);

        let mut grp_lscale = [0_u8; 4];
        for (g, &s) in grp_scales.iter().enumerate() {
            let q = (((s / block_min).log2() / d_log_actual).round()).clamp(0.0, 15.0) as u8;
            if g % 2 == 0 {
                grp_lscale[g / 2] |= q & 0x0F;
            } else {
                grp_lscale[g / 2] |= (q & 0x0F) << 4;
            }
        }

        let total_nz = signs.len();
        let mut packed_signs = vec![0_u8; total_nz.div_ceil(8)];
        for (idx, &is_pos) in signs.iter().enumerate() {
            if is_pos {
                packed_signs[idx / 8] |= 1_u8 << (idx % 8);
            }
        }

        Ok(Self {
            base_off: 0,
            grp_nz_cnt,
            grp_lscale,
            d_log_uq44,
            block_min,
            nz_mask,
            packed_signs,
            total_nz,
        })
    }

    /// Decodes all 1,024 weights with bounds-clamped sign lookup (`R85` OOB guard).
    pub fn decode(&self) -> Vec<f32> {
        let mut out = vec![0.0_f32; SUPERBLOCK_LEN];
        if self.total_nz == 0 || self.packed_signs.is_empty() {
            return out;
        }
        let d_log = f32::from(self.d_log_uq44) * 0.0625;
        let mut running_nz = 0_usize;

        for g in 0..NUM_GROUPS {
            let nibble = if g % 2 == 0 {
                self.grp_lscale[g / 2] & 0x0F
            } else {
                (self.grp_lscale[g / 2] >> 4) & 0x0F
            };
            let scale = self.block_min * (f32::from(nibble) * d_log).exp2();

            for local_idx in 0..GROUP_LEN {
                let global_idx = g * GROUP_LEN + local_idx;
                let mask = self.nz_mask[global_idx / 32];
                if (mask & (1_u32 << (global_idx % 32))) != 0 {
                    let clamped_idx = running_nz.min(self.total_nz - 1);
                    let is_pos = (self.packed_signs[clamped_idx / 8] >> (clamped_idx % 8)) & 1 != 0;
                    out[global_idx] = if is_pos { scale } else { -scale };
                    running_nz += 1;
                }
            }
        }
        out
    }

    /// Nominal bits per weight including the 128-bit aligned header (`0.125` bpw).
    pub fn nominal_bpw(&self) -> f64 {
        let total_bits = 128 + 1024 + self.total_nz;
        (total_bits as f64) / (SUPERBLOCK_LEN as f64)
    }
}

/// 6 valid `2:4` sparsity patterns (`(i, j)` with `0 <= i < j < 4`) encoded in 3 bits.
const PATTERNS_24: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// `BITCOS-2:4` 32-weight packed block (`3` bytes mask + `2` bytes signs = `1.25` bpw payload).
#[derive(Debug, Clone, PartialEq)]
pub struct Bitcos24Block32 {
    pub packed_mask_3b: [u8; 3],
    pub packed_signs_2b: u16,
    pub scale: f32,
}

impl Bitcos24Block32 {
    /// Encodes 32 weights into `2:4` structured ternary (selecting top-2 magnitude per quad).
    pub fn encode(weights: &[f32]) -> EngineResult<Self> {
        if weights.len() != 32 {
            return Err(EngineError::InvalidInput(format!(
                "Bitcos24Block32 requires 32 weights, got {}",
                weights.len()
            )));
        }
        let mut mask_bits = 0_u32;
        let mut sign_bits = 0_u16;
        let mut sum_abs = 0.0_f32;

        for q in 0..8 {
            let quad = &weights[q * 4..(q + 1) * 4];
            let mut order = [0_usize, 1, 2, 3];
            order.sort_by(|&a, &b| quad[b].abs().total_cmp(&quad[a].abs()));
            let (i, j) = if order[0] < order[1] {
                (order[0], order[1])
            } else {
                (order[1], order[0])
            };
            let pat_idx = PATTERNS_24.iter().position(|&p| p == (i, j)).unwrap_or(0) as u32;
            mask_bits |= pat_idx << (q * 3);

            if quad[i] >= 0.0 {
                sign_bits |= 1_u16 << (q * 2);
            }
            if quad[j] >= 0.0 {
                sign_bits |= 1_u16 << (q * 2 + 1);
            }
            sum_abs += quad[i].abs() + quad[j].abs();
        }

        Ok(Self {
            packed_mask_3b: [
                (mask_bits & 0xFF) as u8,
                ((mask_bits >> 8) & 0xFF) as u8,
                ((mask_bits >> 16) & 0xFF) as u8,
            ],
            packed_signs_2b: sign_bits,
            scale: sum_abs / 16.0,
        })
    }

    /// Decodes all 32 weights, preserving exact `2:4` sparsity.
    pub fn decode(&self) -> [f32; 32] {
        let mut out = [0.0_f32; 32];
        let mask_bits = u32::from(self.packed_mask_3b[0])
            | (u32::from(self.packed_mask_3b[1]) << 8)
            | (u32::from(self.packed_mask_3b[2]) << 16);

        for q in 0..8 {
            let pat_idx = ((mask_bits >> (q * 3)) & 0x7).min(5) as usize;
            let (i, j) = PATTERNS_24[pat_idx];
            let s0 = if (self.packed_signs_2b >> (q * 2)) & 1 != 0 {
                self.scale
            } else {
                -self.scale
            };
            let s1 = if (self.packed_signs_2b >> (q * 2 + 1)) & 1 != 0 {
                self.scale
            } else {
                -self.scale
            };
            out[q * 4 + i] = s0;
            out[q * 4 + j] = s1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitcos_1024_round_trip_and_bpw_at_515_sparsity() {
        let mut w = vec![0.0_f32; SUPERBLOCK_LEN];
        // Make 497 nonzeros (~51.5% zeros).
        for (i, val) in w.iter_mut().enumerate().take(497) {
            *val = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let sb = BitcosSuperBlock1024::encode(&w, 0.25).unwrap();
        assert_eq!(sb.total_nz, 497);
        assert!(sb.nominal_bpw() < 1.625, "bpw={}", sb.nominal_bpw());

        let dec = sb.decode();
        for (a, b) in w.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 1e-3, "a={a} b={b}");
        }
    }

    #[test]
    fn bitcos_24_preserves_exact_2_of_4_sparsity() {
        let mut w = [0.0_f32; 32];
        for q in 0..8 {
            w[q * 4] = 1.0;
            w[q * 4 + 2] = -1.0;
            w[q * 4 + 1] = 0.05;
            w[q * 4 + 3] = -0.02;
        }
        let blk = Bitcos24Block32::encode(&w).unwrap();
        let dec = blk.decode();
        for q in 0..8 {
            let nz = dec[q * 4..(q + 1) * 4]
                .iter()
                .filter(|&&v| v != 0.0)
                .count();
            assert_eq!(nz, 2);
        }
    }
}
