//! GGML K-quant block codecs: Q4_K, Q5_K and Q6_K (256 values per super-block).
//!
//! Byte layouts follow GGML's `block_q4_K` (144 bytes), `block_q5_K` (176 bytes) and
//! `block_q6_K` (210 bytes), so blocks written here decode in llama.cpp and blocks read from
//! GGUF files decode here. The encoders choose scales from each sub-block's range (a direct
//! rule, not llama.cpp's iterative search), so their error can be slightly higher than
//! llama.cpp's own quantizer while producing format-identical blocks.

// Sub-block loops read clearer with `chunks_exact` than with `as_chunks` arrays.
#![allow(clippy::chunks_exact_to_as_chunks)]

use anyhow::{bail, Result};

use super::conversion::f32_to_f16;
use crate::profiling::f16_to_f32;

/// Values per K-quant super-block.
pub const QK_K: usize = 256;
pub const Q4_K_BYTES: usize = 144;
pub const Q5_K_BYTES: usize = 176;
pub const Q6_K_BYTES: usize = 210;
const K_SCALE_SIZE: usize = 12;

fn f16_at(bytes: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

/// The 6-bit scale and min of sub-block `j` (0..8) from the packed 12-byte scales field.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Packs eight 6-bit scales and mins into the 12-byte field read by `scale_min_k4`.
fn pack_scale_min_k4(ls: &[u8; 8], lm: &[u8; 8]) -> [u8; K_SCALE_SIZE] {
    let mut q = [0u8; K_SCALE_SIZE];
    for j in 0..8 {
        let (s, m) = (ls[j] & 63, lm[j] & 63);
        if j < 4 {
            q[j] |= s;
            q[j + 4] |= m;
        } else {
            q[j + 4] = (s & 0x0F) | ((m & 0x0F) << 4);
            q[j - 4] |= (s >> 4) << 6;
            q[j] |= (m >> 4) << 6;
        }
    }
    q
}

fn check_block(values: &[f32], name: &str) -> Result<()> {
    if values.len() != QK_K {
        bail!("{name} block requires {QK_K} values, got {}", values.len());
    }
    if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
        bail!("{name} cannot encode non-finite value {bad}");
    }
    Ok(())
}

fn check_len(block: &[u8], expected: usize, name: &str) -> Result<()> {
    if block.len() < expected {
        bail!(
            "{name} block requires {expected} bytes, got {}",
            block.len()
        );
    }
    Ok(())
}

/// Rounds `value` to the f16 grid; the encoders quantize against the scale the decoder sees.
fn f16_round(value: f32) -> (u16, f32) {
    let bits = f32_to_f16(value);
    (bits, f16_to_f32(bits))
}

/// Scale and min (both >= 0) per 32-value sub-block for `levels` quantization steps, and the
/// 6-bit codes against the shared super-block scales `d` and `dmin`.
struct AsymmetricScales {
    d_bits: u16,
    dmin_bits: u16,
    d: f32,
    dmin: f32,
    ls: [u8; 8],
    lm: [u8; 8],
}

fn asymmetric_scales(values: &[f32], levels: f32) -> AsymmetricScales {
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    for (s, sub) in values.chunks_exact(32).enumerate() {
        let lo = sub.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);
        let hi = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        scales[s] = (hi - lo) / levels;
        mins[s] = -lo;
    }
    let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
    let max_min = mins.iter().copied().fold(0.0f32, f32::max);
    let (d_bits, d) = f16_round(max_scale / 63.0);
    let (dmin_bits, dmin) = f16_round(max_min / 63.0);
    let code = |value: f32, unit: f32| -> u8 {
        if unit > 0.0 {
            (value / unit).round().clamp(0.0, 63.0) as u8
        } else {
            0
        }
    };
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for s in 0..8 {
        // Round the scale up so the sub-block's range still fits in `levels` steps.
        ls[s] = if d > 0.0 {
            ((scales[s] / d).ceil()).clamp(0.0, 63.0) as u8
        } else {
            0
        };
        lm[s] = code(mins[s], dmin);
    }
    AsymmetricScales {
        d_bits,
        dmin_bits,
        d,
        dmin,
        ls,
        lm,
    }
}

fn quantize_asym(x: f32, scale: f32, min: f32, max_q: f32) -> u8 {
    if scale > 0.0 {
        ((x + min) / scale).round().clamp(0.0, max_q) as u8
    } else {
        0
    }
}

/// Encodes 256 values as one Q4_K block (144 bytes).
pub fn encode_q4_k_block(values: &[f32], out: &mut Vec<u8>) -> Result<()> {
    check_block(values, "Q4_K")?;
    let s = asymmetric_scales(values, 15.0);
    out.extend_from_slice(&s.d_bits.to_le_bytes());
    out.extend_from_slice(&s.dmin_bits.to_le_bytes());
    out.extend_from_slice(&pack_scale_min_k4(&s.ls, &s.lm));
    for chunk in 0..4 {
        let (a, b) = (2 * chunk, 2 * chunk + 1);
        let (sa, ma) = (s.d * f32::from(s.ls[a]), s.dmin * f32::from(s.lm[a]));
        let (sb, mb) = (s.d * f32::from(s.ls[b]), s.dmin * f32::from(s.lm[b]));
        for l in 0..32 {
            let qa = quantize_asym(values[64 * chunk + l], sa, ma, 15.0);
            let qb = quantize_asym(values[64 * chunk + 32 + l], sb, mb, 15.0);
            out.push(qa | (qb << 4));
        }
    }
    Ok(())
}

/// Decodes one Q4_K block.
pub fn decode_q4_k(block: &[u8]) -> Result<[f32; QK_K]> {
    check_len(block, Q4_K_BYTES, "Q4_K")?;
    let d = f16_at(block, 0);
    let dmin = f16_at(block, 2);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qs = &block[4 + K_SCALE_SIZE..Q4_K_BYTES];
    let mut out = [0.0f32; QK_K];
    for chunk in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * chunk, scales);
        let (sc2, m2) = scale_min_k4(2 * chunk + 1, scales);
        let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
        let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
        for l in 0..32 {
            let q = qs[32 * chunk + l];
            out[64 * chunk + l] = d1 * f32::from(q & 0x0F) - min1;
            out[64 * chunk + 32 + l] = d2 * f32::from(q >> 4) - min2;
        }
    }
    Ok(out)
}

/// Encodes 256 values as one Q5_K block (176 bytes).
pub fn encode_q5_k_block(values: &[f32], out: &mut Vec<u8>) -> Result<()> {
    check_block(values, "Q5_K")?;
    let s = asymmetric_scales(values, 31.0);
    out.extend_from_slice(&s.d_bits.to_le_bytes());
    out.extend_from_slice(&s.dmin_bits.to_le_bytes());
    out.extend_from_slice(&pack_scale_min_k4(&s.ls, &s.lm));
    let mut qh = [0u8; 32];
    let mut ql = [0u8; 128];
    for chunk in 0..4 {
        let (a, b) = (2 * chunk, 2 * chunk + 1);
        let (sa, ma) = (s.d * f32::from(s.ls[a]), s.dmin * f32::from(s.lm[a]));
        let (sb, mb) = (s.d * f32::from(s.ls[b]), s.dmin * f32::from(s.lm[b]));
        for l in 0..32 {
            let qa = quantize_asym(values[64 * chunk + l], sa, ma, 31.0);
            let qb = quantize_asym(values[64 * chunk + 32 + l], sb, mb, 31.0);
            ql[32 * chunk + l] = (qa & 0x0F) | ((qb & 0x0F) << 4);
            qh[l] |= ((qa >> 4) << (2 * chunk)) | ((qb >> 4) << (2 * chunk + 1));
        }
    }
    out.extend_from_slice(&qh);
    out.extend_from_slice(&ql);
    Ok(())
}

/// Decodes one Q5_K block.
pub fn decode_q5_k(block: &[u8]) -> Result<[f32; QK_K]> {
    check_len(block, Q5_K_BYTES, "Q5_K")?;
    let d = f16_at(block, 0);
    let dmin = f16_at(block, 2);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qh = &block[16..48];
    let ql = &block[48..Q5_K_BYTES];
    let mut out = [0.0f32; QK_K];
    for chunk in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * chunk, scales);
        let (sc2, m2) = scale_min_k4(2 * chunk + 1, scales);
        let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
        let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
        let (u1, u2) = (1u8 << (2 * chunk), 1u8 << (2 * chunk + 1));
        for l in 0..32 {
            let q = ql[32 * chunk + l];
            let hi1 = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            let hi2 = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            out[64 * chunk + l] = d1 * (f32::from(q & 0x0F) + hi1) - min1;
            out[64 * chunk + 32 + l] = d2 * (f32::from(q >> 4) + hi2) - min2;
        }
    }
    Ok(out)
}

/// Encodes 256 values as one Q6_K block (210 bytes): 16 sub-blocks of 16 signed 6-bit values.
pub fn encode_q6_k_block(values: &[f32], out: &mut Vec<u8>) -> Result<()> {
    check_block(values, "Q6_K")?;
    let mut sub_scales = [0.0f32; 16];
    for (s, sub) in values.chunks_exact(16).enumerate() {
        sub_scales[s] = sub.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 31.0;
    }
    let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
    let (d_bits, d) = f16_round(max_scale / 127.0);
    let mut sc = [0i8; 16];
    for s in 0..16 {
        sc[s] = if d > 0.0 {
            (sub_scales[s] / d).ceil().clamp(0.0, 127.0) as i8
        } else {
            0
        };
    }
    let mut q = [0u8; QK_K];
    for (i, &x) in values.iter().enumerate() {
        let scale = d * f32::from(sc[i / 16]);
        let level = if scale > 0.0 {
            (x / scale).round().clamp(-32.0, 31.0) as i32
        } else {
            0
        };
        q[i] = (level + 32) as u8;
    }
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    for n in 0..2 {
        for l in 0..32 {
            let base = 128 * n + l;
            let (q1, q2, q3, q4) = (q[base], q[base + 32], q[base + 64], q[base + 96]);
            ql[64 * n + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
            ql[64 * n + 32 + l] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);
            qh[32 * n + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(sc.iter().map(|&s| s as u8));
    out.extend_from_slice(&d_bits.to_le_bytes());
    Ok(())
}

/// Decodes one Q6_K block.
pub fn decode_q6_k(block: &[u8]) -> Result<[f32; QK_K]> {
    check_len(block, Q6_K_BYTES, "Q6_K")?;
    let ql = &block[..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = f16_at(block, 208);
    let mut out = [0.0f32; QK_K];
    for n in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let (a, b, h) = (ql[64 * n + l], ql[64 * n + 32 + l], qh[32 * n + l]);
            let q1 = i32::from((a & 0x0F) | ((h & 3) << 4)) - 32;
            let q2 = i32::from((b & 0x0F) | (((h >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((a >> 4) | (((h >> 4) & 3) << 4)) - 32;
            let q4 = i32::from((b >> 4) | (((h >> 6) & 3) << 4)) - 32;
            let s = |k: usize| f32::from(sc[8 * n + is + k] as i8);
            out[128 * n + l] = d * s(0) * q1 as f32;
            out[128 * n + l + 32] = d * s(2) * q2 as f32;
            out[128 * n + l + 64] = d * s(4) * q3 as f32;
            out[128 * n + l + 96] = d * s(6) * q4 as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: [u8; 2] = [0x00, 0x3C]; // f16 1.0

    #[test]
    fn scale_min_packing_matches_hand_derived_bytes() {
        let ls = [1, 2, 3, 4, 37, 58, 17, 63];
        let lm = [5, 6, 7, 8, 44, 19, 48, 1];
        // Derived by hand from the GGML layout (low 6 bits / high 2 bits / shared nibbles).
        let expected = [129, 194, 67, 196, 133, 70, 199, 8, 197, 58, 1, 31];
        let packed = pack_scale_min_k4(&ls, &lm);
        assert_eq!(packed, expected);
        for j in 0..8 {
            assert_eq!(scale_min_k4(j, &expected), (ls[j], lm[j]), "sub-block {j}");
        }
    }

    #[test]
    fn q4_k_decoder_reads_hand_built_block() {
        let mut block = vec![0u8; Q4_K_BYTES];
        block[0..2].copy_from_slice(&ONE); // d = 1
        block[2..4].copy_from_slice(&ONE); // dmin = 1
        let ls = [1, 2, 1, 1, 1, 1, 1, 1];
        let lm = [1, 0, 0, 0, 0, 0, 0, 0];
        block[4..16].copy_from_slice(&pack_scale_min_k4(&ls, &lm));
        block[16] = 0x21; // low nibble -> value 0 (sub-block 0), high -> value 32 (sub-block 1)
        let out = decode_q4_k(&block).unwrap();
        assert_eq!(out[0], 1.0 * 1.0 - 1.0);
        assert_eq!(out[32], 2.0 * 2.0);
        assert_eq!(out[1], -1.0);
    }

    #[test]
    fn q5_k_decoder_reads_the_high_bits() {
        let mut block = vec![0u8; Q5_K_BYTES];
        block[0..2].copy_from_slice(&ONE);
        block[4..16].copy_from_slice(&pack_scale_min_k4(&[1; 8], &[0; 8]));
        block[16] = 0b0000_0111; // qh[0]: high bit for values 0, 32 and 64
        block[48] = 0x0F; // ql[0]: values 0 (low) and 32 (high nibble 0)
        block[48 + 32] = 0x03; // ql[32]: value 64 (low nibble 3)
        let out = decode_q5_k(&block).unwrap();
        assert_eq!(out[0], 31.0);
        assert_eq!(out[32], 16.0);
        assert_eq!(out[64], 19.0);
        assert_eq!(out[96], 0.0);
    }

    #[test]
    fn q6_k_decoder_reads_hand_built_block() {
        let mut block = vec![0u8; Q6_K_BYTES];
        block[0] = 0x05; // ql[0]: q1 low = 5, q3 low = 0
        block[32] = 0x30; // ql[32]: q2 low = 0, q4 low = 3
        block[128] = 0b10; // qh[0]: q1 high = 2
        block[192] = 2; // sc[0]
        block[194] = 1; // sc[2]
        block[196] = 1; // sc[4]
        block[208..210].copy_from_slice(&ONE);
        let out = decode_q6_k(&block).unwrap();
        assert_eq!(out[0], 2.0 * 5.0); // (5 | 2 << 4) - 32 = 5
        assert_eq!(out[32], -32.0); // q2 = 0 - 32, sc[2] = 1
        assert_eq!(out[64], -32.0); // q3 = 0 - 32, sc[4] = 1
        assert_eq!(out[96], 0.0); // sc[6] = 0
    }

    fn sample(seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..QK_K)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                u * 0.08 + if i % 37 == 0 { 0.2 } else { 0.0 }
            })
            .collect()
    }

    fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
        let err: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let sig: f32 = a.iter().map(|x| x * x).sum();
        (err / sig).sqrt()
    }

    #[test]
    fn round_trips_are_accurate_and_more_bits_mean_less_error() {
        for seed in [1u64, 7, 42] {
            let values = sample(seed);
            let mut b4 = Vec::new();
            let mut b5 = Vec::new();
            let mut b6 = Vec::new();
            encode_q4_k_block(&values, &mut b4).unwrap();
            encode_q5_k_block(&values, &mut b5).unwrap();
            encode_q6_k_block(&values, &mut b6).unwrap();
            assert_eq!(
                (b4.len(), b5.len(), b6.len()),
                (Q4_K_BYTES, Q5_K_BYTES, Q6_K_BYTES)
            );
            let e4 = rel_rms(&values, &decode_q4_k(&b4).unwrap());
            let e5 = rel_rms(&values, &decode_q5_k(&b5).unwrap());
            let e6 = rel_rms(&values, &decode_q6_k(&b6).unwrap());
            assert!(e4 < 0.12, "Q4_K error {e4}");
            assert!(e5 < 0.06, "Q5_K error {e5}");
            assert!(e6 < 0.04, "Q6_K error {e6}");
            assert!(e6 < e5 && e5 < e4, "{e4} {e5} {e6}");
        }
    }

    #[test]
    fn encoders_handle_zero_blocks_and_reject_bad_input() {
        let zeros = vec![0.0f32; QK_K];
        for encode in [encode_q4_k_block, encode_q5_k_block, encode_q6_k_block] {
            let mut out = Vec::new();
            encode(&zeros, &mut out).unwrap();
            assert!(encode(&zeros[..100], &mut Vec::new()).is_err());
            let mut bad = zeros.clone();
            bad[3] = f32::NAN;
            assert!(encode(&bad, &mut Vec::new()).is_err());
        }
        let mut out = Vec::new();
        encode_q4_k_block(&zeros, &mut out).unwrap();
        assert!(decode_q4_k(&out).unwrap().iter().all(|v| *v == 0.0));
        assert!(decode_q6_k(&[0u8; 10]).is_err());
    }
}
