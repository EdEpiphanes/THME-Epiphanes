// SPDX-License-Identifier: AGPL-3.0-only
//! BitNet b1.58 Vectorized Bitstream Packer (Milestone v0.1.8.1)
//!
//! Encodes 1.58-bit ternary states {-1, 0, +1} into dense 2-bit representations
//! (4 weights per byte) with per-channel scale extraction, achieving an 8.0x
//! compression ratio over 16-bit floating point baselines.

use serde::{Deserialize, Serialize};

/// 2-bit ternary encoding scheme:
/// - `0b00` (0) -> -1
/// - `0b01` (1) ->  0
/// - `0b10` (2) -> +1
/// - `0b11` (3) -> Reserved / Padding
const CODE_NEG_ONE: u8 = 0b00;
const CODE_ZERO: u8 = 0b01;
const CODE_POS_ONE: u8 = 0b10;
const CODE_PADDING: u8 = 0b11;

/// Packed BitNet b1.58 tensor representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackedBitNetTensor {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    /// Per-row (channel) FP32/FP16 scale factors: alpha_i = (1 / cols) * sum(|W_ij|)
    pub row_scales: Vec<f32>,
    /// Contiguous 2-bit packed binary payload (4 weights per byte)
    pub packed_data: Vec<u8>,
}

impl PackedBitNetTensor {
    /// Total unpacked elements in the tensor
    #[inline]
    pub fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    /// Size of the packed payload in bytes
    #[inline]
    pub fn packed_byte_size(&self) -> usize {
        self.packed_data.len()
    }

    /// Theoretical FP16 byte size (2 bytes per element)
    #[inline]
    pub fn fp16_byte_size(&self) -> usize {
        self.element_count() * 2
    }

    /// Effective compression ratio achieved against FP16
    #[inline]
    pub fn compression_ratio(&self) -> f32 {
        let total_packed = self.packed_byte_size() + self.row_scales.len() * 4;
        if total_packed == 0 {
            return 1.0;
        }
        self.fp16_byte_size() as f32 / total_packed as f32
    }
}

/// Vectorized packing and unpacking engine for BitNet b1.58 ternary matrices
pub struct BitNetPacker;

impl BitNetPacker {
    /// Pack a 2D floating-point weight matrix (row-major) into a `PackedBitNetTensor`
    pub fn pack_matrix(
        name: &str,
        rows: usize,
        cols: usize,
        weights: &[f32],
    ) -> PackedBitNetTensor {
        if rows == 0 || cols == 0 || weights.is_empty() {
            return PackedBitNetTensor {
                name: name.to_string(),
                rows: 0,
                cols: 0,
                row_scales: Vec::new(),
                packed_data: Vec::new(),
            };
        }

        let mut row_scales = Vec::with_capacity(rows);
        let mut packed_data = Vec::new();

        for r in 0..rows {
            let row_start = r * cols;
            let row_end = (row_start + cols).min(weights.len());
            let row_slice = if row_start < weights.len() {
                &weights[row_start..row_end]
            } else {
                &[]
            };

            // 1. Calculate per-channel absolute mean scale: alpha = mean(|W_ij|)
            let mean_abs = crate::engine::numfmt::absmean(row_slice);
            // An all-zero row stores scale 1.0; its codes are all zero either way.
            let scale = if mean_abs > 0.0 { mean_abs } else { 1.0 };
            row_scales.push(scale);

            // 2. Discretize weights to ternary {-1, 0, +1} and pack 4 per byte
            let mut byte_val: u8 = 0;
            let mut shift: u8 = 0;

            for &w in row_slice {
                let ternary = crate::engine::numfmt::ternary_code(w / scale, 0.5);

                let code = match ternary {
                    -1 => CODE_NEG_ONE,
                    1 => CODE_POS_ONE,
                    _ => CODE_ZERO,
                };

                byte_val |= code << shift;
                shift += 2;

                if shift == 8 {
                    packed_data.push(byte_val);
                    byte_val = 0;
                    shift = 0;
                }
            }

            // Flush remaining partial byte with padding if cols is not multiple of 4
            if shift > 0 {
                while shift < 8 {
                    byte_val |= CODE_PADDING << shift;
                    shift += 2;
                }
                packed_data.push(byte_val);
            }
        }

        PackedBitNetTensor {
            name: name.to_string(),
            rows,
            cols,
            row_scales,
            packed_data,
        }
    }

    /// Pack pre-discretized ternary states {-1, 0, +1} directly with provided per-channel scales
    pub fn pack_ternary_slice(
        name: &str,
        rows: usize,
        cols: usize,
        ternary_weights: &[i8],
        row_scales: &[f32],
    ) -> PackedBitNetTensor {
        let mut packed_data = Vec::new();

        for r in 0..rows {
            let row_start = r * cols;
            let row_end = (row_start + cols).min(ternary_weights.len());
            let row_slice = if row_start < ternary_weights.len() {
                &ternary_weights[row_start..row_end]
            } else {
                &[]
            };

            let mut byte_val: u8 = 0;
            let mut shift: u8 = 0;

            for &t in row_slice {
                let code = match t {
                    -1 => CODE_NEG_ONE,
                    1 => CODE_POS_ONE,
                    _ => CODE_ZERO,
                };

                byte_val |= code << shift;
                shift += 2;

                if shift == 8 {
                    packed_data.push(byte_val);
                    byte_val = 0;
                    shift = 0;
                }
            }

            if shift > 0 {
                while shift < 8 {
                    byte_val |= CODE_PADDING << shift;
                    shift += 2;
                }
                packed_data.push(byte_val);
            }
        }

        PackedBitNetTensor {
            name: name.to_string(),
            rows,
            cols,
            row_scales: row_scales.to_vec(),
            packed_data,
        }
    }

    /// Unpack a `PackedBitNetTensor` back into floating-point weights: W_ij = ternary * scale_i
    pub fn unpack_matrix(packed: &PackedBitNetTensor) -> Vec<f32> {
        let total_elements = packed.rows * packed.cols;
        if total_elements == 0 {
            return Vec::new();
        }

        let mut unpacked = Vec::with_capacity(total_elements);
        let bytes_per_row = packed.cols.div_ceil(4);

        for r in 0..packed.rows {
            let scale = packed.row_scales.get(r).copied().unwrap_or(1.0);
            let row_byte_start = r * bytes_per_row;

            let mut cols_read = 0;
            let mut byte_idx = row_byte_start;

            while cols_read < packed.cols && byte_idx < packed.packed_data.len() {
                let byte_val = packed.packed_data[byte_idx];
                byte_idx += 1;

                for shift in (0..8).step_by(2) {
                    if cols_read >= packed.cols {
                        break;
                    }
                    let code = (byte_val >> shift) & 0b11;
                    let ternary_val = match code {
                        CODE_NEG_ONE => -1.0,
                        CODE_POS_ONE => 1.0,
                        _ => 0.0, // CODE_ZERO, CODE_PADDING, or invalid
                    };

                    unpacked.push(ternary_val * scale);
                    cols_read += 1;
                }
            }
        }

        unpacked
    }

    /// Unpack raw ternary values in {-1, 0, 1} without scale multiplication
    pub fn unpack_raw_ternary(packed: &PackedBitNetTensor) -> Vec<i8> {
        let total_elements = packed.rows * packed.cols;
        if total_elements == 0 {
            return Vec::new();
        }

        let mut unpacked = Vec::with_capacity(total_elements);
        let bytes_per_row = packed.cols.div_ceil(4);

        for r in 0..packed.rows {
            let row_byte_start = r * bytes_per_row;
            let mut cols_read = 0;
            let mut byte_idx = row_byte_start;

            while cols_read < packed.cols && byte_idx < packed.packed_data.len() {
                let byte_val = packed.packed_data[byte_idx];
                byte_idx += 1;

                for shift in (0..8).step_by(2) {
                    if cols_read >= packed.cols {
                        break;
                    }
                    let code = (byte_val >> shift) & 0b11;
                    let val = match code {
                        CODE_NEG_ONE => -1,
                        CODE_POS_ONE => 1,
                        _ => 0,
                    };
                    unpacked.push(val);
                    cols_read += 1;
                }
            }
        }

        unpacked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_and_unpack_exact_ternary_fidelity() {
        let rows = 2;
        let cols = 8;
        let original_ternary: Vec<i8> = vec![-1, 0, 1, -1, 1, 1, 0, -1, 0, 1, 0, -1, -1, 0, 1, 1];
        let scales = vec![0.5, 0.25];

        let packed =
            BitNetPacker::pack_ternary_slice("test_tensor", rows, cols, &original_ternary, &scales);
        assert_eq!(packed.rows, 2);
        assert_eq!(packed.cols, 8);
        assert_eq!(packed.packed_data.len(), 4); // 8 cols = 2 bytes per row * 2 rows = 4 bytes

        let unpacked_ternary = BitNetPacker::unpack_raw_ternary(&packed);
        assert_eq!(unpacked_ternary, original_ternary);

        let unpacked_floats = BitNetPacker::unpack_matrix(&packed);
        assert_eq!(unpacked_floats.len(), 16);
        assert!((unpacked_floats[0] - (-0.5)).abs() < 1e-6);
        assert!((unpacked_floats[1] - (0.0)).abs() < 1e-6);
        assert!((unpacked_floats[2] - (0.5)).abs() < 1e-6);
        assert!((unpacked_floats[9] - (0.25)).abs() < 1e-6);
    }

    #[test]
    fn test_pack_matrix_with_non_multiple_of_four_cols() {
        let rows = 3;
        let cols = 5; // not multiple of 4; requires padding
        let weights = vec![
            0.5, -0.4, 0.0, 0.6, -0.7, 0.1, 0.2, -0.3, 0.0, 0.0, -0.9, 0.8, -0.7, 0.6, -0.5,
        ];

        let packed = BitNetPacker::pack_matrix("odd_tensor", rows, cols, &weights);
        assert_eq!(packed.rows, 3);
        assert_eq!(packed.cols, 5);
        // 5 cols -> 2 bytes per row -> 6 bytes total
        assert_eq!(packed.packed_data.len(), 6);

        let unpacked = BitNetPacker::unpack_matrix(&packed);
        assert_eq!(unpacked.len(), 15);
    }

    #[test]
    fn test_compression_ratio_calculation() {
        let rows = 64;
        let cols = 128;
        let weights = vec![0.1; rows * cols];

        let packed = BitNetPacker::pack_matrix("compression_test", rows, cols, &weights);
        let ratio = packed.compression_ratio();
        // 64 * 128 * 2 = 16384 bytes FP16
        // 64 * (128 / 4) = 2048 bytes packed + 64 * 4 (256 bytes scale) = 2304 bytes
        // 16384 / 2304 ≈ 7.11x with scale overhead; payload alone is 8.0x
        assert!(ratio > 7.0);
    }
}
