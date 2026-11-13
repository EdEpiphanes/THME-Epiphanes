//! Bounded-memory conversion of float tensor sources to GGML block formats.

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};

use crate::exporter::calibration::optimal_ternary_scale;
use crate::exporter::streaming_writer::STREAM_CHUNK_SIZE;
use crate::parser::gguf::GgmlType;

type BlockEncoder = fn(&[f32], &mut Vec<u8>) -> Result<()>;

fn encode_f16_value(values: &[f32], out: &mut Vec<u8>) -> Result<()> {
    for &value in values {
        let half = f32_to_f16(value);
        if !crate::profiling::f16_to_f32(half).is_finite() {
            bail!("Value {value} exceeds the finite FP16 range");
        }
        out.extend_from_slice(&half.to_le_bytes());
    }
    Ok(())
}

/// Values per block, bytes per block and encoder for each supported conversion target.
fn conversion_target(target: GgmlType) -> Result<(usize, usize, BlockEncoder)> {
    use crate::exporter::kquant;
    Ok(match target {
        GgmlType::F16 => (1, 2, encode_f16_value),
        GgmlType::Q8_0 => (32, 34, encode_q8_block),
        GgmlType::Q4_0 => (32, 18, encode_q4_0_block),
        GgmlType::TQ1_0 => (256, 54, encode_tq1_block),
        GgmlType::Q4_K => (kquant::QK_K, kquant::Q4_K_BYTES, kquant::encode_q4_k_block),
        GgmlType::Q5_K => (kquant::QK_K, kquant::Q5_K_BYTES, kquant::encode_q5_k_block),
        GgmlType::Q6_K => (kquant::QK_K, kquant::Q6_K_BYTES, kquant::encode_q6_k_block),
        _ => bail!("Unsupported float conversion target {}", target.type_name()),
    })
}

/// True when `encode_float_source` can write `target`.
pub fn is_conversion_target(target: GgmlType) -> bool {
    conversion_target(target).is_ok()
}

/// Encode an exact-length F32/F16/BF16 source into a GGML stream of `target`
/// (F16, Q8_0, Q4_0, TQ1_0, Q4_K, Q5_K or Q6_K).
pub fn encode_float_source<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    source_dtype: &str,
    target: GgmlType,
    element_count: usize,
) -> Result<u64> {
    let (block, byte_size, encode) = conversion_target(target)?;
    if !matches!(source_dtype, "F16" | "F32" | "BF16") {
        bail!("Conversion source dtype must be F32, F16 or BF16, got {source_dtype}");
    }
    if !element_count.is_multiple_of(block) {
        bail!(
            "Tensor element count {element_count} must be divisible by {block} for {}",
            target.type_name()
        );
    }
    let source_width = if source_dtype == "F32" { 4 } else { 2 };
    let mut input = [0u8; STREAM_CHUNK_SIZE];
    let mut written = 0u64;
    let mut remaining = element_count;
    while remaining > 0 {
        let count = (remaining.min(STREAM_CHUNK_SIZE / source_width) / block) * block;
        let input_len = count
            .checked_mul(source_width)
            .context("Input chunk size overflow")?;
        reader
            .read_exact(&mut input[..input_len])
            .context("Float tensor source ended before expected element count")?;
        let mut values = Vec::with_capacity(count);
        for bytes in input[..input_len].chunks_exact(source_width) {
            let value = match source_dtype {
                "F32" => f32::from_le_bytes(bytes.try_into().context("Invalid F32 source chunk")?),
                "BF16" => crate::profiling::bf16_to_f32(u16::from_le_bytes(
                    bytes.try_into().context("Invalid BF16 source chunk")?,
                )),
                _ => crate::profiling::f16_to_f32(u16::from_le_bytes(
                    bytes.try_into().context("Invalid F16 source chunk")?,
                )),
            };
            if !value.is_finite() {
                bail!("Float source contains NaN or infinity");
            }
            values.push(value);
        }
        for chunk in values.chunks_exact(block) {
            let mut encoded = Vec::with_capacity(byte_size);
            encode(chunk, &mut encoded)?;
            writer
                .write_all(&encoded)
                .context("Failed writing converted tensor block")?;
            written = written
                .checked_add(encoded.len() as u64)
                .context("Converted payload size overflow")?;
        }
        remaining -= count;
    }
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!("Float tensor source contains bytes beyond expected element count");
    }
    Ok(written)
}

pub(crate) fn encode_q8_block(values: &[f32], out: &mut Vec<u8>) -> Result<()> {
    let amax = values
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    let scale = amax / 127.0;
    let half_scale = f32_to_f16(scale);
    if !crate::profiling::f16_to_f32(half_scale).is_finite() {
        bail!("Q8_0 scale exceeds finite FP16 range");
    }
    if scale > 0.0 && half_scale == 0 {
        bail!("Q8_0 scale underflows to zero in FP16");
    }
    out.extend_from_slice(&half_scale.to_le_bytes());
    for value in values {
        let quant = if scale == 0.0 {
            0
        } else {
            (value / scale).round().clamp(-127.0, 127.0) as i8
        };
        out.push(quant as u8);
    }
    Ok(())
}

/// Encodes 32 values as a GGML Q4_0 block: an FP16 scale `d` and 4-bit codes `q`
/// with `value ≈ d * (q - 8)`. Follows the GGML reference: `d` is the signed value of
/// largest magnitude divided by -8, so that extreme maps exactly to code 0.
pub(crate) fn encode_q4_0_block(values: &[f32], out: &mut Vec<u8>) -> Result<()> {
    if values.len() != 32 {
        bail!("Q4_0 block requires 32 values, got {}", values.len());
    }
    let extreme = values
        .iter()
        .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
    let scale = extreme / -8.0;
    let half_scale = f32_to_f16(scale);
    let stored = crate::profiling::f16_to_f32(half_scale);
    if !stored.is_finite() {
        bail!("Q4_0 scale exceeds finite FP16 range");
    }
    if scale != 0.0 && half_scale & 0x7fff == 0 {
        bail!("Q4_0 scale underflows to zero in FP16");
    }
    let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    let code = |v: f32| -> u8 { ((v * inv + 8.5) as i32).clamp(0, 15) as u8 };
    out.extend_from_slice(&half_scale.to_le_bytes());
    for i in 0..16 {
        out.push(code(values[i]) | (code(values[i + 16]) << 4));
    }
    Ok(())
}

pub(crate) fn encode_tq1_block(values: &[f32], out: &mut Vec<u8>) -> Result<()> {
    let scale = optimal_ternary_scale(values)?.scale;
    let half_scale = f32_to_f16(scale);
    if !crate::profiling::f16_to_f32(half_scale).is_finite() {
        bail!("TQ1_0 scale exceeds finite FP16 range");
    }
    if scale > 0.0 && half_scale == 0 {
        bail!("TQ1_0 scale underflows to zero in FP16");
    }
    let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    let mut quant = values
        .iter()
        .map(|value| (value * inv).round().clamp(-1.0, 1.0) as i16 + 1);
    let q: Vec<u8> = quant.by_ref().map(|v| v as u8).collect();
    let mut qs = [0u8; 48];
    // GGML groups 5 values from each 32-value lane, then maps base-3 [0,242] to byte [0,255].
    for (j, byte) in qs.iter_mut().enumerate() {
        let (base, stride) = if j < 32 { (j, 32) } else { (160 + j - 32, 16) };
        let mut packed = 0u16;
        for n in 0..5 {
            packed = packed * 3 + q[base + n * stride] as u16;
        }
        *byte = (packed * 256).div_ceil(243) as u8;
    }
    let mut qh = [0u8; 4];
    for (j, byte) in qh.iter_mut().enumerate() {
        let mut packed = 0u16;
        for n in 0..4 {
            packed = packed * 3 + q[240 + j + n * 4] as u16;
        }
        packed *= 3; // qh stores four trits followed by a zero padding trit.
        *byte = (packed * 256).div_ceil(243) as u8;
    }
    out.extend_from_slice(&qs);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&half_scale.to_le_bytes());
    Ok(())
}

pub(crate) fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7fffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return sign | ((mantissa + 0x1000) >> 13) as u16;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exponent as u16) << 10) | ((mantissa + 0x1000) >> 13) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn q8_uses_standard_34_byte_block_and_reconstructs_extrema() {
        let values: Vec<f32> = (0..32).map(|v| v as f32 - 16.0).collect();
        let mut output = Vec::new();
        encode_float_source(
            &mut Cursor::new(f32_bytes(&values)),
            &mut output,
            "F32",
            GgmlType::Q8_0,
            32,
        )
        .unwrap();
        assert_eq!(output.len(), 34);
        assert_eq!(
            crate::parser::gguf::GgufBlockDecoder::unpack_q8_0(&output)
                .unwrap()
                .len(),
            32
        );
    }

    #[test]
    fn tq1_matches_standard_block_layout_and_accepts_zeroes() {
        let values = vec![0.0; 256];
        let mut output = Vec::new();
        encode_float_source(
            &mut Cursor::new(f32_bytes(&values)),
            &mut output,
            "F32",
            GgmlType::TQ1_0,
            256,
        )
        .unwrap();
        assert_eq!(output.len(), 54);
        assert_eq!(&output[..48], &[128; 48]);
        assert_eq!(&output[48..52], &[127; 4]);
        assert_eq!(&output[52..], &[0, 0]);
    }

    #[test]
    fn tq1_nonuniform_values_roundtrip_with_reference_trit_order() {
        let values: Vec<f32> = (0..256)
            .map(|index| match index % 5 {
                0 => -2.0,
                1 => -1.0,
                2 => 0.0,
                3 => 1.0,
                _ => 2.0,
            })
            .collect();
        let scale = optimal_ternary_scale(&values).unwrap().scale;
        let mut block = Vec::new();
        encode_tq1_block(&values, &mut block).unwrap();
        let decoded = crate::parser::gguf::GgufBlockDecoder::unpack_tq1_0(&block).unwrap();
        let stored_scale = crate::profiling::f16_to_f32(u16::from_le_bytes([block[52], block[53]]));
        for (index, (actual, value)) in decoded.iter().zip(values).enumerate() {
            let expected = ((value / scale).round().clamp(-1.0, 1.0) as i8) as f32 * stored_scale;
            assert_eq!(*actual, expected, "trit mismatch at value index {index}");
        }
    }

    #[test]
    fn positive_scales_that_underflow_fp16_are_rejected() {
        let mut q8 = Vec::new();
        assert!(encode_q8_block(&[1.0e-10; 32], &mut q8).is_err());
        let mut tq1 = Vec::new();
        assert!(encode_tq1_block(&[1.0e-10; 256], &mut tq1).is_err());
    }

    #[test]
    fn conversion_rejects_partial_blocks_and_nonfinite_values() {
        let mut out = Vec::new();
        assert!(encode_float_source(
            &mut Cursor::new(vec![0; 4]),
            &mut out,
            "F32",
            GgmlType::Q8_0,
            1
        )
        .is_err());
        let mut values = vec![0.0; 32];
        values[0] = f32::NAN;
        assert!(encode_float_source(
            &mut Cursor::new(f32_bytes(&values)),
            &mut out,
            "F32",
            GgmlType::Q8_0,
            32
        )
        .is_err());
    }
}
