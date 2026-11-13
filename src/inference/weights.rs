//! Tensor payload decoding and precision round-tripping for the reference forward pass.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::exporter::conversion::{
    encode_q4_0_block, encode_q8_block, encode_tq1_block, f32_to_f16,
};
use crate::exporter::kquant::{self, QK_K};
use crate::ir::{TensorFormat, TensorIR};
use crate::parser::gguf::GgufBlockDecoder;
use crate::profiling::{bf16_to_f32, f16_to_f32};
use crate::transform::byoa::AlgorithmDefinition;

const Q8_BLOCK_VALUES: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;
const Q4_BLOCK_BYTES: usize = 18;
const TQ1_BLOCK_VALUES: usize = 256;
const TQ1_BLOCK_BYTES: usize = 54;

/// Reads a file-backed tensor payload and decodes it to `f32` values.
pub fn read_tensor_f32(tensor: &TensorIR) -> Result<Vec<f32>> {
    let bytes = read_tensor_bytes(tensor)?;
    decode_payload(&bytes, &tensor.original_dtype, tensor.numel)
        .with_context(|| format!("Failed to decode tensor '{}'", tensor.name))
}

/// Reads a tensor's raw payload bytes exactly as stored in its source file.
pub fn read_tensor_bytes(tensor: &TensorIR) -> Result<Vec<u8>> {
    let source = tensor.source.as_ref().with_context(|| {
        format!(
            "Tensor '{}' has no file-backed payload; import the model from disk first",
            tensor.name
        )
    })?;
    let len = usize::try_from(source.byte_len)
        .with_context(|| format!("Tensor '{}' payload length exceeds usize", tensor.name))?;
    let mut file = File::open(&source.path)
        .with_context(|| format!("Failed to open payload file for tensor '{}'", tensor.name))?;
    file.seek(SeekFrom::Start(source.byte_offset))
        .with_context(|| format!("Failed to seek to payload of tensor '{}'", tensor.name))?;
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)
        .with_context(|| format!("Payload of tensor '{}' is truncated", tensor.name))?;
    Ok(bytes)
}

/// Decodes a raw payload of `numel` elements stored as `dtype` into `f32` values.
pub fn decode_payload(bytes: &[u8], dtype: &str, numel: usize) -> Result<Vec<f32>> {
    let expect_len = |expected: Option<usize>| -> Result<()> {
        match expected {
            Some(expected) if expected == bytes.len() => Ok(()),
            Some(expected) => bail!(
                "{dtype} payload of {numel} elements requires {expected} bytes, found {}",
                bytes.len()
            ),
            None => bail!("{dtype} payload size overflows usize"),
        }
    };
    let blocks = |values: usize, block_bytes: usize| -> Result<Option<usize>> {
        if !numel.is_multiple_of(values) {
            bail!("{dtype} requires an element count divisible by {values}, found {numel}");
        }
        Ok((numel / values).checked_mul(block_bytes))
    };

    let mut out = Vec::with_capacity(numel);
    match dtype {
        "F32" => {
            expect_len(numel.checked_mul(4))?;
            out.extend(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
        }
        "F16" => {
            expect_len(numel.checked_mul(2))?;
            out.extend(
                bytes
                    .chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            );
        }
        "BF16" => {
            expect_len(numel.checked_mul(2))?;
            out.extend(
                bytes
                    .chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            );
        }
        "Q8_0" => {
            expect_len(blocks(Q8_BLOCK_VALUES, Q8_BLOCK_BYTES)?)?;
            for block in bytes.chunks_exact(Q8_BLOCK_BYTES) {
                out.extend_from_slice(&GgufBlockDecoder::unpack_q8_0(block)?);
            }
        }
        "Q4_0" => {
            expect_len(blocks(Q8_BLOCK_VALUES, Q4_BLOCK_BYTES)?)?;
            for block in bytes.chunks_exact(Q4_BLOCK_BYTES) {
                out.extend_from_slice(&GgufBlockDecoder::unpack_q4_0(block)?);
            }
        }
        "TQ1_0" => {
            expect_len(blocks(TQ1_BLOCK_VALUES, TQ1_BLOCK_BYTES)?)?;
            for block in bytes.chunks_exact(TQ1_BLOCK_BYTES) {
                out.extend_from_slice(&GgufBlockDecoder::unpack_tq1_0(block)?);
            }
        }
        "Q4_K" => {
            expect_len(blocks(QK_K, kquant::Q4_K_BYTES)?)?;
            for block in bytes.chunks_exact(kquant::Q4_K_BYTES) {
                out.extend_from_slice(&kquant::decode_q4_k(block)?);
            }
        }
        "Q5_K" => {
            expect_len(blocks(QK_K, kquant::Q5_K_BYTES)?)?;
            for block in bytes.chunks_exact(kquant::Q5_K_BYTES) {
                out.extend_from_slice(&kquant::decode_q5_k(block)?);
            }
        }
        "Q6_K" => {
            expect_len(blocks(QK_K, kquant::Q6_K_BYTES)?)?;
            for block in bytes.chunks_exact(kquant::Q6_K_BYTES) {
                out.extend_from_slice(&kquant::decode_q6_k(block)?);
            }
        }
        other => bail!("Unsupported tensor dtype '{other}' for the reference forward pass"),
    }
    Ok(out)
}

/// Metadata key holding a BYOA `AlgorithmDefinition` (JSON) for tensors whose selected
/// format is `TensorFormat::Custom(id)`. Attached by `thme evaluate` after resolving the id.
pub const BYOA_METADATA_KEY: &str = "byoa_definition";

/// Applies a tensor's selected format to its values: a built-in block format, or the BYOA
/// algorithm attached to it. Algorithms that are not valid weight transforms are refused.
pub fn apply_selected_format(tensor: &TensorIR, values: &mut [f32]) -> Result<()> {
    match &tensor.selected_format {
        TensorFormat::Custom(id) => {
            let json = tensor.metadata.get(BYOA_METADATA_KEY).with_context(|| {
                format!(
                    "Tensor '{}' uses algorithm '{id}' but no definition is attached; resolve it with the BYOA registry first",
                    tensor.name
                )
            })?;
            let algorithm = AlgorithmDefinition::from_json(json)
                .with_context(|| format!("Invalid definition for algorithm '{id}'"))?;
            algorithm.measurable().map_err(|e| anyhow::anyhow!(e))?;
            use crate::transform::awq::{parse_format, ChannelStats, CALIBRATION_METADATA_KEY};
            let calibration: Option<ChannelStats> = tensor
                .metadata
                .get(CALIBRATION_METADATA_KEY)
                .map(|json| serde_json::from_str(json))
                .transpose()
                .with_context(|| format!("Invalid calibration statistics on '{}'", tensor.name))?;
            let round = |v: &mut [f32], name: &str| -> Result<(), String> {
                let format = parse_format(name).map_err(|e| format!("{e:#}"))?;
                fake_quantize(v, &format).map_err(|e| format!("{e:#}"))
            };
            let ctx = crate::transform::byoa::ApplyContext {
                row_width: tensor.shape.last().copied(),
                calibration: calibration.as_ref(),
                round: Some(&round),
            };
            algorithm.apply_with(values, &ctx).map_err(|e| {
                anyhow::anyhow!("Algorithm '{id}' failed on '{}': {e}", tensor.name)
            })?;
            Ok(())
        }
        other => fake_quantize(values, other),
    }
}

/// Replaces `values` with what they become after being stored in `format`.
///
/// Uses the exporter's own block encoders and the parser's decoders, so the
/// result matches what an exported artifact would hold for that format:
/// `Q8_0` is the GGML 32-value block and `Ternary` is the GGML `TQ1_0`
/// 256-value block with an MSE-optimal scale.
pub fn fake_quantize(values: &mut [f32], format: &TensorFormat) -> Result<()> {
    match format {
        TensorFormat::FP32 => {}
        TensorFormat::FP16 => {
            for value in values.iter_mut() {
                *value = f16_to_f32(f32_to_f16(*value));
            }
        }
        TensorFormat::BF16 => {
            for value in values.iter_mut() {
                *value = bf16_to_f32((value.to_bits() >> 16) as u16);
            }
        }
        TensorFormat::Q8_0 => {
            if !values.len().is_multiple_of(Q8_BLOCK_VALUES) {
                bail!(
                    "Q8_0 requires an element count divisible by {Q8_BLOCK_VALUES}, found {}",
                    values.len()
                );
            }
            let mut encoded = Vec::with_capacity(Q8_BLOCK_BYTES);
            for chunk in values.chunks_exact_mut(Q8_BLOCK_VALUES) {
                encoded.clear();
                encode_q8_block(chunk, &mut encoded)?;
                chunk.copy_from_slice(&GgufBlockDecoder::unpack_q8_0(&encoded)?);
            }
        }
        TensorFormat::Q4_0 => {
            if !values.len().is_multiple_of(Q8_BLOCK_VALUES) {
                bail!(
                    "Q4_0 requires an element count divisible by {Q8_BLOCK_VALUES}, found {}",
                    values.len()
                );
            }
            let mut encoded = Vec::with_capacity(Q4_BLOCK_BYTES);
            for chunk in values.chunks_exact_mut(Q8_BLOCK_VALUES) {
                encoded.clear();
                encode_q4_0_block(chunk, &mut encoded)?;
                chunk.copy_from_slice(&GgufBlockDecoder::unpack_q4_0(&encoded)?);
            }
        }
        TensorFormat::Q4_K | TensorFormat::Q5_K | TensorFormat::Q6_K => {
            type Encode = fn(&[f32], &mut Vec<u8>) -> Result<()>;
            type Decode = fn(&[u8]) -> Result<[f32; QK_K]>;
            let (encode, decode): (Encode, Decode) = match format {
                TensorFormat::Q4_K => (kquant::encode_q4_k_block, kquant::decode_q4_k),
                TensorFormat::Q5_K => (kquant::encode_q5_k_block, kquant::decode_q5_k),
                _ => (kquant::encode_q6_k_block, kquant::decode_q6_k),
            };
            if !values.len().is_multiple_of(QK_K) {
                bail!(
                    "{} requires an element count divisible by {QK_K}, found {}",
                    format.display_name(),
                    values.len()
                );
            }
            let mut encoded = Vec::with_capacity(kquant::Q6_K_BYTES);
            for chunk in values.chunks_exact_mut(QK_K) {
                encoded.clear();
                encode(chunk, &mut encoded)?;
                chunk.copy_from_slice(&decode(&encoded)?);
            }
        }
        TensorFormat::Ternary => {
            if !values.len().is_multiple_of(TQ1_BLOCK_VALUES) {
                bail!(
                    "Ternary (TQ1_0) requires an element count divisible by {TQ1_BLOCK_VALUES}, found {}",
                    values.len()
                );
            }
            let mut encoded = Vec::with_capacity(TQ1_BLOCK_BYTES);
            for chunk in values.chunks_exact_mut(TQ1_BLOCK_VALUES) {
                encoded.clear();
                encode_tq1_block(chunk, &mut encoded)?;
                chunk.copy_from_slice(&GgufBlockDecoder::unpack_tq1_0(&encoded)?);
            }
        }
        other => bail!(
            "Precision round-trip for {} is not implemented in the reference forward pass",
            other.display_name()
        ),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 400.0)
            .collect()
    }

    #[test]
    fn decodes_f32_and_rejects_wrong_length() {
        let values = ramp(8);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(decode_payload(&bytes, "F32", 8).unwrap(), values);
        assert!(decode_payload(&bytes[..28], "F32", 8).is_err());
        assert!(decode_payload(&bytes, "I64", 8).is_err());
    }

    #[test]
    fn q8_round_trip_stays_within_half_a_step() {
        let original = ramp(64);
        let mut values = original.clone();
        fake_quantize(&mut values, &TensorFormat::Q8_0).unwrap();
        for (block, quantized) in original.chunks(32).zip(values.chunks(32)) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let step = amax / 127.0;
            for (a, b) in block.iter().zip(quantized) {
                assert!((a - b).abs() <= step * 0.51, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn q4_0_round_trip_uses_sixteen_levels_and_keeps_the_extreme() {
        let original = ramp(64);
        let mut values = original.clone();
        fake_quantize(&mut values, &TensorFormat::Q4_0).unwrap();
        for (block, quantized) in original.chunks(32).zip(values.chunks(32)) {
            let extreme = block
                .iter()
                .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
            let step = extreme.abs() / 8.0;
            let mut levels: Vec<i32> = quantized
                .iter()
                .map(|q| (q / step).round() as i32)
                .collect();
            levels.sort_unstable();
            levels.dedup();
            assert!(levels.len() <= 16);
            // The extreme maps exactly (code 0); values of the opposite sign can only
            // reach +7 steps, so their error is at most one step. Everything else rounds.
            let at_extreme = block.iter().position(|&v| v == extreme).unwrap();
            assert!((quantized[at_extreme] - extreme).abs() <= extreme.abs() * 1e-3);
            for (a, b) in block.iter().zip(quantized) {
                let same_side = a.signum() == extreme.signum() || *a == 0.0;
                let bound = if same_side { step * 0.51 } else { step * 1.01 };
                assert!((a - b).abs() <= bound + 1e-4, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn ternary_round_trip_leaves_three_levels_per_block() {
        let mut values = ramp(512);
        fake_quantize(&mut values, &TensorFormat::Ternary).unwrap();
        for block in values.chunks(256) {
            let scale = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(scale > 0.0);
            for v in block {
                assert!(*v == 0.0 || v.abs() == scale);
            }
        }
    }

    #[test]
    fn k_quant_payloads_decode_and_fake_quantize_matches_the_codec() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 400.0)
            .collect();
        let mut q4 = Vec::new();
        let mut q5 = Vec::new();
        let mut q6 = Vec::new();
        for chunk in values.chunks_exact(QK_K) {
            kquant::encode_q4_k_block(chunk, &mut q4).unwrap();
            kquant::encode_q5_k_block(chunk, &mut q5).unwrap();
            kquant::encode_q6_k_block(chunk, &mut q6).unwrap();
        }
        for (bytes, dtype) in [(&q4, "Q4_K"), (&q5, "Q5_K"), (&q6, "Q6_K")] {
            let decoded = decode_payload(bytes, dtype, 512).unwrap();
            assert_eq!(decoded.len(), 512);
            let worst = values
                .iter()
                .zip(&decoded)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 0.02, "{dtype} worst error {worst}");
        }
        assert!(decode_payload(&q4[..100], "Q4_K", 512).is_err());

        for (bytes, format, dtype) in [
            (&q4, TensorFormat::Q4_K, "Q4_K"),
            (&q5, TensorFormat::Q5_K, "Q5_K"),
            (&q6, TensorFormat::Q6_K, "Q6_K"),
        ] {
            let mut fake = values.clone();
            fake_quantize(&mut fake, &format).unwrap();
            assert_eq!(fake, decode_payload(bytes, dtype, 512).unwrap(), "{dtype}");
        }
    }

    #[test]
    fn unsupported_formats_and_partial_blocks_are_errors() {
        let mut values = ramp(33);
        assert!(fake_quantize(&mut values, &TensorFormat::Q8_0).is_err());
        assert!(fake_quantize(&mut values, &TensorFormat::Ternary).is_err());
        assert!(fake_quantize(&mut values, &TensorFormat::Q4_K).is_err());
        assert!(fake_quantize(&mut values, &TensorFormat::Q2_K).is_err());
        assert!(fake_quantize(&mut values, &TensorFormat::Q4_0).is_err());
        fake_quantize(&mut values, &TensorFormat::FP16).unwrap();
    }
}
