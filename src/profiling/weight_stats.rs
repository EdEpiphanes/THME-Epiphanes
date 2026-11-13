use crate::ir::WeightProfile;

/// IEEE half precision to f32 (shared implementation in `engine::minifloat`).
pub fn f16_to_f32(h: u16) -> f32 {
    crate::engine::minifloat::f16_to_f32(h)
}

/// Bfloat16 to f32 (shared implementation in `engine::minifloat`).
pub fn bf16_to_f32(b: u16) -> f32 {
    crate::engine::minifloat::bf16_to_f32(b)
}

pub struct WeightAnalyzer;

impl WeightAnalyzer {
    /// Decode a representative subsample of floating-point values from raw tensor bytes
    pub fn decode_samples(raw_data: &[u8], dtype: &str, max_samples: usize) -> Vec<f32> {
        let lower = dtype.to_lowercase();
        let mut samples = Vec::with_capacity(max_samples);

        if lower.contains("f32") || lower.contains("float32") {
            let num_floats = raw_data.len() / 4;
            let stride = (num_floats / max_samples).max(1);
            for i in (0..num_floats).step_by(stride).take(max_samples) {
                let offset = i * 4;
                if offset + 4 <= raw_data.len() {
                    let bytes = [
                        raw_data[offset],
                        raw_data[offset + 1],
                        raw_data[offset + 2],
                        raw_data[offset + 3],
                    ];
                    samples.push(f32::from_le_bytes(bytes));
                }
            }
        } else if lower.contains("bf16") || lower.contains("bfloat16") {
            let num_floats = raw_data.len() / 2;
            let stride = (num_floats / max_samples).max(1);
            for i in (0..num_floats).step_by(stride).take(max_samples) {
                let offset = i * 2;
                if offset + 2 <= raw_data.len() {
                    let u = u16::from_le_bytes([raw_data[offset], raw_data[offset + 1]]);
                    samples.push(bf16_to_f32(u));
                }
            }
        } else if lower.contains("f16") || lower.contains("float16") {
            let num_floats = raw_data.len() / 2;
            let stride = (num_floats / max_samples).max(1);
            for i in (0..num_floats).step_by(stride).take(max_samples) {
                let offset = i * 2;
                if offset + 2 <= raw_data.len() {
                    let u = u16::from_le_bytes([raw_data[offset], raw_data[offset + 1]]);
                    samples.push(f16_to_f32(u));
                }
            }
        } else if lower.contains("q4_0") {
            let block_size = 18;
            let num_blocks = raw_data.len() / block_size;
            let stride = (num_blocks / (max_samples / 32).max(1)).max(1);
            for i in (0..num_blocks).step_by(stride) {
                let offset = i * block_size;
                if offset + block_size <= raw_data.len() {
                    if let Ok(weights) = crate::parser::gguf::GgufBlockDecoder::unpack_q4_0(
                        &raw_data[offset..offset + block_size],
                    ) {
                        for w in weights {
                            samples.push(w);
                            if samples.len() >= max_samples {
                                break;
                            }
                        }
                    }
                }
                if samples.len() >= max_samples {
                    break;
                }
            }
        } else if lower.contains("q8_0") {
            let block_size = 34;
            let num_blocks = raw_data.len() / block_size;
            let stride = (num_blocks / (max_samples / 32).max(1)).max(1);
            for i in (0..num_blocks).step_by(stride) {
                let offset = i * block_size;
                if offset + block_size <= raw_data.len() {
                    if let Ok(weights) = crate::parser::gguf::GgufBlockDecoder::unpack_q8_0(
                        &raw_data[offset..offset + block_size],
                    ) {
                        for w in weights {
                            samples.push(w);
                            if samples.len() >= max_samples {
                                break;
                            }
                        }
                    }
                }
                if samples.len() >= max_samples {
                    break;
                }
            }
        } else if lower.contains("q4_1") {
            let block_size = 20;
            let num_blocks = raw_data.len() / block_size;
            let stride = (num_blocks / (max_samples / 32).max(1)).max(1);
            for i in (0..num_blocks).step_by(stride) {
                let offset = i * block_size;
                if offset + block_size <= raw_data.len() {
                    if let Ok(weights) = crate::parser::gguf::GgufBlockDecoder::unpack_q4_1(
                        &raw_data[offset..offset + block_size],
                    ) {
                        for w in weights {
                            samples.push(w);
                            if samples.len() >= max_samples {
                                break;
                            }
                        }
                    }
                }
                if samples.len() >= max_samples {
                    break;
                }
            }
        } else if lower.contains("q5_0") {
            let block_size = 22;
            let num_blocks = raw_data.len() / block_size;
            let stride = (num_blocks / (max_samples / 32).max(1)).max(1);
            for i in (0..num_blocks).step_by(stride) {
                let offset = i * block_size;
                if offset + block_size <= raw_data.len() {
                    if let Ok(weights) = crate::parser::gguf::GgufBlockDecoder::unpack_q5_0(
                        &raw_data[offset..offset + block_size],
                    ) {
                        for w in weights {
                            samples.push(w);
                            if samples.len() >= max_samples {
                                break;
                            }
                        }
                    }
                }
                if samples.len() >= max_samples {
                    break;
                }
            }
        } else if lower.contains("i8") || lower.contains("int8") || lower.contains("q8") {
            let num_bytes = raw_data.len();
            let stride = (num_bytes / max_samples).max(1);
            for i in (0..num_bytes).step_by(stride).take(max_samples) {
                let b = raw_data[i] as i8;
                samples.push(b as f32 / 127.0);
            }
        } else {
            // Default generic fallback: sample bytes as normalized values
            let num_bytes = raw_data.len();
            let stride = (num_bytes / max_samples).max(1);
            for i in (0..num_bytes).step_by(stride).take(max_samples) {
                let b = raw_data[i];
                samples.push((b as f32 - 128.0) / 128.0);
            }
        }

        samples
    }

    /// Analyze raw tensor bytes and produce comprehensive WeightProfile statistics
    pub fn profile_tensor(raw_data: &[u8], dtype: &str) -> WeightProfile {
        if raw_data.is_empty() {
            return WeightProfile::default();
        }

        // Decode up to 16,384 samples across the tensor for high statistical accuracy
        let samples = Self::decode_samples(raw_data, dtype, 16384);
        if samples.is_empty() {
            return WeightProfile::default();
        }

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut zero_count = 0;

        for &val in &samples {
            if val.is_finite() {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
                sum += val as f64;
                if val.abs() < 1e-7 {
                    zero_count += 1;
                }
            }
        }

        if !min_val.is_finite() {
            min_val = -0.15;
        }
        if !max_val.is_finite() {
            max_val = 0.15;
        }

        let n = samples.len() as f64;
        let mean_val = (sum / n) as f32;

        let mut var_sum = 0.0_f64;
        for &val in &samples {
            if val.is_finite() {
                let diff = val as f64 - mean_val as f64;
                var_sum += diff * diff;
            }
        }
        let std_val = (var_sum / n).sqrt().max(1e-6) as f32;
        let sparsity = (zero_count as f32) / (samples.len() as f32);

        // Outlier ratio: |x - mean| > 3 * std
        let outlier_thresh = 3.0 * std_val;
        let mut outlier_count = 0;
        for &val in &samples {
            if (val - mean_val).abs() > outlier_thresh {
                outlier_count += 1;
            }
        }
        let outlier_ratio = (outlier_count as f32) / (samples.len() as f32);

        // 32-bin histogram calculation
        let mut histogram = [0.0_f32; 32];
        let range = (max_val - min_val).max(1e-6);
        for &val in &samples {
            if val.is_finite() {
                let norm = ((val - min_val) / range).clamp(0.0, 0.9999);
                let bin = (norm * 32.0).floor() as usize;
                histogram[bin] += 1.0;
            }
        }
        let hist_sum: f32 = histogram.iter().sum();
        if hist_sum > 0.0 {
            for h in histogram.iter_mut() {
                *h /= hist_sum;
            }
        }

        // Preview of first 64 values for UI matrix slice inspector
        let sample_weights = samples.iter().take(64).copied().collect();

        WeightProfile {
            min_val,
            max_val,
            mean_val,
            std_val,
            sparsity,
            outlier_ratio,
            histogram,
            sample_weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        // 0.0 in f16 is 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 1.0 in f16 is 0x3C00
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // -1.0 in f16 is 0xBC00
        assert_eq!(f16_to_f32(0xBC00), -1.0);
    }

    #[test]
    fn test_weight_profiling_f32() {
        let vals = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0, 0.0, 0.2];
        let mut raw = Vec::new();
        for v in vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }

        let profile = WeightAnalyzer::profile_tensor(&raw, "F32");
        assert!(profile.min_val <= -1.0);
        assert!(profile.max_val >= 1.0);
        assert!(profile.sparsity > 0.0);
        assert_eq!(profile.histogram.len(), 32);
    }
}
