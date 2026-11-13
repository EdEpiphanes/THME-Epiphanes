// SPDX-License-Identifier: AGPL-3.0-only
use crate::profiling::hardware::PrecisionType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelType {
    LinearQProj,
    LinearKProj,
    LinearVProj,
    LinearOutProj,
    AttentionStandard,
    AttentionFlashV2,
    SwigluGateUp,
    SwigluDown,
    SwigluFused,
    RmsNorm,
    CompleteLayerMonolithic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionPhase {
    Prefill { seq_len: usize },
    Decode { context_len: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelIntensityReport {
    pub kernel: KernelType,
    pub name: String,
    pub total_flops: u64,
    pub dram_bytes: u64,
    pub arithmetic_intensity: f64,
    pub execution_phase: ExecutionPhase,
    pub precision: PrecisionType,
}

pub struct IntensityCalculator;

impl IntensityCalculator {
    #[allow(clippy::too_many_arguments)]
    pub fn linear_projection(
        name: &str,
        kernel: KernelType,
        batch_size: usize,
        seq_len: usize,
        dim_in: usize,
        dim_out: usize,
        weight_precision: PrecisionType,
        act_precision: PrecisionType,
    ) -> KernelIntensityReport {
        let b = batch_size.max(1);
        let s = seq_len.max(1);
        let m = b * s;
        let flops = 2 * (m as u64) * (dim_in as u64) * (dim_out as u64);
        let weight_bytes =
            ((dim_in * dim_out) as f64 * weight_precision.bytes_per_element()) as u64;
        let act_bytes =
            ((m * (dim_in + dim_out)) as f64 * act_precision.bytes_per_element()) as u64;
        let total_bytes = weight_bytes + act_bytes;
        let intensity = if total_bytes == 0 {
            0.0
        } else {
            flops as f64 / total_bytes as f64
        };

        KernelIntensityReport {
            kernel,
            name: name.to_string(),
            total_flops: flops,
            dram_bytes: total_bytes,
            arithmetic_intensity: intensity,
            execution_phase: ExecutionPhase::Prefill { seq_len: s },
            precision: weight_precision,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn calculate_linear(
        name: &str,
        kernel: KernelType,
        batch_size: usize,
        seq_len: usize,
        dim_in: usize,
        dim_out: usize,
        weight_precision: PrecisionType,
        act_precision: PrecisionType,
    ) -> KernelIntensityReport {
        Self::linear_projection(
            name,
            kernel,
            batch_size,
            seq_len,
            dim_in,
            dim_out,
            weight_precision,
            act_precision,
        )
    }

    pub fn swiglu_ffn(
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        intermediate_size: usize,
        weight_prec: PrecisionType,
        act_prec: PrecisionType,
    ) -> KernelIntensityReport {
        let b = batch_size.max(1);
        let s = seq_len.max(1);
        let m = (b * s) as u64;
        let d = hidden_size as u64;
        let ffn = intermediate_size as u64;

        let flops = 6 * m * d * ffn + 3 * m * ffn;
        let weight_bytes = (3 * d * ffn) as f64 * weight_prec.bytes_per_element();
        let act_bytes = if m == 1 {
            (m * (2 * d)) as f64 * act_prec.bytes_per_element()
        } else {
            (m * (2 * d + 2 * ffn)) as f64 * act_prec.bytes_per_element()
        };
        let total_bytes = (weight_bytes + act_bytes) as u64;
        let intensity = if total_bytes == 0 {
            0.0
        } else {
            flops as f64 / total_bytes as f64
        };

        KernelIntensityReport {
            kernel: KernelType::SwigluFused,
            name: "SwiGLU FFN Sub-Block".to_string(),
            total_flops: flops,
            dram_bytes: total_bytes,
            arithmetic_intensity: intensity,
            execution_phase: ExecutionPhase::Prefill { seq_len: s },
            precision: weight_prec,
        }
    }

    pub fn calculate_swiglu(
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        intermediate_size: usize,
        weight_prec: PrecisionType,
        act_prec: PrecisionType,
    ) -> KernelIntensityReport {
        Self::swiglu_ffn(
            batch_size,
            seq_len,
            hidden_size,
            intermediate_size,
            weight_prec,
            act_prec,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn standard_attention(
        batch_size: usize,
        seq_len: usize,
        context_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        act_prec: PrecisionType,
    ) -> KernelIntensityReport {
        let b = batch_size.max(1) as u64;
        let s = seq_len.max(1) as u64;
        let c = context_len.max(1) as u64;
        let h = num_heads as u64;
        let h_kv = num_kv_heads as u64;
        let d = head_dim as u64;

        let flops = 4 * b * h * s * c * d + 3 * b * h * s * c;

        let q_bytes = b * s * (h * d);
        let k_bytes = b * c * (h_kv * d);
        let v_bytes = b * c * (h_kv * d);
        let o_bytes = b * s * (h * d);
        let score_matrix_roundtrip = 2 * b * h * s * c;
        let total_elements = q_bytes + k_bytes + v_bytes + o_bytes + score_matrix_roundtrip;
        let dram_bytes = (total_elements as f64 * act_prec.bytes_per_element()) as u64;
        let intensity = if dram_bytes == 0 {
            0.0
        } else {
            flops as f64 / dram_bytes as f64
        };

        KernelIntensityReport {
            kernel: KernelType::AttentionStandard,
            name: "Standard Scaled Dot-Product Attention".to_string(),
            total_flops: flops,
            dram_bytes,
            arithmetic_intensity: intensity,
            execution_phase: ExecutionPhase::Prefill {
                seq_len: s as usize,
            },
            precision: act_prec,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn calculate_attention(
        batch_size: usize,
        seq_len: usize,
        context_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        act_prec: PrecisionType,
    ) -> KernelIntensityReport {
        Self::standard_attention(
            batch_size,
            seq_len,
            context_len,
            num_heads,
            num_kv_heads,
            head_dim,
            act_prec,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_v2(
        batch_size: usize,
        seq_len: usize,
        context_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        act_prec: PrecisionType,
    ) -> KernelIntensityReport {
        let b = batch_size.max(1) as u64;
        let s = seq_len.max(1) as u64;
        let c = context_len.max(1) as u64;
        let h = num_heads as u64;
        let h_kv = num_kv_heads as u64;
        let d = head_dim as u64;

        let flops = 4 * b * h * s * c * d + 3 * b * h * s * c;

        let q_bytes = b * s * (h * d);
        let k_bytes = b * c * (h_kv * d);
        let v_bytes = b * c * (h_kv * d);
        let o_bytes = b * s * (h * d);
        let total_elements = q_bytes + k_bytes + v_bytes + o_bytes;
        let dram_bytes = (total_elements as f64 * act_prec.bytes_per_element()) as u64;
        let intensity = if dram_bytes == 0 {
            0.0
        } else {
            flops as f64 / dram_bytes as f64
        };

        KernelIntensityReport {
            kernel: KernelType::AttentionFlashV2,
            name: "FlashAttention-2 Kernel".to_string(),
            total_flops: flops,
            dram_bytes,
            arithmetic_intensity: intensity,
            execution_phase: ExecutionPhase::Prefill {
                seq_len: s as usize,
            },
            precision: act_prec,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn calculate_flash_attention_v2(
        batch_size: usize,
        seq_len: usize,
        context_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        act_prec: PrecisionType,
    ) -> KernelIntensityReport {
        Self::flash_attention_v2(
            batch_size,
            seq_len,
            context_len,
            num_heads,
            num_kv_heads,
            head_dim,
            act_prec,
        )
    }

    pub fn rms_norm(
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        act_prec: PrecisionType,
    ) -> KernelIntensityReport {
        let b = batch_size.max(1) as u64;
        let s = seq_len.max(1) as u64;
        let d = hidden_size as u64;
        let m = b * s;

        let flops = 3 * m * d;
        let act_bytes = (2 * m * d) as f64 * act_prec.bytes_per_element();
        let weight_bytes = d as f64 * act_prec.bytes_per_element();
        let total_bytes = (act_bytes + weight_bytes) as u64;
        let intensity = if total_bytes == 0 {
            0.0
        } else {
            flops as f64 / total_bytes as f64
        };

        KernelIntensityReport {
            kernel: KernelType::RmsNorm,
            name: "RMSNorm Kernel".to_string(),
            total_flops: flops,
            dram_bytes: total_bytes,
            arithmetic_intensity: intensity,
            execution_phase: ExecutionPhase::Prefill {
                seq_len: s as usize,
            },
            precision: act_prec,
        }
    }
}

pub type LayerArithmeticIntensity = IntensityCalculator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_intensity_arithmetic_formulas() {
        let linear = IntensityCalculator::linear_projection(
            "Q Projection",
            KernelType::LinearQProj,
            1,
            2048,
            4096,
            4096,
            PrecisionType::FP16,
            PrecisionType::FP16,
        );

        assert_eq!(linear.total_flops, 2 * 2048 * 4096 * 4096);
        let expected_weight_bytes = 4096 * 4096 * 2;
        let expected_act_bytes = 2048 * (4096 + 4096) * 2;
        assert_eq!(
            linear.dram_bytes,
            expected_weight_bytes + expected_act_bytes
        );
        assert!(linear.arithmetic_intensity > 0.0);

        let swiglu = IntensityCalculator::swiglu_ffn(
            1,
            2048,
            4096,
            11008,
            PrecisionType::FP16,
            PrecisionType::FP16,
        );
        let expected_swiglu_flops = 6 * 2048 * 4096 * 11008 + 3 * 2048 * 11008;
        assert_eq!(swiglu.total_flops, expected_swiglu_flops);
        assert!(swiglu.arithmetic_intensity > 100.0);
    }

    #[test]
    fn test_flash_attention_vs_standard_attention() {
        let std_attn = IntensityCalculator::standard_attention(
            1,
            2048,
            2048,
            32,
            32,
            128,
            PrecisionType::FP16,
        );
        let flash_attn = IntensityCalculator::flash_attention_v2(
            1,
            2048,
            2048,
            32,
            32,
            128,
            PrecisionType::FP16,
        );

        assert_eq!(std_attn.total_flops, flash_attn.total_flops);
        assert!(flash_attn.dram_bytes < std_attn.dram_bytes);
        assert!(flash_attn.arithmetic_intensity > std_attn.arithmetic_intensity);
    }
}
