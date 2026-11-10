use std::collections::HashMap;

use super::precision::{AllocationPlan, LayerAllocation};
use crate::ir::{ModelIR, TensorFormat, TensorRole};

pub struct ParetoAllocator;

impl ParetoAllocator {
    /// Generate a heterogeneous precision plan constrained by either target VRAM (GB)
    /// or maximum tolerated degradation.
    pub fn plan_allocation(
        model: &ModelIR,
        target_vram_gb: Option<f64>,
        max_degradation_pct: f64,
    ) -> AllocationPlan {
        let mut tensor_allocations = HashMap::new();
        let original_bytes = model.original_byte_size;
        let original_gb = original_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        // Compute desired compression ratio from target_vram_gb if provided
        let target_ratio = if let Some(target_gb) = target_vram_gb {
            if target_gb > 0.0 {
                (original_gb / target_gb).max(1.0)
            } else {
                4.0
            }
        } else {
            // Default compression ratio derived from degradation tolerance
            1.0 + (max_degradation_pct / 2.0).clamp(0.5, 4.0)
        };

        // Sensitivity thresholds adjust according to aggressiveness
        let (high_sens_thresh, med_sens_thresh) = if target_ratio > 3.0 {
            // Aggressive target: push more into Q4 and Ternary
            (0.75, 0.45)
        } else if target_ratio > 2.0 {
            // Balanced
            (0.60, 0.35)
        } else {
            // Conservative
            (0.45, 0.25)
        };

        let mut planned_bytes: usize = 0;
        let mut total_quality_loss: f64 = 0.0;

        for tensor in &model.tensors {
            let format = match tensor.role {
                // Safeguard critical normalization layers in high precision
                TensorRole::FinalNorm | TensorRole::LayerNorm => TensorFormat::FP16,

                // Embeddings and Output Heads: preserve at least INT8 unless ultra-compressed
                TensorRole::Embedding | TensorRole::OutputHead => {
                    if target_ratio > 3.5 {
                        TensorFormat::Q4_0
                    } else {
                        TensorFormat::INT8
                    }
                }

                // Transformer layers: allocate based on manifold sensitivity score
                _ => {
                    if tensor.sensitivity_score >= high_sens_thresh {
                        TensorFormat::INT8
                    } else if tensor.sensitivity_score >= med_sens_thresh {
                        TensorFormat::Q4_0
                    } else {
                        TensorFormat::Ternary
                    }
                }
            };

            let bits = format.bits_per_weight();
            let estimated_tensor_bytes = ((tensor.numel as f64 * bits) / 8.0).ceil() as usize;
            planned_bytes += estimated_tensor_bytes;

            // Estimated quality impact proportional to sensitivity * compression factor
            let loss_factor = (16.0 - bits) / 16.0;
            total_quality_loss += tensor.sensitivity_score * loss_factor;

            tensor_allocations.insert(tensor.name.clone(), format);
        }

        let planned_gb = planned_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let actual_compression_ratio = if planned_bytes == 0 {
            1.0
        } else {
            original_bytes as f64 / planned_bytes as f64
        };

        // Normalize estimated retention percentage
        let avg_loss = if !model.tensors.is_empty() {
            total_quality_loss / model.tensors.len() as f64
        } else {
            0.0
        };
        let estimated_retention = (100.0 - (avg_loss * 12.0)).clamp(60.0, 99.9);

        // Summarize by layer
        let mut layer_allocations = Vec::new();
        for (layer_idx, tensors) in model.tensors_by_layer() {
            if let Some(l) = layer_idx {
                let orig_mb =
                    tensors.iter().map(|t| t.byte_size).sum::<usize>() as f64 / (1024.0 * 1024.0);
                let opt_mb = tensors
                    .iter()
                    .map(|t| {
                        let fmt = tensor_allocations
                            .get(&t.name)
                            .unwrap_or(&TensorFormat::FP16);
                        ((t.numel as f64 * fmt.bits_per_weight()) / 8.0).ceil() as usize
                    })
                    .sum::<usize>() as f64
                    / (1024.0 * 1024.0);

                // Majority format in this layer weighted by parameter volume
                let mut fmt_volume = std::collections::HashMap::new();
                for t in tensors {
                    if let Some(f) = tensor_allocations.get(&t.name) {
                        *fmt_volume.entry(f.clone()).or_insert(0u64) += t.numel as u64;
                    }
                }
                let rep_fmt = fmt_volume
                    .into_iter()
                    .max_by_key(|(_, vol)| *vol)
                    .map(|(f, _)| f)
                    .unwrap_or(TensorFormat::FP16);

                layer_allocations.push(LayerAllocation {
                    layer_index: l,
                    assigned_format: rep_fmt,
                    original_mb: orig_mb,
                    optimized_mb: opt_mb,
                });
            }
        }
        layer_allocations.sort_by_key(|l| l.layer_index);

        AllocationPlan {
            model_name: model.name.clone(),
            original_size_gb: original_gb,
            planned_size_gb: planned_gb,
            compression_ratio: actual_compression_ratio,
            target_vram_gb,
            estimated_quality_retention_pct: estimated_retention,
            layer_allocations,
            tensor_allocations,
        }
    }
}
