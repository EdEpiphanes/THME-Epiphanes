use crate::ir::model::ModelIR;
use crate::profiling::hardware::{HardwareDevice, PrecisionType};
use crate::profiling::intensity::{IntensityCalculator, KernelType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundClassification {
    MemoryBound {
        memory_bound_factor: f64,
        headroom_to_knee: f64,
    },
    ComputeBound {
        compute_saturation_ratio: f64,
    },
}

impl PartialEq for BoundClassification {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::MemoryBound {
                    memory_bound_factor: f1,
                    headroom_to_knee: h1,
                },
                Self::MemoryBound {
                    memory_bound_factor: f2,
                    headroom_to_knee: h2,
                },
            ) => (f1 - f2).abs() < 1e-4 && (h1 - h2).abs() < 1e-4,
            (
                Self::ComputeBound {
                    compute_saturation_ratio: r1,
                },
                Self::ComputeBound {
                    compute_saturation_ratio: r2,
                },
            ) => (r1 - r2).abs() < 1e-4,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRooflinePoint {
    pub name: String,
    pub arithmetic_intensity: f64,
    pub attainable_tflops: f64,
    pub classification: BoundClassification,
    pub efficiency_pct: f64,
    pub recommended_optimizations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineReport {
    pub hardware_name: String,
    pub precision: PrecisionType,
    pub peak_compute_tflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub knee_intensity_flop_per_byte: f64,
    pub points: Vec<LayerRooflinePoint>,
}

impl RooflineReport {
    pub fn format_ascii_report(
        &self,
        _hw: &HardwareDevice,
        effective_peak: f64,
        knee: f64,
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "================ ROOFLINE CEILING MODEL [{}] ================\n",
            self.hardware_name
        ));
        out.push_str(&format!(
            "Peak Compute: {:.1} TFLOPS ({}) | Bandwidth: {:.1} GB/s | Knee: {:.2} FLOP/Byte\n\n",
            effective_peak,
            self.precision.display_name(),
            self.memory_bandwidth_gbps,
            knee
        ));
        out.push_str("   TFLOPS\n");
        out.push_str(&format!(
            "{:7.1} |-------------------------------+-----------------------------+ (Peak Compute Ceiling)\n",
            effective_peak
        ));
        out.push_str("        |                              /|\n");
        out.push_str("        |                             / |\n");
        out.push_str("        |                            /  |\n");
        out.push_str("        |                           /   |\n");
        out.push_str("        |       MEMORY BOUND       /    |        COMPUTE BOUND\n");
        out.push_str("        |                         /     |\n");
        out.push_str("        |                        /      |\n");
        out.push_str("        |                       /       |\n");
        out.push_str("        |                      /        |\n");
        out.push_str("        |                     /         |\n");
        out.push_str("        |                    /          |\n");
        out.push_str("        |                   /           |\n");
        out.push_str("        |                  /            |\n");
        out.push_str("        |                 /             |\n");
        out.push_str("        |                /              |\n");
        out.push_str("        |               /               |\n");
        out.push_str("    0.1 +--------------+----------------+-----------------------------+---> Arithmetic Intensity (FLOP/Byte)\n");
        out.push_str(&format!(
            "       0.1            1.0             {:.1} (Knee)                   1000.0\n\n",
            knee
        ));

        out.push_str("OPERATIONAL KERNEL MAPPINGS:\n");
        for (i, p) in self.points.iter().enumerate() {
            let class_str = match p.classification {
                BoundClassification::MemoryBound { .. } => "MEMORY-BOUND",
                BoundClassification::ComputeBound { .. } => "COMPUTE-BOUND",
            };
            out.push_str(&format!(
                " [{}] {:<28} | Int: {:7.2} FLOP/B | Attain: {:6.1} TFLOPS | {}\n",
                i + 1,
                p.name,
                p.arithmetic_intensity,
                p.attainable_tflops,
                class_str
            ));
        }
        out
    }
}

pub type RooflineProfileReport = RooflineReport;

pub struct RooflineEngine {
    device: HardwareDevice,
    precision: PrecisionType,
}

impl RooflineEngine {
    pub fn new(device: HardwareDevice, precision: PrecisionType) -> Self {
        Self { device, precision }
    }

    pub fn evaluate_point(&self, name: &str, intensity: f64) -> LayerRooflinePoint {
        self.evaluate_point_throttled(name, intensity, 1.0)
    }

    pub fn evaluate_point_throttled(
        &self,
        name: &str,
        intensity: f64,
        throttle_factor: f64,
    ) -> LayerRooflinePoint {
        let knee = self.device.machine_balance(self.precision);
        let peak =
            self.device.peak_throughput_tflops(self.precision) * throttle_factor.clamp(0.1, 1.0);
        let bw = self.device.dram_bandwidth_gbps;

        let (attainable, class, eff, recs) = if intensity < knee {
            let attainable = ((intensity * bw) / 1000.0).min(peak);
            let factor = if knee > 0.0 { intensity / knee } else { 0.0 };
            let mut recs = vec![
                "Quantize weights to BitNet b1.58 or Q4_K to reduce memory traffic.".to_string(),
                "Apply kernel fusion (SwiGLU, LayerNorm) to keep intermediates in SRAM."
                    .to_string(),
            ];
            if name.contains("Attention") {
                recs.push("Substitute Standard Attention with FlashAttention-2.".to_string());
            }
            (
                attainable,
                BoundClassification::MemoryBound {
                    memory_bound_factor: factor,
                    headroom_to_knee: (knee - intensity).max(0.0),
                },
                factor * 100.0,
                recs,
            )
        } else {
            let recs = vec![
                "Apply 2:4 structured sparsity to double effective tensor compute.".to_string(),
                "Utilize low-precision Tensor Cores (INT8/FP8) for compute speedup.".to_string(),
            ];
            (
                peak,
                BoundClassification::ComputeBound {
                    compute_saturation_ratio: 1.0,
                },
                100.0,
                recs,
            )
        };

        LayerRooflinePoint {
            name: name.to_string(),
            arithmetic_intensity: intensity,
            attainable_tflops: attainable,
            classification: class,
            efficiency_pct: eff,
            recommended_optimizations: recs,
        }
    }

    pub fn generate_ascii_diagram(&self, points: &[LayerRooflinePoint]) -> String {
        let knee = self.device.machine_balance(self.precision);
        let peak = self.device.peak_throughput_tflops(self.precision);
        let report = RooflineReport {
            hardware_name: self.device.name.clone(),
            precision: self.precision,
            peak_compute_tflops: peak,
            memory_bandwidth_gbps: self.device.dram_bandwidth_gbps,
            knee_intensity_flop_per_byte: knee,
            points: points.to_vec(),
        };
        report.format_ascii_report(&self.device, peak, knee)
    }

    pub fn profile_model(
        model: &ModelIR,
        hw: &HardwareDevice,
        batch_size: usize,
        seq_len: usize,
        flash_attn: bool,
        thermal_factor: f32,
    ) -> anyhow::Result<RooflineReport> {
        let precision = PrecisionType::FP16;
        let engine = Self::new(hw.clone(), precision);

        let hidden = model.hidden_size.max(512);
        let heads = model.num_attention_heads.max(1);
        let kv_heads = model.num_key_value_heads.max(1);
        let head_dim = if model.head_dim > 0 {
            model.head_dim
        } else {
            hidden / heads
        };
        let intermediate = (8 * hidden) / 3;

        let q_proj = IntensityCalculator::linear_projection(
            "Q Projection",
            KernelType::LinearQProj,
            batch_size,
            seq_len,
            hidden,
            heads * head_dim,
            precision,
            precision,
        );
        let k_proj = IntensityCalculator::linear_projection(
            "K Projection",
            KernelType::LinearKProj,
            batch_size,
            seq_len,
            hidden,
            kv_heads * head_dim,
            precision,
            precision,
        );
        let v_proj = IntensityCalculator::linear_projection(
            "V Projection",
            KernelType::LinearVProj,
            batch_size,
            seq_len,
            hidden,
            kv_heads * head_dim,
            precision,
            precision,
        );
        let o_proj = IntensityCalculator::linear_projection(
            "Output Projection",
            KernelType::LinearOutProj,
            batch_size,
            seq_len,
            heads * head_dim,
            hidden,
            precision,
            precision,
        );

        let attn = if flash_attn {
            IntensityCalculator::flash_attention_v2(
                batch_size, seq_len, seq_len, heads, kv_heads, head_dim, precision,
            )
        } else {
            IntensityCalculator::standard_attention(
                batch_size, seq_len, seq_len, heads, kv_heads, head_dim, precision,
            )
        };

        let ffn = IntensityCalculator::swiglu_ffn(
            batch_size,
            seq_len,
            hidden,
            intermediate,
            precision,
            precision,
        );
        let norm = IntensityCalculator::rms_norm(batch_size, seq_len, hidden, precision);

        let throttle = thermal_factor as f64;
        let points = vec![
            engine.evaluate_point_throttled(&q_proj.name, q_proj.arithmetic_intensity, throttle),
            engine.evaluate_point_throttled(&k_proj.name, k_proj.arithmetic_intensity, throttle),
            engine.evaluate_point_throttled(&v_proj.name, v_proj.arithmetic_intensity, throttle),
            engine.evaluate_point_throttled(&o_proj.name, o_proj.arithmetic_intensity, throttle),
            engine.evaluate_point_throttled(&attn.name, attn.arithmetic_intensity, throttle),
            engine.evaluate_point_throttled(&ffn.name, ffn.arithmetic_intensity, throttle),
            engine.evaluate_point_throttled(&norm.name, norm.arithmetic_intensity, throttle),
        ];

        let knee = hw.machine_balance(precision);
        let peak = hw.peak_throughput_tflops(precision) * throttle;

        Ok(RooflineReport {
            hardware_name: hw.name.clone(),
            precision,
            peak_compute_tflops: peak,
            memory_bandwidth_gbps: hw.dram_bandwidth_gbps,
            knee_intensity_flop_per_byte: knee,
            points,
        })
    }
}

pub type RooflineModel = RooflineEngine;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roofline_classification_memory_vs_compute() {
        let h100 = HardwareDevice::predefined_h100();
        let engine = RooflineEngine::new(h100, PrecisionType::FP16);
        let knee = engine.device.machine_balance(PrecisionType::FP16);

        let mem_point = engine.evaluate_point("Decode Projection", 1.5);
        assert!(mem_point.arithmetic_intensity < knee);
        match mem_point.classification {
            BoundClassification::MemoryBound {
                memory_bound_factor,
                headroom_to_knee,
            } => {
                assert!(memory_bound_factor < 1.0);
                assert!(headroom_to_knee > 0.0);
            }
            BoundClassification::ComputeBound { .. } => {
                panic!("Expected MemoryBound classification for low intensity");
            }
        }
        assert!(mem_point.efficiency_pct < 100.0);

        let compute_point = engine.evaluate_point("Prefill Dense GEMM", 1000.0);
        assert!(compute_point.arithmetic_intensity >= knee);
        match compute_point.classification {
            BoundClassification::ComputeBound {
                compute_saturation_ratio,
            } => {
                assert_eq!(compute_saturation_ratio, 1.0);
            }
            BoundClassification::MemoryBound { .. } => {
                panic!("Expected ComputeBound classification for high intensity");
            }
        }
        assert_eq!(compute_point.efficiency_pct, 100.0);
    }

    #[test]
    fn test_ascii_roofline_diagram_generation() {
        let h100 = HardwareDevice::predefined_h100();
        let engine = RooflineEngine::new(h100, PrecisionType::FP16);
        let p1 = engine.evaluate_point("RMSNorm", 0.5);
        let p2 = engine.evaluate_point("FlashAttention-2", 1500.0);

        let diagram = engine.generate_ascii_diagram(&[p1, p2]);
        assert!(diagram.contains("ROOFLINE CEILING MODEL"));
        assert!(diagram.contains("Knee:"));
        assert!(diagram.contains("RMSNorm"));
        assert!(diagram.contains("FlashAttention-2"));
        assert!(diagram.contains("MEMORY-BOUND"));
        assert!(diagram.contains("COMPUTE-BOUND"));
    }
}
