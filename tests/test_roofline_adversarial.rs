//! Adversarial Stress & Boundary Challenge Test Suite for THME Roofline Engine (v0.1.7.0)
//!
//! EMPIRICAL CHALLENGER 1 ADVERSARIAL HARNESS:
//! 1. Mathematical formulations stress under boundary conditions (zero batch, extreme context S > 32768, zero dimensions, extreme temperatures)
//! 2. Cache simulator hierarchy scaling from 1 KB up to 1 TB
//! 3. Roofline engine machine balance knee points and asymptotic extremes
//! 4. CLI input resilience and error propagation

use thme::ir::{ModelArchitecture, ModelIR, TensorIR};
use thme::profiling::cache_sim::{CacheSimulator4Tier, CacheTierLevel};
use thme::profiling::hardware::{HardwareDevice, HardwareTarget, PrecisionType};
use thme::profiling::intensity::{IntensityCalculator, KernelType};
use thme::profiling::roofline::{BoundClassification, RooflineEngine};
use thme::profiling::thermal::{DynamicThermalModel, ThermalConfig};

// Helper: minimal synthetic transformer model for boundary profiling
fn make_minimal_model(hidden: usize, heads: usize, kv_heads: usize, head_dim: usize) -> ModelIR {
    let tensors = vec![
        TensorIR::new(
            "model.embed_tokens.weight".to_string(),
            vec![1000, hidden],
            "F16".to_string(),
            1000 * hidden * 2,
        ),
        TensorIR::new(
            "lm_head.weight".to_string(),
            vec![1000, hidden],
            "F16".to_string(),
            1000 * hidden * 2,
        ),
    ];
    ModelIR::new(
        "adversarial-test-model".to_string(),
        ModelArchitecture::Llama,
        1,
        hidden,
        heads,
        kv_heads,
        1000,
        head_dim,
        tensors,
    )
}

#[test]
fn test_adversarial_arithmetic_intensity_zero_batch_and_dims() {
    // 1. Zero batch size: calculator must clamp to 1 internally and not produce NaN or panic
    let lin_zero_b = IntensityCalculator::linear_projection(
        "Zero Batch Linear",
        KernelType::LinearQProj,
        0, // B = 0
        2048,
        4096,
        4096,
        PrecisionType::FP16,
        PrecisionType::FP16,
    );
    assert!(!lin_zero_b.arithmetic_intensity.is_nan());
    assert!(!lin_zero_b.arithmetic_intensity.is_infinite());
    assert!(lin_zero_b.arithmetic_intensity > 0.0);
    assert!(lin_zero_b.total_flops > 0);
    assert!(lin_zero_b.dram_bytes > 0);

    // 2. Zero sequence length: calculator must clamp to 1
    let lin_zero_s = IntensityCalculator::linear_projection(
        "Zero Seq Linear",
        KernelType::LinearQProj,
        1,
        0, // S = 0
        4096,
        4096,
        PrecisionType::FP16,
        PrecisionType::FP16,
    );
    assert!(!lin_zero_s.arithmetic_intensity.is_nan());
    assert!(lin_zero_s.arithmetic_intensity > 0.0);

    // 3. Zero dimensions: dim_in = 0, dim_out = 0 -> FLOPs = 0, bytes = 0 -> intensity must be 0.0, NOT NaN!
    let lin_zero_dim = IntensityCalculator::linear_projection(
        "Zero Dim Linear",
        KernelType::LinearQProj,
        1,
        128,
        0, // dim_in = 0
        0, // dim_out = 0
        PrecisionType::FP16,
        PrecisionType::FP16,
    );
    assert_eq!(lin_zero_dim.total_flops, 0);
    assert_eq!(lin_zero_dim.dram_bytes, 0);
    assert_eq!(lin_zero_dim.arithmetic_intensity, 0.0);
    assert!(!lin_zero_dim.arithmetic_intensity.is_nan());

    // 4. Zero dimensions in SwiGLU FFN
    let swiglu_zero = IntensityCalculator::swiglu_ffn(
        0, // B = 0
        0, // S = 0
        0, // hidden = 0
        0, // intermediate = 0
        PrecisionType::FP16,
        PrecisionType::FP16,
    );
    assert_eq!(swiglu_zero.total_flops, 0);
    assert_eq!(swiglu_zero.dram_bytes, 0);
    assert_eq!(swiglu_zero.arithmetic_intensity, 0.0);
    assert!(!swiglu_zero.arithmetic_intensity.is_nan());

    // 5. Zero dimensions in Standard Attention
    let attn_zero = IntensityCalculator::standard_attention(
        0,
        0,
        0,
        0, // num_heads = 0
        0, // num_kv_heads = 0
        0, // head_dim = 0
        PrecisionType::FP16,
    );
    assert_eq!(attn_zero.total_flops, 0);
    assert_eq!(attn_zero.dram_bytes, 0);
    assert_eq!(attn_zero.arithmetic_intensity, 0.0);
    assert!(!attn_zero.arithmetic_intensity.is_nan());

    // 6. Zero dimensions in FlashAttention-2
    let flash_zero = IntensityCalculator::flash_attention_v2(0, 0, 0, 0, 0, 0, PrecisionType::FP16);
    assert_eq!(flash_zero.total_flops, 0);
    assert_eq!(flash_zero.dram_bytes, 0);
    assert_eq!(flash_zero.arithmetic_intensity, 0.0);
    assert!(!flash_zero.arithmetic_intensity.is_nan());

    // 7. Zero dimensions in RMSNorm
    let norm_zero = IntensityCalculator::rms_norm(
        0,
        0,
        0, // hidden = 0
        PrecisionType::FP16,
    );
    assert_eq!(norm_zero.total_flops, 0);
    assert_eq!(norm_zero.dram_bytes, 0);
    assert_eq!(norm_zero.arithmetic_intensity, 0.0);
    assert!(!norm_zero.arithmetic_intensity.is_nan());
}

#[test]
fn test_adversarial_arithmetic_intensity_extreme_contexts() {
    // Extreme contexts: S = 32768, 65536, 131072, 262144
    let contexts = [32768, 65536, 131072, 262144];
    let heads = 32;
    let kv_heads = 8;
    let head_dim = 128;

    for &ctx in &contexts {
        let std_attn = IntensityCalculator::standard_attention(
            1,
            ctx,
            ctx,
            heads,
            kv_heads,
            head_dim,
            PrecisionType::FP16,
        );
        let flash_attn = IntensityCalculator::flash_attention_v2(
            1,
            ctx,
            ctx,
            heads,
            kv_heads,
            head_dim,
            PrecisionType::FP16,
        );

        // Verification 1: Both algorithms compute identical FLOPs
        assert_eq!(std_attn.total_flops, flash_attn.total_flops);
        assert!(std_attn.total_flops > 0);

        // Verification 2: Neither produces NaN or infinity
        assert!(!std_attn.arithmetic_intensity.is_nan());
        assert!(!std_attn.arithmetic_intensity.is_infinite());
        assert!(!flash_attn.arithmetic_intensity.is_nan());
        assert!(!flash_attn.arithmetic_intensity.is_infinite());

        // Verification 3: FlashAttention-2 traffic is strictly smaller than Standard Attention
        assert!(flash_attn.dram_bytes < std_attn.dram_bytes);

        // Verification 4: FlashAttention-2 arithmetic intensity scaling advantage expands with S
        let advantage_ratio = flash_attn.arithmetic_intensity / std_attn.arithmetic_intensity;
        assert!(
            advantage_ratio > 10.0,
            "Expected >10x advantage at S={ctx}, got {advantage_ratio:.2}x"
        );
    }
}

#[test]
fn test_adversarial_cache_simulator_hierarchy_1kb_to_1tb() {
    let device = HardwareDevice::predefined_h100();
    let sim = CacheSimulator4Tier::new(device.clone());

    // Scale from 1 KB up to 1 TB
    let working_sets: Vec<(usize, &str)> = vec![
        (1_024, "1 KB"),
        (65_536, "64 KB"),
        (1_000_000, "1 MB"),
        (10_000_000, "10 MB"),
        (33_800_000, "33.8 MB (L1 boundary)"),
        (40_000_000, "40 MB (Spill past L1)"),
        (50_000_000, "50 MB (L2 boundary)"),
        (100_000_000, "100 MB (Spill into DRAM)"),
        (1_000_000_000, "1 GB"),
        (10_000_000_000, "10 GB"),
        (80_000_000_000, "80 GB (DRAM capacity boundary)"),
        (500_000_000_000, "500 GB"),
        (1_000_000_000_000, "1 TB"),
        (10_000_000_000_000, "10 TB"),
        (100_000_000_000_000, "100 TB"),
    ];

    let mut prev_latency = 0.0;
    let mut prev_spill = 0;

    for (ws_bytes, label) in working_sets {
        let summary = sim.simulate_working_set(ws_bytes);

        // Invariant 1: Total working set bytes matches input
        assert_eq!(
            summary.total_working_set_bytes, ws_bytes,
            "Mismatch at {label}"
        );

        // Invariant 2: Latency and bandwidth are positive and finite
        assert!(
            !summary.effective_latency_ns.is_nan(),
            "NaN latency at {label}"
        );
        assert!(
            !summary.effective_bandwidth_gbps.is_nan(),
            "NaN bandwidth at {label}"
        );
        assert!(
            summary.effective_latency_ns > 0.0,
            "Non-positive latency at {label}"
        );
        assert!(
            summary.effective_bandwidth_gbps > 0.0,
            "Non-positive bandwidth at {label}"
        );

        // Invariant 3: Effective latency is monotonically non-decreasing
        assert!(
            summary.effective_latency_ns >= prev_latency,
            "Latency decreased from {prev_latency} to {} at {label}",
            summary.effective_latency_ns
        );
        prev_latency = summary.effective_latency_ns;

        // Invariant 4: DRAM spill is monotonically non-decreasing
        assert!(
            summary.dram_spill_bytes >= prev_spill,
            "Spill decreased from {prev_spill} to {} at {label}",
            summary.dram_spill_bytes
        );
        prev_spill = summary.dram_spill_bytes;

        // Invariant 5: Effective bandwidth bounded between DRAM bus and L1 cache
        assert!(summary.effective_bandwidth_gbps <= 22000.0);
        assert!(summary.effective_bandwidth_gbps >= device.dram_bandwidth_gbps * 0.9);

        // Invariant 6: Tier hit and miss ratios sum to 1.0 within epsilon
        for tier in &summary.tiers {
            assert!((tier.hit_ratio + tier.miss_ratio - 1.0).abs() < 1e-4);
            assert!(tier.hit_ratio >= 0.0 && tier.hit_ratio <= 1.0);
            assert!(tier.miss_ratio >= 0.0 && tier.miss_ratio <= 1.0);
        }
    }

    // Zero working set boundary
    let zero_summary = sim.simulate_working_set(0);
    assert_eq!(zero_summary.total_working_set_bytes, 0);
    assert_eq!(zero_summary.dram_spill_bytes, 0);
    assert!(!zero_summary.effective_latency_ns.is_nan());
    assert!(!zero_summary.effective_bandwidth_gbps.is_nan());
}

#[test]
fn test_adversarial_cache_simulator_all_devices_including_4_tier_apple() {
    let targets = [
        HardwareTarget::H100,
        HardwareTarget::RTX4090,
        HardwareTarget::AppleM3Max,
        HardwareTarget::MI300X,
        HardwareTarget::IntelEmeraldRapids,
    ];

    for target in targets {
        let dev = HardwareDevice::from_target(target);
        let sim = CacheSimulator4Tier::new(dev.clone());

        let summary = sim.simulate_working_set(100_000_000); // 100 MB
        assert!(!summary.effective_latency_ns.is_nan());
        assert!(!summary.effective_bandwidth_gbps.is_nan());
        assert!(summary.dram_spill_bytes > 0);

        if target == HardwareTarget::AppleM3Max {
            // Apple M3 has L1, L2, L3 (SLC), and DRAM (4 distinct tiers)
            assert_eq!(summary.tiers.len(), 4);
            assert_eq!(summary.tiers[0].level, CacheTierLevel::L1);
            assert_eq!(summary.tiers[1].level, CacheTierLevel::L2);
            assert_eq!(summary.tiers[2].level, CacheTierLevel::L3);
            assert_eq!(summary.tiers[3].level, CacheTierLevel::DRAM);
        }
    }
}

#[test]
fn test_adversarial_roofline_exact_knee_and_asymptotic_extremes() {
    let dev = HardwareDevice::predefined_h100();
    let engine = RooflineEngine::new(dev.clone(), PrecisionType::FP16);
    let knee = dev.machine_balance(PrecisionType::FP16);
    let peak = dev.peak_throughput_tflops(PrecisionType::FP16);

    // 1. Exactly below knee by 1e-9: Must be MemoryBound
    let below_knee = engine.evaluate_point("Just Below Knee", knee - 1e-9);
    match below_knee.classification {
        BoundClassification::MemoryBound {
            memory_bound_factor,
            headroom_to_knee,
        } => {
            assert!(memory_bound_factor <= 1.0);
            assert!(headroom_to_knee > 0.0);
        }
        BoundClassification::ComputeBound { .. } => {
            panic!("Expected MemoryBound just below knee");
        }
    }
    assert!(below_knee.attainable_tflops <= peak);

    // 2. Exactly at knee: Must be ComputeBound
    let at_knee = engine.evaluate_point("Exactly At Knee", knee);
    match at_knee.classification {
        BoundClassification::ComputeBound {
            compute_saturation_ratio,
        } => {
            assert_eq!(compute_saturation_ratio, 1.0);
        }
        BoundClassification::MemoryBound { .. } => {
            panic!("Expected ComputeBound exactly at knee");
        }
    }
    assert!((at_knee.attainable_tflops - peak).abs() < 1e-4);
    assert_eq!(at_knee.efficiency_pct, 100.0);

    // 3. Exactly above knee by 1e-9: Must be ComputeBound
    let above_knee = engine.evaluate_point("Just Above Knee", knee + 1e-9);
    match above_knee.classification {
        BoundClassification::ComputeBound {
            compute_saturation_ratio,
        } => {
            assert_eq!(compute_saturation_ratio, 1.0);
        }
        BoundClassification::MemoryBound { .. } => {
            panic!("Expected ComputeBound just above knee");
        }
    }
    assert!((above_knee.attainable_tflops - peak).abs() < 1e-4);

    // 4. Zero intensity boundary
    let zero_point = engine.evaluate_point("Zero Intensity", 0.0);
    assert_eq!(zero_point.arithmetic_intensity, 0.0);
    assert_eq!(zero_point.attainable_tflops, 0.0);
    assert_eq!(zero_point.efficiency_pct, 0.0);
    match zero_point.classification {
        BoundClassification::MemoryBound {
            memory_bound_factor,
            headroom_to_knee,
        } => {
            assert_eq!(memory_bound_factor, 0.0);
            assert!((headroom_to_knee - knee).abs() < 1e-4);
        }
        _ => panic!("Expected MemoryBound for zero intensity"),
    }

    // 5. Huge asymptotic intensity: 1e9 FLOP/Byte
    let huge_point = engine.evaluate_point("Huge Compute", 1e9);
    assert_eq!(huge_point.attainable_tflops, peak);
    assert_eq!(huge_point.efficiency_pct, 100.0);
    match huge_point.classification {
        BoundClassification::ComputeBound {
            compute_saturation_ratio,
        } => {
            assert_eq!(compute_saturation_ratio, 1.0);
        }
        _ => panic!("Expected ComputeBound for huge intensity"),
    }

    // 6. Inf intensity
    let inf_point = engine.evaluate_point("Inf Compute", f64::INFINITY);
    assert_eq!(inf_point.attainable_tflops, peak);
    assert_eq!(inf_point.efficiency_pct, 100.0);

    // 7. Negative intensity adversarial input (must not panic or return NaN)
    let neg_point = engine.evaluate_point("Negative Intensity", -1.0);
    assert!(!neg_point.arithmetic_intensity.is_nan());
    assert!(!neg_point.attainable_tflops.is_nan());
    match neg_point.classification {
        BoundClassification::MemoryBound {
            memory_bound_factor,
            headroom_to_knee,
        } => {
            assert!(memory_bound_factor <= 0.0);
            assert!(headroom_to_knee > knee);
        }
        _ => panic!("Expected MemoryBound for negative intensity"),
    }
}

#[test]
fn test_adversarial_thermal_model_extreme_temperatures() {
    let dev = HardwareDevice::predefined_h100();
    // 1. Absolute zero / deep space cryogenic temperature (-273.15 C)
    let cryo_factor = DynamicThermalModel::calculate_throttling_factor(-273.15, dev.tdp_watts);
    assert_eq!(
        cryo_factor, 1.0,
        "Cryogenic temp should have no throttling (factor = 1.0)"
    );

    // 2. Exact throttle knee (83.0 C)
    let knee_factor = DynamicThermalModel::calculate_throttling_factor(83.0, dev.tdp_watts);
    assert_eq!(
        knee_factor, 1.0,
        "Exact 83C throttle knee should be factor 1.0"
    );

    // 3. High temperature throttle (90.0 C) -> 1.0 - 0.05 * 7 = 0.65
    let hot_factor = DynamicThermalModel::calculate_throttling_factor(90.0, dev.tdp_watts);
    assert!((hot_factor - 0.65).abs() < 1e-4);

    // 4. Extreme thermal runaway (150 C, 500 C, 1000 C) -> clamped to 0.40 minimum floor
    let runaway_150 = DynamicThermalModel::calculate_throttling_factor(150.0, dev.tdp_watts);
    assert_eq!(runaway_150, 0.40, "Extreme runaway must clamp to 0.40");

    let runaway_1000 = DynamicThermalModel::calculate_throttling_factor(1000.0, dev.tdp_watts);
    assert_eq!(runaway_1000, 0.40, "1000C extreme must clamp to 0.40");

    // 5. Dynamic simulation step stability under extreme ambient temperatures
    let extreme_hot_config = ThermalConfig {
        ambient_temp_c: 150.0,
        ..Default::default()
    };
    let mut hot_model = DynamicThermalModel::new(dev.clone(), extreme_hot_config);
    assert_eq!(hot_model.current_temperature(), 150.0);

    let step = hot_model.step_simulation(1.0, 1.0, 1.0);
    assert!(step.is_throttling);
    assert_eq!(step.throttle_factor, 0.40);
    assert!(!step.die_temperature_c.is_nan());
    assert!(!step.effective_tflops.is_nan());
    assert_eq!(step.effective_tflops, dev.peak_fp16_tflops * 0.40);

    // 6. Sub-zero ambient temperature (-50 C)
    let subzero_config = ThermalConfig {
        ambient_temp_c: -50.0,
        ..Default::default()
    };
    let mut cold_model = DynamicThermalModel::new(dev.clone(), subzero_config);
    let cold_step = cold_model.step_simulation(1.0, 1.0, 1.0);
    assert!(!cold_step.is_throttling);
    assert_eq!(cold_step.throttle_factor, 1.0);
    assert!(!cold_step.die_temperature_c.is_nan());

    // 7. Long-horizon convergence stress test (100,000 seconds simulation)
    let mut long_model = DynamicThermalModel::new(dev, ThermalConfig::default());
    for _ in 0..1000 {
        let step = long_model.step_simulation(10.0, 1.0, 1.0);
        assert!(!step.die_temperature_c.is_nan());
        assert!(!step.die_temperature_c.is_infinite());
        assert!(step.die_temperature_c > 25.0 && step.die_temperature_c < 200.0);
    }
}

#[test]
fn test_adversarial_profile_model_edge_parameters() {
    let h100 = HardwareDevice::predefined_h100();
    let model = make_minimal_model(4096, 32, 8, 128);

    // Batch size 0, seq_len 0: Infallible clamping must prevent panics and NaNs
    let report_zero = RooflineEngine::profile_model(&model, &h100, 0, 0, true, 1.0);
    assert!(report_zero.is_ok());
    let r0 = report_zero.unwrap();
    assert_eq!(r0.points.len(), 7);
    for pt in &r0.points {
        assert!(!pt.arithmetic_intensity.is_nan());
        assert!(!pt.attainable_tflops.is_nan());
        assert!(!pt.efficiency_pct.is_nan());
    }

    // Extreme sequence length: 131,072
    let report_extreme_s = RooflineEngine::profile_model(&model, &h100, 1, 131072, true, 1.0);
    assert!(report_extreme_s.is_ok());
    let r_ext = report_extreme_s.unwrap();
    for pt in &r_ext.points {
        assert!(!pt.arithmetic_intensity.is_nan());
        assert!(!pt.attainable_tflops.is_nan());
        assert!(pt.attainable_tflops > 0.0);
    }
}
