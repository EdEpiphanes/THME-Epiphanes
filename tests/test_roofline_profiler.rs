//! End-to-End Integration & Unit Test Suite for THME Roofline & Micro-Kernel Profiler (v0.1.7.0)
//!
//! Covers:
//! 1. Hardware Database & Machine Balance Knee Calculator (H100, RTX 4090, Apple M3, AMD MI300X, Intel Emerald Rapids)
//! 2. Layer Arithmetic Intensity Calculator for Linear GEMM Projections (Decode vs Prefill, Quantized Weight Scaling)
//! 3. Attention Arithmetic Intensity: Standard O(S^2) Attention vs FlashAttention-2 O(S) SRAM Tiling
//! 4. SwiGLU FFN Arithmetic Intensity (Gate, Up, Down Projections and Hadamard Activation)
//! 5. 4-Tier Memory Hierarchy Cache Simulator (L1, L2, L3, DRAM Hit Rates and Eviction)
//! 6. Empirical Roofline Curve Generator & Layer Classifier (MemoryBound vs ComputeBound)
//! 7. Dynamic TDP & Thermal Throttling Model (Steady-State Temperature & DVFS Downclocking)
//! 8. End-to-End Model Profiler on Multi-Layer Transformer Architecture
//! 9. Roofline CLI JSON Serialization and Round-Trip Deserialization
//! 10. MCP JSON-RPC Roofline Tool Schema Validation & Execution

use serde_json::json;
use std::str::FromStr;

use thme::ir::{ModelArchitecture, ModelIR, ParadigmConverter, TensorIR};
use thme::profiling::cache_sim::{CacheSimulator4Tier, CacheTierLevel};
use thme::profiling::hardware::{HardwareDevice, HardwareSpec, HardwareTarget, PrecisionType};
use thme::profiling::intensity::{KernelType, LayerArithmeticIntensity};
use thme::profiling::roofline::{BoundClassification, RooflineModel, RooflineReport};
use thme::profiling::thermal::{DynamicThermalModel, ThermalConfig};

/// Helper to generate a synthetic 12-layer transformer model for profiling tests.
fn create_test_transformer_model(num_layers: usize, hidden_size: usize) -> ModelIR {
    let mut tensors = Vec::new();
    let intermediate = (hidden_size * 8) / 3;

    tensors.push(TensorIR::new(
        "model.embed_tokens.weight".to_string(),
        vec![32000, hidden_size],
        "F16".to_string(),
        32000 * hidden_size * 2,
    ));

    for i in 0..num_layers {
        tensors.push(TensorIR::new(
            format!("layers.{i}.self_attn.q_proj.weight"),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{i}.self_attn.k_proj.weight"),
            vec![hidden_size, hidden_size / 4],
            "F16".to_string(),
            hidden_size * (hidden_size / 4) * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{i}.self_attn.v_proj.weight"),
            vec![hidden_size, hidden_size / 4],
            "F16".to_string(),
            hidden_size * (hidden_size / 4) * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{i}.self_attn.o_proj.weight"),
            vec![hidden_size, hidden_size],
            "F16".to_string(),
            hidden_size * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{i}.mlp.gate_proj.weight"),
            vec![intermediate, hidden_size],
            "F16".to_string(),
            intermediate * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{i}.mlp.up_proj.weight"),
            vec![intermediate, hidden_size],
            "F16".to_string(),
            intermediate * hidden_size * 2,
        ));
        tensors.push(TensorIR::new(
            format!("layers.{i}.mlp.down_proj.weight"),
            vec![hidden_size, intermediate],
            "F16".to_string(),
            hidden_size * intermediate * 2,
        ));
    }

    tensors.push(TensorIR::new(
        "lm_head.weight".to_string(),
        vec![32000, hidden_size],
        "F16".to_string(),
        32000 * hidden_size * 2,
    ));

    ModelIR::new(
        "synthetic-llama-test".to_string(),
        ModelArchitecture::Llama,
        num_layers,
        hidden_size,
        32,
        8,
        32000,
        4096,
        tensors,
    )
}

#[test]
fn test_hardware_database_predefined_targets() {
    let targets = vec![
        HardwareTarget::H100,
        HardwareTarget::RTX4090,
        HardwareTarget::AppleM3Max,
        HardwareTarget::MI300X,
        HardwareTarget::IntelEmeraldRapids,
    ];

    for target in targets {
        let dev = HardwareSpec::from_target(target);
        assert!(!dev.name.is_empty());
        assert!(!dev.vendor.is_empty());
        assert!(!dev.architecture.is_empty());
        assert!(dev.peak_fp32_tflops > 0.0);
        assert!(dev.peak_fp16_tflops > 0.0);
        assert!(dev.peak_int8_tops > 0.0);
        assert!(dev.dram_bandwidth_gbps > 0.0);
        assert!(dev.memory_bandwidth_tbps() > 0.0);
        assert!(dev.tdp_watts > 0.0);
        assert!(dev.idle_power_watts > 0.0);
        assert!(dev.idle_power_watts < dev.tdp_watts);
        assert!(!dev.caches.is_empty());

        for cache in &dev.caches {
            assert!(cache.capacity_bytes > 0);
            assert!(cache.bandwidth_gbps > 0.0);
            assert!(cache.latency_ns > 0.0);
        }

        // Validate machine balance knee point formulas: Knee = Peak FLOPs / Bandwidth Bytes
        let knee_fp16 = dev.machine_balance_knee(PrecisionType::FP16);
        let expected_fp16 = (dev.peak_fp16_tflops * 1000.0) / dev.dram_bandwidth_gbps;
        assert!((knee_fp16 - expected_fp16).abs() < 1e-4);

        let knee_fp32 = dev.machine_balance_knee(PrecisionType::FP32);
        let expected_fp32 = (dev.peak_fp32_tflops * 1000.0) / dev.dram_bandwidth_gbps;
        assert!((knee_fp32 - expected_fp32).abs() < 1e-4);

        // INT8 knee should be higher due to double the compute TOPS
        let knee_int8 = dev.machine_balance_knee(PrecisionType::INT8);
        assert!(knee_int8 >= knee_fp16);
    }

    // Exact empirical benchmark calibration checks
    let h100 = HardwareDevice::predefined_h100();
    assert_eq!(h100.name, "NVIDIA H100 SXM5");
    assert!((h100.dram_bandwidth_gbps - 3350.0).abs() < 1e-3);
    assert!((h100.peak_fp16_tflops - 989.4).abs() < 1e-3);
    assert!((h100.machine_balance(PrecisionType::FP16) - 295.343).abs() < 0.1);

    let rtx4090 = HardwareDevice::predefined_rtx4090();
    assert_eq!(rtx4090.name, "NVIDIA GeForce RTX 4090");
    assert!((rtx4090.dram_bandwidth_gbps - 1008.0).abs() < 1e-3);
    assert!((rtx4090.peak_fp16_tflops - 165.2).abs() < 1e-3);
    assert!((rtx4090.machine_balance(PrecisionType::FP16) - 163.888).abs() < 0.1);

    let m3 = HardwareDevice::predefined_apple_m3_max();
    assert_eq!(m3.name, "Apple M3 Max");
    assert!((m3.dram_bandwidth_gbps - 400.0).abs() < 1e-3);
    assert!((m3.peak_fp16_tflops - 32.4).abs() < 1e-3);
    assert!((m3.machine_balance(PrecisionType::FP16) - 81.0).abs() < 0.1);

    // Substring parsing resolution
    assert_eq!(
        HardwareTarget::from_str("h100").unwrap(),
        HardwareTarget::H100
    );
    assert_eq!(
        HardwareTarget::from_str("rtx-4090").unwrap(),
        HardwareTarget::RTX4090
    );
    assert_eq!(
        HardwareTarget::from_str("apple_m3").unwrap(),
        HardwareTarget::AppleM3Max
    );
    assert_eq!(
        HardwareTarget::from_str("mi300x").unwrap(),
        HardwareTarget::MI300X
    );
    assert_eq!(
        HardwareTarget::from_str("emerald_rapids").unwrap(),
        HardwareTarget::IntelEmeraldRapids
    );
    assert!(HardwareDevice::find_device("hopper").is_some());
    assert!(HardwareDevice::find_device("unknown_chip").is_none());
}

#[test]
fn test_layer_arithmetic_intensity_linear() {
    let dim_in = 4096;
    let dim_out = 4096;

    // Case 1: Decode Phase (M = B * S = 1 * 1 = 1)
    let decode_report = LayerArithmeticIntensity::calculate_linear(
        "q_proj_decode",
        KernelType::LinearQProj,
        1,
        1,
        dim_in,
        dim_out,
        PrecisionType::FP16,
        PrecisionType::FP16,
    );

    // Theoretical decode FLOPs: 2 * 1 * 4096 * 4096 = 33,554,432
    assert_eq!(
        decode_report.total_flops,
        2 * (dim_in as u64) * (dim_out as u64)
    );
    // Weight bytes: 4096 * 4096 * 2 = 33,554,432. Activation bytes: 1 * (4096 + 4096) * 2 = 16,384. Total = 33,570,816
    let expected_decode_bytes = (dim_in * dim_out * 2 + (dim_in + dim_out) * 2) as u64;
    assert_eq!(decode_report.dram_bytes, expected_decode_bytes);
    // Arithmetic intensity in decode should be ~1.0 FLOP/Byte (heavily memory bound)
    assert!((decode_report.arithmetic_intensity - 1.0).abs() < 0.01);
    assert!(decode_report.arithmetic_intensity < 2.0);

    // Case 2: Prefill Phase (B = 1, S = 2048)
    let prefill_report = LayerArithmeticIntensity::calculate_linear(
        "q_proj_prefill",
        KernelType::LinearQProj,
        1,
        2048,
        dim_in,
        dim_out,
        PrecisionType::FP16,
        PrecisionType::FP16,
    );

    assert_eq!(
        prefill_report.total_flops,
        2 * 2048 * (dim_in as u64) * (dim_out as u64)
    );
    // High weight amortization elevates arithmetic intensity to ~1024 FLOPs/Byte
    assert!(prefill_report.arithmetic_intensity > 500.0);
    assert!(prefill_report.arithmetic_intensity > decode_report.arithmetic_intensity * 500.0);

    // Case 3: Quantized BitNet b1.58 Ternary Weights
    let ternary_decode_report = LayerArithmeticIntensity::calculate_linear(
        "q_proj_ternary_decode",
        KernelType::LinearQProj,
        1,
        1,
        dim_in,
        dim_out,
        PrecisionType::Ternary,
        PrecisionType::FP16,
    );

    // 0.2 bytes per parameter (1.58 bits) vs 2.0 bytes for FP16 -> ~10x byte traffic reduction
    assert!(ternary_decode_report.dram_bytes < decode_report.dram_bytes / 8);
    assert!(ternary_decode_report.arithmetic_intensity > decode_report.arithmetic_intensity * 7.0);
}

#[test]
fn test_layer_arithmetic_intensity_attention_and_flash_attention() {
    let batch = 1;
    let seq_len = 2048;
    let context_len = 2048;
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = 128;

    // Standard Attention (writes quadratic S x C matrix to DRAM)
    let std_attn = LayerArithmeticIntensity::calculate_attention(
        batch,
        seq_len,
        context_len,
        num_heads,
        num_kv_heads,
        head_dim,
        PrecisionType::FP16,
    );

    // FlashAttention-2 (tiles online softmax in fast SRAM, avoids O(S*C) DRAM roundtrips)
    let flash_attn = LayerArithmeticIntensity::calculate_flash_attention_v2(
        batch,
        seq_len,
        context_len,
        num_heads,
        num_kv_heads,
        head_dim,
        PrecisionType::FP16,
    );

    // Both algorithms perform identical computational FLOPs: 4 B H S C d + 3 B H S C
    assert_eq!(std_attn.total_flops, flash_attn.total_flops);

    // FlashAttention-2 drastically reduces DRAM bytes by eliminating intermediate attention map
    assert!(std_attn.dram_bytes > flash_attn.dram_bytes * 10);
    assert!(flash_attn.arithmetic_intensity > std_attn.arithmetic_intensity * 10.0);

    // At long sequence length (S = 4096), the disparity expands linearly with sequence length
    let std_long = LayerArithmeticIntensity::calculate_attention(
        batch,
        4096,
        4096,
        num_heads,
        num_kv_heads,
        head_dim,
        PrecisionType::FP16,
    );
    let flash_long = LayerArithmeticIntensity::calculate_flash_attention_v2(
        batch,
        4096,
        4096,
        num_heads,
        num_kv_heads,
        head_dim,
        PrecisionType::FP16,
    );

    assert!(std_long.dram_bytes > flash_long.dram_bytes * 20);
    assert!(flash_long.arithmetic_intensity > std_long.arithmetic_intensity * 20.0);
}

#[test]
fn test_layer_arithmetic_intensity_swiglu() {
    let batch = 1;
    let seq_len = 2048;
    let hidden = 4096;
    let intermediate = 11008;

    let swiglu_report = LayerArithmeticIntensity::calculate_swiglu(
        batch,
        seq_len,
        hidden,
        intermediate,
        PrecisionType::FP16,
        PrecisionType::FP16,
    );

    let m = (batch * seq_len) as u64;
    let d = hidden as u64;
    let ffn = intermediate as u64;

    // Mathematical formula verification: FLOPs = 6 * M * D * FFN + 3 * M * FFN
    let expected_flops = 6 * m * d * ffn + 3 * m * ffn;
    assert_eq!(swiglu_report.total_flops, expected_flops);

    // Weight bytes: 3 * D * FFN * 2.0; Activation bytes: M * (2 * D + 2 * FFN) * 2.0
    let expected_weight_bytes = 3 * d * ffn * 2;
    let expected_act_bytes = m * (2 * d + 2 * ffn) * 2;
    assert_eq!(
        swiglu_report.dram_bytes,
        expected_weight_bytes + expected_act_bytes
    );

    // Prefill SwiGLU has high intensity (> 1000 FLOP/Byte)
    assert!(swiglu_report.arithmetic_intensity > 1000.0);

    // Decode SwiGLU (M = 1) is dominated by weight bytes (~1.0 FLOP/Byte)
    let swiglu_decode = LayerArithmeticIntensity::calculate_swiglu(
        1,
        1,
        hidden,
        intermediate,
        PrecisionType::FP16,
        PrecisionType::FP16,
    );
    assert!((swiglu_decode.arithmetic_intensity - 1.0).abs() < 0.05);
    assert!(swiglu_decode.arithmetic_intensity > 0.9);
}

#[test]
fn test_cache_simulator_4_tier() {
    let h100 = HardwareDevice::predefined_h100();
    let sim = CacheSimulator4Tier::new(h100);

    // Case 1: Small working set (1 MB) fits entirely in L1 SRAM (33.8 MB)
    let small_res = sim.simulate_working_set(1_000_000);
    assert_eq!(small_res.tiers[0].level, CacheTierLevel::L1);
    assert_eq!(small_res.tiers[0].hit_ratio, 0.95);
    // L1 compulsory misses (50,000 bytes) are serviced by L2 (95% hit), leaving 2,500 bytes DRAM spill
    assert_eq!(small_res.dram_spill_bytes, 2500);
    assert!(small_res.effective_latency_ns < 10.0);

    // Case 2: Medium working set (40 MB) spills past L1 (33.8 MB) into L2 (50 MB)
    let med_res = sim.simulate_working_set(40_000_000);
    assert!(med_res.tiers[0].hit_ratio < 0.95);
    assert!(med_res.tiers[0].hit_ratio > 0.70);
    assert!(med_res.effective_latency_ns > small_res.effective_latency_ns);

    // Case 3: Large working set (100 MB) exceeds L2 (50 MB) and spills into DRAM
    let large_res = sim.simulate_working_set(100_000_000);
    assert!(large_res.tiers[1].hit_ratio < 0.60);
    assert!(large_res.dram_spill_bytes > 30_000_000);
    assert!(large_res.effective_latency_ns > med_res.effective_latency_ns);
    // Effective bandwidth should be lower due to DRAM access penalties
    assert!(large_res.effective_bandwidth_gbps < small_res.effective_bandwidth_gbps);
}

#[test]
fn test_roofline_curve_and_layer_classification() {
    let h100 = HardwareDevice::predefined_h100();
    let engine = RooflineModel::new(h100, PrecisionType::FP16);
    let knee = 295.343;

    // Test 1: Memory-Bound Point (e.g. Decode Projection at I = 1.0 FLOP/Byte)
    let mem_point = engine.evaluate_point("Decode GEMM", 1.0);
    assert!(mem_point.arithmetic_intensity < knee);
    match mem_point.classification {
        BoundClassification::MemoryBound {
            memory_bound_factor,
            headroom_to_knee,
        } => {
            assert!(memory_bound_factor < 0.01);
            assert!(headroom_to_knee > 290.0);
        }
        BoundClassification::ComputeBound { .. } => {
            panic!("Intensity 1.0 must be classified as MemoryBound");
        }
    }
    assert!(mem_point.attainable_tflops < 5.0); // 1.0 * 3350 / 1000 = 3.35 TFLOPS
    assert!(mem_point.efficiency_pct < 1.0);
    assert!(mem_point
        .recommended_optimizations
        .iter()
        .any(|r| r.contains("BitNet")));

    // Test 2: Compute-Bound Point (e.g. Prefill SwiGLU at I = 1200.0 FLOP/Byte)
    let comp_point = engine.evaluate_point("Prefill SwiGLU FFN", 1200.0);
    assert!(comp_point.arithmetic_intensity > knee);
    match comp_point.classification {
        BoundClassification::ComputeBound {
            compute_saturation_ratio,
        } => {
            assert_eq!(compute_saturation_ratio, 1.0);
        }
        BoundClassification::MemoryBound { .. } => {
            panic!("Intensity 1200.0 must be classified as ComputeBound");
        }
    }
    assert!((comp_point.attainable_tflops - 989.4).abs() < 1e-3);
    assert_eq!(comp_point.efficiency_pct, 100.0);
    assert!(comp_point
        .recommended_optimizations
        .iter()
        .any(|r| r.contains("sparsity")));

    // Test 3: ASCII Diagram Generation
    let ascii_chart = engine.generate_ascii_diagram(&[mem_point, comp_point]);
    assert!(ascii_chart.contains("ROOFLINE CEILING MODEL [NVIDIA H100 SXM5]"));
    assert!(ascii_chart.contains("Knee: 295.34 FLOP/Byte"));
    assert!(ascii_chart.contains("MEMORY BOUND"));
    assert!(ascii_chart.contains("COMPUTE BOUND"));
    assert!(ascii_chart.contains("Decode GEMM"));
    assert!(ascii_chart.contains("Prefill SwiGLU FFN"));
}

#[test]
fn test_thermal_throttling_model() {
    let rtx4090 = HardwareDevice::predefined_rtx4090();
    let config = ThermalConfig::default();
    let mut model = DynamicThermalModel::new(rtx4090, config);

    assert_eq!(model.current_temperature(), 25.0);

    // Initial nominal step at ambient 25°C
    let step1 = model.step_simulation(1.0, 1.0, 1.0);
    assert!(!step1.is_throttling);
    assert_eq!(step1.throttle_factor, 1.0);
    assert_eq!(step1.effective_tflops, 165.2);
    assert!(model.current_temperature() > 25.0);

    // Steady-state temperature under full compute utilization
    let steady_temp = model.steady_state_temp(1.0, 1.0);
    assert!(steady_temp > 50.0);

    // Inject high junction temperature to trigger DVFS throttling (> 83.0°C)
    model.set_temperature(89.0);
    let throttled_step = model.step_simulation(1.0, 1.0, 1.0);
    assert!(throttled_step.is_throttling);
    // Delta = 89 - 83 = 6.0°C -> Throttle factor = 1.0 - 0.05 * 6 = 0.70
    assert!((throttled_step.throttle_factor - 0.70).abs() < 0.02);
    assert!(throttled_step.effective_tflops < 165.2 * 0.75);
    // Dynamic power dissipation downscaled quadratically
    assert!(throttled_step.dynamic_power_watts < (450.0 - 35.0) * 0.55);

    // Helper calculate_throttling_factor
    assert_eq!(
        DynamicThermalModel::calculate_throttling_factor(65.0, 450.0),
        1.0
    );
    let factor_hot = DynamicThermalModel::calculate_throttling_factor(85.0, 450.0);
    assert!((factor_hot - 0.90).abs() < 1e-4);
}

#[test]
fn test_end_to_end_roofline_model_profiler() {
    let model = create_test_transformer_model(12, 2048);
    let h100 = HardwareSpec::from_target(HardwareTarget::H100);

    // Profile model in Prefill phase (B = 1, S = 2048) with FlashAttention-2 enabled
    let report = RooflineModel::profile_model(&model, &h100, 1, 2048, true, 1.0).unwrap();

    assert_eq!(report.hardware_name, "NVIDIA H100 SXM5");
    assert_eq!(report.precision, PrecisionType::FP16);
    assert!((report.peak_compute_tflops - 989.4).abs() < 1e-3);
    assert!((report.knee_intensity_flop_per_byte - 295.343).abs() < 0.1);
    assert_eq!(report.points.len(), 7); // Q, K, V, O proj, Attention, SwiGLU FFN, RMSNorm

    // Verify FlashAttention-2 and SwiGLU are compute bound in prefill
    let flash_pt = report
        .points
        .iter()
        .find(|p| p.name.contains("FlashAttention-2"))
        .unwrap();
    assert_eq!(
        flash_pt.classification,
        BoundClassification::ComputeBound {
            compute_saturation_ratio: 1.0
        }
    );

    let swiglu_pt = report
        .points
        .iter()
        .find(|p| p.name.contains("SwiGLU"))
        .unwrap();
    assert_eq!(
        swiglu_pt.classification,
        BoundClassification::ComputeBound {
            compute_saturation_ratio: 1.0
        }
    );

    // Verify ASCII formatted report contains model metrics
    let ascii = report.format_ascii_report(
        &h100,
        report.peak_compute_tflops,
        report.knee_intensity_flop_per_byte,
    );
    assert!(ascii.contains("ROOFLINE CEILING MODEL [NVIDIA H100 SXM5]"));
    assert!(ascii.contains("FlashAttention-2"));
    assert!(ascii.contains("SwiGLU FFN"));
    assert!(ascii.contains("COMPUTE-BOUND"));

    // Compare with standard attention (non-flash): attention intensity drops
    let report_std = RooflineModel::profile_model(&model, &h100, 1, 2048, false, 1.0).unwrap();
    let std_attn_pt = report_std
        .points
        .iter()
        .find(|p| p.name.contains("Attention"))
        .unwrap();
    assert!(std_attn_pt.arithmetic_intensity < flash_pt.arithmetic_intensity);
}

#[test]
fn test_roofline_cli_json_serialization() {
    let model = ParadigmConverter::create_synthetic_baseline();
    let rtx4090 = HardwareSpec::from_target(HardwareTarget::RTX4090);

    let report = RooflineModel::profile_model(&model, &rtx4090, 1, 1024, true, 0.95).unwrap();

    // Serialize to JSON
    let json_str =
        serde_json::to_string_pretty(&report).expect("Serialization to JSON must succeed");
    assert!(!json_str.is_empty());
    assert!(json_str.contains("\"hardware_name\": \"NVIDIA GeForce RTX 4090\""));
    assert!(json_str.contains("\"precision\": \"FP16\""));
    assert!(json_str.contains("\"peak_compute_tflops\""));
    assert!(json_str.contains("\"knee_intensity_flop_per_byte\""));
    assert!(json_str.contains("\"points\""));

    // Round-trip deserialize from JSON string back to RooflineReport
    let deserialized: RooflineReport =
        serde_json::from_str(&json_str).expect("Deserialization from JSON must succeed");

    assert_eq!(deserialized.hardware_name, report.hardware_name);
    assert_eq!(deserialized.precision, report.precision);
    assert!((deserialized.peak_compute_tflops - report.peak_compute_tflops).abs() < 1e-4);
    assert!((deserialized.memory_bandwidth_gbps - report.memory_bandwidth_gbps).abs() < 1e-4);
    assert_eq!(deserialized.points.len(), report.points.len());

    for (p1, p2) in report.points.iter().zip(deserialized.points.iter()) {
        assert_eq!(p1.name, p2.name);
        assert!((p1.arithmetic_intensity - p2.arithmetic_intensity).abs() < 1e-4);
        assert!((p1.attainable_tflops - p2.attainable_tflops).abs() < 1e-4);
        assert_eq!(p1.recommended_optimizations, p2.recommended_optimizations);

        match (&p1.classification, &p2.classification) {
            (
                BoundClassification::MemoryBound {
                    memory_bound_factor: f1,
                    headroom_to_knee: h1,
                },
                BoundClassification::MemoryBound {
                    memory_bound_factor: f2,
                    headroom_to_knee: h2,
                },
            ) => {
                assert!((f1 - f2).abs() < 1e-4);
                assert!((h1 - h2).abs() < 1e-4);
            }
            (
                BoundClassification::ComputeBound {
                    compute_saturation_ratio: r1,
                },
                BoundClassification::ComputeBound {
                    compute_saturation_ratio: r2,
                },
            ) => {
                assert!((r1 - r2).abs() < 1e-4);
            }
            _ => panic!("Classification variant mismatch after JSON roundtrip"),
        }
    }
}

#[test]
fn test_mcp_roofline_tool_invocation() {
    // 1. Validate MCP JSON-RPC Request structure for thme_profile_roofline
    let rpc_req = json!({
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "thme_profile_roofline",
            "arguments": {
                "hardware": "h100",
                "batch_size": 1,
                "seq_len": 2048,
                "enable_flash_attention": true,
                "thermal_throttling": false
            }
        }
    });

    assert_eq!(rpc_req["method"], "tools/call");
    assert_eq!(rpc_req["params"]["name"], "thme_profile_roofline");
    let args = &rpc_req["params"]["arguments"];
    assert_eq!(args["hardware"], "h100");
    assert_eq!(args["batch_size"], 1);
    assert_eq!(args["seq_len"], 2048);

    // 2. Execute underlying profiler pipeline matching MCP handler contract
    let model = ParadigmConverter::create_synthetic_baseline();
    let hw_target = HardwareTarget::from_str(args["hardware"].as_str().unwrap()).unwrap();
    let hw = HardwareSpec::from_target(hw_target);
    let batch_size = args["batch_size"].as_u64().unwrap() as usize;
    let seq_len = args["seq_len"].as_u64().unwrap() as usize;
    let flash_attn = args["enable_flash_attention"].as_bool().unwrap();

    let report =
        RooflineModel::profile_model(&model, &hw, batch_size, seq_len, flash_attn, 1.0).unwrap();

    // 3. Format simulated MCP response markdown
    let mut mem_bound_count = 0;
    let mut comp_bound_count = 0;
    for pt in &report.points {
        match pt.classification {
            BoundClassification::MemoryBound { .. } => mem_bound_count += 1,
            BoundClassification::ComputeBound { .. } => comp_bound_count += 1,
        }
    }

    let mcp_output = format!(
        "### THME Roofline & Micro-Kernel Profiling Report\n\
        - **Target Hardware**: `{}`\n\
        - **Machine Balance Knee**: {:.2} FLOPs/Byte\n\
        - **Memory-Bound Kernels**: {}\n\
        - **Compute-Bound Kernels**: {}\n",
        hw.name, report.knee_intensity_flop_per_byte, mem_bound_count, comp_bound_count
    );

    assert!(mcp_output.contains("NVIDIA H100 SXM5"));
    assert!(mcp_output.contains("Machine Balance Knee"));
    assert!(mem_bound_count > 0);
    assert!(comp_bound_count > 0);
}
