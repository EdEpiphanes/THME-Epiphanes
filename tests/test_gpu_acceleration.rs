// SPDX-License-Identifier: AGPL-3.0-only
//! Integration test suite for the GPU acceleration pipeline:
//! WGPU compute backend, WGSL shaders for GEMV & fused dequantization (Q4_K, Q5_K, IQ4_XS, F32),
//! hardware profiling integration via neutron-gpu, and fallback semantics.

use std::sync::Arc;
use thme::exporter::conversion::f32_to_f16;
use thme::exporter::kquant::{
    decode_iq4_xs, decode_q4_k, decode_q5_k, encode_q4_k_block, encode_q5_k_block, IQ4_XS_BYTES,
};
use thme::inference::gpu::{
    dispatch_linear, dispatch_score_head, GpuBackend, GpuContext, QuantDtype, QuantizedTensor,
};
use thme::inference::llama::{linear, score_head, synthetic_demo};
use thme::profiling::hardware::HardwareDevice;

fn pseudo_random(seed: &mut u64, len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((*seed >> 33) as f32 / 2_147_483_648.0) - 1.0) * scale
        })
        .collect()
}

#[test]
fn test_wgpu_vulkan_init_and_ada_lovelace_recipe() {
    let gpu = match GpuContext::init() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("Notice: GPU unavailable on host ({err}); skipping physical device test");
            return;
        }
    };

    let info = &gpu.adapter_info;
    println!(
        "GPU Adapter Initialized: name='{}', vendor=0x{:X}, device=0x{:X}, backend={:?}",
        info.name, info.vendor, info.device, info.backend
    );

    // Verify all 4 compute pipelines were precompiled
    assert!(gpu
        .pipelines
        .contains_key(&thme::inference::gpu::ShaderKind::Q4_K));
    assert!(gpu
        .pipelines
        .contains_key(&thme::inference::gpu::ShaderKind::Q5_K));
    assert!(gpu
        .pipelines
        .contains_key(&thme::inference::gpu::ShaderKind::IQ4_XS));
    assert!(gpu
        .pipelines
        .contains_key(&thme::inference::gpu::ShaderKind::F32));

    // Connect to neutron-gpu recipe lookup
    let vendor_id = info.vendor as u16;
    let device_id = info.device as u16;

    if vendor_id == 0x10DE {
        // NVIDIA vendor
        if let Some(recipe) = neutron_gpu::get_recipe(vendor_id, device_id) {
            println!(
                "Matched neutron-gpu CardRecipe: '{}' (ChipFamily: {:?}, chip_id: 0x{:X}, Tensor Cores: {})",
                recipe.name,
                recipe.family,
                recipe.chip_id,
                recipe.has_tensor_cores()
            );
            assert!(recipe.has_tensor_cores());
            if device_id == 0x2803 || device_id == 0x2805 {
                assert_eq!(recipe.family, neutron_gpu::ChipFamily::AdaLovelace);
                assert_eq!(recipe.chip_id, 0x196); // AD106 silicon
            }
        }

        // Verify HardwareDevice profile connection
        if let Some(profile) = HardwareDevice::from_pci_recipe(vendor_id, device_id) {
            println!(
                "HardwareDevice Profile: '{}', DRAM BW: {} GB/s, Peak FP32: {} TFLOPS",
                profile.name, profile.dram_bandwidth_gbps, profile.peak_fp32_tflops
            );
            assert_eq!(profile.dram_bandwidth_gbps, 288.0);
            assert_eq!(profile.peak_fp32_tflops, 22.06);
        }
    }
}

#[test]
fn test_f32_gemv_numerical_parity() {
    let gpu = match GpuContext::init() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let mut seed = 42u64;
    let in_dim = 256;
    let out_dim = 64;

    let x = pseudo_random(&mut seed, in_dim, 1.0);
    let w = pseudo_random(&mut seed, out_dim * in_dim, 0.5);

    // CPU reference
    let y_cpu = linear(&x, &w, in_dim, out_dim);

    // GPU execution
    let tensor =
        QuantizedTensor::new_f32("test_f32", vec![out_dim, in_dim], w).expect("create f32 tensor");
    let y_gpu = gpu
        .compute_linear(&x, &tensor, in_dim, out_dim)
        .expect("gpu compute_linear f32");

    assert_eq!(y_cpu.len(), out_dim);
    assert_eq!(y_gpu.len(), out_dim);

    let mut max_diff = 0.0f32;
    for (c, g) in y_cpu.iter().zip(&y_gpu) {
        assert!(g.is_finite(), "GPU output must be finite");
        let diff = (c - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("F32 GEMV Parity: max_diff = {max_diff:.6e}");
    assert!(
        max_diff < 1e-4,
        "F32 GEMV output diff {max_diff} exceeded tolerance 1e-4"
    );
}

#[test]
fn test_q4_k_dequant_gemv_parity() {
    let gpu = match GpuContext::init() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let mut seed = 123u64;
    let in_dim = 256; // 1 Q4_K super-block
    let out_dim = 16;

    // Generate random weights and quantize them into Q4_K blocks
    let mut raw_bytes = Vec::new();
    let mut w_f32_reconstructed = Vec::new();

    for _ in 0..out_dim {
        let row_values = pseudo_random(&mut seed, in_dim, 2.0);
        let mut block_bytes = Vec::new();
        encode_q4_k_block(&row_values, &mut block_bytes).expect("encode Q4_K block");
        assert_eq!(block_bytes.len(), 144);

        // Decode on CPU to establish reference decoded weights
        let decoded = decode_q4_k(&block_bytes).expect("decode Q4_K");
        w_f32_reconstructed.extend_from_slice(&decoded);
        raw_bytes.extend_from_slice(&block_bytes);
    }

    let x = pseudo_random(&mut seed, in_dim, 1.0);

    // CPU reference multiplication with decoded weights
    let y_cpu = linear(&x, &w_f32_reconstructed, in_dim, out_dim);

    // GPU execution with raw quantized bytes and in-shader dequantization
    let tensor = QuantizedTensor::new_quantized(
        "q4_k_test",
        vec![out_dim, in_dim],
        QuantDtype::Q4_K,
        Arc::from(raw_bytes.into_boxed_slice()),
    )
    .expect("create Q4_K tensor");

    let y_gpu = gpu
        .compute_linear(&x, &tensor, in_dim, out_dim)
        .expect("gpu compute_linear Q4_K");

    assert_eq!(y_cpu.len(), out_dim);
    assert_eq!(y_gpu.len(), out_dim);

    let mut max_diff = 0.0f32;
    for (i, (&c, &g)) in y_cpu.iter().zip(&y_gpu).enumerate() {
        assert!(g.is_finite(), "GPU output[{i}] must be finite");
        let diff = (c - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Q4_K Dequant + GEMV Parity: max_diff = {max_diff:.6e}");
    assert!(
        max_diff < 1e-3,
        "Q4_K GEMV output diff {max_diff} exceeded tolerance 1e-3"
    );
}

#[test]
fn test_q5_k_dequant_gemv_parity() {
    let gpu = match GpuContext::init() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let mut seed = 456u64;
    let in_dim = 256; // 1 Q5_K super-block
    let out_dim = 16;

    let mut raw_bytes = Vec::new();
    let mut w_f32_reconstructed = Vec::new();

    for _ in 0..out_dim {
        let row_values = pseudo_random(&mut seed, in_dim, 2.5);
        let mut block_bytes = Vec::new();
        encode_q5_k_block(&row_values, &mut block_bytes).expect("encode Q5_K block");
        assert_eq!(block_bytes.len(), 176);

        let decoded = decode_q5_k(&block_bytes).expect("decode Q5_K");
        w_f32_reconstructed.extend_from_slice(&decoded);
        raw_bytes.extend_from_slice(&block_bytes);
    }

    let x = pseudo_random(&mut seed, in_dim, 1.0);

    let y_cpu = linear(&x, &w_f32_reconstructed, in_dim, out_dim);

    let tensor = QuantizedTensor::new_quantized(
        "q5_k_test",
        vec![out_dim, in_dim],
        QuantDtype::Q5_K,
        Arc::from(raw_bytes.into_boxed_slice()),
    )
    .expect("create Q5_K tensor");

    let y_gpu = gpu
        .compute_linear(&x, &tensor, in_dim, out_dim)
        .expect("gpu compute_linear Q5_K");

    assert_eq!(y_cpu.len(), out_dim);
    assert_eq!(y_gpu.len(), out_dim);

    let mut max_diff = 0.0f32;
    for (i, (&c, &g)) in y_cpu.iter().zip(&y_gpu).enumerate() {
        assert!(g.is_finite(), "GPU output[{i}] must be finite");
        let diff = (c - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Q5_K Dequant + GEMV Parity: max_diff = {max_diff:.6e}");
    assert!(
        max_diff < 1e-3,
        "Q5_K GEMV output diff {max_diff} exceeded tolerance 1e-3"
    );
}

#[test]
fn test_iq4_xs_dequant_gemv_parity() {
    let gpu = match GpuContext::init() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let in_dim = 256;
    let out_dim = 8;

    let mut raw_bytes = Vec::new();
    let mut w_f32_reconstructed = Vec::new();

    // Construct valid IQ4_XS blocks
    for row in 0..out_dim {
        let mut block = vec![0u8; IQ4_XS_BYTES];
        let d_val = 0.05 * (row + 1) as f32;
        block[0..2].copy_from_slice(&f32_to_f16(d_val).to_le_bytes());
        block[2] = (row as u8) & 3; // scales_h
        block[3] = 0;
        block[4] = 0x12; // scales_l
        block[5] = 0x34;
        block[6] = 0x56;
        block[7] = 0x78;
        for (i, byte) in block[8..136].iter_mut().enumerate() {
            let idx = i + 8;
            *byte = ((idx * 17 + row * 13) % 256) as u8;
        }

        let decoded = decode_iq4_xs(&block).expect("decode IQ4_XS");
        w_f32_reconstructed.extend_from_slice(&decoded);
        raw_bytes.extend_from_slice(&block);
    }

    let mut seed = 789u64;
    let x = pseudo_random(&mut seed, in_dim, 1.0);

    let y_cpu = linear(&x, &w_f32_reconstructed, in_dim, out_dim);

    let tensor = QuantizedTensor::new_quantized(
        "iq4_xs_test",
        vec![out_dim, in_dim],
        QuantDtype::IQ4_XS,
        Arc::from(raw_bytes.into_boxed_slice()),
    )
    .expect("create IQ4_XS tensor");

    let y_gpu = gpu
        .compute_linear(&x, &tensor, in_dim, out_dim)
        .expect("gpu compute_linear IQ4_XS");

    assert_eq!(y_cpu.len(), out_dim);
    assert_eq!(y_gpu.len(), out_dim);

    let mut max_diff = 0.0f32;
    for (i, (&c, &g)) in y_cpu.iter().zip(&y_gpu).enumerate() {
        assert!(g.is_finite(), "GPU output[{i}] must be finite");
        let diff = (c - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("IQ4_XS Dequant + GEMV Parity: max_diff = {max_diff:.6e}");
    assert!(
        max_diff < 1e-3,
        "IQ4_XS GEMV output diff {max_diff} exceeded tolerance 1e-3"
    );
}

#[test]
fn test_score_head_vocab_projection_and_argmax() {
    let gpu = match GpuContext::init() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let mut seed = 999u64;
    let d = 256;
    let vocab = 512;
    let tokens = [1u32, 42u32, 108u32]; // 3 tokens -> 2 exit rows

    let normed = pseudo_random(&mut seed, tokens.len() * d, 1.0);
    let head = pseudo_random(&mut seed, vocab * d, 0.2);

    let cpu_exit = score_head(&head, d, vocab, &normed, &tokens, 1);

    let head_tensor =
        QuantizedTensor::new_f32("head_test", vec![vocab, d], head).expect("create head tensor");
    let gpu_exit = dispatch_score_head(Some(&gpu), &head_tensor, d, vocab, &normed, &tokens, 1)
        .expect("dispatch_score_head on GPU");

    assert_eq!(
        cpu_exit.top1, gpu_exit.top1,
        "Top-1 predicted tokens must match 100%"
    );
    assert_eq!(cpu_exit.nll.len(), gpu_exit.nll.len());

    for (r, (&c_nll, &g_nll)) in cpu_exit.nll.iter().zip(&gpu_exit.nll).enumerate() {
        assert!(g_nll.is_finite());
        let diff = (c_nll - g_nll).abs();
        println!(
            "Score head row {r}: CPU NLL = {c_nll:.4}, GPU NLL = {g_nll:.4}, diff = {diff:.6e}"
        );
        assert!(diff < 0.01, "NLL difference {diff} exceeds tolerance 0.01");
    }
}

#[test]
fn test_real_medgemma_or_synthetic_forward_pass() {
    let path = std::env::var("THME_MEDGEMMA_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
                std::path::PathBuf::from(local_app_data)
                    .join("hermes/models/medgemma-4b-it-UD-Q4_K_XL.gguf")
            } else {
                std::path::PathBuf::from("models/medgemma-4b-it-UD-Q4_K_XL.gguf")
            }
        });

    if path.exists() {
        println!("Testing on real host MedGemma GGUF model: {:?}", path);
        let model_ir = thme::parser::ModelImporter::import(&path).expect("import medgemma model");
        let arch_cfg = thme::inference::arch::ArchConfig::detect(&model_ir, &path)
            .expect("detect architecture");

        let cpu_model = arch_cfg
            .load(&model_ir, thme::inference::WeightPrecision::AsStored)
            .expect("load CPU model");

        let tokens = [2u32, 106u32];
        let cpu_exits = cpu_model
            .forward_scheduled(&tokens, &[0], &[1])
            .expect("CPU forward_scheduled layer 0");
        assert_eq!(cpu_exits.len(), 1);
        println!("CPU MedGemma layer 0 NLL: {:.4}", cpu_exits[0].nll[0]);

        if let Ok(gpu_ctx) = GpuContext::init() {
            println!(
                "Testing GPU acceleration on host: {}",
                gpu_ctx.adapter_info.name
            );
            let gpu = Arc::new(gpu_ctx);

            if let thme::inference::arch::ArchConfig::Gemma(cfg) = arch_cfg {
                let gpu_model = thme::inference::GemmaModel::load(
                    &model_ir,
                    cfg,
                    thme::inference::WeightPrecision::AsStored,
                )
                .expect("load GPU Gemma model")
                .with_gpu(gpu.clone());

                let gpu_exits = gpu_model
                    .forward_scheduled(&tokens, &[0], &[1])
                    .expect("GPU forward_scheduled layer 0");

                println!(
                    "GPU MedGemma layer 0 NLL: {:.4}, top1: {:?}",
                    gpu_exits[0].nll[0], gpu_exits[0].top1
                );

                assert!(gpu_exits[0].nll[0].is_finite());
                assert_eq!(
                    cpu_exits[0].top1, gpu_exits[0].top1,
                    "Predicted token must match"
                );
                let diff = (cpu_exits[0].nll[0] - gpu_exits[0].nll[0]).abs();
                assert!(
                    diff < 0.05,
                    "MedGemma NLL diff {diff} exceeds tolerance 0.05"
                );
            }
        }
    } else {
        println!("MedGemma not found on disk; testing synthetic model forward pass");
        let (cpu_model, _) = synthetic_demo(42);
        let tokens = [5u32, 12u32, 28u32];
        let cpu_exits = cpu_model
            .forward_scheduled(&tokens, &[0, 1], &[2])
            .expect("CPU synthetic forward");

        if let Ok(gpu_ctx) = GpuContext::init() {
            let gpu = Arc::new(gpu_ctx);
            let (mut gpu_model, _) = synthetic_demo(42);
            gpu_model.set_gpu(Some(gpu));
            let gpu_exits = gpu_model
                .forward_scheduled(&tokens, &[0, 1], &[2])
                .expect("GPU synthetic forward");

            assert_eq!(cpu_exits[0].top1, gpu_exits[0].top1);
        }
    }
}

#[test]
fn test_graceful_fallback_when_gpu_disabled() {
    let mut seed = 777u64;
    let in_dim = 128;
    let out_dim = 32;

    let x = pseudo_random(&mut seed, in_dim, 1.0);
    let w = pseudo_random(&mut seed, out_dim * in_dim, 0.5);

    // Call dispatch_linear with backend = None (CPU fallback)
    let tensor = QuantizedTensor::new_f32("fallback_w", vec![out_dim, in_dim], w.clone())
        .expect("create tensor");
    let y_fallback = dispatch_linear(None, &x, &tensor, in_dim, out_dim)
        .expect("dispatch_linear fallback to CPU");

    let y_cpu = linear(&x, &w, in_dim, out_dim);
    assert_eq!(
        y_fallback, y_cpu,
        "Fallback to CPU must equal linear reference output"
    );
}
