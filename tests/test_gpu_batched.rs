// SPDX-License-Identifier: AGPL-3.0-only
//! The batched GPU path: one dispatch for all prompt rows, many linears per submit,
//! weights resident on the device, and parity with the CPU path.
//!
//! Tests that need an adapter print a skip reason and return when there is none.
//! Real-model tests are opt-in: set THME_REAL_MODEL_TESTS=1 and THME_TEST_MODELS_DIR to a
//! folder holding `pythia-14m` and `smollm-135m`.

use std::path::PathBuf;
use std::sync::Arc;
use thme::inference::gpu::{GpuBackend, GpuContext, QuantDtype, QuantizedTensor, ThmeError};
use thme::inference::llama::{linear, synthetic_demo};
use thme::inference::{load_reference_model, DeviceArg};

/// Absolute tolerance for GPU vs CPU logits and activations (matches the 0.005 HF cross-check bound).
const PARITY_TOL: f32 = 5e-3;

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

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::init() {
        Ok(gpu) => Some(gpu),
        Err(err) => {
            eprintln!("SKIP: no GPU adapter on this host ({err})");
            None
        }
    }
}

fn assert_close(cpu: &[f32], gpu: &[f32], tol: f32, what: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{what}: length");
    let worst = cpu
        .iter()
        .zip(gpu)
        .map(|(c, g)| (c - g).abs() / c.abs().max(1.0))
        .fold(0.0f32, f32::max);
    assert!(worst < tol, "{what}: worst relative diff {worst} >= {tol}");
}

#[test]
fn batched_matmul_matches_cpu_across_tile_edges() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut seed = 7u64;
    for (rows, in_dim, out_dim) in [
        (1, 70, 33),
        (2, 16, 64),
        (17, 70, 100),
        (64, 64, 64),
        (65, 129, 65),
        (130, 576, 200),
    ] {
        let x = pseudo_random(&mut seed, rows * in_dim, 1.0);
        let w = pseudo_random(&mut seed, out_dim * in_dim, 0.5);
        let got = gpu
            .compute_linear_slices(&x, &[(&w, out_dim)], in_dim)
            .expect("gpu batched linear");
        assert_close(
            &linear(&x, &w, in_dim, out_dim),
            &got[0],
            1e-4,
            &format!("rows={rows} in={in_dim} out={out_dim}"),
        );
    }
}

#[test]
fn prompt_rows_go_in_one_dispatch_and_one_submit() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut seed = 11u64;
    let (rows, in_dim, out_dim) = (128, 64, 96);
    let x = pseudo_random(&mut seed, rows * in_dim, 1.0);
    let w = pseudo_random(&mut seed, out_dim * in_dim, 0.5);
    let before = gpu.stats();
    gpu.compute_linear_slices(&x, &[(&w, out_dim)], in_dim)
        .expect("gpu batched linear");
    let after = gpu.stats();
    assert_eq!(after.dispatches - before.dispatches, 1, "dispatches");
    assert_eq!(after.submits - before.submits, 1, "submits");
}

#[test]
fn many_linears_share_one_submit_and_match_separate_calls() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut seed = 13u64;
    let (rows, in_dim) = (5, 48);
    let x = pseudo_random(&mut seed, rows * in_dim, 1.0);
    let (wq, wk, wv) = (
        pseudo_random(&mut seed, 48 * in_dim, 0.5),
        pseudo_random(&mut seed, 24 * in_dim, 0.5),
        pseudo_random(&mut seed, 24 * in_dim, 0.5),
    );
    let ops: [(&[f32], usize); 3] = [(&wq, 48), (&wk, 24), (&wv, 24)];
    let before = gpu.stats();
    let together = gpu
        .compute_linear_slices(&x, &ops, in_dim)
        .expect("gpu multi linear");
    let after = gpu.stats();
    assert_eq!(
        after.submits - before.submits,
        1,
        "one submit for three ops"
    );
    assert_eq!(after.dispatches - before.dispatches, 3, "three dispatches");
    for (i, (w, out_dim)) in ops.iter().enumerate() {
        assert_close(
            &linear(&x, w, in_dim, *out_dim),
            &together[i],
            1e-4,
            &format!("op {i}"),
        );
    }
}

#[test]
fn weights_stay_resident_between_calls() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut seed = 17u64;
    let in_dim = 32;
    let x = pseudo_random(&mut seed, in_dim, 1.0);
    let w = pseudo_random(&mut seed, 16 * in_dim, 0.5);
    let w2 = pseudo_random(&mut seed, 16 * in_dim, 0.5);
    let s0 = gpu.stats();
    gpu.compute_linear_slices(&x, &[(&w, 16)], in_dim)
        .expect("first");
    let s1 = gpu.stats();
    gpu.compute_linear_slices(&x, &[(&w, 16)], in_dim)
        .expect("second");
    let s2 = gpu.stats();
    gpu.compute_linear_slices(&x, &[(&w2, 16)], in_dim)
        .expect("other weights");
    let s3 = gpu.stats();
    assert_eq!(
        s1.weight_uploads - s0.weight_uploads,
        1,
        "first call uploads"
    );
    assert_eq!(
        s2.weight_uploads - s1.weight_uploads,
        0,
        "second call reuses"
    );
    assert_eq!(
        s3.weight_uploads - s2.weight_uploads,
        1,
        "new weights upload"
    );
}

#[test]
fn changed_weights_at_same_address_are_not_served_stale() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut seed = 19u64;
    let in_dim = 32;
    let x = pseudo_random(&mut seed, in_dim, 1.0);
    let mut w = pseudo_random(&mut seed, 16 * in_dim, 0.5);
    gpu.compute_linear_slices(&x, &[(&w, 16)], in_dim)
        .expect("first");
    for v in w.iter_mut() {
        *v = -*v;
    }
    let got = gpu
        .compute_linear_slices(&x, &[(&w, 16)], in_dim)
        .expect("after in-place change");
    assert_close(
        &linear(&x, &w, in_dim, 16),
        &got[0],
        1e-4,
        "mutated weights",
    );
}

#[test]
fn rows_beyond_the_workgroup_limit_still_work_for_a_single_row() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut seed = 23u64;
    let (in_dim, out_dim) = (16, 70_000);
    let x = pseudo_random(&mut seed, in_dim, 1.0);
    let w = pseudo_random(&mut seed, out_dim * in_dim, 0.5);
    let got = gpu
        .compute_linear_slices(&x, &[(&w, out_dim)], in_dim)
        .expect("wide output");
    assert_close(
        &linear(&x, &w, in_dim, out_dim),
        &got[0],
        1e-4,
        "70000 outputs",
    );
}

#[test]
fn quantized_shaders_handle_many_rows_in_one_dispatch() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut seed = 29u64;
    let (rows, in_dim, out_dim) = (4, 256, 8);
    let mut raw = Vec::new();
    let mut decoded = Vec::new();
    for _ in 0..out_dim {
        let row = pseudo_random(&mut seed, in_dim, 2.0);
        let mut block = Vec::new();
        thme::exporter::kquant::encode_q4_k_block(&row, &mut block).expect("encode");
        decoded.extend(thme::exporter::kquant::decode_q4_k(&block).expect("decode"));
        raw.extend(block);
    }
    let tensor = QuantizedTensor::new_quantized(
        "q4k_rows",
        vec![out_dim, in_dim],
        QuantDtype::Q4_K,
        Arc::from(raw.into_boxed_slice()),
    )
    .expect("tensor");
    let x = pseudo_random(&mut seed, rows * in_dim, 1.0);
    let before = gpu.stats();
    let got = gpu
        .compute_linear(&x, &tensor, in_dim, out_dim)
        .expect("q4_k rows");
    assert_eq!(
        gpu.stats().dispatches - before.dispatches,
        1,
        "one dispatch"
    );
    assert_close(
        &linear(&x, &decoded, in_dim, out_dim),
        &got,
        1e-3,
        "q4_k rows",
    );
}

/// CPU-only stand-in for a device, so CI without a GPU still covers the dispatch plumbing.
struct FakeBackend;

impl GpuBackend for FakeBackend {
    fn compute_linear(
        &self,
        x: &[f32],
        weights: &QuantizedTensor,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, ThmeError> {
        Ok(linear(x, weights.as_f32_slice()?, in_dim, out_dim))
    }
    fn is_format_supported(&self, _dtype: QuantDtype) -> bool {
        true
    }
    fn device_name(&self) -> &str {
        "fake"
    }
}

#[test]
fn fake_backend_default_batch_matches_cpu() {
    let mut seed = 31u64;
    let (rows, in_dim) = (3, 20);
    let x = pseudo_random(&mut seed, rows * in_dim, 1.0);
    let (a, b) = (
        pseudo_random(&mut seed, 7 * in_dim, 0.5),
        pseudo_random(&mut seed, 9 * in_dim, 0.5),
    );
    let got = FakeBackend
        .compute_linear_slices(&x, &[(&a, 7), (&b, 9)], in_dim)
        .expect("default batch");
    assert_eq!(got[0], linear(&x, &a, in_dim, 7));
    assert_eq!(got[1], linear(&x, &b, in_dim, 9));
}

#[test]
fn fake_backend_model_matches_cpu_exactly() {
    let tokens: Vec<u32> = (0..24).map(|i| (i * 37 % 256) as u32).collect();
    let (cpu_model, _) = synthetic_demo(5);
    let (mut gpu_model, _) = synthetic_demo(5);
    gpu_model.set_gpu(Some(Arc::new(FakeBackend)));
    assert_eq!(
        cpu_model.forward_nll(&tokens).expect("cpu"),
        gpu_model.forward_nll(&tokens).expect("fake gpu")
    );
}

#[test]
fn synthetic_model_gpu_matches_cpu_within_tolerance() {
    let Some(gpu) = gpu_or_skip() else { return };
    let tokens: Vec<u32> = (0..40).map(|i| (i * 53 % 256) as u32).collect();
    let (cpu_model, _) = synthetic_demo(9);
    let (mut gpu_model, _) = synthetic_demo(9);
    gpu_model.set_gpu(Some(Arc::new(gpu)));
    let cpu = cpu_model.forward_logits(&tokens).expect("cpu logits");
    let got = gpu_model.forward_logits(&tokens).expect("gpu logits");
    let worst = cpu
        .iter()
        .zip(&got)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < PARITY_TOL, "max abs logit diff {worst}");
    let nll_cpu = cpu_model.forward_nll(&tokens).expect("cpu nll");
    let nll_gpu = gpu_model.forward_nll(&tokens).expect("gpu nll");
    for (c, g) in nll_cpu.iter().zip(&nll_gpu) {
        assert!((c - g).abs() < PARITY_TOL, "nll diff {c} vs {g}");
    }
}

fn real_model_dir(name: &str) -> Option<PathBuf> {
    if std::env::var_os("THME_REAL_MODEL_TESTS").is_none() {
        eprintln!("SKIP: set THME_REAL_MODEL_TESTS=1 to run real-model tests");
        return None;
    }
    let Some(root) = std::env::var_os("THME_TEST_MODELS_DIR") else {
        eprintln!("SKIP: set THME_TEST_MODELS_DIR to the folder holding {name}");
        return None;
    };
    let dir = PathBuf::from(root).join(name);
    if !dir.exists() {
        eprintln!("SKIP: {name} not found under THME_TEST_MODELS_DIR");
        return None;
    }
    Some(dir)
}

fn real_model_parity(name: &str) {
    let Some(dir) = real_model_dir(name) else {
        return;
    };
    if gpu_or_skip().is_none() {
        return;
    }
    let (_, _, cpu_model) = load_reference_model(&dir, DeviceArg::Cpu).expect("cpu load");
    let (_, _, gpu_model) = load_reference_model(&dir, DeviceArg::Gpu).expect("gpu load");
    let vocab = cpu_model.vocab_size();
    for len in [2usize, 128] {
        let tokens: Vec<u32> = (0..len).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();
        let cpu = cpu_model.forward_logits(&tokens).expect("cpu logits");
        let got = gpu_model.forward_logits(&tokens).expect("gpu logits");
        let worst = cpu
            .iter()
            .zip(&got)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        println!("{name} len={len}: max abs logit diff {worst:.3e}");
        assert!(worst < PARITY_TOL, "{name} len={len}: diff {worst}");
        let top = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|p| p.0)
        };
        assert_eq!(top(&cpu), top(&got), "{name} len={len}: top-1 token");
    }
}

#[test]
fn smollm_135m_gpu_matches_cpu() {
    real_model_parity("smollm-135m");
}

#[test]
fn pythia_14m_gpu_matches_cpu() {
    real_model_parity("pythia-14m");
}

#[test]
fn whole_forward_pass_is_one_submit() {
    let Some(gpu) = gpu_or_skip() else { return };
    let gpu = Arc::new(gpu);
    let (mut model, _) = synthetic_demo(3);
    model.set_gpu(Some(gpu.clone()));
    let tokens: Vec<u32> = (0..16).map(|i| (i * 41 % 256) as u32).collect();
    let before = gpu.stats();
    model.forward_logits(&tokens).expect("gpu logits");
    let after = gpu.stats();
    assert_eq!(
        after.submits - before.submits,
        1,
        "one submit for the stack"
    );
    assert!(
        after.dispatches - before.dispatches > 8,
        "layers were dispatched"
    );
}

#[test]
fn context_longer_than_the_device_limit_falls_back_to_the_per_op_path() {
    let Some(gpu) = gpu_or_skip() else { return };
    let tokens: Vec<u32> = (0..2100).map(|i| (i * 29 % 256) as u32).collect();
    let (cpu_model, _) = synthetic_demo(4);
    let (mut gpu_model, _) = synthetic_demo(4);
    gpu_model.set_gpu(Some(Arc::new(gpu)));
    let cpu = cpu_model.forward_logits(&tokens).expect("cpu");
    let got = gpu_model.forward_logits(&tokens).expect("gpu");
    let worst = cpu
        .iter()
        .zip(&got)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < PARITY_TOL, "max abs logit diff {worst}");
}
