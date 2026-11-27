// SPDX-License-Identifier: AGPL-3.0-only
//! Golden-file integration tests for `thme pathfinder` (THME-3057).

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use thme::cli::pathfinder::{plan, render_json, render_table, KvArg, PathfinderArgs};
use thme::profiling::run_memory::KvType;
use thme::profiling::run_plan::RunPlan;

/// Generates the mock LLaMA model matching `examples/generate_mock_model.rs` inside `out_dir`.
fn write_mock_llama(out_dir: &Path) -> PathBuf {
    let model_dir = out_dir.join("mock_llama");
    std::fs::create_dir_all(&model_dir).expect("create mock_llama directory");

    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "torch_dtype": "float16"
    });
    let mut f_config = File::create(model_dir.join("config.json")).expect("create config.json");
    f_config
        .write_all(config.to_string().as_bytes())
        .expect("write config.json");

    let mut buffers: Vec<Vec<u8>> = Vec::new();
    let mut entries: Vec<(String, Vec<usize>, usize)> = Vec::new();

    let embed_idx = buffers.len();
    buffers.push(vec![0u8; 32000 * 256 * 2]);
    entries.push((
        "model.embed_tokens.weight".to_string(),
        vec![32000, 256],
        embed_idx,
    ));

    for l in 0..4 {
        let q_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 256 * 2]);
        entries.push((
            format!("model.layers.{l}.self_attn.q_proj.weight"),
            vec![256, 256],
            q_idx,
        ));

        let k_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 128 * 2]);
        entries.push((
            format!("model.layers.{l}.self_attn.k_proj.weight"),
            vec![256, 128],
            k_idx,
        ));

        let v_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 128 * 2]);
        entries.push((
            format!("model.layers.{l}.self_attn.v_proj.weight"),
            vec![256, 128],
            v_idx,
        ));

        let o_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 256 * 2]);
        entries.push((
            format!("model.layers.{l}.self_attn.o_proj.weight"),
            vec![256, 256],
            o_idx,
        ));

        let gate_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 1024 * 2]);
        entries.push((
            format!("model.layers.{l}.mlp.gate_proj.weight"),
            vec![256, 1024],
            gate_idx,
        ));

        let up_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 1024 * 2]);
        entries.push((
            format!("model.layers.{l}.mlp.up_proj.weight"),
            vec![256, 1024],
            up_idx,
        ));

        let down_idx = buffers.len();
        buffers.push(vec![0u8; 1024 * 256 * 2]);
        entries.push((
            format!("model.layers.{l}.mlp.down_proj.weight"),
            vec![1024, 256],
            down_idx,
        ));
    }

    let norm_idx = buffers.len();
    buffers.push(vec![0u8; 256 * 2]);
    entries.push(("model.norm.weight".to_string(), vec![256], norm_idx));

    let mut data_map = HashMap::new();
    for (name, shape, buf_idx) in &entries {
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F16,
            shape.clone(),
            &buffers[*buf_idx],
        )
        .expect("create TensorView");
        data_map.insert(name.clone(), view);
    }

    let serialized =
        safetensors::tensor::serialize(&data_map, &None).expect("serialize safetensors");
    let mut sf = File::create(model_dir.join("model.safetensors")).expect("create safetensors");
    sf.write_all(&serialized).expect("write safetensors");

    model_dir
}

#[test]
fn pathfinder_mock_model_matches_golden_json_and_table() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let model_dir = write_mock_llama(dir.path());

    let args = PathfinderArgs {
        model: model_dir,
        budget_gb: None,
        context: None,
        kv: None,
        json: true,
    };
    let run_plan = plan(&args).expect("plan mock model");
    let json_text = render_json(&run_plan).expect("render RunPlan JSON");

    let golden_text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/pathfinder_mock.json"
    ))
    .expect("read golden file");

    let actual_val: serde_json::Value =
        serde_json::from_str(&json_text).expect("parse actual JSON");
    let golden_val: serde_json::Value =
        serde_json::from_str(&golden_text).expect("parse golden JSON");
    assert_eq!(actual_val, golden_val);

    let golden_plan: RunPlan =
        serde_json::from_str(&golden_text).expect("deserialize golden RunPlan");
    assert_eq!(run_plan, golden_plan);
    assert_eq!(json_text.trim(), golden_text.replace("\r\n", "\n").trim());

    let table = render_table(&run_plan);
    assert!(table.contains("mock_llama"));
    assert!(table.contains("4096"));
    assert!(table.contains("F16"));
    assert!(table.contains("24252928"));
    assert!(table.contains("8388608"));
    assert!(table.contains("143360"));
    assert!(table.contains("32784896"));
    assert!(table.contains("predicted"));
    assert!(table.to_lowercase().contains("nothing was measured"));
}

#[test]
fn pathfinder_respects_fixed_flags_under_tight_budget() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let model_dir = write_mock_llama(dir.path());

    // 0.027 GiB = 28,991,029 bytes.
    // Unconstrained F16 total is 32,784,896 bytes; Int8 is 29,114,880 bytes; Int4 is 27,017,728 bytes.
    let auto_kv = plan(&PathfinderArgs {
        model: model_dir.clone(),
        budget_gb: Some(0.027),
        context: None,
        kv: None,
        json: false,
    })
    .expect("auto KV shrink should fit at Int4");
    assert_eq!(auto_kv.settings.kv, KvType::Int4);
    assert_eq!(auto_kv.settings.context, 4096);
    assert_eq!(auto_kv.total_bytes(), 27_017_728);

    // Fixing --kv f16 keeps KV at F16 and shrinks context 4096 -> 2048 (total 28,525,056 bytes).
    let fixed_kv = plan(&PathfinderArgs {
        model: model_dir,
        budget_gb: Some(0.027),
        context: None,
        kv: Some(KvArg::F16),
        json: false,
    })
    .expect("fixed F16 KV should shrink context to 2048");
    assert_eq!(fixed_kv.settings.kv, KvType::F16);
    assert_eq!(fixed_kv.settings.context, 2048);
    assert_eq!(fixed_kv.total_bytes(), 28_525_056);
}
