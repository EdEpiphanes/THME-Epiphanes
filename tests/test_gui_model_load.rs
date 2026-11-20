// SPDX-License-Identifier: AGPL-3.0-only
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use thme::cli::epiph::{pack, PackArgs};
use thme::gui::UniversalModel;

fn ensure_test_model() -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("create temp dir");
    let model_path = dir.path().join("model");
    create_dir_all(&model_path).expect("create model dir");

    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 4,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "torch_dtype": "float16"
    });
    let mut f_config = File::create(model_path.join("config.json")).expect("create config");
    f_config
        .write_all(config.to_string().as_bytes())
        .expect("write config");

    let mut data_map = HashMap::new();
    let embed_data = vec![0u8; 256 * 64 * 2];
    let view_embed =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![256, 64], &embed_data)
            .expect("create embed view");
    data_map.insert("model.embed_tokens.weight".to_string(), view_embed);

    let layer_data = vec![0u8; 64 * 64 * 2];
    let view_layer =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![64, 64], &layer_data)
            .expect("create layer view");
    data_map.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        view_layer,
    );

    let serialized = safetensors::serialize(&data_map, &None).expect("serialize safetensors");
    let mut f_st = File::create(model_path.join("model.safetensors")).expect("create safetensors");
    f_st.write_all(&serialized).expect("write safetensors");

    (dir, model_path)
}

#[test]
fn test_gui_loads_mock_llama() {
    let (_tmp, mock_path) = if Path::new("examples/mock_llama").exists() {
        (
            tempdir().expect("tempdir"),
            PathBuf::from("examples/mock_llama"),
        )
    } else {
        ensure_test_model()
    };
    let model =
        UniversalModel::load_universal(&mock_path).expect("Failed to load mock llama in GUI");
    assert_eq!(model.num_layers, 4);
    assert!(!model.tensors.is_empty());
}

#[test]
fn test_gui_loads_epiph_container() {
    let (_tmp, mock_path) = if Path::new("examples/mock_llama").exists() {
        (
            tempdir().expect("tempdir"),
            PathBuf::from("examples/mock_llama"),
        )
    } else {
        ensure_test_model()
    };
    let out_dir = tempdir().expect("out dir");
    let temp_epiph = out_dir.path().join("test_gui.epiph");
    let args = PackArgs {
        model: mock_path,
        out: temp_epiph.clone(),
        recipe: None,
        calibration: None,
        quality: Vec::new(),
        parent: Vec::new(),
        annotate: Vec::new(),
        quantize: false,
        allow_partial: false,
        placement: None,
        compress: None,
    };
    pack(&args).expect("pack mock to epiph");

    let model = UniversalModel::load_universal(&temp_epiph).expect("Failed to load .epiph in GUI");
    assert_eq!(model.num_layers, 4);
    assert!(!model.tensors.is_empty());
}
