use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use thme::parser::safetensors::SafetensorsScanner;
use thme::parser::ModelImporter;

#[test]
fn test_synthetic_safetensors_import() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let model_dir = dir.path();

    // 1. Create a config.json
    let config_json = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 1000,
        "max_position_embeddings": 2048,
        "torch_dtype": "float16"
    });
    let mut config_file = File::create(model_dir.join("config.json"))?;
    config_file.write_all(config_json.to_string().as_bytes())?;

    // 2. Create a synthetic safetensors file
    let tensor1_data = vec![0u8; 128 * 128 * 2]; // 128x128 FP16
    let tensor2_data = vec![0u8; 128 * 256 * 2]; // 128x256 FP16
    let norm_data = vec![0u8; 128 * 2];

    let mut data_map = HashMap::new();
    let view1 = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F16,
        vec![128, 128],
        &tensor1_data,
    )?;
    let view2 = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F16,
        vec![128, 256],
        &tensor2_data,
    )?;
    let view3 = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F16,
        vec![128],
        &norm_data,
    )?;

    data_map.insert("model.layers.0.self_attn.q_proj.weight".to_string(), view1);
    data_map.insert("model.layers.0.mlp.down_proj.weight".to_string(), view2);
    data_map.insert("model.norm.weight".to_string(), view3);

    let serialized = safetensors::tensor::serialize(&data_map, &None)?;
    let mut sf_file = File::create(model_dir.join("model.safetensors"))?;
    sf_file.write_all(&serialized)?;

    // 3. Scan and verify
    let files = SafetensorsScanner::find_safetensors_files(model_dir)?;
    assert_eq!(files.len(), 1);

    let tensors = SafetensorsScanner::scan_model_dir(model_dir)?;
    assert_eq!(tensors.len(), 3);

    // 4. Test full model importer
    let model = ModelImporter::import_from_dir(model_dir)?;
    assert_eq!(model.num_layers, 2);
    assert_eq!(model.hidden_size, 128);
    assert_eq!(model.num_attention_heads, 4);
    assert_eq!(model.num_key_value_heads, 2);

    Ok(())
}
