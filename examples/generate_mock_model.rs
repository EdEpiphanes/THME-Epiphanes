// SPDX-License-Identifier: AGPL-3.0-only
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let out_dir = Path::new("examples/mock_llama");
    create_dir_all(out_dir)?;

    // 1. Create config.json
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
    let mut f_config = File::create(out_dir.join("config.json"))?;
    f_config.write_all(config.to_string().as_bytes())?;

    // 2. Create mock safetensors weights - keep data alive outside loop
    let mut buffers: Vec<Vec<u8>> = Vec::new();
    let mut entries: Vec<(String, Vec<usize>, usize)> = Vec::new();

    // Embedding
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
            format!("model.layers.{}.self_attn.q_proj.weight", l),
            vec![256, 256],
            q_idx,
        ));

        let k_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 128 * 2]);
        entries.push((
            format!("model.layers.{}.self_attn.k_proj.weight", l),
            vec![256, 128],
            k_idx,
        ));

        let v_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 128 * 2]);
        entries.push((
            format!("model.layers.{}.self_attn.v_proj.weight", l),
            vec![256, 128],
            v_idx,
        ));

        let o_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 256 * 2]);
        entries.push((
            format!("model.layers.{}.self_attn.o_proj.weight", l),
            vec![256, 256],
            o_idx,
        ));

        let gate_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 1024 * 2]);
        entries.push((
            format!("model.layers.{}.mlp.gate_proj.weight", l),
            vec![256, 1024],
            gate_idx,
        ));

        let up_idx = buffers.len();
        buffers.push(vec![0u8; 256 * 1024 * 2]);
        entries.push((
            format!("model.layers.{}.mlp.up_proj.weight", l),
            vec![256, 1024],
            up_idx,
        ));

        let down_idx = buffers.len();
        buffers.push(vec![0u8; 1024 * 256 * 2]);
        entries.push((
            format!("model.layers.{}.mlp.down_proj.weight", l),
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
        )?;
        data_map.insert(name.clone(), view);
    }

    let serialized = safetensors::tensor::serialize(&data_map, &None)?;
    let mut sf = File::create(out_dir.join("model.safetensors"))?;
    sf.write_all(&serialized)?;

    println!("Mock LLaMA created successfully at examples/mock_llama");
    Ok(())
}
