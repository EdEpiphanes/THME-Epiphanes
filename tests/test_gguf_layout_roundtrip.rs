// SPDX-License-Identifier: AGPL-3.0-only
//! A GGUF in llama.cpp layout has permuted q/k rows. THME's reader must undo that, so the logits
//! after export and re-import equal the logits of the original checkpoint (THME-3103).

use thme::exporter::safetensors_writer::SafeTensorsWriter;
use thme::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
use thme::inference::{ArchConfig, WeightPrecision};
use thme::ir::model::ModelMetadataValue;
use thme::ir::{ModelArchitecture, ModelIR, TensorFormat};
use thme::parser::{ModelImporter, SafetensorsScanner};

const HIDDEN: usize = 8;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const INTER: usize = 16;
const VOCAB: usize = 10;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Deterministic values in [-scale, scale].
fn values(len: usize, seed: &mut u64, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((*seed >> 33) as f32 / 2_147_483_648.0) - 1.0) * scale
        })
        .collect()
}

fn write_tiny_checkpoint(dir: &std::path::Path) -> ModelIR {
    let kv_dim = HIDDEN / HEADS * KV_HEADS;
    let mut seed = 99u64;
    let mut w = SafeTensorsWriter::new();
    let mut add = |name: &str, shape: Vec<usize>, scale: f32, ones: bool| {
        let n: usize = shape.iter().product();
        let v = if ones {
            vec![1.0; n]
        } else {
            values(n, &mut seed, scale)
        };
        w.add_tensor(name, "F32", shape, f32_bytes(&v));
    };
    add("model.embed_tokens.weight", vec![VOCAB, HIDDEN], 0.5, false);
    add(
        "model.layers.0.input_layernorm.weight",
        vec![HIDDEN],
        0.0,
        true,
    );
    add(
        "model.layers.0.self_attn.q_proj.weight",
        vec![HIDDEN, HIDDEN],
        0.5,
        false,
    );
    add(
        "model.layers.0.self_attn.k_proj.weight",
        vec![kv_dim, HIDDEN],
        0.5,
        false,
    );
    add(
        "model.layers.0.self_attn.v_proj.weight",
        vec![kv_dim, HIDDEN],
        0.5,
        false,
    );
    add(
        "model.layers.0.self_attn.o_proj.weight",
        vec![HIDDEN, HIDDEN],
        0.5,
        false,
    );
    add(
        "model.layers.0.post_attention_layernorm.weight",
        vec![HIDDEN],
        0.0,
        true,
    );
    add(
        "model.layers.0.mlp.gate_proj.weight",
        vec![INTER, HIDDEN],
        0.5,
        false,
    );
    add(
        "model.layers.0.mlp.up_proj.weight",
        vec![INTER, HIDDEN],
        0.5,
        false,
    );
    add(
        "model.layers.0.mlp.down_proj.weight",
        vec![HIDDEN, INTER],
        0.5,
        false,
    );
    add("model.norm.weight", vec![HIDDEN], 0.0, true);
    add("lm_head.weight", vec![VOCAB, HIDDEN], 0.5, false);
    let path = dir.join("tiny.safetensors");
    w.write_to_file(&path).expect("write checkpoint");
    let tensors = SafetensorsScanner::parse_metadata_from_file(&path).expect("scan checkpoint");
    ModelIR::new(
        "tiny".into(),
        ModelArchitecture::Llama,
        1,
        HIDDEN,
        HEADS,
        KV_HEADS,
        VOCAB,
        32,
        tensors,
    )
}

fn write_tokenizer(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("tokenizer.json");
    std::fs::write(
        &path,
        r#"{"model":{"type":"BPE","vocab":{"<s>":0,"</s>":1,"a":2,"b":3,"ab":4},"merges":["a b"]},
"added_tokens":[{"id":0,"content":"<s>","special":true},{"id":1,"content":"</s>","special":true}],
"pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"}}"#,
    )
    .expect("write tokenizer");
    path
}

fn logits(model: &ModelIR, path: &std::path::Path, tokens: &[u32]) -> Vec<f32> {
    let config = ArchConfig::detect(model, path).expect("detect architecture");
    let loaded = config
        .load(model, WeightPrecision::AsStored)
        .expect("load weights");
    loaded.forward_logits(tokens).expect("forward")
}

fn export_tiny_gguf(
    dir: &std::path::Path,
    llama_cpp_layout: bool,
) -> (ModelIR, std::path::PathBuf) {
    let mut source = write_tiny_checkpoint(dir);
    // Keep every matrix in f32 so rounding to f16 cannot hide or fake a layout error.
    for tensor in &mut source.tensors {
        tensor.selected_format = TensorFormat::FP32;
    }
    // Real checkpoints record their norm epsilon in config.json; the test does the same so the
    // only difference between the two loads is the file layout.
    std::fs::write(
        dir.join("config.json"),
        r#"{"rms_norm_eps": 1e-5, "rope_theta": 10000.0}"#,
    )
    .expect("write config.json");

    let gguf = dir.join("tiny.gguf");
    ModelExporter::export(
        &source,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::PerTensor,
            output_path: gguf.clone(),
            compute_checksum: false,
            llama_cpp_layout,
            tokenizer_json_path: llama_cpp_layout.then(|| write_tokenizer(dir)),
        },
    )
    .expect("export GGUF");

    (source, gguf)
}

fn assert_logits_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "largest logit gap {worst}");
}

#[test]
fn a_llama_cpp_layout_gguf_reads_back_to_the_checkpoints_logits() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (source, gguf) = export_tiny_gguf(dir.path(), true);
    let tokens = [1u32, 5, 3, 7, 2];
    let want = logits(&source, dir.path(), &tokens);

    let reread = ModelImporter::import(&gguf).expect("import the GGUF");
    let got = logits(&reread, &gguf, &tokens);
    assert_logits_close(&got, &want);
}

#[test]
fn a_gguf_with_hugging_face_names_is_not_unpermuted() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (source, gguf) = export_tiny_gguf(dir.path(), false);
    let tokens = [1u32, 5, 3, 7, 2];
    let want = logits(&source, dir.path(), &tokens);

    let reread = ModelImporter::import(&gguf).expect("import the GGUF");
    let got = logits(&reread, &gguf, &tokens);
    assert_logits_close(&got, &want);
}

#[test]
fn the_unpermute_is_keyed_on_the_llama_architecture_string() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (_, gguf) = export_tiny_gguf(dir.path(), true);

    let reread = ModelImporter::import(&gguf).expect("import the GGUF");
    assert_eq!(
        reread.source_gguf_metadata.get("general.architecture"),
        Some(&ModelMetadataValue::String("llama".to_string()))
    );
}

#[test]
fn the_round_trip_holds_for_several_token_sequences() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (source, gguf) = export_tiny_gguf(dir.path(), true);
    let reread = ModelImporter::import(&gguf).expect("import the GGUF");

    let sequences: &[&[u32]] = &[
        &[0, 9],
        &[2, 4, 6],
        &[9, 1, 8, 0],
        &[1, 5, 3, 7, 2],
        &[3, 0, 7, 4, 8, 5],
    ];
    for &tokens in sequences {
        let want = logits(&source, dir.path(), tokens);
        let got = logits(&reread, &gguf, tokens);
        assert_logits_close(&got, &want);
    }
}
