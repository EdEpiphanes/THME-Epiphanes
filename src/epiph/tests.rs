// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::cli::evaluate::tests::synthetic_checkpoint;
use crate::inference::{
    collect_calibration, read_tensor_f32, read_token_file, ArchConfig, WeightPrecision,
};
use crate::parser::ModelImporter;
use serde_json::json;

fn packed(dir: &Path) -> (ModelIR, std::path::PathBuf, std::path::PathBuf) {
    let (model_dir, tokens) = synthetic_checkpoint(dir);
    let ir = ModelImporter::import(&model_dir).unwrap();
    let out = dir.join("model.epiph");
    let options = PackOptions {
        recipe: Some(json!({"name": "demo", "operations": []})),
        quality: vec![json!({"evidence": "tested", "gate": "demo", "verdict": "pass"})],
        annotations: BTreeMap::from([("owner".into(), "epiphanes".into())]),
        created_unix: Some(1_800_000_000),
        ..PackOptions::default()
    };
    write(&ir, &options, &out).unwrap();
    (ir, out, tokens)
}

/// Round trip: same model description, identical tensor bytes, same forward pass.
#[test]
fn round_trip_preserves_model_bytes_and_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let (ir, out, tokens) = packed(dir.path());
    let (back, manifest) = read(&out).unwrap();

    let strip = |m: &ModelIR| serde_json::to_value(m).unwrap();
    assert_eq!(strip(&back), strip(&ir), "model description survives");
    for (a, b) in ir.tensors.iter().zip(&back.tensors) {
        assert_eq!(
            read_tensor_bytes(a).unwrap(),
            read_tensor_bytes(b).unwrap(),
            "{}",
            a.name
        );
    }
    assert_eq!(manifest.recipe.unwrap()["name"], "demo");
    assert_eq!(manifest.annotations["owner"], "epiphanes");
    assert_eq!(manifest.provenance.created_unix, 1_800_000_000);
    assert_eq!(manifest.provenance.tool, "thme");
    assert!(!manifest.provenance.sources.is_empty());

    // Runs without any sibling config.json, through the ordinary importer.
    let imported = ModelImporter::import(&out).unwrap();
    let token_ids = read_token_file(&tokens).unwrap();
    let nll = |m: &ModelIR, path: &Path| {
        ArchConfig::detect(m, path)
            .unwrap()
            .load(m, WeightPrecision::AsStored)
            .unwrap()
            .forward_nll(&token_ids[..40])
            .unwrap()
    };
    assert_eq!(nll(&imported, &out), nll(&ir, &dir.path().join("model")));
    assert_eq!(verify(&out).unwrap().tensors, ir.tensors.len());
}

#[test]
fn header_and_layout_follow_the_specification() {
    let dir = tempfile::tempdir().unwrap();
    let (_, out, _) = packed(dir.path());
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(&bytes[..8], b"\x89EPIPH\r\n");
    assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 1);
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 0);
    let data = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    assert_eq!(data % 64, 0);
    let (_, manifest) = read(&out).unwrap();
    assert!(manifest.tensors.iter().all(|t| t.offset % 64 == 0));
    // No directory or user name leaks into the manifest.
    let text = String::from_utf8_lossy(&bytes[32..data as usize]).to_string();
    let home = dir.path().to_string_lossy().replace('\\', "\\\\");
    assert!(!text.contains(&home), "absolute path in manifest");
}

#[test]
fn calibration_travels_with_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let (model_dir, tokens) = synthetic_checkpoint(dir.path());
    let ir = ModelImporter::import(&model_dir).unwrap();
    let lm = ArchConfig::detect(&ir, &model_dir)
        .unwrap()
        .load(&ir, WeightPrecision::AsStored)
        .unwrap();
    let ids = read_token_file(&tokens).unwrap();
    let calib = collect_calibration(&*lm, &ir, &ids, 32, Some(64)).unwrap();
    let out = dir.path().join("cal.epiph");
    let options = PackOptions {
        calibration: Some(calib.clone()),
        created_unix: Some(0),
        ..PackOptions::default()
    };
    assert_eq!(write(&ir, &options, &out).unwrap().attachments, 1);
    assert_eq!(read_calibration(&out).unwrap(), Some(calib));
    assert_eq!(verify(&out).unwrap().attachments, 1);
}

#[test]
fn corruption_and_bad_headers_are_errors_not_panics() {
    let dir = tempfile::tempdir().unwrap();
    let (ir, out, _) = packed(dir.path());
    let good = std::fs::read(&out).unwrap();
    let data = u64::from_le_bytes(good[24..32].try_into().unwrap()) as usize;
    let try_bytes = |bytes: &[u8]| {
        let p = dir.path().join("bad.epiph");
        std::fs::write(&p, bytes).unwrap();
        (read(&p).err(), verify(&p).err())
    };
    // A flipped payload byte is caught by verify, naming the tensor.
    let mut flipped = good.clone();
    flipped[data] ^= 0xFF;
    let (_, err) = try_bytes(&flipped);
    assert!(format!("{:#}", err.unwrap()).contains(&ir.tensors[0].name));
    // Wrong magic, future major version, reserved flag.
    for (at, value) in [(1usize, b'X'), (8, 2), (12, 1)] {
        let mut bad = good.clone();
        bad[at] = value;
        assert!(try_bytes(&bad).0.is_some(), "byte {at}");
    }
    // Every truncation is refused without panicking.
    for cut in [0, 7, 31, 40, data - 1, data + 3, good.len() - 1] {
        let (r, v) = try_bytes(&good[..cut]);
        assert!(r.is_some() || v.is_some(), "cut at {cut}");
    }
    // A quality entry without an evidence label is refused at write time.
    let bad = PackOptions {
        quality: vec![json!({"verdict": "pass"})],
        ..PackOptions::default()
    };
    assert!(write(&ir, &bad, &dir.path().join("x.epiph")).is_err());
}

/// The format is a full citizen: converting an .epiph model to GGUF through the ordinary
/// exporter gives exactly the bytes that converting its source gives.
#[test]
fn epiph_converts_to_gguf_like_its_source() {
    let dir = tempfile::tempdir().unwrap();
    let (ir, out, _) = packed(dir.path());
    let (back, _) = read(&out).unwrap();
    let export = |m: &ModelIR, name: &str| {
        let path = dir.path().join(name);
        let config = crate::exporter::ExportConfig {
            format: crate::exporter::ExportFormat::GGUFv3,
            quant_type: crate::exporter::ExportQuantType::PerTensor,
            output_path: path.clone(),
            compute_checksum: false,
        };
        crate::exporter::ModelExporter::export(m, &config).unwrap();
        std::fs::read(path).unwrap()
    };
    let from_source = export(&ir, "source.gguf");
    let from_epiph = export(&back, "epiph.gguf");
    assert!(!from_source.is_empty());
    assert_eq!(from_epiph, from_source);
    let reimported = ModelImporter::import(dir.path().join("epiph.gguf")).unwrap();
    assert_eq!(reimported.tensors.len(), ir.tensors.len());
    assert!(read_tensor_f32(&reimported.tensors[0]).is_ok());
}

/// THME-3013 done-means: a recipe candidate packs with each tensor in its selected format; the
/// packed file runs exactly like the in-memory candidate and converts to the same GGUF bytes.
#[test]
fn quantized_pack_runs_like_the_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let (model_dir, tokens) = synthetic_checkpoint(dir.path());
    let ir = ModelImporter::import(&model_dir).unwrap();
    let recipe = crate::cli::evaluate::tests::write_recipe(dir.path(), false);
    let candidate = crate::cli::export::apply_recipe(&ir, &recipe, false).unwrap();

    let plain = dir.path().join("plain.epiph");
    let quant = dir.path().join("quant.epiph");
    write(&ir, &PackOptions::default(), &plain).unwrap();
    let options = PackOptions {
        store_selected: true,
        ..PackOptions::default()
    };
    write(&candidate, &options, &quant).unwrap();
    assert!(
        std::fs::metadata(&quant).unwrap().len() < std::fs::metadata(&plain).unwrap().len(),
        "quantized file is smaller"
    );
    assert_eq!(verify(&quant).unwrap().tensors, ir.tensors.len());

    let (packed, _) = read(&quant).unwrap();
    let down = packed
        .tensors
        .iter()
        .find(|t| t.name.contains("down_proj"))
        .unwrap();
    assert_eq!(down.original_dtype, "Q8_0");
    assert_eq!(down.metadata["epiph.source_dtype"], "F32");

    let ids = read_token_file(&tokens).unwrap();
    let nll = |m: &ModelIR, path: &Path, precision| {
        ArchConfig::detect(m, path)
            .unwrap()
            .load(m, precision)
            .unwrap()
            .forward_nll(&ids[..40])
            .unwrap()
    };
    assert_eq!(
        nll(&packed, &quant, WeightPrecision::AsStored),
        nll(&candidate, &model_dir, WeightPrecision::SelectedFormat),
        "packed weights equal the candidate's round-tripped weights"
    );

    let export = |m: &ModelIR, name: &str| {
        let path = dir.path().join(name);
        let config = crate::exporter::ExportConfig {
            format: crate::exporter::ExportFormat::GGUFv3,
            quant_type: crate::exporter::ExportQuantType::PerTensor,
            output_path: path.clone(),
            compute_checksum: false,
        };
        crate::exporter::ModelExporter::export(m, &config).unwrap();
        std::fs::read(path).unwrap()
    };
    assert_eq!(export(&packed, "a.gguf"), export(&candidate, "b.gguf"));
}

fn gguf_with_tokenizer(dir: &Path, template: &str) -> std::path::PathBuf {
    use crate::parser::gguf::{GgmlType, GgufValue as V};
    let mut w = crate::exporter::GgufWriter::new();
    w.add_metadata("general.architecture", V::String("llama".into()));
    w.add_metadata("general.name", V::String("tiny".into()));
    w.add_metadata("llama.block_count", V::Uint32(1));
    w.add_metadata("llama.rope.freq_base", V::Float32(10_000.0));
    w.add_metadata("test.u8", V::Uint8(7));
    w.add_metadata("test.i64", V::Int64(-5));
    w.add_metadata("test.f64", V::Float64(0.1));
    w.add_metadata("test.bool", V::Bool(true));
    w.add_metadata(
        "tokenizer.ggml.tokens",
        V::Array(vec![
            V::String("<s>".into()),
            V::String("hello".into()),
            V::String("exec(".into()),
        ]),
    );
    w.add_metadata(
        "tokenizer.ggml.scores",
        V::Array(vec![V::Float32(0.0), V::Float32(-1.5), V::Float32(-2.25)]),
    );
    w.add_metadata(
        "tokenizer.ggml.token_type",
        V::Array(vec![V::Int32(3), V::Int32(1), V::Int32(1)]),
    );
    w.add_metadata("tokenizer.chat_template", V::String(template.into()));
    let data: Vec<u8> = (0..64u32)
        .flat_map(|i| (i as f32 * 0.01).to_le_bytes())
        .collect();
    w.add_tensor("blk.0.attn_q.weight", vec![8, 8], GgmlType::F32, data);
    let path = dir.join("tiny.gguf");
    std::fs::write(&path, w.write_to_bytes().unwrap()).unwrap();
    path
}

/// THME-3014 done-means: GGUF -> .epiph -> GGUF keeps every metadata key with its exact type,
/// including the tokenizer; the GGUF written from the .epiph equals the one written from the
/// source.
#[test]
fn gguf_metadata_survives_the_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = gguf_with_tokenizer(dir.path(), "{{ messages }}");
    let source = ModelImporter::import(&gguf).unwrap();
    assert!(source
        .source_gguf_metadata
        .contains_key("tokenizer.ggml.tokens"));
    let packed = dir.path().join("tiny.epiph");
    write(&source, &PackOptions::default(), &packed).unwrap();
    let (back, manifest) = read(&packed).unwrap();
    assert!(manifest
        .attachments
        .iter()
        .any(|a| a.kind == "gguf-metadata"));
    assert_eq!(
        back.source_gguf_metadata, source.source_gguf_metadata,
        "every key and type"
    );

    let export = |m: &ModelIR, name: &str| {
        let path = dir.path().join(name);
        let config = crate::exporter::ExportConfig {
            format: crate::exporter::ExportFormat::GGUFv3,
            quant_type: crate::exporter::ExportQuantType::PerTensor,
            output_path: path.clone(),
            compute_checksum: false,
        };
        crate::exporter::ModelExporter::export(m, &config).unwrap();
        path
    };
    let a = export(&source, "a.gguf");
    let b = export(&back, "b.gguf");
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    let again = ModelImporter::import(&b).unwrap();
    assert_eq!(
        again.source_gguf_metadata["tokenizer.ggml.tokens"],
        source.source_gguf_metadata["tokenizer.ggml.tokens"]
    );
}

#[test]
fn hf_files_attach_extract_and_names_are_safe() {
    let dir = tempfile::tempdir().unwrap();
    let (model_dir, _) = synthetic_checkpoint(dir.path());
    std::fs::write(
        model_dir.join("tokenizer.json"),
        r#"{"model": {"type": "BPE"}}"#,
    )
    .unwrap();
    std::fs::write(
        model_dir.join("tokenizer_config.json"),
        r#"{"chat_template": "{{ x }}"}"#,
    )
    .unwrap();
    let ir = ModelImporter::import(&model_dir).unwrap();
    let names = ["config.json", "tokenizer.json", "tokenizer_config.json"];
    let options = PackOptions {
        hf_files: names.iter().map(|n| model_dir.join(n)).collect(),
        calibration: None,
        ..PackOptions::default()
    };
    let packed = dir.path().join("hf.epiph");
    assert_eq!(write(&ir, &options, &packed).unwrap().attachments, 3);
    assert!(
        verify(&packed).is_ok(),
        "several attachments do not overlap"
    );
    let out = dir.path().join("extracted");
    let written = extract_hf_files(&packed, &out).unwrap();
    assert_eq!(written.len(), 3);
    for n in names {
        assert_eq!(
            std::fs::read(out.join(n)).unwrap(),
            std::fs::read(model_dir.join(n)).unwrap(),
            "{n}"
        );
    }
    for bad in ["", ".", "..", "../x", "a/b", r"a\b", "C:x", "x\0y"] {
        assert!(!is_plain_file_name(bad), "{bad:?}");
    }
    assert!(is_plain_file_name("tokenizer.json"));
}

#[test]
fn scan_checks_templates_carried_inside_epiph_files() {
    let dir = tempfile::tempdir().unwrap();
    let evil = "{{ cycler.__init__.__globals__.os.popen('id').read() }}";
    let gguf = gguf_with_tokenizer(dir.path(), evil);
    let packed = dir.path().join("evil.epiph");
    write(
        &ModelImporter::import(&gguf).unwrap(),
        &PackOptions::default(),
        &packed,
    )
    .unwrap();
    let report = crate::scan::scan(&packed).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.rule == "template-injection"),
        "{:?}",
        report.findings
    );
    // The vocabulary token "exec(" is data, not a finding.
    assert!(!report
        .findings
        .iter()
        .any(|f| f.detail.contains("tokenizer.ggml.tokens")));
}
