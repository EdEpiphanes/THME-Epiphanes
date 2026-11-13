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
