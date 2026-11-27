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
            llama_cpp_layout: false,
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
            llama_cpp_layout: false,
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
            llama_cpp_layout: false,
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
    // Scan on import refuses the injected template; the test builds the file on purpose.
    assert!(ModelImporter::import(&gguf).is_err());
    write(
        &ModelImporter::import_unscanned(&gguf).unwrap(),
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

fn imported(dir: &Path) -> ModelIR {
    let (model_dir, _) = synthetic_checkpoint(dir);
    ModelImporter::import(&model_dir).unwrap()
}

fn pack_placed(ir: &ModelIR, out: &Path, placement: Option<placement::Budget>) {
    let options = PackOptions {
        placement,
        created_unix: Some(0),
        ..PackOptions::default()
    };
    write(ir, &options, out).unwrap();
}

#[test]
fn placement_is_stored_and_read_back() {
    let dir = tempfile::tempdir().unwrap();
    let ir = imported(dir.path());
    let total: u64 = ir
        .tensors
        .iter()
        .map(|t| read_tensor_bytes(t).unwrap().len() as u64)
        .sum();
    let out = dir.path().join("placed.epiph");
    let budget = placement::Budget {
        vram_bytes: total - 1,
        host_bytes: 1 << 20,
    };
    pack_placed(&ir, &out, Some(budget));
    let (_, manifest) = read(&out).unwrap();
    let plan = manifest.placement.clone().expect("plan is stored");
    assert_eq!(plan.budget, budget);
    assert_eq!(plan.gpu_bytes + plan.cpu_bytes + plan.stream_bytes, total);
    let lengths: Vec<(String, u64)> = manifest
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.length))
        .collect();
    assert!(plan.check(&lengths).is_ok(), "plan is in tensor order");
    assert_eq!(verify(&out).unwrap().placement, Some(plan));
}

#[test]
fn files_without_a_plan_are_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let ir = imported(dir.path());
    let out = dir.path().join("plain.epiph");
    pack_placed(&ir, &out, None);
    let (_, manifest) = read(&out).unwrap();
    assert!(manifest.placement.is_none());
    assert!(verify(&out).unwrap().placement.is_none());
    let bytes = std::fs::read(&out).unwrap();
    let data = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    let text = String::from_utf8_lossy(&bytes[32..data as usize]).to_string();
    assert!(!text.contains("placement"));
}

#[test]
fn a_plan_that_disagrees_with_the_tensors_fails_verify() {
    let dir = tempfile::tempdir().unwrap();
    let ir = imported(dir.path());
    let out = dir.path().join("placed.epiph");
    let budget = placement::Budget {
        vram_bytes: 1 << 40,
        host_bytes: 1 << 20,
    };
    pack_placed(&ir, &out, Some(budget));
    let mut bytes = std::fs::read(&out).unwrap();
    let needle = br#""cpu_bytes":0"#;
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap();
    bytes[at + needle.len() - 1] = b'7';
    let bad = dir.path().join("bad.epiph");
    std::fs::write(&bad, &bytes).unwrap();
    let err = verify(&bad).unwrap_err();
    assert!(format!("{err:#}").contains("placement"), "{err:#}");
}

#[test]
fn format_version_is_1_3_and_older_files_still_verify() {
    let dir = tempfile::tempdir().unwrap();
    let ir = imported(dir.path());
    let out = dir.path().join("plain.epiph");
    pack_placed(&ir, &out, None);
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(u16::from_le_bytes([bytes[10], bytes[11]]), 3);
    assert_eq!(read(&out).unwrap().1.version, "1.3");

    // The same file as a 1.2 writer would have left it: minor 2, manifest "1.2". Identity
    // payloads never carry the 1.3 fields, so nothing else differs.
    let mut old = bytes.clone();
    old[10] = 2;
    let needle = br#""version":"1.3""#;
    let at = old.windows(needle.len()).position(|w| w == needle).unwrap();
    old[at + needle.len() - 2] = b'2';
    let old_path = dir.path().join("old.epiph");
    std::fs::write(&old_path, &old).unwrap();
    let (back, manifest) = read(&old_path).unwrap();
    assert_eq!(manifest.version, "1.2");
    assert!(manifest.placement.is_none());
    assert!(manifest.tensors.iter().all(|t| t.decoded_length.is_none()));
    assert!(verify(&old_path).is_ok());
    for (a, b) in ir.tensors.iter().zip(&back.tensors) {
        assert_eq!(read_tensor_bytes(a).unwrap(), read_tensor_bytes(b).unwrap());
    }
}

#[test]
fn signed_file_with_a_plan_verifies() {
    let dir = tempfile::tempdir().unwrap();
    let ir = imported(dir.path());
    let out = dir.path().join("signed.epiph");
    let budget = placement::Budget {
        vram_bytes: 1 << 40,
        host_bytes: 1 << 20,
    };
    pack_placed(&ir, &out, Some(budget));
    let pair = minisign::KeyPair::generate_unencrypted_keypair().unwrap();
    sign::sign(&out, &pair.sk, Some(&pair.pk)).unwrap();
    assert!(matches!(
        sign::check(&out, Some(&pair.pk)).unwrap(),
        sign::SignatureStatus::Valid { .. }
    ));
    assert!(verify(&out).unwrap().placement.is_some());
}

// ---- THME-3015: lossless payload compression (`encoding` deflate) ----

const RAMP_LEN: usize = 4096;

/// Compressible F32 payload: a short repeating pattern.
fn ramp_bytes() -> Vec<u8> {
    (0..RAMP_LEN / 4)
        .flat_map(|i| ((i % 7) as f32).to_le_bytes())
        .collect()
}

/// Incompressible F32 payload from a fixed xorshift stream. Every word is kept finite (an
/// all-ones exponent is NaN or infinity), so the Q8_0 conversion accepts it.
fn noise_bytes() -> Vec<u8> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut out: Vec<u8> = (0..RAMP_LEN)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect();
    for word in out.as_chunks_mut::<4>().0 {
        if word[3] & 0x7F == 0x7F && word[2] & 0x80 != 0 {
            word[3] &= 0xBF;
        }
    }
    out
}

/// The synthetic checkpoint's description with its tensors replaced by F32 payloads stored in
/// files under `dir`.
fn model_with(dir: &Path, payloads: &[(&str, Vec<u8>)]) -> ModelIR {
    let mut ir = imported(dir);
    ir.tensors = payloads
        .iter()
        .map(|(name, bytes)| {
            let path = dir.join(format!("{name}.raw"));
            std::fs::write(&path, bytes).unwrap();
            let mut tensor = crate::ir::TensorIR::new(
                (*name).to_string(),
                vec![bytes.len() / 4],
                "F32".into(),
                bytes.len(),
            );
            tensor.source = Some(TensorSource {
                path,
                byte_offset: 0,
                byte_len: bytes.len() as u64,
                encoding: crate::ir::tensor::SourceEncoding::Identity,
            });
            tensor
        })
        .collect();
    ir
}

fn options(compress: Option<Encoding>) -> PackOptions {
    PackOptions {
        compress,
        created_unix: Some(0),
        ..PackOptions::default()
    }
}

/// Writes a one-tensor file by hand: `manifest` as given, then `stored` as the only payload.
fn forge(path: &Path, manifest: &Manifest, stored: &[u8]) {
    let json = serde_json::to_vec(manifest).unwrap();
    let data = align(HEADER_LEN + json.len() as u64);
    let mut bytes = MAGIC.to_vec();
    bytes.extend(MAJOR.to_le_bytes());
    bytes.extend(MINOR.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend((json.len() as u64).to_le_bytes());
    bytes.extend(data.to_le_bytes());
    bytes.extend(&json);
    bytes.resize(data as usize, 0);
    bytes.extend(stored);
    std::fs::write(path, bytes).unwrap();
}

/// Forges `out` from the one-tensor file `base`: its tensor is declared DEFLATE with `stored`
/// as payload, the given decoded length and decoded digest.
fn forge_deflate(
    base: &Path,
    out: &Path,
    stored: &[u8],
    decoded_length: u64,
    decoded_sha256: String,
) {
    let (_, mut manifest) = read(base).unwrap();
    let entry = &mut manifest.tensors[0];
    entry.encoding = Encoding::Deflate;
    entry.length = stored.len() as u64;
    entry.sha256 = compute_sha256(stored);
    entry.decoded_length = Some(decoded_length);
    entry.decoded_sha256 = Some(decoded_sha256);
    forge(out, &manifest, stored);
}

#[test]
fn deflate_round_trip_is_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    let ir = model_with(
        dir.path(),
        &[
            ("ramp.weight", ramp_bytes()),
            ("noise.weight", noise_bytes()),
        ],
    );
    let plain = dir.path().join("plain.epiph");
    let small = dir.path().join("small.epiph");
    write(&ir, &options(None), &plain).unwrap();
    let summary = write(&ir, &options(Some(Encoding::Deflate)), &small).unwrap();

    let (back, manifest) = read(&small).unwrap();
    assert_eq!(manifest.tensors[0].encoding, Encoding::Deflate);
    assert_eq!(manifest.tensors[1].encoding, Encoding::Identity);
    assert_eq!(manifest.tensors[0].decoded_length, Some(RAMP_LEN as u64));
    for (original, stored) in ir.tensors.iter().zip(&back.tensors) {
        assert_eq!(
            read_tensor_bytes(stored).unwrap(),
            read_tensor_bytes(original).unwrap(),
            "{}",
            original.name
        );
    }
    assert_eq!(read_tensor_bytes(&back.tensors[0]).unwrap(), ramp_bytes());
    assert!(verify(&small).is_ok());
    assert!(std::fs::metadata(&small).unwrap().len() < std::fs::metadata(&plain).unwrap().len());
    assert_eq!(summary.compressed_tensors, 1);
}

#[test]
fn incompressible_payloads_stay_identity() {
    let dir = tempfile::tempdir().unwrap();
    let ir = model_with(dir.path(), &[("noise.weight", noise_bytes())]);
    let out = dir.path().join("noise.epiph");
    let summary = write(&ir, &options(Some(Encoding::Deflate)), &out).unwrap();
    let (back, manifest) = read(&out).unwrap();
    assert_eq!(manifest.tensors[0].encoding, Encoding::Identity);
    assert_eq!(manifest.tensors[0].decoded_length, None);
    assert_eq!(manifest.tensors[0].length, RAMP_LEN as u64);
    assert_eq!(summary.compressed_tensors, 0);
    assert_eq!(summary.tensor_stored_bytes, summary.tensor_decoded_bytes);
    assert_eq!(read_tensor_bytes(&back.tensors[0]).unwrap(), noise_bytes());
}

#[test]
fn pack_summary_reports_stored_decoded_and_ratio() {
    let dir = tempfile::tempdir().unwrap();
    let ir = model_with(
        dir.path(),
        &[
            ("ramp.weight", ramp_bytes()),
            ("noise.weight", noise_bytes()),
        ],
    );
    let out = dir.path().join("ratio.epiph");
    let summary = write(&ir, &options(Some(Encoding::Deflate)), &out).unwrap();
    assert_eq!(summary.compressed_tensors, 1);
    assert_eq!(summary.tensor_decoded_bytes, (RAMP_LEN * 2) as u64);
    assert!(summary.tensor_stored_bytes < summary.tensor_decoded_bytes);
    assert!(summary.ratio() > 1.0);
    let plain = write(&ir, &options(None), &dir.path().join("p.epiph")).unwrap();
    assert_eq!(plain.ratio(), 1.0);
}

#[test]
fn inflate_refuses_more_or_less_than_declared() {
    let zeros = vec![0u8; 1 << 20];
    let stream = deflate_payload(&zeros).unwrap();
    assert_eq!(inflate_payload(&stream, 1 << 20).unwrap(), zeros);
    let too_small = format!("{:#}", inflate_payload(&stream, 1024).unwrap_err());
    assert!(too_small.contains("declared 1024"), "{too_small}");
    assert!(inflate_payload(&stream, (1 << 20) + 1).is_err());
    assert!(inflate_payload(&stream[..stream.len() - 4], 1 << 20).is_err());
    let mut padded = stream.clone();
    padded.extend_from_slice(b"junk");
    let trailing = format!("{:#}", inflate_payload(&padded, 1 << 20).unwrap_err());
    assert!(trailing.contains("trailing"), "{trailing}");
}

#[test]
fn a_decompression_bomb_is_refused_at_its_declared_length() {
    let dir = tempfile::tempdir().unwrap();
    let ir = model_with(dir.path(), &[("solo.weight", ramp_bytes())]);
    let base = dir.path().join("base.epiph");
    write(&ir, &options(None), &base).unwrap();
    // About 1 KiB stored, expanding to 1 MiB, under the declared length of the tensor (4096).
    let bomb = deflate_payload(&vec![0u8; 1 << 20]).unwrap();
    assert!(bomb.len() < RAMP_LEN);
    let out = dir.path().join("bomb.epiph");
    forge_deflate(
        &base,
        &out,
        &bomb,
        RAMP_LEN as u64,
        compute_sha256(&vec![0u8; RAMP_LEN]),
    );
    let err = format!("{:#}", verify(&out).unwrap_err());
    assert!(err.contains("solo.weight"), "{err}");
    assert!(err.contains("declared"), "{err}");
    let (back, _) = read(&out).unwrap();
    assert!(read_tensor_bytes(&back.tensors[0]).is_err());
    let report = crate::scan::scan(&out).unwrap();
    assert_ne!(report.verdict(), "no findings");
}

#[test]
fn absurd_declared_lengths_are_refused_before_any_allocation() {
    let dir = tempfile::tempdir().unwrap();
    let ir = model_with(dir.path(), &[("solo.weight", ramp_bytes())]);
    let base = dir.path().join("base.epiph");
    write(&ir, &options(None), &base).unwrap();
    let stored = deflate_payload(&ramp_bytes()).unwrap();
    for (name, declared) in [
        ("max", u64::MAX),
        ("cap", 1 << 33),
        ("ratio", stored.len() as u64 * 1032 + 1),
    ] {
        let out = dir.path().join(format!("{name}.epiph"));
        forge_deflate(
            &base,
            &out,
            &stored,
            declared,
            compute_sha256(&ramp_bytes()),
        );
        let err = format!("{:#}", verify(&out).unwrap_err());
        assert!(err.contains("solo.weight"), "{name}: {err}");
    }
}

#[test]
fn truncated_and_padded_streams_are_clear_errors() {
    let dir = tempfile::tempdir().unwrap();
    let ir = model_with(dir.path(), &[("solo.weight", ramp_bytes())]);
    let base = dir.path().join("base.epiph");
    write(&ir, &options(None), &base).unwrap();
    let honest = deflate_payload(&ramp_bytes()).unwrap();
    let digest = compute_sha256(&ramp_bytes());

    let truncated = dir.path().join("truncated.epiph");
    let cut = &honest[..honest.len() - 4];
    forge_deflate(&base, &truncated, cut, RAMP_LEN as u64, digest.clone());
    let err = format!("{:#}", verify(&truncated).unwrap_err());
    assert!(err.contains("solo.weight"), "{err}");

    let mut padded_stream = honest.clone();
    padded_stream.extend_from_slice(b"junk");
    let padded = dir.path().join("padded.epiph");
    forge_deflate(&base, &padded, &padded_stream, RAMP_LEN as u64, digest);
    let err = format!("{:#}", verify(&padded).unwrap_err());
    assert!(
        err.contains("trailing") && err.contains("solo.weight"),
        "{err}"
    );
}

#[test]
fn digest_mismatches_name_the_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let ir = model_with(
        dir.path(),
        &[
            ("ramp.weight", ramp_bytes()),
            ("noise.weight", noise_bytes()),
        ],
    );
    let out = dir.path().join("digests.epiph");
    write(&ir, &options(Some(Encoding::Deflate)), &out).unwrap();
    // A flipped byte in the stored (compressed) ramp payload fails its stored digest.
    let mut bytes = std::fs::read(&out).unwrap();
    let data = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
    bytes[data] ^= 0xFF;
    let bad = dir.path().join("flipped.epiph");
    std::fs::write(&bad, &bytes).unwrap();
    let err = format!("{:#}", verify(&bad).unwrap_err());
    assert!(err.contains("ramp.weight"), "{err}");

    // An honest stored payload whose decoded digest is wrong fails the decoded check, by name.
    let base = dir.path().join("base.epiph");
    write(
        &model_with(dir.path(), &[("solo.weight", ramp_bytes())]),
        &options(None),
        &base,
    )
    .unwrap();
    let honest = deflate_payload(&ramp_bytes()).unwrap();
    let wrong = dir.path().join("wrong.epiph");
    forge_deflate(
        &base,
        &wrong,
        &honest,
        RAMP_LEN as u64,
        compute_sha256(&noise_bytes()),
    );
    let err = format!("{:#}", verify(&wrong).unwrap_err());
    assert!(
        err.contains("solo.weight") && err.contains("SHA-256"),
        "{err}"
    );
}

#[test]
fn signed_file_with_compressed_payloads_verifies() {
    let dir = tempfile::tempdir().unwrap();
    let ir = model_with(
        dir.path(),
        &[
            ("ramp.weight", ramp_bytes()),
            ("noise.weight", noise_bytes()),
        ],
    );
    let out = dir.path().join("signed.epiph");
    write(&ir, &options(Some(Encoding::Deflate)), &out).unwrap();
    let (_, manifest) = read(&out).unwrap();
    assert_eq!(manifest.tensors[0].encoding, Encoding::Deflate);
    let pair = minisign::KeyPair::generate_unencrypted_keypair().unwrap();
    sign::sign(&out, &pair.sk, Some(&pair.pk)).unwrap();
    assert!(matches!(
        sign::check(&out, Some(&pair.pk)).unwrap(),
        sign::SignatureStatus::Valid { .. }
    ));
    assert!(verify(&out).is_ok());
}

#[test]
fn compressed_file_converts_to_the_same_gguf_bytes() {
    use crate::exporter::ExportQuantType;
    let dir = tempfile::tempdir().unwrap();
    // Two compressible tensors with small finite values: Q8_0 needs scales that fit in FP16.
    let steps: Vec<u8> = (0..RAMP_LEN / 4)
        .flat_map(|i| (((i * 3) % 11) as f32 * 0.5).to_le_bytes())
        .collect();
    let ir = model_with(
        dir.path(),
        &[("ramp.weight", ramp_bytes()), ("steps.weight", steps)],
    );
    let plain = dir.path().join("plain.epiph");
    let small = dir.path().join("small.epiph");
    write(&ir, &options(None), &plain).unwrap();
    write(&ir, &options(Some(Encoding::Deflate)), &small).unwrap();
    let (back_plain, _) = read(&plain).unwrap();
    let (back_small, small_manifest) = read(&small).unwrap();
    assert!(small_manifest
        .tensors
        .iter()
        .all(|t| t.encoding == Encoding::Deflate));
    let export = |m: &ModelIR, name: &str, quant: ExportQuantType| {
        let path = dir.path().join(name);
        let config = crate::exporter::ExportConfig {
            format: crate::exporter::ExportFormat::GGUFv3,
            quant_type: quant,
            output_path: path.clone(),
            compute_checksum: false,
            llama_cpp_layout: false,
        };
        crate::exporter::ModelExporter::export(m, &config).unwrap();
        std::fs::read(path).unwrap()
    };
    for (quant, label) in [
        (ExportQuantType::PerTensor, "per-tensor"),
        (ExportQuantType::Q8_0, "q8_0"),
    ] {
        let from_plain = export(&back_plain, &format!("plain-{label}.gguf"), quant);
        let from_small = export(&back_small, &format!("small-{label}.gguf"), quant);
        assert!(!from_plain.is_empty());
        assert_eq!(from_small, from_plain, "{label}");
    }
}
