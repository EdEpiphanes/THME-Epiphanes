// SPDX-License-Identifier: AGPL-3.0-only
//! Fixtures are built in temporary folders from bytes. They imitate the shape of known attack
//! patterns (an `os.system` import, a template-injection string, an ELF header) but contain no
//! working payload, and nothing is ever loaded or executed.

use super::*;
use crate::cli::evaluate::tests::synthetic_checkpoint;
use crate::exporter::GgufWriter;
use crate::parser::gguf::{GgmlType, GgufValue};

fn rules(report: &ScanReport) -> Vec<&'static str> {
    report.findings.iter().map(|f| f.rule).collect()
}

fn fake_elf() -> Vec<u8> {
    let mut b = b"\x7fELF".to_vec();
    b.extend_from_slice(&[0u8; 60]);
    b
}

fn zip_entries(entries: &[(&str, u16, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, method, data) in entries {
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0u8; 8]); // time, date, crc
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
    }
    out
}

fn zip_stored(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let items: Vec<(&str, u16, &[u8])> = entries.iter().map(|(n, d)| (*n, 0, *d)).collect();
    zip_entries(&items)
}

/// Deflates each payload, the way the PyTorch container stores pickles.
fn zip_deflated(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let items: Vec<(&str, u16, Vec<u8>)> =
        entries.iter().map(|(n, d)| (*n, 8, deflate(d))).collect();
    let refs: Vec<(&str, u16, &[u8])> = items
        .iter()
        .map(|(n, m, d)| (*n, *m, d.as_slice()))
        .collect();
    zip_entries(&refs)
}

fn deflate(data: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

/// Deflates `len` zero bytes without holding them uncompressed; a zip-bomb stand-in.
fn deflate_zeros(len: u64) -> Vec<u8> {
    use std::io::Write;
    let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
    let chunk = [0u8; 1 << 16];
    let mut left = len;
    while left > 0 {
        let take = left.min(chunk.len() as u64) as usize;
        enc.write_all(&chunk[..take]).unwrap();
        left -= take as u64;
    }
    enc.finish().unwrap()
}

fn gguf_bytes(template: Option<&str>) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.add_metadata("general.architecture", GgufValue::String("llama".into()));
    if let Some(t) = template {
        w.add_metadata("tokenizer.chat_template", GgufValue::String(t.into()));
    }
    let data: Vec<u8> = (0..64u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    w.add_tensor("blk.0.attn_q.weight", vec![8, 8], GgmlType::F32, data);
    w.write_to_bytes().unwrap()
}

#[test]
fn clean_model_folder_has_no_findings() {
    let dir = tempfile::tempdir().unwrap();
    let (model, _) = synthetic_checkpoint(dir.path());
    let report = scan(&model).unwrap();
    assert!(report.files_scanned > 0);
    assert_eq!(report.verdict(), "no findings", "{:?}", report.findings);
}

#[test]
fn pickles_that_import_code_are_critical_and_torch_pickles_are_not() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("plain.pkl"),
        b"cos\nsystem\n(S'echo fixture'\ntR.",
    )
    .unwrap();
    // PyTorch-style zip with a STACK_GLOBAL builtins.eval pickle.
    let mut evil = vec![0x80, 4];
    for s in ["builtins", "eval"] {
        evil.push(0x8c);
        evil.push(s.len() as u8);
        evil.extend_from_slice(s.as_bytes());
    }
    evil.extend_from_slice(b"\x93\x8c\x011\x85R.");
    std::fs::write(
        dir.path().join("model.pt"),
        zip_stored(&[("archive/data.pkl", &evil)]),
    )
    .unwrap();
    // A benign checkpoint pickle: torch rebuild helpers and an OrderedDict.
    let benign =
        b"\x80\x02ccollections\nOrderedDict\nq\x00)Rq\x01ctorch._utils\n_rebuild_tensor_v2\nq\x02.";
    std::fs::write(
        dir.path().join("ok.pt"),
        zip_stored(&[("archive/data.pkl", benign)]),
    )
    .unwrap();

    let report = scan(dir.path()).unwrap();
    let critical: Vec<&str> = report
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Critical)
        .map(|f| f.file.as_str())
        .collect();
    assert!(critical.contains(&"plain.pkl"), "{:?}", report.findings);
    assert!(critical.contains(&"model.pt!archive/data.pkl"));
    assert!(
        !report.findings.iter().any(|f| f.file.starts_with("ok.pt")),
        "{:?}",
        report.findings
    );
    assert_eq!(report.verdict(), "dangerous");
}

#[test]
fn hidden_bytes_and_template_injection_in_gguf_are_found() {
    let dir = tempfile::tempdir().unwrap();
    let clean = dir.path().join("clean.gguf");
    std::fs::write(
        &clean,
        gguf_bytes(Some("{% for m in messages %}{{ m.content }}{% endfor %}")),
    )
    .unwrap();
    assert_eq!(scan(&clean).unwrap().verdict(), "no findings");

    let mut smuggled = gguf_bytes(None);
    smuggled.extend_from_slice(&[0u8; 64]);
    smuggled.extend_from_slice(&fake_elf());
    let path = dir.path().join("smuggled.gguf");
    std::fs::write(&path, smuggled).unwrap();
    assert!(rules(&scan(&path).unwrap()).contains(&"hidden-executable"));

    let mut padded = gguf_bytes(None);
    padded.extend_from_slice(&[7u8; 100]);
    let path = dir.path().join("padded.gguf");
    std::fs::write(&path, padded).unwrap();
    assert!(rules(&scan(&path).unwrap()).contains(&"unaccounted-bytes"));

    let injected = "{{ cycler.__init__.__globals__.os.popen('id').read() }}";
    let path = dir.path().join("template.gguf");
    std::fs::write(&path, gguf_bytes(Some(injected))).unwrap();
    let report = scan(&path).unwrap();
    assert!(
        rules(&report).contains(&"template-injection"),
        "{:?}",
        report.findings
    );
}

#[test]
fn remote_code_executables_and_broken_files_are_reported_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(
        root.join("config.json"),
        r#"{"auto_map": {"AutoModel": "modeling_x.Model"}}"#,
    )
    .unwrap();
    std::fs::write(root.join("modeling_x.py"), "class Model: pass\n").unwrap();
    std::fs::write(root.join("helper.bin"), fake_elf()).unwrap();
    let mut truncated = gguf_bytes(None);
    truncated.truncate(40);
    std::fs::write(root.join("cut.gguf"), truncated).unwrap();
    std::fs::write(
        root.join("tokenizer_config.json"),
        r#"{"chat_template": "{{ ''.__class__.__mro__[1].__subclasses__() }}"}"#,
    )
    .unwrap();
    let report = scan(root).unwrap();
    let r = rules(&report);
    for expected in [
        "remote-code",
        "code-file",
        "executable-file",
        "malformed-header",
        "template-injection",
    ] {
        assert!(
            r.contains(&expected),
            "{expected} missing: {:?}",
            report.findings
        );
    }
    let code = report
        .findings
        .iter()
        .find(|f| f.rule == "code-file")
        .unwrap();
    assert_eq!(
        code.severity,
        Severity::High,
        "auto_map makes the code file high"
    );
    // Findings use relative paths only.
    let root_text = root.to_string_lossy().to_string();
    assert!(report.findings.iter().all(|f| !f.file.contains(&root_text)));
    // Random bytes with model extensions never panic.
    let mut state = 3u64;
    for (i, ext) in ["pkl", "pt", "gguf", "safetensors", "epiph", "bin"]
        .iter()
        .enumerate()
    {
        let junk: Vec<u8> = (0..(200 + i * 37))
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 56) as u8
            })
            .collect();
        let p = root.join(format!("junk.{ext}"));
        std::fs::write(&p, junk).unwrap();
        let _ = scan(&p).unwrap();
    }
}

#[test]
fn epiph_files_are_verified() {
    let dir = tempfile::tempdir().unwrap();
    let (model, _) = synthetic_checkpoint(dir.path());
    let ir = crate::parser::ModelImporter::import(&model).unwrap();
    let out = dir.path().join("m.epiph");
    crate::epiph::write(&ir, &crate::epiph::PackOptions::default(), &out).unwrap();
    assert_eq!(scan(&out).unwrap().verdict(), "no findings");
    let mut bytes = std::fs::read(&out).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&out, bytes).unwrap();
    assert!(rules(&scan(&out).unwrap()).contains(&"epiph-integrity"));
}

#[test]
fn deflated_pickle_importing_code_is_critical_like_the_stored_one() {
    let dir = tempfile::tempdir().unwrap();
    let evil = b"cos\nsystem\n(S'echo fixture'\ntR.";
    std::fs::write(
        dir.path().join("stored.pt"),
        zip_stored(&[("archive/data.pkl", evil)]),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("deflated.pt"),
        zip_deflated(&[("archive/data.pkl", evil)]),
    )
    .unwrap();

    let report = scan(dir.path()).unwrap();
    let finding = |file: &str| {
        report
            .findings
            .iter()
            .find(|f| f.file == file)
            .unwrap_or_else(|| panic!("no finding for {file}: {:?}", report.findings))
    };
    let stored = finding("stored.pt!archive/data.pkl");
    assert_eq!(stored.severity, Severity::Critical);
    let deflated = finding("deflated.pt!archive/data.pkl");
    assert_eq!(deflated.severity, stored.severity);
    assert_eq!(deflated.rule, stored.rule);
    assert_eq!(deflated.detail, stored.detail);
}

#[test]
fn deflated_clean_pickle_has_no_finding() {
    let dir = tempfile::tempdir().unwrap();
    let benign =
        b"\x80\x02ccollections\nOrderedDict\nq\x00)Rq\x01ctorch._utils\n_rebuild_tensor_v2\nq\x02.";
    std::fs::write(
        dir.path().join("ok.pt"),
        zip_deflated(&[("archive/data.pkl", benign)]),
    )
    .unwrap();
    let report = scan(dir.path()).unwrap();
    assert_eq!(report.verdict(), "no findings", "{:?}", report.findings);
}

#[test]
fn deflated_pickle_past_the_size_limit_is_reported_too_large() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("bomb.pt"),
        zip_entries(&[(
            "archive/data.pkl",
            8,
            deflate_zeros(PICKLE_LIMIT + 1).as_slice(),
        )]),
    )
    .unwrap();
    let report = scan(dir.path()).unwrap();
    let finding = report
        .findings
        .iter()
        .find(|f| f.rule == "pickle-not-inspected")
        .unwrap_or_else(|| panic!("expected a not-inspected finding: {:?}", report.findings));
    assert!(
        finding.detail.contains("too large"),
        "detail was {}",
        finding.detail
    );
}

const TEST_LIMIT: u64 = 64;

/// `body` followed by zero padding so the whole thing is well past `TEST_LIMIT`.
fn padded(body: &[u8]) -> Vec<u8> {
    let mut v = body.to_vec();
    v.extend_from_slice(&[0u8; 200]);
    v
}

fn assert_evil_and_note(report: &ScanReport) {
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.rule == "pickle-code-execution" && f.severity == Severity::Critical),
        "{:?}",
        report.findings
    );
    assert!(
        rules(report).contains(&"pickle-not-inspected"),
        "{:?}",
        report.findings
    );
}

#[test]
fn oversize_stored_pickle_is_still_judged_on_its_prefix() {
    let mut report = ScanReport::default();
    inspect_zip_pickle(
        &padded(EVIL_PICKLE),
        0,
        "a.pt",
        "d.pkl",
        TEST_LIMIT,
        &mut report,
    );
    assert_evil_and_note(&report);
}

#[test]
fn oversize_deflated_pickle_is_still_judged_on_its_prefix() {
    let mut report = ScanReport::default();
    let body = deflate(&padded(EVIL_PICKLE));
    inspect_zip_pickle(&body, 8, "a.pt", "d.pkl", TEST_LIMIT, &mut report);
    assert_evil_and_note(&report);
}

#[test]
fn oversize_loose_pickle_is_still_judged_on_its_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("evil.pkl");
    std::fs::write(&path, padded(EVIL_PICKLE)).unwrap();
    let mut report = ScanReport::default();
    scan_loose_pickle(&path, "evil.pkl", TEST_LIMIT, &mut report);
    assert_evil_and_note(&report);
}

#[test]
fn oversize_clean_pickle_gets_only_the_size_note() {
    let mut report = ScanReport::default();
    inspect_zip_pickle(
        &padded(CLEAN_PICKLE),
        0,
        "a.pt",
        "d.pkl",
        TEST_LIMIT,
        &mut report,
    );
    assert!(
        rules(&report).contains(&"pickle-not-inspected"),
        "{:?}",
        report.findings
    );
    assert!(
        report
            .findings
            .iter()
            .all(|f| f.severity <= Severity::Medium),
        "{:?}",
        report.findings
    );
}

#[test]
fn pickle_of_exactly_the_limit_gets_no_size_note() {
    let limit = CLEAN_PICKLE.len() as u64;
    let mut report = ScanReport::default();
    inspect_zip_pickle(CLEAN_PICKLE, 0, "a.pt", "d.pkl", limit, &mut report);
    assert!(
        !rules(&report).contains(&"pickle-not-inspected"),
        "{:?}",
        report.findings
    );
}

#[test]
fn stored_keras_config_past_the_limit_is_noted_and_skipped_on_purpose() {
    // A cut JSON prefix cannot be parsed, so inspect_zip_keras skips it (`Some((_, false))` gate).
    let mut report = ScanReport::default();
    let body = zip_entry_body(
        &padded(b"{}"),
        0,
        "Keras config",
        "config.json",
        "keras-not-inspected",
        "m.keras",
        TEST_LIMIT,
        &mut report,
    );
    assert!(matches!(body, Some((_, true))));
    assert_eq!(rules(&report), ["keras-not-inspected"]);
}

#[test]
fn object_npy_past_the_limit_gets_the_size_note() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.npy");
    std::fs::write(&path, npy_bytes((1, 0), "|O", &padded(EVIL_PICKLE))).unwrap();
    let mut report = ScanReport::default();
    scan_npy(&path, "big.npy", TEST_LIMIT, &mut report);
    assert_eq!(
        rules(&report),
        ["numpy-not-inspected"],
        "{:?}",
        report.findings
    );
}

#[test]
fn corrupt_deflate_stream_is_a_finding_not_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let mut body = deflate(b"cos\nsystem\n(S'echo fixture'\ntR.");
    body.truncate(body.len() / 2);
    std::fs::write(
        dir.path().join("corrupt.pt"),
        zip_entries(&[("archive/data.pkl", 8, &body)]),
    )
    .unwrap();
    let report = scan(dir.path()).unwrap();
    let finding = report
        .findings
        .iter()
        .find(|f| f.rule == "pickle-not-inspected")
        .unwrap_or_else(|| panic!("expected a not-inspected finding: {:?}", report.findings));
    assert!(
        finding.detail.contains("corrupt"),
        "detail was {}",
        finding.detail
    );
}

/// Builds a `.npy` file: magic, version, little-endian header length, dict header, payload.
/// `descr` is a dtype token in the header's own quoting, so `'O'` and `"'O'"` both reach
/// `descr` as `'O'`; the fixture covers the quoting a real `.npy` file uses.
fn npy_bytes_raw(version: (u8, u8), descr: &str, payload: &[u8]) -> Vec<u8> {
    let len_size = if version.0 == 1 { 2 } else { 4 };
    let mut header =
        format!("{{'descr': {descr}, 'fortran_order': False, 'shape': (2,), }}").into_bytes();
    while (8 + len_size + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut out = b"\x93NUMPY".to_vec();
    out.push(version.0);
    out.push(version.1);
    if len_size == 2 {
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    } else {
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    }
    out.extend_from_slice(&header);
    out.extend_from_slice(payload);
    out
}

/// A dtype code without its quotes, the way `'|O'` appears in a header.
fn npy_bytes(version: (u8, u8), descr: &str, payload: &[u8]) -> Vec<u8> {
    npy_bytes_raw(version, &format!("'{descr}'"), payload)
}

const EVIL_PICKLE: &[u8] = b"cos\nsystem\n(S'echo fixture'\ntR.";
const CLEAN_PICKLE: &[u8] =
    b"\x80\x02ccollections\nOrderedDict\nq\x00)Rq\x01ctorch._utils\n_rebuild_tensor_v2\nq\x02.";

#[test]
fn numeric_npy_has_no_finding() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.npy");
    std::fs::write(&path, npy_bytes((1, 0), "<f8", &[0u8; 16])).unwrap();
    let report = scan(&path).unwrap();
    assert_eq!(report.verdict(), "no findings", "{:?}", report.findings);
}

#[test]
fn object_npy_with_code_execution_pickle_is_critical() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("evil.npy");
    std::fs::write(&path, npy_bytes((1, 0), "|O", EVIL_PICKLE)).unwrap();
    let report = scan(&path).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.rule == "pickle-code-execution" && f.severity == Severity::Critical),
        "{:?}",
        report.findings
    );
}

#[test]
fn object_npy_with_clean_pickle_only_needs_allow_pickle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.npy");
    std::fs::write(&path, npy_bytes((1, 0), "|O", CLEAN_PICKLE)).unwrap();
    let report = scan(&path).unwrap();
    assert_eq!(report.findings.len(), 1, "{:?}", report.findings);
    assert_eq!(report.findings[0].rule, "numpy-allow-pickle");
    assert_eq!(report.findings[0].severity, Severity::Medium);
    assert_eq!(report.verdict(), "suspicious");
}

#[test]
fn npy_version_2_header_is_parsed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2.npy");
    std::fs::write(&path, npy_bytes((2, 0), "<f8", &[0u8; 16])).unwrap();
    let report = scan(&path).unwrap();
    assert_eq!(report.verdict(), "no findings", "{:?}", report.findings);
}

#[test]
fn object_npy_without_byte_order_marker_is_flagged() {
    // `np.dtype('O')` is the object dtype; a header that writes `descr` as `'O'`
    // (no byte-order marker) still loads as a pickle.
    let dir = tempfile::tempdir().unwrap();
    let mut files = Vec::new();
    for (i, descr) in ["'O'", "\"O\""].into_iter().enumerate() {
        let path = dir.path().join(format!("obj-{i}.npy"));
        std::fs::write(&path, npy_bytes_raw((1, 0), descr, EVIL_PICKLE)).unwrap();
        files.push(path);
    }
    // The same descr inside an `.npz` entry.
    std::fs::write(
        dir.path().join("z.npz"),
        zip_stored(&[(
            "arr.npy",
            npy_bytes_raw((1, 0), "'O'", EVIL_PICKLE).as_slice(),
        )]),
    )
    .unwrap();

    let report = scan(dir.path()).unwrap();
    let flagged: Vec<&str> = report
        .findings
        .iter()
        .filter(|f| f.rule == "pickle-code-execution" && f.severity == Severity::Critical)
        .map(|f| f.file.as_str())
        .collect();
    assert_eq!(flagged.len(), 3, "{:?}", report.findings);
    assert!(flagged.contains(&"z.npz!arr.npy"), "{:?}", report.findings);
    // The always-present allow_pickle medium accompanies each object array.
    let medium: Vec<&str> = report
        .findings
        .iter()
        .filter(|f| f.rule == "numpy-allow-pickle")
        .map(|f| f.file.as_str())
        .collect();
    assert_eq!(medium.len(), 3, "{:?}", report.findings);
}

#[test]
fn truncated_npy_header_is_a_finding_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = npy_bytes((1, 0), "|O", EVIL_PICKLE);
    bytes.truncate(12);
    let path = dir.path().join("cut.npy");
    std::fs::write(&path, bytes).unwrap();
    let report = scan(&path).unwrap();
    assert!(
        rules(&report).contains(&"malformed-header"),
        "{:?}",
        report.findings
    );
}

#[test]
fn npz_entry_with_object_array_is_flagged() {
    let dir = tempfile::tempdir().unwrap();
    let evil = npy_bytes((1, 0), "|O", EVIL_PICKLE);
    std::fs::write(
        dir.path().join("stored.npz"),
        zip_stored(&[("arr.npy", evil.as_slice())]),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("deflated.npz"),
        zip_deflated(&[("arr.npy", evil.as_slice())]),
    )
    .unwrap();

    let report = scan(dir.path()).unwrap();
    let critical: Vec<&str> = report
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Critical)
        .map(|f| f.file.as_str())
        .collect();
    assert!(
        critical.contains(&"stored.npz!arr.npy"),
        "{:?}",
        report.findings
    );
    assert!(
        critical.contains(&"deflated.npz!arr.npy"),
        "{:?}",
        report.findings
    );
}

// -------------------------------------------------------------------------------------------
// Keras `.keras` / legacy `.h5`

fn write_keras(dir: &Path, name: &str, config: &[u8], deflated: bool) -> PathBuf {
    let bytes = if deflated {
        zip_deflated(&[("config.json", config)])
    } else {
        zip_stored(&[("config.json", config)])
    };
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn keras_lambda_layer_is_high_and_deflated_matches() {
    let dir = tempfile::tempdir().unwrap();
    let config = br#"{
        "config": {"layers": [
            {"class_name": "Lambda", "config": {"name": "scale", "function": {
                "class_name": "__lambda__", "config": {"code": "AAAA"}}}}
        ]}
    }"#;
    write_keras(dir.path(), "stored.keras", config, false);
    write_keras(dir.path(), "deflated.keras", config, true);

    let report = scan(dir.path()).unwrap();
    for name in ["stored.keras", "deflated.keras"] {
        let at = format!("{name}!config.json");
        let named: Vec<&Finding> = report
            .findings
            .iter()
            .filter(|f| f.file == at && f.rule == "keras-lambda-layer")
            .collect();
        assert!(
            named.iter().any(|f| {
                f.severity == Severity::High
                    && f.detail.contains("scale")
                    && f.detail.contains("Lambda")
            }),
            "{at}: {:?}",
            report.findings
        );
    }
}

#[test]
fn keras_tf_op_lambda_layer_is_high() {
    let dir = tempfile::tempdir().unwrap();
    let config = br#"{"config": {"layers": [
        {"class_name": "TFOpLambda", "config": {"name": "cast", "function": "math.multiply"}}
    ]}}"#;
    write_keras(dir.path(), "tf.keras", config, false);
    let report = scan(&dir.path().join("tf.keras")).unwrap();
    let finding = report
        .findings
        .iter()
        .find(|f| f.rule == "keras-lambda-layer")
        .unwrap_or_else(|| panic!("{:?}", report.findings));
    assert_eq!(finding.severity, Severity::High);
    assert_eq!(finding.file, "tf.keras!config.json");
    assert!(finding.detail.contains("cast"), "{}", finding.detail);
    assert!(finding.detail.contains("TFOpLambda"), "{}", finding.detail);
}

#[test]
fn plain_keras_model_has_no_findings() {
    let dir = tempfile::tempdir().unwrap();
    let config = br#"{
        "module": "keras", "class_name": "Functional", "registered_name": "Functional",
        "config": {"name": "model", "layers": [
            {"class_name": "InputLayer", "module": "keras.layers", "registered_name": null,
             "config": {"name": "input"}},
            {"class_name": "Dense", "module": "keras.layers", "registered_name": "Dense",
             "config": {"name": "dense"}}
        ]}
    }"#;
    write_keras(dir.path(), "plain.keras", config, false);
    let report = scan(&dir.path().join("plain.keras")).unwrap();
    assert_eq!(report.verdict(), "no findings", "{:?}", report.findings);
    assert!(report.findings.is_empty(), "{:?}", report.findings);
}

#[test]
fn keras_custom_object_is_medium() {
    let dir = tempfile::tempdir().unwrap();
    let config = br#"{"config": {"layers": [
        {"class_name": "MyLayer", "module": "my_pkg.layers", "registered_name": "Custom>MyLayer",
         "config": {"name": "custom"}},
        {"class_name": "Other", "registered_name": "Custom>Other", "config": {"name": "other"}}
    ]}}"#;
    write_keras(dir.path(), "custom.keras", config, false);
    let report = scan(&dir.path().join("custom.keras")).unwrap();
    let custom: Vec<&Finding> = report
        .findings
        .iter()
        .filter(|f| f.rule == "keras-custom-object")
        .collect();
    assert_eq!(custom.len(), 2, "{:?}", report.findings);
    assert!(custom.iter().all(|f| f.severity == Severity::Medium));
    assert!(
        custom
            .iter()
            .any(|f| f.detail.contains("my_pkg.layers") && f.detail.contains("MyLayer")),
        "{:?}",
        custom
    );
    assert!(
        custom.iter().any(|f| f.detail.contains("Other")),
        "{:?}",
        custom
    );
}

#[test]
fn invalid_keras_config_json_is_a_medium_finding_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    write_keras(dir.path(), "broken.keras", b"{not json", false);
    let report = scan(&dir.path().join("broken.keras")).unwrap();
    let finding = report
        .findings
        .iter()
        .find(|f| f.rule == "keras-not-inspected")
        .unwrap_or_else(|| panic!("{:?}", report.findings));
    assert_eq!(finding.severity, Severity::Medium);
    assert_eq!(finding.file, "broken.keras!config.json");
}

#[test]
fn legacy_hdf5_file_is_one_info_finding() {
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = b"\x89HDF\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(&[0u8; 64]);
    let path = dir.path().join("legacy.h5");
    std::fs::write(&path, bytes).unwrap();
    let report = scan(&path).unwrap();
    assert_eq!(report.findings.len(), 1, "{:?}", report.findings);
    assert_eq!(report.findings[0].severity, Severity::Info);
    assert_eq!(report.findings[0].rule, "hdf5-not-inspected");
    assert_eq!(report.findings[0].file, "legacy.h5");
    assert!(
        report.findings[0].detail.contains("not inspected"),
        "{}",
        report.findings[0].detail
    );
    assert_eq!(report.verdict(), "no findings");
}

#[test]
fn deeply_nested_keras_config_is_reported_without_overflowing_the_stack() {
    let dir = tempfile::tempdir().unwrap();
    // serde_json's own 128-level parser limit turns this into the same medium finding;
    // 100 levels is past the scanner's 64-level cap but still parses.
    let deep = format!("{}1{}", "{\"a\":".repeat(200), "}".repeat(200));
    write_keras(dir.path(), "deep.keras", deep.as_bytes(), false);
    let report = scan(&dir.path().join("deep.keras")).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.rule == "keras-not-inspected" && f.severity == Severity::Medium),
        "{:?}",
        report.findings
    );

    let capped = format!("{}1{}", "{\"a\":".repeat(100), "}".repeat(100));
    write_keras(dir.path(), "capped.keras", capped.as_bytes(), false);
    let report = scan(&dir.path().join("capped.keras")).unwrap();
    let finding = report
        .findings
        .iter()
        .find(|f| f.rule == "keras-not-inspected")
        .unwrap_or_else(|| panic!("{:?}", report.findings));
    assert_eq!(finding.severity, Severity::Medium);
    assert!(finding.detail.contains("64"), "{}", finding.detail);
}

// ---------------------------------------------------------------------------------------------
// ONNX

mod onnx_files {
    use super::*;
    use crate::scan::onnx::tests::{model, pb_field, tensor};

    fn scan_bytes(bytes: &[u8]) -> ScanReport {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.onnx");
        std::fs::write(&p, bytes).unwrap();
        scan(&p).unwrap()
    }

    fn initializer(location: &str) -> Vec<u8> {
        model(&pb_field(5, 2, &tensor(location)))
    }

    #[test]
    fn local_external_data_has_no_finding() {
        assert!(scan_bytes(&initializer("weights.bin")).findings.is_empty());
    }

    #[test]
    fn traversal_in_initializer_constant_and_subgraph_is_high() {
        let attr_t = pb_field(5, 2, &tensor("../../etc/passwd"));
        let node = |attr: &[u8]| pb_field(1, 2, &pb_field(5, 2, attr));
        let subgraph = pb_field(6, 2, &pb_field(5, 2, &tensor("/etc/passwd")));
        for bytes in [
            initializer("../../etc/passwd"),
            model(&node(&attr_t)),
            model(&node(&subgraph)),
        ] {
            let report = scan_bytes(&bytes);
            assert_eq!(rules(&report), ["onnx-external-data-path"], "{report:?}");
            assert_eq!(report.verdict(), "dangerous");
        }
    }

    #[test]
    fn custom_domain_is_info_and_standard_domains_are_silent() {
        let node = |d: &str| model(&pb_field(1, 2, &pb_field(7, 2, d.as_bytes())));
        let report = scan_bytes(&node("com.example.custom"));
        assert_eq!(rules(&report), ["onnx-custom-domain"]);
        assert_eq!(report.findings[0].severity, Severity::Info);
        for d in ["", "ai.onnx", "ai.onnx.ml"] {
            assert!(scan_bytes(&node(d)).findings.is_empty(), "{d}");
        }
    }

    #[test]
    fn attribute_tensor_and_graph_lists_are_flagged() {
        let node = |attr: &[u8]| model(&pb_field(1, 2, &pb_field(5, 2, attr)));
        let graph = pb_field(5, 2, &tensor("/etc/passwd"));
        for attr in [pb_field(10, 2, &tensor("../x")), pb_field(11, 2, &graph)] {
            assert_eq!(
                rules(&scan_bytes(&node(&attr))),
                ["onnx-external-data-path"]
            );
        }
    }

    #[test]
    fn executable_renamed_onnx_is_high() {
        let report = scan_bytes(b"\x7fELF renamed program");
        assert_eq!(rules(&report), ["executable-file"]);
    }

    #[test]
    fn deep_nesting_is_malformed_not_a_stack_overflow() {
        let mut graph = pb_field(5, 2, &tensor("../x"));
        for _ in 0..100 {
            graph = pb_field(1, 2, &pb_field(5, 2, &pb_field(6, 2, &graph)));
        }
        assert!(rules(&scan_bytes(&model(&graph))).contains(&"onnx-malformed"));
    }

    #[test]
    fn cut_file_is_malformed_not_a_panic() {
        let bytes = initializer("weights.bin");
        let report = scan_bytes(&bytes[..bytes.len() - 1]);
        assert_eq!(rules(&report), ["onnx-malformed"]);
    }
}
