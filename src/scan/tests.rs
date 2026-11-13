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

fn zip_stored(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, data) in entries {
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
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
