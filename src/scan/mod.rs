// SPDX-License-Identifier: AGPL-3.0-only
//! Static security scan of model files and folders (`thme scan`).
//!
//! Nothing is loaded or executed: files are read as bytes and checked by rules. The scan looks
//! for the ways a model file can carry code or hide data:
//!
//! - **Pickle imports** in `.bin`/`.pt`/`.pth`/`.ckpt`/`.pkl` files (plain or inside the
//!   PyTorch zip), and in NumPy `.npy`/`.npz` object arrays, whose payload is a pickle.
//!   Loading such a file needs `allow_pickle=True` and can run code.
//! - **Keras Lambda layers** in `.keras` files: a layer whose `class_name` is `Lambda`,
//!   `TFOpLambda` or a `__lambda__` function carries Python code that runs when the model is
//!   loaded with `safe_mode` disabled. Custom layer classes must be imported from user code;
//!   legacy HDF5 `.h5` files are reported as not inspected.
//! - **ONNX external data** in `.onnx` files: a tensor whose `location` is an absolute path or
//!   contains `..` makes a loader read a file outside the model folder; custom operator domains
//!   are noted.
//! - **Malformed headers** in GGUF and SafeTensors files, which target parser bugs.
//! - **Unaccounted bytes** between or after tensor payloads, and executables hidden there.
//! - **Code in text**: template-injection patterns in chat templates, shell or Python in
//!   metadata strings, and large encoded blobs.
//! - **Code shipped with a model**: `auto_map` (runs with `trust_remote_code`), scripts and
//!   executables in the folder.
//! - **`.epiph` integrity**: every digest and tensor checked.
//!
//! A clean result means no rule fired. It is not proof that a file is safe.

pub mod onnx;
pub mod pickle;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::ir::model::ModelMetadataValue;
use pickle::{scan_pickle, PickleImport};

/// How serious a finding is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Medium,
    High,
    Critical,
}

/// One rule that fired.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Finding {
    pub severity: Severity,
    pub rule: &'static str,
    /// Path relative to the scanned root, with `/` separators.
    pub file: String,
    pub detail: String,
}

/// Result of a scan.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ScanReport {
    pub files_scanned: usize,
    pub findings: Vec<Finding>,
}

impl ScanReport {
    pub fn worst(&self) -> Option<Severity> {
        self.findings.iter().map(|f| f.severity).max()
    }

    /// `dangerous` (high or critical), `suspicious` (medium) or `no findings`.
    pub fn verdict(&self) -> &'static str {
        match self.worst() {
            Some(Severity::Critical | Severity::High) => "dangerous",
            Some(Severity::Medium) => "suspicious",
            _ => "no findings",
        }
    }

    fn add(&mut self, severity: Severity, rule: &'static str, file: &str, detail: String) {
        self.findings.push(Finding {
            severity,
            rule,
            file: file.to_string(),
            detail,
        });
    }
}

/// Largest number of bytes read from a single pickle stream.
const PICKLE_LIMIT: u64 = 256 << 20;
const MAX_FILES: usize = 10_000;
const MAX_DEPTH: usize = 6;

/// Scans a file or a folder (recursively, up to a fixed depth and file count).
pub fn scan(path: &Path) -> Result<ScanReport> {
    let mut report = ScanReport::default();
    if path.is_dir() {
        let mut files = Vec::new();
        collect(path, 0, &mut files)?;
        if files.len() > MAX_FILES {
            report.add(
                Severity::Medium,
                "scan-truncated",
                ".",
                format!("more than {MAX_FILES} files; only the first {MAX_FILES} were scanned"),
            );
            files.truncate(MAX_FILES);
        }
        let remote_code = folder_declares_remote_code(path);
        for file in files {
            let rel = relative(path, &file);
            scan_file(&file, &rel, remote_code, &mut report);
        }
    } else if path.is_file() {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        scan_file(path, &name, false, &mut report);
    } else {
        bail!("{path:?} is neither a file nor a folder");
    }
    report.findings.sort_by(|a, b| {
        b.severity
            .cmp(&a.severity)
            .then_with(|| a.file.cmp(&b.file))
    });
    Ok(report)
}

fn collect(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) -> Result<()> {
    if depth > MAX_DEPTH || out.len() > MAX_FILES {
        return Ok(());
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to list {dir:?}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for entry in entries {
        // Symlinks are not followed, so a link cannot pull files from outside the folder.
        let meta = std::fs::symlink_metadata(&entry)?;
        if meta.is_dir() {
            collect(&entry, depth + 1, out)?;
        } else if meta.is_file() {
            out.push(entry);
        }
    }
    Ok(())
}

fn relative(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .unwrap_or(file)
        .to_string_lossy()
        .replace('\\', "/")
}

fn folder_declares_remote_code(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .is_some_and(|v| v.get("auto_map").is_some())
}

fn head(path: &Path, n: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(n);
    if let Ok(file) = File::open(path) {
        let _ = file.take(n as u64).read_to_end(&mut buf);
    }
    buf
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn scan_file(path: &Path, rel: &str, remote_code: bool, report: &mut ScanReport) {
    report.files_scanned += 1;
    let magic = head(path, 4096);
    let ext = extension(path);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    if magic.starts_with(&crate::epiph::MAGIC) {
        match crate::epiph::verify(path) {
            Err(e) => report.add(Severity::High, "epiph-integrity", rel, format!("{e:#}")),
            Ok(_) => scan_epiph_contents(path, rel, report),
        }
    } else if magic.starts_with(b"GGUF") {
        scan_gguf(path, rel, report);
    } else if magic.starts_with(b"\x93NUMPY") {
        scan_npy(path, rel, PICKLE_LIMIT, report);
    } else if magic.starts_with(b"PK\x03\x04") {
        scan_zip(path, rel, report);
    } else if magic.starts_with(b"\x89HDF\r\n\x1a\n") {
        report.add(
            Severity::Info,
            "hdf5-not-inspected",
            rel,
            "HDF5 or legacy Keras file; layers (including Lambda) are not inspected".to_string(),
        );
    } else if ext == "safetensors" {
        scan_safetensors(path, rel, report);
    } else if let Some(kind) = executable_kind(&magic, 0) {
        report.add(
            Severity::High,
            "executable-file",
            rel,
            format!("{kind} executable shipped with the model"),
        );
    } else if ext == "onnx" {
        scan_onnx_file(path, rel, report);
    } else if is_pickle_candidate(&magic, &ext) {
        scan_loose_pickle(path, rel, PICKLE_LIMIT, report);
    } else if matches!(
        ext.as_str(),
        "py" | "sh" | "bat" | "cmd" | "ps1" | "js" | "vbs" | "pyc"
    ) {
        let (severity, detail) = if remote_code {
            (
                Severity::High,
                "code file in a model whose config.json declares auto_map: it runs when the model is loaded with trust_remote_code",
            )
        } else {
            (Severity::Medium, "code file shipped with the model")
        };
        report.add(severity, "code-file", rel, detail.to_string());
    } else if name == "config.json" || name == "tokenizer_config.json" {
        if let Some(json) = read_json(path) {
            check_hf_json(&name, &json, rel, report);
        }
    }
}

/// Rules for an ONNX file: tensor data locations that leave the model folder, a broken protobuf,
/// and operator domains that need a custom-op library.
fn scan_onnx_file(path: &Path, rel: &str, report: &mut ScanReport) {
    const STANDARD_DOMAINS: [&str; 5] = [
        "",
        "ai.onnx",
        "ai.onnx.ml",
        "ai.onnx.training",
        "ai.onnx.preview.training",
    ];
    let data = match read_limited(path, 2 << 30) {
        Ok(d) => d,
        Err(e) => {
            report.add(Severity::Medium, "unreadable", rel, format!("{e:#}"));
            return;
        }
    };
    let scan = onnx::scan_onnx(&data);
    for location in scan
        .external_locations
        .iter()
        .filter(|l| onnx::is_unsafe_location(l))
    {
        report.add(
            Severity::High,
            "onnx-external-data-path",
            rel,
            format!("tensor data location '{location}' points outside the model folder"),
        );
    }
    if let Some(at) = scan.malformed_at {
        report.add(
            Severity::Medium,
            "onnx-malformed",
            rel,
            format!("protobuf ends or breaks at byte {at}"),
        );
    }
    for domain in scan
        .domains
        .iter()
        .filter(|d| !STANDARD_DOMAINS.contains(&d.as_str()))
    {
        report.add(
            Severity::Info,
            "onnx-custom-domain",
            rel,
            format!("operator domain '{domain}' needs a custom-op library; the scan loads none"),
        );
    }
}

/// Judges a loose pickle. Loaders stop at the first STOP opcode, so padding past `limit` must not
/// hide a payload: the first `limit` bytes are judged and the cut is reported.
// The limit is a parameter only so tests can use a small one.
fn scan_loose_pickle(path: &Path, rel: &str, limit: u64, report: &mut ScanReport) {
    match read_limited(path, limit.saturating_add(1)) {
        Ok(mut bytes) => {
            if bytes.len() as u64 > limit {
                bytes.truncate(limit as usize);
                report.add(
                    Severity::Medium,
                    "pickle-not-inspected",
                    rel,
                    format!(
                        "larger than the {limit}-byte limit; only the first {limit} bytes were inspected"
                    ),
                );
            }
            judge_pickle(&bytes, rel, "", report);
        }
        Err(e) => report.add(Severity::Medium, "unreadable", rel, format!("{e:#}")),
    }
}

/// Rules for Hugging Face JSON files, loose or attached to an `.epiph` file.
fn check_hf_json(name: &str, json: &serde_json::Value, rel: &str, report: &mut ScanReport) {
    if name == "config.json" {
        if let Some(map) = json.get("auto_map") {
            report.add(
                Severity::High,
                "remote-code",
                rel,
                format!("auto_map {map} executes custom Python when loaded with trust_remote_code"),
            );
        }
    } else if name == "tokenizer_config.json" {
        if let Some(template) = json.get("chat_template") {
            check_text(&template.to_string(), rel, "chat_template", true, report);
        }
    }
}

/// An `.epiph` file's GGUF metadata and attached Hugging Face files get the same rules as
/// loose files.
fn scan_epiph_contents(path: &Path, rel: &str, report: &mut ScanReport) {
    match crate::epiph::read(path) {
        Ok((model, _)) => check_gguf_metadata(&model.source_gguf_metadata, rel, report),
        Err(e) => report.add(Severity::High, "epiph-integrity", rel, format!("{e:#}")),
    }
    match crate::epiph::read_attachments(path, "hf-file") {
        Ok(files) => {
            for (name, bytes) in files {
                let name = name.unwrap_or_default();
                let at = format!("{rel}!{name}");
                if !crate::epiph::is_plain_file_name(&name) {
                    report.add(
                        Severity::High,
                        "unsafe-attachment-name",
                        rel,
                        format!("attachment name '{name}' is not a plain file name"),
                    );
                } else if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    check_hf_json(&name, &json, &at, report);
                }
            }
        }
        Err(e) => report.add(Severity::High, "epiph-integrity", rel, format!("{e:#}")),
    }
}

/// Text rules over GGUF metadata strings; vocabulary arrays are data and are skipped.
fn check_gguf_metadata(
    metadata: &std::collections::BTreeMap<String, ModelMetadataValue>,
    rel: &str,
    report: &mut ScanReport,
) {
    for (key, value) in metadata {
        if key.starts_with("tokenizer.ggml.") {
            continue; // vocabulary, merges and token types are data, not instructions
        }
        let mut strings = Vec::new();
        collect_strings(value, &mut strings);
        let is_template = key.contains("chat_template");
        for s in strings {
            check_text(&s, rel, key, is_template, report);
        }
    }
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    File::open(path)?.take(limit).read_to_end(&mut buf)?;
    Ok(buf)
}

fn is_pickle_candidate(magic: &[u8], ext: &str) -> bool {
    let protocol_header = magic.first() == Some(&0x80) && magic.get(1).is_some_and(|&p| p <= 5);
    protocol_header || matches!(ext, "pkl" | "pickle" | "pt" | "pth" | "ckpt" | "bin")
}

// ---------------------------------------------------------------------------------------------
// Pickle

const DANGEROUS_MODULES: [&str; 22] = [
    "os",
    "posix",
    "nt",
    "subprocess",
    "sys",
    "socket",
    "shutil",
    "runpy",
    "pty",
    "webbrowser",
    "ctypes",
    "importlib",
    "marshal",
    "code",
    "commands",
    "multiprocessing",
    "asyncio",
    "http",
    "urllib",
    "requests",
    "pip",
    "signal",
];
const DANGEROUS_BUILTINS: [&str; 13] = [
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "input",
];

fn classify(import: &PickleImport) -> Option<(Severity, &'static str)> {
    let root = import.module.split('.').next().unwrap_or("");
    if DANGEROUS_MODULES.contains(&root) {
        return Some((Severity::Critical, "pickle-code-execution"));
    }
    if matches!(root, "builtins" | "__builtin__") {
        return DANGEROUS_BUILTINS
            .contains(&import.name.as_str())
            .then_some((Severity::Critical, "pickle-code-execution"));
    }
    let known = matches!(
        root,
        "torch" | "numpy" | "collections" | "_codecs" | "copyreg"
    ) || (root == "__torch__");
    (!known).then_some((Severity::Medium, "pickle-unknown-import"))
}

fn judge_pickle(bytes: &[u8], rel: &str, inner: &str, report: &mut ScanReport) {
    let at = if inner.is_empty() {
        rel.to_string()
    } else {
        format!("{rel}!{inner}")
    };
    judge_pickle_at(bytes, &at, report);
}

/// Pickle rules with the finding path already resolved.
fn judge_pickle_at(bytes: &[u8], at: &str, report: &mut ScanReport) {
    let scan = scan_pickle(bytes);
    for import in &scan.imports {
        if let Some((severity, rule)) = classify(import) {
            report.add(
                severity,
                rule,
                at,
                format!(
                    "loading this pickle imports {}.{}",
                    import.module, import.name
                ),
            );
        }
    }
    if scan.unresolved_globals > 0 {
        report.add(
            Severity::Medium,
            "pickle-unresolved-import",
            at,
            format!(
                "{} imports whose names are computed at load time",
                scan.unresolved_globals
            ),
        );
    }
    if let Some(offset) = scan.malformed_at {
        report.add(
            Severity::Medium,
            "pickle-malformed",
            at,
            format!("pickle stream ends or breaks at byte {offset}"),
        );
    }
}

/// Why a deflate stream could not be inflated.
enum InflateFailure {
    /// It is not a valid deflate stream.
    Corrupt(String),
}

/// Inflates a raw deflate stream (zip method 8), stopping at `limit` bytes. The flag says the
/// stream held more, so the bytes returned are only its first `limit`.
fn inflate_limited(
    bytes: &[u8],
    limit: u64,
) -> std::result::Result<(Vec<u8>, bool), InflateFailure> {
    let decoder = flate2::read::DeflateDecoder::new(bytes);
    let mut out = Vec::new();
    // Read one byte past the limit so an oversized stream is detected.
    match decoder.take(limit.saturating_add(1)).read_to_end(&mut out) {
        Ok(_) => {
            let truncated = out.len() as u64 > limit;
            out.truncate(limit as usize);
            Ok((out, truncated))
        }
        Err(e) => Err(InflateFailure::Corrupt(e.to_string())),
    }
}

/// An archive entry's uncompressed bytes and whether they were cut at `limit`: stored entries
/// as-is, deflate inflated. Reports why an entry cannot be inspected instead of failing; a cut
/// entry gets its size note here and the caller decides whether the prefix is worth judging.
#[allow(clippy::too_many_arguments)]
fn zip_entry_body(
    bytes: &[u8],
    method: usize,
    kind: &str,
    name: &str,
    rule: &'static str,
    rel: &str,
    limit: u64,
    report: &mut ScanReport,
) -> Option<(Vec<u8>, bool)> {
    let body = match method {
        0 => {
            let cut = bytes.len() as u64 > limit;
            let kept = if cut { &bytes[..limit as usize] } else { bytes };
            Some((kept.to_vec(), cut))
        }
        8 => match inflate_limited(bytes, limit) {
            Ok(inflated) => Some(inflated),
            Err(InflateFailure::Corrupt(e)) => {
                report.add(
                    Severity::Medium,
                    rule,
                    rel,
                    format!("{kind} '{name}' has a corrupt deflate stream ({e}); not inspected"),
                );
                None
            }
        },
        _ => {
            report.add(
                Severity::Medium,
                rule,
                rel,
                format!("{kind} '{name}' is compressed (method {method}); not inspected"),
            );
            None
        }
    };
    if body.as_ref().is_some_and(|(_, cut)| *cut) {
        report.add(
            Severity::Medium,
            rule,
            rel,
            format!(
                "{kind} '{name}' is larger than the {limit}-byte limit; too large to inspect past its first {limit} bytes"
            ),
        );
    }
    body
}

/// Inspects one `.pkl` entry: stored bytes as they are, deflate (method 8) inflated first.
/// An oversize entry is noted and its first `limit` bytes are still judged.
fn inspect_zip_pickle(
    bytes: &[u8],
    method: usize,
    rel: &str,
    name: &str,
    limit: u64,
    report: &mut ScanReport,
) {
    if let Some((body, _)) = zip_entry_body(
        bytes,
        method,
        "pickle",
        name,
        "pickle-not-inspected",
        rel,
        limit,
        report,
    ) {
        judge_pickle(&body, rel, name, report);
    }
}

/// Deepest JSON nesting walked in a Keras config; deeper configs are reported, not searched.
const KERAS_JSON_DEPTH: usize = 64;

/// Inspects a Keras 3 `config.json` entry: Lambda layers carry serialized Python code.
fn inspect_zip_keras(bytes: &[u8], method: usize, rel: &str, name: &str, report: &mut ScanReport) {
    let Some((body, false)) = zip_entry_body(
        bytes,
        method,
        "Keras config",
        name,
        "keras-not-inspected",
        rel,
        PICKLE_LIMIT,
        report,
    ) else {
        return;
    };
    let at = format!("{rel}!{name}");
    let json = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(json) => json,
        Err(e) => {
            report.add(
                Severity::Medium,
                "keras-not-inspected",
                &at,
                format!("Keras config '{name}' is not valid JSON ({e}); not inspected"),
            );
            return;
        }
    };
    walk_keras_config(&json, 0, &at, name, report);
}

/// Walks a Keras config, reporting Lambda layers and custom classes, and stops at the depth cap.
fn walk_keras_config(
    value: &serde_json::Value,
    depth: usize,
    at: &str,
    name: &str,
    report: &mut ScanReport,
) {
    if depth > KERAS_JSON_DEPTH {
        report.add(
            Severity::Medium,
            "keras-not-inspected",
            at,
            format!(
                "Keras config '{name}' nests deeper than {KERAS_JSON_DEPTH} levels; not inspected past there"
            ),
        );
        return;
    }
    match value {
        serde_json::Value::Object(map) => {
            check_keras_object(map, at, report);
            for child in map.values() {
                walk_keras_config(child, depth + 1, at, name, report);
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                walk_keras_config(child, depth + 1, at, name, report);
            }
        }
        _ => {}
    }
}

/// Rules for one object with a `class_name`: Lambda layers carry code; custom classes must be
/// imported from user code.
fn check_keras_object(
    map: &serde_json::Map<String, serde_json::Value>,
    at: &str,
    report: &mut ScanReport,
) {
    let Some(class) = map.get("class_name").and_then(|v| v.as_str()) else {
        return;
    };
    if matches!(class, "Lambda" | "TFOpLambda" | "__lambda__") {
        let name = map
            .get("config")
            .and_then(|config| config.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("unnamed");
        report.add(
            Severity::High,
            "keras-lambda-layer",
            at,
            format!(
                "layer '{name}' ({class}) carries Python code that runs when the model is loaded with safe_mode disabled"
            ),
        );
        return;
    }
    let module = map.get("module").and_then(|v| v.as_str());
    let registered = map.get("registered_name").and_then(|v| v.as_str());
    let custom = module.is_some_and(|m| !m.starts_with("keras") && !m.starts_with("tensorflow"))
        || registered.is_some_and(|r| r.starts_with("Custom>"));
    if custom {
        let origin = module
            .map(|m| format!("module '{m}'"))
            .unwrap_or_else(|| "no module".to_string());
        report.add(
            Severity::Medium,
            "keras-custom-object",
            at,
            format!(
                "custom class '{class}' ({origin}) must be imported from user code when the model is loaded"
            ),
        );
    }
}

/// Inspects one `.npy` entry inside a `.npz`: the same header rules as a loose file.
fn inspect_zip_npy(bytes: &[u8], method: usize, rel: &str, name: &str, report: &mut ScanReport) {
    let Some((body, false)) = zip_entry_body(
        bytes,
        method,
        "NumPy array",
        name,
        "numpy-not-inspected",
        rel,
        PICKLE_LIMIT,
        report,
    ) else {
        return;
    };
    let at = format!("{rel}!{name}");
    match parse_npy(&body) {
        Ok(info) => {
            if let Some(payload) = body.get(info.data_at..) {
                judge_npy(&info, payload, &at, report);
            }
        }
        Err(e) => report.add(
            Severity::High,
            "malformed-header",
            &at,
            format!(".npy header does not parse ({e})"),
        ),
    }
}

/// Walks the local headers of a zip (the PyTorch checkpoint container) and scans its pickles.
fn scan_zip(path: &Path, rel: &str, report: &mut ScanReport) {
    let data = match read_limited(path, 4 << 30) {
        Ok(d) => d,
        Err(e) => {
            report.add(Severity::Medium, "unreadable", rel, format!("{e:#}"));
            return;
        }
    };
    let u16_at = |i: usize| {
        data.get(i..i + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
    };
    let u32_at = |i: usize| {
        data.get(i..i + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    let mut at = 0usize;
    while data.get(at..at + 4) == Some(b"PK\x03\x04") {
        let parsed = (|| {
            let flags = u16_at(at + 6)?;
            let method = u16_at(at + 8)?;
            let size = u32_at(at + 18)?;
            let name_len = u16_at(at + 26)?;
            let extra_len = u16_at(at + 28)?;
            let name_start = at + 30;
            let name =
                String::from_utf8_lossy(data.get(name_start..name_start + name_len)?).into_owned();
            let body = name_start + name_len + extra_len;
            Some((flags, method, size, name, body))
        })();
        let Some((flags, method, size, name, body)) = parsed else {
            report.add(
                Severity::High,
                "malformed-archive",
                rel,
                format!("zip entry header at byte {at} is truncated"),
            );
            return;
        };
        if flags & 0x8 != 0 && size == 0 {
            report.add(
                Severity::Medium,
                "archive-not-inspected",
                rel,
                format!(
                    "entry '{name}' has no stored size; the rest of the archive was not inspected"
                ),
            );
            return;
        }
        let Some(bytes) = data.get(body..body.saturating_add(size)) else {
            report.add(
                Severity::High,
                "malformed-archive",
                rel,
                format!("entry '{name}' runs past the end of the file"),
            );
            return;
        };
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".pkl") {
            inspect_zip_pickle(bytes, method, rel, &name, PICKLE_LIMIT, report);
        } else if lower.ends_with(".npy") {
            inspect_zip_npy(bytes, method, rel, &name, report);
        } else if lower == "config.json" {
            inspect_zip_keras(bytes, method, rel, &name, report);
        } else if let Some(kind) = executable_kind(bytes, 0) {
            report.add(
                Severity::High,
                "executable-file",
                &format!("{rel}!{name}"),
                format!("{kind} executable inside the archive"),
            );
        } else if [".py", ".sh", ".bat", ".ps1", ".so", ".dll"]
            .iter()
            .any(|e| lower.ends_with(e))
        {
            report.add(
                Severity::High,
                "code-file",
                &format!("{rel}!{name}"),
                "code inside a model archive".into(),
            );
        }
        at = body + size;
    }
}

// ---------------------------------------------------------------------------------------------
// NumPy `.npy` / `.npz`

/// Magic at the start of every `.npy` file.
const NPY_MAGIC: &[u8] = b"\x93NUMPY";
/// Largest header held in memory; a corrupt length field must not drive a huge allocation.
const MAX_NPY_HEADER: usize = 64 << 20;

/// A parsed `.npy` header: the dtype description and where the payload starts.
struct NpyInfo {
    descr: String,
    data_at: usize,
}

/// Parses a `.npy` blob, magic included. Versions 1.0, 2.0 and 3.0 differ only in the
/// width of the little-endian header-length field: 2 bytes for 1.0, 4 for 2.0 and 3.0.
fn parse_npy(data: &[u8]) -> std::result::Result<NpyInfo, String> {
    if !data.starts_with(NPY_MAGIC) {
        return Err("missing \\x93NUMPY magic".into());
    }
    let major = *data.get(6).ok_or("truncated header")?;
    let (len_size, len) = match major {
        1 => {
            let b = data.get(8..10).ok_or("truncated header")?;
            (2, usize::from(u16::from_le_bytes([b[0], b[1]])))
        }
        2 | 3 => {
            let b = data.get(8..12).ok_or("truncated header")?;
            (4, u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        }
        other => return Err(format!("unsupported version {other}.0")),
    };
    if len > MAX_NPY_HEADER {
        return Err(format!("header length {len} is implausible"));
    }
    let start: usize = 8 + len_size;
    let end = start.checked_add(len).ok_or("header length overflows")?;
    let header = data
        .get(start..end)
        .ok_or("header runs past the end of the file")?;
    let text = std::str::from_utf8(header).map_err(|_| "header is not UTF-8")?;
    if !text.trim_start().starts_with('{') {
        return Err("header is not a Python dict".into());
    }
    let descr = header_descr(text).ok_or("no 'descr' field in the header")?;
    Ok(NpyInfo {
        descr: descr.to_string(),
        data_at: end,
    })
}

/// The value of `descr` in a NumPy header dict literal: a quoted string, a bracketed list
/// of `(name, dtype)` tuples, or a bare token.
fn header_descr(header: &str) -> Option<&str> {
    let after_key = &header[header.find("descr")?..];
    let rest = &after_key[after_key.find(':')? + 1..];
    let start = rest.find(|c: char| !c.is_whitespace())?;
    let tail = &rest[start..];
    let bytes = tail.as_bytes();
    let end = match *bytes.first()? {
        q @ (b'\'' | b'"') => tail[1..].find(q as char)? + 2,
        b'[' => {
            // Matching bracket, skipping the quotes around field names and type codes.
            let mut depth = 0usize;
            let mut quote = None;
            let mut at = 0usize;
            loop {
                let b = *bytes.get(at)?;
                match quote {
                    Some(q) if b == q => quote = None,
                    Some(_) => {}
                    None if b == b'\'' || b == b'"' => quote = Some(b),
                    None if b == b'[' => depth += 1,
                    None if b == b']' => {
                        depth -= 1;
                        if depth == 0 {
                            break at + 1;
                        }
                    }
                    None => {}
                }
                at += 1;
            }
        }
        _ => tail.find([',', '}']).unwrap_or(tail.len()),
    };
    Some(&tail[..end])
}

/// True for the object dtype: `O` with any byte-order marker, a quoted `O` field type in a
/// structured dtype, or a bare `O`. Field names are quoted too, so this matches the dtype
/// code rather than any capital O.
fn is_object_descr(descr: &str) -> bool {
    const OBJECT_CODES: [&str; 5] = ["|O", "<O", ">O", "'O'", "\"O\""];
    let descr = descr.trim();
    descr == "O" || OBJECT_CODES.iter().any(|code| descr.contains(code))
}

/// Reads just the header of a `.npy` file, so a large numeric array is never loaded whole.
fn read_npy_head(path: &Path) -> std::result::Result<Vec<u8>, String> {
    use std::io::{Seek, SeekFrom};
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut prefix = [0u8; 12];
    file.read_exact(&mut prefix).map_err(|e| e.to_string())?;
    let major = prefix[6];
    let (len_size, len) = match major {
        1 => (2, usize::from(u16::from_le_bytes([prefix[8], prefix[9]]))),
        2 | 3 => (
            4,
            u32::from_le_bytes([prefix[8], prefix[9], prefix[10], prefix[11]]) as usize,
        ),
        other => return Err(format!("unsupported version {other}.0")),
    };
    if len > MAX_NPY_HEADER {
        return Err(format!("header length {len} is implausible"));
    }
    let total = 8 + len_size + len;
    let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if total as u64 > file_len {
        return Err(format!(
            "header claims {total} bytes but the file holds {file_len}"
        ));
    }
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; total];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Scans a `.npy` file: only an object dtype stores its payload as a pickle.
// The limit is a parameter only so tests can use a small one.
fn scan_npy(path: &Path, rel: &str, limit: u64, report: &mut ScanReport) {
    let info = match read_npy_head(path).and_then(|head| parse_npy(&head)) {
        Ok(info) => info,
        Err(e) => {
            report.add(
                Severity::High,
                "malformed-header",
                rel,
                format!(".npy header does not parse ({e})"),
            );
            return;
        }
    };
    if !is_object_descr(&info.descr) {
        return; // numeric payload: nothing to inspect
    }
    match read_range(path, info.data_at as u64, limit.saturating_add(1)) {
        Ok(payload) if payload.len() as u64 > limit => report.add(
            Severity::Medium,
            "numpy-not-inspected",
            rel,
            format!(
                "object array payload is larger than the {limit}-byte limit; too large to inspect"
            ),
        ),
        Ok(payload) => judge_npy(&info, &payload, rel, report),
        Err(e) => {
            // The object dtype is already known, so the allow_pickle medium stands
            // even when the payload cannot be read.
            add_numpy_allow_pickle(rel, report);
            report.add(Severity::Medium, "unreadable", rel, format!("{e:#}"));
        }
    }
}

/// The medium finding every object `.npy` carries: loading it needs `allow_pickle=True`.
fn add_numpy_allow_pickle(at: &str, report: &mut ScanReport) {
    report.add(
        Severity::Medium,
        "numpy-allow-pickle",
        at,
        "loading this .npy needs allow_pickle=True, which runs its embedded pickle".to_string(),
    );
}

/// An object dtype's payload is a pickle, so it goes through the pickle rules; loading
/// such a file needs `allow_pickle=True` either way.
fn judge_npy(info: &NpyInfo, payload: &[u8], at: &str, report: &mut ScanReport) {
    if !is_object_descr(&info.descr) {
        return;
    }
    add_numpy_allow_pickle(at, report);
    judge_pickle_at(payload, at, report);
}
// ---------------------------------------------------------------------------------------------
// Tensor containers

fn scan_gguf(path: &Path, rel: &str, report: &mut ScanReport) {
    let model = match crate::parser::gguf::GgufParser::parse_from_file(path) {
        Ok(m) => m,
        Err(e) => {
            report.add(
                Severity::High,
                "malformed-header",
                rel,
                format!("GGUF header does not parse ({e:#}); malformed headers target loader bugs"),
            );
            return;
        }
    };
    check_gguf_metadata(&model.source_gguf_metadata, rel, report);
    let alignment = match model.source_gguf_metadata.get("general.alignment") {
        Some(ModelMetadataValue::Uint32(a)) => u64::from(*a).max(1),
        _ => 32,
    };
    let ranges: Vec<(u64, u64)> = model
        .tensors
        .iter()
        .filter_map(|t| t.source.as_ref())
        .map(|s| (s.byte_offset, s.byte_len))
        .collect();
    check_coverage(path, rel, ranges, alignment, report);
}

fn collect_strings(value: &ModelMetadataValue, out: &mut Vec<String>) {
    match value {
        ModelMetadataValue::String(s) => out.push(s.clone()),
        ModelMetadataValue::Array(items) => items.iter().for_each(|i| collect_strings(i, out)),
        _ => {}
    }
}

fn scan_safetensors(path: &Path, rel: &str, report: &mut ScanReport) {
    let tensors =
        match crate::parser::safetensors::SafetensorsScanner::parse_metadata_from_file(path) {
            Ok(t) => t,
            Err(e) => {
                report.add(
                    Severity::High,
                    "malformed-header",
                    rel,
                    format!("SafeTensors header does not parse ({e:#})"),
                );
                return;
            }
        };
    // Free-form `__metadata__` strings in the JSON header.
    let header = head(path, 8);
    if let Some(len) = header
        .get(..8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    {
        if let Ok(bytes) = read_limited(path, 8 + len.min(100 << 20)) {
            if let Some(meta) = bytes
                .get(8..)
                .and_then(|h| serde_json::from_slice::<serde_json::Value>(h).ok())
                .and_then(|v| v.get("__metadata__").cloned())
            {
                if let Some(map) = meta.as_object() {
                    for (key, value) in map {
                        let text = value
                            .as_str()
                            .map_or_else(|| value.to_string(), str::to_string);
                        check_text(&text, rel, key, key.contains("template"), report);
                    }
                }
            }
        }
    }
    let ranges = tensors
        .iter()
        .filter_map(|t| t.source.as_ref())
        .map(|s| (s.byte_offset, s.byte_len))
        .collect();
    check_coverage(path, rel, ranges, 1, report);
}

/// Bytes after the first tensor that no tensor uses, beyond alignment padding, are reported;
/// an executable found there is critical.
fn check_coverage(
    path: &Path,
    rel: &str,
    mut ranges: Vec<(u64, u64)>,
    alignment: u64,
    report: &mut ScanReport,
) {
    let Ok(file_len) = std::fs::metadata(path).map(|m| m.len()) else {
        return;
    };
    ranges.sort_unstable();
    let mut gaps = Vec::new();
    let mut end = ranges.first().map_or(file_len, |r| r.0);
    for &(offset, len) in &ranges {
        if offset > end {
            gaps.push((end, offset));
        }
        end = end.max(offset.saturating_add(len));
    }
    if file_len > end {
        gaps.push((end, file_len));
    }
    for (start, stop) in gaps {
        if stop - start < alignment {
            continue; // alignment padding
        }
        let Ok(bytes) = read_range(path, start, (stop - start).min(64 << 20)) else {
            continue;
        };
        if let Some(kind) = (0..bytes.len().min(1 << 20)).find_map(|i| executable_kind(&bytes, i)) {
            report.add(
                Severity::Critical,
                "hidden-executable",
                rel,
                format!("{kind} executable hidden in unused bytes {start}..{stop}"),
            );
        } else if bytes.iter().any(|&b| b != 0) {
            report.add(
                Severity::Medium,
                "unaccounted-bytes",
                rel,
                format!(
                    "{} bytes at {start}..{stop} belong to no tensor and are not zero padding",
                    stop - start
                ),
            );
        }
    }
}

fn read_range(path: &Path, start: u64, len: u64) -> Result<Vec<u8>> {
    use std::io::{Seek, SeekFrom};
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    let mut buf = Vec::new();
    file.take(len).read_to_end(&mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------------------------
// Content rules

/// The kind of executable starting at `at`, if any (ELF, Mach-O, PE, script with a shebang).
pub fn executable_kind(bytes: &[u8], at: usize) -> Option<&'static str> {
    let rest = bytes.get(at..)?;
    if rest.starts_with(b"\x7fELF") {
        return Some("ELF");
    }
    if rest.starts_with(&[0xcf, 0xfa, 0xed, 0xfe]) || rest.starts_with(&[0xce, 0xfa, 0xed, 0xfe]) {
        return Some("Mach-O");
    }
    if rest.starts_with(b"MZ") {
        let pe = rest
            .get(0x3c..0x40)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)?;
        if pe < 4096 && rest.get(pe..pe + 4) == Some(b"PE\0\0") {
            return Some("Windows PE");
        }
    }
    if at == 0 && rest.starts_with(b"#!/") {
        return Some("script");
    }
    None
}

const CODE_PATTERNS: [&str; 16] = [
    "__globals__",
    "__subclasses__",
    "__builtins__",
    "__import__",
    "__class__.__mro__",
    "os.popen",
    "os.system",
    "subprocess",
    "import os",
    "eval(",
    "exec(",
    "#!/bin/",
    "/bin/sh",
    "powershell",
    "cmd.exe",
    "<script",
];

/// Checks one metadata string. In a chat template, code patterns are template injection.
pub fn check_text(text: &str, rel: &str, key: &str, is_template: bool, report: &mut ScanReport) {
    let lower = text.to_ascii_lowercase();
    let hits: Vec<&str> = CODE_PATTERNS
        .iter()
        .copied()
        .filter(|p| lower.contains(p))
        .collect();
    if !hits.is_empty() {
        let (rule, what) = if is_template {
            ("template-injection", "chat template")
        } else {
            ("code-in-metadata", "metadata string")
        };
        report.add(
            Severity::High,
            rule,
            rel,
            format!("{what} '{key}' contains {}", hits.join(", ")),
        );
    }
    if let Some(kind) = executable_kind(text.as_bytes(), 0) {
        report.add(
            Severity::Critical,
            "executable-in-metadata",
            rel,
            format!("metadata '{key}' holds a {kind} executable"),
        );
    }
    let longest_b64 = text
        .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=')))
        .map(str::len)
        .max()
        .unwrap_or(0);
    if longest_b64 >= 4096 {
        report.add(
            Severity::Medium,
            "encoded-blob",
            rel,
            format!("metadata '{key}' holds a {longest_b64}-character encoded run"),
        );
    }
}

#[cfg(test)]
mod tests;
