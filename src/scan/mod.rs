// SPDX-License-Identifier: AGPL-3.0-only
//! Static security scan of model files and folders (`thme scan`).
//!
//! Nothing is loaded or executed: files are read as bytes and checked by rules. The scan looks
//! for the ways a model file can carry code or hide data:
//!
//! - **Pickle imports** in `.bin`/`.pt`/`.pth`/`.ckpt`/`.pkl` files (plain or inside the
//!   PyTorch zip), since loading a pickle can call any importable function.
//! - **Malformed headers** in GGUF and SafeTensors files, which target parser bugs.
//! - **Unaccounted bytes** between or after tensor payloads, and executables hidden there.
//! - **Code in text**: template-injection patterns in chat templates, shell or Python in
//!   metadata strings, and large encoded blobs.
//! - **Code shipped with a model**: `auto_map` (runs with `trust_remote_code`), scripts and
//!   executables in the folder.
//! - **`.epiph` integrity**: every digest and tensor checked.
//!
//! A clean result means no rule fired. It is not proof that a file is safe.

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
        if let Err(e) = crate::epiph::verify(path) {
            report.add(Severity::High, "epiph-integrity", rel, format!("{e:#}"));
        }
    } else if magic.starts_with(b"GGUF") {
        scan_gguf(path, rel, report);
    } else if magic.starts_with(b"PK\x03\x04") {
        scan_zip(path, rel, report);
    } else if ext == "safetensors" {
        scan_safetensors(path, rel, report);
    } else if let Some(kind) = executable_kind(&magic, 0) {
        report.add(
            Severity::High,
            "executable-file",
            rel,
            format!("{kind} executable shipped with the model"),
        );
    } else if is_pickle_candidate(&magic, &ext) {
        match read_limited(path, PICKLE_LIMIT) {
            Ok(bytes) => judge_pickle(&bytes, rel, "", report),
            Err(e) => report.add(Severity::Medium, "unreadable", rel, format!("{e:#}")),
        }
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
    } else if name == "config.json" {
        if let Some(map) = read_json(path).and_then(|v| v.get("auto_map").cloned()) {
            report.add(
                Severity::High,
                "remote-code",
                rel,
                format!("auto_map {map} executes custom Python when loaded with trust_remote_code"),
            );
        }
    } else if name == "tokenizer_config.json" {
        if let Some(template) = read_json(path).and_then(|v| v.get("chat_template").cloned()) {
            let text = template.to_string();
            check_text(&text, rel, "chat_template", true, report);
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
    let scan = scan_pickle(bytes);
    for import in &scan.imports {
        if let Some((severity, rule)) = classify(import) {
            report.add(
                severity,
                rule,
                &at,
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
            &at,
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
            &at,
            format!("pickle stream ends or breaks at byte {offset}"),
        );
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
            if method == 0 {
                judge_pickle(bytes, rel, &name, report);
            } else {
                report.add(
                    Severity::Medium,
                    "pickle-not-inspected",
                    rel,
                    format!("pickle '{name}' is compressed (method {method}); not inspected"),
                );
            }
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
    for (key, value) in &model.source_gguf_metadata {
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
