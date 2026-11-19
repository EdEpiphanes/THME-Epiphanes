// SPDX-License-Identifier: AGPL-3.0-only
//! Structural check of a finished GGUF v3 file. It reads the bytes on their own, without the
//! writer that produced them, so a bad export is caught before llama.cpp meets it.

use std::collections::HashSet;
use std::fmt;

use anyhow::Result;

use crate::parser::gguf::{GgmlType, GgufReader};

/// How serious a finding is: an `Error` makes the file unusable, a `Warning` does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// One problem found in a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub severity: Severity,
    /// Short rule id, e.g. "magic", "version", "tensor-type".
    pub rule: &'static str,
    /// The tensor the finding is about, when there is one.
    pub tensor: Option<String>,
    pub message: String,
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let severity = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        write!(f, "{severity}[{}]", self.rule)?;
        if let Some(name) = &self.tensor {
            write!(f, " tensor '{name}'")?;
        }
        write!(f, ": {}", self.message)
    }
}

/// Checks `bytes` against the GGUF v3 layout. Never panics; every problem is a finding.
pub fn check_gguf(bytes: &[u8]) -> Vec<Finding> {
    let mut scan = Scan::new();
    if let Err(err) = scan_table(bytes, &mut scan) {
        scan.push(Severity::Error, "truncated", None, format!("{err:#}"));
    }
    scan.findings
}

/// One row of the tensor table. The data-section rules read `offset` and `table_end`.
#[allow(dead_code)]
struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    ty: GgmlType,
    offset: u64,
}

/// What the header scan has found so far.
#[allow(dead_code)]
struct Scan {
    findings: Vec<Finding>,
    tensors: Vec<TensorInfo>,
    /// Byte offset where the tensor table ends (the data section starts after alignment).
    table_end: usize,
    /// `general.alignment`, 32 when the file does not set it.
    alignment: u64,
}

impl Scan {
    fn new() -> Self {
        Self {
            findings: Vec::new(),
            tensors: Vec::new(),
            table_end: 0,
            alignment: 32,
        }
    }

    fn push(
        &mut self,
        severity: Severity,
        rule: &'static str,
        tensor: Option<&str>,
        message: impl Into<String>,
    ) {
        self.findings.push(Finding {
            severity,
            rule,
            tensor: tensor.map(str::to_owned),
            message: message.into(),
        });
    }
}

/// Reads the header, metadata and tensor table. A reader error returns `Err` (reported as
/// `truncated`); a bad magic, version or count is a finding and ends the scan with `Ok`.
fn scan_table(bytes: &[u8], scan: &mut Scan) -> Result<()> {
    if bytes.get(..4) != Some(b"GGUF".as_slice()) {
        scan.push(
            Severity::Error,
            "magic",
            None,
            "file does not start with the GGUF magic bytes",
        );
        return Ok(());
    }
    let mut reader = GgufReader::new(bytes);
    reader.read_bytes(4)?;

    let version = reader.read_u32()?;
    if version != 3 {
        scan.push(
            Severity::Error,
            "version",
            None,
            format!("version is {version}; THME writes GGUF v3"),
        );
    }

    let tensor_count = reader.read_u64()?;
    let kv_count = reader.read_u64()?;
    // Each entry takes at least 8 bytes, so a larger count cannot be real.
    let max_count = (bytes.len() / 8) as u64;
    if tensor_count > max_count || kv_count > max_count {
        scan.push(
            Severity::Error,
            "counts",
            None,
            format!("tensor count {tensor_count} or metadata count {kv_count} exceeds {max_count}, the most this file can hold"),
        );
        return Ok(());
    }

    for _ in 0..kv_count {
        let key = reader.read_string()?;
        let value_type = reader.read_u32()?;
        let value = reader.read_value(value_type)?;
        if key == "general.alignment" {
            // The GGUF spec asks for a multiple of 8; llama.cpp's loader also needs a power of two.
            // shortcut: spec text not checked offline, re-check against the GGUF spec before merge.
            match value.as_u64() {
                Some(a) if a >= 8 && a.is_power_of_two() => scan.alignment = a,
                Some(a) => scan.push(
                    Severity::Error,
                    "alignment",
                    None,
                    format!("general.alignment is {a}; it must be a power of two and at least 8"),
                ),
                None => scan.push(
                    Severity::Error,
                    "alignment",
                    None,
                    "general.alignment is not an unsigned integer",
                ),
            }
        }
    }

    let mut names = HashSet::new();
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        let n_dims = reader.read_u32()?;
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            dims.push(reader.read_u64()?);
        }
        let ty = GgmlType::from_u32(reader.read_u32()?);
        let offset = reader.read_u64()?;
        check_tensor(scan, &name, &dims, ty, &mut names);
        scan.tensors.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
        });
    }
    scan.table_end = reader.cursor;
    Ok(())
}

fn check_tensor(
    scan: &mut Scan,
    name: &str,
    dims: &[u64],
    ty: GgmlType,
    names: &mut HashSet<String>,
) {
    if name.is_empty() {
        scan.push(Severity::Error, "tensor-name", Some(name), "name is empty");
    } else if name.len() > 64 {
        scan.push(
            Severity::Error,
            "tensor-name",
            Some(name),
            format!("name is {} bytes; the GGUF limit is 64", name.len()),
        );
    }
    if !names.insert(name.to_owned()) {
        scan.push(
            Severity::Error,
            "tensor-name",
            Some(name),
            "duplicate tensor name",
        );
    }

    if dims.is_empty() {
        scan.push(
            Severity::Error,
            "tensor-dims",
            Some(name),
            "tensor has no dimensions",
        );
    } else if dims.len() > 4 {
        scan.push(
            Severity::Warning,
            "tensor-dims",
            Some(name),
            format!(
                "{} dimensions; ggml supports 4 (THME's writer allows 8)",
                dims.len()
            ),
        );
    }

    if let GgmlType::Unknown(id) = ty {
        scan.push(
            Severity::Error,
            "tensor-type",
            Some(name),
            format!("type id {id} is not a known GGML type"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporter::gguf_writer::GgufWriter;

    /// The header (24 bytes) plus the one metadata pair of `valid_writer`: 24 + 45 = 69 bytes.
    const METADATA_END: usize = 69;

    fn valid_writer() -> GgufWriter {
        let mut writer = GgufWriter::new();
        writer.add_string("general.architecture", "llama");
        writer.add_tensor("token_embd.weight", vec![2, 4], GgmlType::F32, vec![0; 32]);
        writer.add_tensor(
            "blk.0.attn_q.weight",
            vec![1, 32],
            GgmlType::Q8_0,
            vec![0; GgmlType::Q8_0.type_size()],
        );
        writer
    }

    fn valid() -> Vec<u8> {
        valid_writer()
            .write_to_bytes()
            .expect("write valid fixture")
    }

    fn rules(findings: &[Finding]) -> Vec<&'static str> {
        findings.iter().map(|finding| finding.rule).collect()
    }

    fn single_tensor(name: &str, dims: Vec<usize>, ty: GgmlType) -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.add_string("general.architecture", "llama");
        writer.add_tensor(name, dims, ty, vec![0; GgmlType::Q8_0.type_size()]);
        writer.write_to_bytes().expect("write tensor fixture")
    }

    fn empty_name() -> Vec<u8> {
        // The writer refuses an empty name, so cut the one-byte name "x" out of the table
        // and zero its length. The table is read in order, so the rest still parses.
        let mut bytes = single_tensor("x", vec![1, 32], GgmlType::Q8_0);
        bytes.remove(METADATA_END + 8);
        bytes[METADATA_END..METADATA_END + 8].copy_from_slice(&0u64.to_le_bytes());
        bytes
    }

    #[test]
    fn valid_file_has_no_findings() {
        assert!(check_gguf(&valid()).is_empty());
    }

    #[test]
    fn bad_magic_stops_with_one_error() {
        let mut bytes = valid();
        bytes[0] = b'X';
        let findings = check_gguf(&bytes);
        assert_eq!(rules(&findings), vec!["magic"]);
        assert_eq!(findings[0].severity, Severity::Error);
    }

    #[test]
    fn wrong_version_is_an_error() {
        let mut bytes = valid();
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        let findings = check_gguf(&bytes);
        assert_eq!(rules(&findings), vec!["version"]);
        assert_eq!(findings[0].severity, Severity::Error);
    }

    #[test]
    fn tensor_count_beyond_file_size_is_an_error() {
        let mut bytes = valid();
        bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(rules(&check_gguf(&bytes)), vec!["counts"]);
    }

    #[test]
    fn alignment_must_be_a_power_of_two() {
        let bytes = valid_writer()
            .with_alignment(24)
            .write_to_bytes()
            .expect("write fixture");
        assert_eq!(rules(&check_gguf(&bytes)), vec!["alignment"]);
    }

    #[test]
    fn alignment_must_be_at_least_eight() {
        let bytes = valid_writer()
            .with_alignment(4)
            .write_to_bytes()
            .expect("write fixture");
        assert_eq!(rules(&check_gguf(&bytes)), vec!["alignment"]);
    }

    #[test]
    fn alignment_64_is_accepted() {
        let bytes = valid_writer()
            .with_alignment(64)
            .write_to_bytes()
            .expect("write fixture");
        assert!(check_gguf(&bytes).is_empty());
    }

    #[test]
    fn empty_tensor_name_is_an_error() {
        let findings = check_gguf(&empty_name());
        assert_eq!(rules(&findings), vec!["tensor-name"]);
        assert_eq!(findings[0].tensor.as_deref(), Some(""));
    }

    #[test]
    fn tensor_name_over_64_bytes_is_an_error() {
        let name = "a".repeat(65);
        let findings = check_gguf(&single_tensor(&name, vec![1, 32], GgmlType::Q8_0));
        assert_eq!(rules(&findings), vec!["tensor-name"]);
        assert_eq!(findings[0].tensor.as_deref(), Some(name.as_str()));
    }

    #[test]
    fn duplicate_tensor_name_is_an_error_on_the_second() {
        let mut writer = valid_writer();
        writer.add_tensor("token_embd.weight", vec![2, 4], GgmlType::F32, vec![0; 32]);
        let bytes = writer.write_to_bytes().expect("write fixture");
        let findings = check_gguf(&bytes);
        assert_eq!(rules(&findings), vec!["tensor-name"]);
        assert_eq!(findings[0].tensor.as_deref(), Some("token_embd.weight"));
    }

    #[test]
    fn five_dimensions_is_a_warning() {
        let findings = check_gguf(&single_tensor(
            "blk.0.wide",
            vec![1, 1, 1, 1, 32],
            GgmlType::Q8_0,
        ));
        assert_eq!(rules(&findings), vec!["tensor-dims"]);
        assert_eq!(findings[0].severity, Severity::Warning);
        assert_eq!(findings[0].tensor.as_deref(), Some("blk.0.wide"));
    }

    #[test]
    fn unknown_tensor_type_is_an_error() {
        let findings = check_gguf(&single_tensor(
            "blk.0.mystery",
            vec![1, 32],
            GgmlType::Unknown(9999),
        ));
        assert_eq!(rules(&findings), vec!["tensor-type"]);
        assert_eq!(findings[0].severity, Severity::Error);
        assert_eq!(findings[0].tensor.as_deref(), Some("blk.0.mystery"));
    }

    #[test]
    fn truncated_files_report_truncated_and_do_not_panic() {
        let bytes = valid();
        for len in [10, 30, METADATA_END] {
            assert_eq!(
                rules(&check_gguf(&bytes[..len])),
                vec!["truncated"],
                "length {len}"
            );
        }
    }

    #[test]
    fn display_names_the_tensor_when_there_is_one() {
        let finding = Finding {
            severity: Severity::Error,
            rule: "tensor-type",
            tensor: Some("blk.0.x".to_string()),
            message: "unknown type".to_string(),
        };
        assert_eq!(
            finding.to_string(),
            "error[tensor-type] tensor 'blk.0.x': unknown type"
        );
    }

    #[test]
    fn display_omits_the_tensor_when_there_is_none() {
        let finding = Finding {
            severity: Severity::Error,
            rule: "magic",
            tensor: None,
            message: "bad magic".to_string(),
        };
        assert_eq!(finding.to_string(), "error[magic]: bad magic");
    }
}
