// SPDX-License-Identifier: AGPL-3.0-only
//! Structural check of a finished GGUF v3 file. It reads the bytes on their own, without the
//! writer that produced them, so a bad export is caught before llama.cpp meets it.

use std::collections::HashSet;
use std::fmt;

use anyhow::Result;

use crate::parser::gguf::{GgmlType, GgufReader, GgufValue};

/// Keys an llama loader reads; a file without them loads with defaults or fails later.
// shortcut: key names recalled from llama.cpp's loader, not checked offline, re-check before merge.
const LLAMA_KEYS: [&str; 5] = [
    "llama.context_length",
    "llama.embedding_length",
    "llama.block_count",
    "llama.feed_forward_length",
    "llama.attention.head_count",
];

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
    /// `general.alignment`, 32 when the file does not set it; `None` when it is invalid, so the
    /// data section cannot be read against it.
    alignment: Option<u64>,
    /// Every metadata key in the file.
    keys: HashSet<String>,
    /// The value of `general.architecture`, when it is a string.
    architecture: Option<String>,
}

impl Scan {
    fn new() -> Self {
        Self {
            findings: Vec::new(),
            tensors: Vec::new(),
            table_end: 0,
            alignment: Some(32),
            keys: HashSet::new(),
            architecture: None,
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
                Some(a) if a >= 8 && a.is_power_of_two() => scan.alignment = Some(a),
                Some(a) => {
                    scan.alignment = None;
                    scan.push(
                        Severity::Error,
                        "alignment",
                        None,
                        format!(
                            "general.alignment is {a}; it must be a power of two and at least 8"
                        ),
                    );
                }
                None => {
                    scan.alignment = None;
                    scan.push(
                        Severity::Error,
                        "alignment",
                        None,
                        "general.alignment is not an unsigned integer",
                    );
                }
            }
        } else if key == "general.architecture" {
            match value {
                GgufValue::String(name) => scan.architecture = Some(name),
                _ => scan.push(
                    Severity::Error,
                    "meta-architecture",
                    None,
                    "general.architecture is not a string",
                ),
            }
        }
        scan.keys.insert(key);
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
    check_data_section(scan, bytes.len());
    check_metadata(scan);
    Ok(())
}

/// Rules 6 and 7: the architecture key and, for `llama`, the keys its loader reads.
fn check_metadata(scan: &mut Scan) {
    if !scan.keys.contains("general.architecture") {
        scan.push(
            Severity::Error,
            "meta-architecture",
            None,
            "general.architecture is missing",
        );
    }
    if scan.architecture.as_deref() != Some("llama") {
        return;
    }
    for key in LLAMA_KEYS {
        if !scan.keys.contains(key) {
            scan.push(
                Severity::Warning,
                "meta-arch-keys",
                None,
                format!("architecture llama expects the key '{key}', which is missing"),
            );
        }
    }
}

/// Rules 1 to 5: the data section start, then each tensor's offset, size, bounds and overlap.
/// Skipped when `general.alignment` is invalid, since the offsets cannot be read against it.
// shortcut: data start = table end rounded up to the alignment and offsets as multiples of it are
// recalled from the GGUF spec, not checked offline; re-check before merge.
fn check_data_section(scan: &mut Scan, file_len: usize) {
    let Some(alignment) = scan.alignment else {
        return;
    };
    let Some(data_start) = (scan.table_end as u64)
        .div_ceil(alignment)
        .checked_mul(alignment)
    else {
        scan.push(
            Severity::Error,
            "alignment",
            None,
            "the start of the data section overflows 64 bits",
        );
        return;
    };
    let file_len = file_len as u64;

    let tensors = std::mem::take(&mut scan.tensors);
    // (offset, end, name) of each tensor whose size is known, for the overlap rule.
    let mut placed: Vec<(u64, u64, &str)> = Vec::new();
    for tensor in &tensors {
        let name = tensor.name.as_str();
        if tensor.offset % alignment != 0 {
            scan.push(
                Severity::Error,
                "tensor-offset",
                Some(name),
                format!(
                    "offset {} is not a multiple of the alignment {alignment}",
                    tensor.offset
                ),
            );
        }
        match payload_bytes(tensor) {
            Err(message) => scan.push(Severity::Error, "tensor-size", Some(name), message),
            Ok(None) => {}
            Ok(Some(expected)) => {
                let end = data_start
                    .checked_add(tensor.offset)
                    .and_then(|start| start.checked_add(expected));
                if end.is_none_or(|end| end > file_len) {
                    scan.push(
                        Severity::Error,
                        "tensor-bounds",
                        Some(name),
                        format!(
                            "{expected} bytes at offset {} run past the {file_len}-byte file",
                            tensor.offset
                        ),
                    );
                }
                placed.push((tensor.offset, tensor.offset.saturating_add(expected), name));
            }
        }
    }

    placed.sort_by_key(|&(offset, _, _)| offset);
    let mut covered = 0u64;
    for &(offset, end, name) in &placed {
        if offset < covered {
            scan.push(
                Severity::Error,
                "tensor-overlap",
                Some(name),
                format!(
                    "starts at {offset}, inside the data of the tensor before it (up to {covered})"
                ),
            );
        }
        covered = covered.max(end);
    }
    scan.tensors = tensors;
}

/// Bytes the tensor's payload must take. `Ok(None)` for an unknown type or no dimensions, which
/// their own rules report. `Err` when the shape does not fit whole blocks of the type.
fn payload_bytes(tensor: &TensorInfo) -> Result<Option<u64>, String> {
    let Some(&row) = tensor.dims.first() else {
        return Ok(None);
    };
    if matches!(tensor.ty, GgmlType::Unknown(_)) {
        return Ok(None);
    }
    let numel = tensor
        .dims
        .iter()
        .try_fold(1u64, |n, &d| n.checked_mul(d))
        .ok_or_else(|| "the element count overflows 64 bits".to_string())?;
    let block = tensor.ty.block_size() as u64;
    if numel % block != 0 || row % block != 0 {
        return Err(format!(
            "{numel} elements in rows of {row}; both must be a multiple of the {} block size {block}",
            tensor.ty.type_name()
        ));
    }
    (numel / block)
        .checked_mul(tensor.ty.type_size() as u64)
        .map(Some)
        .ok_or_else(|| "the payload size overflows 64 bits".to_string())
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

    /// Metadata of a llama file, without the key `skip` (pass "" to keep all of them).
    fn metadata_writer(skip: &str) -> GgufWriter {
        let mut writer = GgufWriter::new();
        writer.add_string("general.architecture", "llama");
        for key in LLAMA_KEYS {
            if key != skip {
                writer.add_u32(key, 1);
            }
        }
        writer
    }

    fn add_tensors(writer: &mut GgufWriter) {
        writer.add_tensor("token_embd.weight", vec![2, 4], GgmlType::F32, vec![0; 32]);
        writer.add_tensor(
            "blk.0.attn_q.weight",
            vec![1, 32],
            GgmlType::Q8_0,
            vec![0; GgmlType::Q8_0.type_size()],
        );
    }

    fn valid_writer() -> GgufWriter {
        let mut writer = metadata_writer("");
        add_tensors(&mut writer);
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
        let mut writer = metadata_writer("");
        writer.add_tensor(name, dims, ty, vec![0; GgmlType::Q8_0.type_size()]);
        writer.write_to_bytes().expect("write tensor fixture")
    }

    /// Position of a tensor's name bytes. Found by search, so the metadata layout can change.
    fn name_at(bytes: &[u8], name: &str) -> usize {
        bytes
            .windows(name.len())
            .position(|window| window == name.as_bytes())
            .expect("tensor name in fixture")
    }

    /// A tensor entry is a `u64` name length, the name, a `u32` dimension count, one `u64` per
    /// dimension (fastest first), a `u32` type and a `u64` offset.
    fn dims_at(bytes: &[u8], name: &str) -> usize {
        name_at(bytes, name) + name.len() + 4
    }

    fn offset_at(bytes: &[u8], name: &str, n_dims: usize) -> usize {
        dims_at(bytes, name) + 8 * n_dims + 4
    }

    fn patch_u64(bytes: &mut [u8], at: usize, value: u64) {
        bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn empty_name() -> Vec<u8> {
        // The writer refuses an empty name, so cut the name bytes out of the table and zero
        // its length. The table is read in order, so the rest still parses.
        let name = "blk.0.empty";
        let mut bytes = single_tensor(name, vec![1, 32], GgmlType::Q8_0);
        let at = name_at(&bytes, name);
        bytes.drain(at..at + name.len());
        patch_u64(&mut bytes, at - 8, 0);
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
        let inside_table = name_at(&bytes, "token_embd.weight");
        // 30 bytes would hit the `counts` guard first: the file cannot hold its own key count.
        for len in [10, 60, inside_table] {
            assert_eq!(
                rules(&check_gguf(&bytes[..len])),
                vec!["truncated"],
                "length {len}"
            );
        }
    }

    #[test]
    fn truncated_tensor_data_is_a_bounds_error_on_the_last_tensor() {
        let bytes = valid();
        // Each payload is padded to 32 bytes, so the 30 pad bytes after the 34-byte Q8_0
        // payload are not data. Cut 5 bytes into that payload instead of into the padding.
        let padding = 32 - GgmlType::Q8_0.type_size() % 32;
        let findings = check_gguf(&bytes[..bytes.len() - padding - 5]);
        assert_eq!(rules(&findings), vec!["tensor-bounds"]);
        assert_eq!(findings[0].tensor.as_deref(), Some("blk.0.attn_q.weight"));
    }

    #[test]
    fn row_width_not_a_multiple_of_the_block_is_a_tensor_size_error() {
        let mut bytes = valid();
        let at = dims_at(&bytes, "blk.0.attn_q.weight");
        patch_u64(&mut bytes, at, 31);
        let findings = check_gguf(&bytes);
        assert_eq!(rules(&findings), vec!["tensor-size"]);
        assert_eq!(findings[0].tensor.as_deref(), Some("blk.0.attn_q.weight"));
    }

    #[test]
    fn dimensions_that_overflow_are_a_tensor_size_error() {
        let mut bytes = valid();
        let at = dims_at(&bytes, "blk.0.attn_q.weight");
        patch_u64(&mut bytes, at, u64::MAX);
        patch_u64(&mut bytes, at + 8, u64::MAX);
        assert_eq!(rules(&check_gguf(&bytes)), vec!["tensor-size"]);
    }

    #[test]
    fn offset_off_the_alignment_is_a_tensor_offset_error() {
        let mut bytes = valid();
        let at = offset_at(&bytes, "blk.0.attn_q.weight", 2);
        patch_u64(&mut bytes, at, 33);
        let findings = check_gguf(&bytes);
        assert_eq!(rules(&findings), vec!["tensor-offset"]);
        assert_eq!(findings[0].tensor.as_deref(), Some("blk.0.attn_q.weight"));
        assert!(findings[0]
            .to_string()
            .starts_with("error[tensor-offset] tensor 'blk.0.attn_q.weight'"));
    }

    #[test]
    fn tensor_starting_inside_the_previous_one_is_an_overlap_error() {
        let mut bytes = valid();
        let at = offset_at(&bytes, "blk.0.attn_q.weight", 2);
        patch_u64(&mut bytes, at, 0);
        let findings = check_gguf(&bytes);
        assert_eq!(rules(&findings), vec!["tensor-overlap"]);
        assert_eq!(findings[0].tensor.as_deref(), Some("blk.0.attn_q.weight"));
    }

    #[test]
    fn missing_architecture_is_an_error() {
        let mut writer = GgufWriter::new();
        add_tensors(&mut writer);
        let bytes = writer.write_to_bytes().expect("write fixture");
        assert_eq!(rules(&check_gguf(&bytes)), vec!["meta-architecture"]);
    }

    #[test]
    fn architecture_that_is_not_a_string_is_an_error() {
        let mut writer = GgufWriter::new();
        writer.add_u32("general.architecture", 7);
        add_tensors(&mut writer);
        let bytes = writer.write_to_bytes().expect("write fixture");
        assert_eq!(rules(&check_gguf(&bytes)), vec!["meta-architecture"]);
    }

    #[test]
    fn llama_without_block_count_is_a_warning_naming_the_key() {
        let mut writer = metadata_writer("llama.block_count");
        add_tensors(&mut writer);
        let findings = check_gguf(&writer.write_to_bytes().expect("write fixture"));
        assert_eq!(rules(&findings), vec!["meta-arch-keys"]);
        assert_eq!(findings[0].severity, Severity::Warning);
        assert!(findings[0].message.contains("llama.block_count"));
    }

    #[test]
    fn other_architectures_get_no_key_warnings() {
        let mut writer = GgufWriter::new();
        writer.add_string("general.architecture", "foo");
        add_tensors(&mut writer);
        assert!(check_gguf(&writer.write_to_bytes().expect("write fixture")).is_empty());
    }

    #[test]
    fn random_bytes_after_a_valid_header_never_panic() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..256 {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend_from_slice(&3u32.to_le_bytes());
            for _ in 0..64 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bytes.push(state as u8);
            }
            let _ = check_gguf(&bytes);
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
