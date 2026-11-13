//! The Epiphanes model format, `.epiph` (open specification: `docs/formats/epiph-v1.md`).
//!
//! One file holds a model's weights, its full description, and what is known about it:
//! recipe, provenance, quality-gate results, calibration statistics and lineage. Reading a
//! `.epiph` file yields an ordinary [`ModelIR`] whose tensors point into the file, so every
//! THME command (evaluate, search, export to GGUF or SafeTensors, the Studio) can use it.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::exporter::{compute_file_sha256, compute_sha256};
use crate::inference::{decode_payload, read_tensor_bytes, CalibrationFile};
use crate::ir::tensor::TensorSource;
use crate::ir::ModelIR;

/// File extension, without the dot.
pub const EXTENSION: &str = "epiph";
/// First eight bytes of every file: `\x89EPIPH\r\n`.
pub const MAGIC: [u8; 8] = [0x89, b'E', b'P', b'I', b'P', b'H', 0x0D, 0x0A];
/// Format version this module reads and writes.
pub const MAJOR: u16 = 1;
pub const MINOR: u16 = 0;
/// Every payload starts on a multiple of this many bytes.
pub const ALIGN: u64 = 64;
const HEADER_LEN: u64 = 32;
/// Largest manifest accepted, to refuse corrupt length fields before allocating.
const MAX_MANIFEST: u64 = 1 << 30;

/// Where one payload lives in the data section, and its digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
}

/// A non-tensor payload, such as calibration statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attachment {
    pub kind: String,
    pub encoding: String,
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
}

/// A file the model was packed from: its name (never a directory) and digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceFile {
    pub file_name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub tool: String,
    pub tool_version: String,
    pub created_unix: u64,
    #[serde(default)]
    pub sources: Vec<SourceFile>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Lineage {
    /// SHA-256 digests of the `.epiph` files this one was derived from.
    #[serde(default)]
    pub parents: Vec<String>,
}

/// The JSON manifest at the start of every file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: String,
    pub model: Value,
    pub tensors: Vec<TensorEntry>,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
    #[serde(default)]
    pub recipe: Option<Value>,
    #[serde(default)]
    pub quality: Vec<Value>,
    pub provenance: Provenance,
    #[serde(default)]
    pub lineage: Lineage,
    #[serde(default)]
    pub annotations: BTreeMap<String, String>,
}

/// Everything besides the weights that a packed file should carry.
#[derive(Debug, Clone, Default)]
pub struct PackOptions {
    pub recipe: Option<Value>,
    /// Quality-gate results; each must carry an `evidence` label.
    pub quality: Vec<Value>,
    pub calibration: Option<CalibrationFile>,
    pub parents: Vec<String>,
    pub annotations: BTreeMap<String, String>,
    /// Fixed creation time for reproducible files; the current time when `None`.
    pub created_unix: Option<u64>,
}

/// What a write produced.
#[derive(Debug, Clone, PartialEq)]
pub struct PackSummary {
    pub tensors: usize,
    pub attachments: usize,
    pub file_bytes: u64,
    pub sha256: String,
}

/// True when `path` names an `.epiph` file (by extension).
pub fn is_epiph(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(EXTENSION))
}

fn align(n: u64) -> u64 {
    n.div_ceil(ALIGN) * ALIGN
}

const EVIDENCE: [&str; 5] = ["measured", "tested", "estimate", "simulated", "imported"];

/// Writes `model` (every tensor must be file-backed) and `options` to `out`.
pub fn write(model: &ModelIR, options: &PackOptions, out: &Path) -> Result<PackSummary> {
    for (i, q) in options.quality.iter().enumerate() {
        let label = q.get("evidence").and_then(Value::as_str).unwrap_or("");
        if !EVIDENCE.contains(&label) {
            bail!("Quality entry {i} has no valid evidence label (one of {EVIDENCE:?})");
        }
    }
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(model.tensors.len() + 1);
    let mut tensors = Vec::with_capacity(model.tensors.len());
    let mut offset = 0u64;
    for tensor in &model.tensors {
        let bytes = read_tensor_bytes(tensor)
            .with_context(|| format!("Cannot pack tensor '{}'", tensor.name))?;
        tensors.push(TensorEntry {
            name: tensor.name.clone(),
            offset,
            length: bytes.len() as u64,
            sha256: compute_sha256(&bytes),
        });
        offset = align(offset + bytes.len() as u64);
        payloads.push(bytes);
    }
    let mut attachments = Vec::new();
    if let Some(calibration) = &options.calibration {
        let bytes = serde_json::to_vec(calibration)?;
        attachments.push(Attachment {
            kind: "calibration".into(),
            encoding: "json".into(),
            offset,
            length: bytes.len() as u64,
            sha256: compute_sha256(&bytes),
        });
        payloads.push(bytes);
    }

    let mut sources = BTreeMap::new();
    for source in model.tensors.iter().filter_map(|t| t.source.as_ref()) {
        if !sources.contains_key(&source.path) {
            sources.insert(source.path.clone(), compute_file_sha256(&source.path)?);
        }
    }
    let created_unix = match options.created_unix {
        Some(t) => t,
        None => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    let manifest = Manifest {
        format: "epiph".into(),
        version: format!("{MAJOR}.{MINOR}"),
        model: serde_json::to_value(model)?,
        tensors,
        attachments,
        recipe: options.recipe.clone(),
        quality: options.quality.clone(),
        provenance: Provenance {
            tool: "thme".into(),
            tool_version: env!("CARGO_PKG_VERSION").into(),
            created_unix,
            sources: sources
                .into_iter()
                .map(|(path, sha256)| SourceFile {
                    file_name: path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    sha256,
                })
                .collect(),
        },
        lineage: Lineage {
            parents: options.parents.clone(),
        },
        annotations: options.annotations.clone(),
    };
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let data_offset = align(HEADER_LEN + manifest_bytes.len() as u64);

    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("Failed to create {parent:?}"))?;
    }
    let mut file = File::create(out).with_context(|| format!("Failed to create {out:?}"))?;
    file.write_all(&MAGIC)?;
    file.write_all(&MAJOR.to_le_bytes())?;
    file.write_all(&MINOR.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;
    file.write_all(&(manifest_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&data_offset.to_le_bytes())?;
    file.write_all(&manifest_bytes)?;
    let mut written = HEADER_LEN + manifest_bytes.len() as u64;
    let pad = |file: &mut File, to: u64, written: &mut u64| -> Result<()> {
        let zeros = vec![0u8; usize::try_from(to - *written)?];
        file.write_all(&zeros)?;
        *written = to;
        Ok(())
    };
    pad(&mut file, data_offset, &mut written)?;
    let places = manifest
        .tensors
        .iter()
        .map(|t| t.offset)
        .chain(manifest.attachments.iter().map(|a| a.offset));
    for (bytes, at) in payloads.iter().zip(places) {
        pad(&mut file, data_offset + at, &mut written)?;
        file.write_all(bytes)?;
        written += bytes.len() as u64;
    }
    file.flush()?;
    drop(file);
    Ok(PackSummary {
        tensors: manifest.tensors.len(),
        attachments: manifest.attachments.len(),
        file_bytes: written,
        sha256: compute_file_sha256(out)?,
    })
}

/// Header fields and manifest of a file, after the structural checks of the specification.
struct Opened {
    manifest: Manifest,
    data_offset: u64,
}

fn open(path: &Path) -> Result<Opened> {
    let mut file = File::open(path).with_context(|| format!("Failed to open {path:?}"))?;
    let file_len = file.metadata()?.len();
    let mut header = [0u8; HEADER_LEN as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("{path:?} is too short to be an .epiph file"))?;
    let field = |range: std::ops::Range<usize>| header.get(range).unwrap_or(&[]);
    if field(0..8) != MAGIC {
        bail!("{path:?} is not an .epiph file (wrong magic bytes)");
    }
    let u16_at =
        |at: usize| -> Result<u16> { Ok(u16::from_le_bytes(field(at..at + 2).try_into()?)) };
    let u32_at =
        |at: usize| -> Result<u32> { Ok(u32::from_le_bytes(field(at..at + 4).try_into()?)) };
    let u64_at =
        |at: usize| -> Result<u64> { Ok(u64::from_le_bytes(field(at..at + 8).try_into()?)) };
    let major = u16_at(8)?;
    if major != MAJOR {
        bail!("{path:?} is .epiph version {major}.x; this reader supports {MAJOR}.x");
    }
    let flags = u32_at(12)?;
    if flags != 0 {
        bail!("{path:?} sets reserved flags {flags:#x}; refusing to guess their meaning");
    }
    let manifest_len = u64_at(16)?;
    let data_offset = u64_at(24)?;
    if manifest_len > MAX_MANIFEST || HEADER_LEN + manifest_len > file_len {
        bail!("{path:?} has a manifest length {manifest_len} beyond the file");
    }
    if data_offset % ALIGN != 0 || data_offset < HEADER_LEN + manifest_len || data_offset > file_len
    {
        bail!("{path:?} has an invalid data offset {data_offset}");
    }
    let mut manifest_bytes = vec![0u8; usize::try_from(manifest_len)?];
    file.read_exact(&mut manifest_bytes)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("{path:?} has an unreadable manifest"))?;
    if manifest.format != "epiph" {
        bail!("{path:?} manifest declares format '{}'", manifest.format);
    }
    // Payload ranges: aligned, inside the file, not overlapping.
    let data_len = file_len - data_offset;
    let mut ranges: Vec<(u64, u64, String)> = manifest
        .tensors
        .iter()
        .map(|t| (t.offset, t.length, format!("tensor '{}'", t.name)))
        .chain(
            manifest
                .attachments
                .iter()
                .map(|a| (a.offset, a.length, format!("attachment '{}'", a.kind))),
        )
        .collect();
    ranges.sort_by_key(|r| r.0);
    let mut end = 0u64;
    for (offset, length, what) in &ranges {
        let stop = offset.checked_add(*length);
        if offset % ALIGN != 0 || *offset < end || stop.is_none_or(|s| s > data_len) {
            bail!("{path:?}: {what} has an invalid or overlapping payload range");
        }
        end = stop.unwrap_or(u64::MAX);
    }
    Ok(Opened {
        manifest,
        data_offset,
    })
}

/// Reads a file into a model whose tensors point into it, plus its manifest.
pub fn read(path: &Path) -> Result<(ModelIR, Manifest)> {
    let Opened {
        manifest,
        data_offset,
    } = open(path)?;
    let mut model: ModelIR = serde_json::from_value(manifest.model.clone())
        .with_context(|| format!("{path:?} has an unreadable model description"))?;
    if model.tensors.len() != manifest.tensors.len() {
        bail!(
            "{path:?} describes {} tensors but stores {}",
            model.tensors.len(),
            manifest.tensors.len()
        );
    }
    let mut seen = BTreeSet::new();
    for (tensor, entry) in model.tensors.iter_mut().zip(&manifest.tensors) {
        if tensor.name != entry.name || !seen.insert(entry.name.clone()) {
            bail!(
                "{path:?}: tensor table does not match the model at '{}'",
                entry.name
            );
        }
        tensor.source = Some(TensorSource {
            path: path.to_path_buf(),
            byte_offset: data_offset + entry.offset,
            byte_len: entry.length,
        });
    }
    Ok((model, manifest))
}

fn read_range(file: &mut File, at: u64, length: u64) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; usize::try_from(length)?];
    file.seek(SeekFrom::Start(at))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// The calibration statistics attached to a file, if any (digest checked).
pub fn read_calibration(path: &Path) -> Result<Option<CalibrationFile>> {
    let Opened {
        manifest,
        data_offset,
    } = open(path)?;
    let Some(entry) = manifest
        .attachments
        .iter()
        .find(|a| a.kind == "calibration")
    else {
        return Ok(None);
    };
    let mut file = File::open(path)?;
    let bytes = read_range(&mut file, data_offset + entry.offset, entry.length)?;
    if compute_sha256(&bytes) != entry.sha256 {
        bail!("{path:?}: calibration attachment fails its SHA-256 check");
    }
    Ok(Some(serde_json::from_slice(&bytes)?))
}

/// Result of a full verification.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyReport {
    pub tensors: usize,
    pub attachments: usize,
    pub payload_bytes: u64,
}

/// Checks every payload digest and that each tensor decodes at its declared shape and type.
pub fn verify(path: &Path) -> Result<VerifyReport> {
    let (model, manifest) = read(path)?;
    let Opened { data_offset, .. } = open(path)?;
    let mut file = File::open(path)?;
    let mut payload_bytes = 0u64;
    for (tensor, entry) in model.tensors.iter().zip(&manifest.tensors) {
        let bytes = read_range(&mut file, data_offset + entry.offset, entry.length)?;
        if compute_sha256(&bytes) != entry.sha256 {
            bail!("{path:?}: tensor '{}' fails its SHA-256 check", entry.name);
        }
        decode_payload(&bytes, &tensor.original_dtype, tensor.numel).with_context(|| {
            format!(
                "{path:?}: tensor '{}' does not match its shape and type",
                entry.name
            )
        })?;
        payload_bytes += entry.length;
    }
    for entry in &manifest.attachments {
        let bytes = read_range(&mut file, data_offset + entry.offset, entry.length)?;
        if compute_sha256(&bytes) != entry.sha256 {
            bail!(
                "{path:?}: attachment '{}' fails its SHA-256 check",
                entry.kind
            );
        }
        payload_bytes += entry.length;
    }
    Ok(VerifyReport {
        tensors: manifest.tensors.len(),
        attachments: manifest.attachments.len(),
        payload_bytes,
    })
}

#[cfg(test)]
mod tests;
