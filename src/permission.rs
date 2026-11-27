// SPDX-License-Identifier: AGPL-3.0-only
//! Permission and ownership record (v0.2.7.4, THME-3008, dossier WP19).
//!
//! Reverse-engineering features (architecture inference, closed-format analysis) call
//! [`require`] first. It refuses unless a record with a basis and a purpose statement sits next
//! to the model file and names the model's current SHA-256, so a record cannot be reused for a
//! different or modified file. The record is the user's own statement: THME stores and checks
//! it, it does not verify that the statement is true or lawful. The wording shown to users
//! still needs legal review (open item in the ticket).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::exporter::compute_file_sha256;

/// Why the user may analyse the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Basis {
    /// The user owns the model.
    Owner,
    /// A licence grants the right.
    Licence,
    /// The rights holder gave written permission.
    WrittenPermission,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRecord {
    pub version: u32,
    pub model_sha256: String,
    pub basis: Basis,
    pub purpose: String,
    /// Seconds since the Unix epoch, supplied by the caller.
    pub recorded_at: u64,
}

/// Where the record for `model` is stored: `<model file name>.thme-permission.json` beside it.
pub fn record_path(model: &Path) -> PathBuf {
    let mut name = model.as_os_str().to_os_string();
    name.push(".thme-permission.json");
    PathBuf::from(name)
}

/// Stores a record for the model file's current hash.
pub fn grant(
    model: &Path,
    basis: Basis,
    purpose: &str,
    recorded_at: u64,
) -> Result<PermissionRecord> {
    if purpose.trim().is_empty() {
        bail!("a purpose statement is required to record permission");
    }
    let record = PermissionRecord {
        version: 1,
        model_sha256: hash(model)?,
        basis,
        purpose: purpose.trim().to_string(),
        recorded_at,
    };
    let json = serde_json::to_string_pretty(&record)?;
    let at = record_path(model);
    std::fs::write(&at, json).with_context(|| format!("writing {}", at.display()))?;
    Ok(record)
}

/// Returns the stored record, or an error that says what is missing. Call before any
/// reverse-engineering feature runs.
pub fn require(model: &Path) -> Result<PermissionRecord> {
    let at = record_path(model);
    let text = std::fs::read_to_string(&at).with_context(|| {
        format!(
            "no permission record at {}: record the basis and purpose for this model first",
            at.display()
        )
    })?;
    let record: PermissionRecord = serde_json::from_str(&text)
        .with_context(|| format!("permission record {} is not valid", at.display()))?;
    if record.version != 1 {
        bail!(
            "permission record version {} is not supported",
            record.version
        );
    }
    if record.purpose.trim().is_empty() {
        bail!("permission record {} has an empty purpose", at.display());
    }
    if record.model_sha256 != hash(model)? {
        bail!(
            "permission record {} was made for a different model file (hash differs): record it again",
            at.display()
        );
    }
    Ok(record)
}

fn hash(model: &Path) -> Result<String> {
    if !model.is_file() {
        bail!("{} is not a model file", model.display());
    }
    compute_file_sha256(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(dir: &Path, bytes: &[u8]) -> PathBuf {
        let p = dir.join("m.bin");
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn refuses_without_a_record() {
        let dir = tempfile::tempdir().unwrap();
        let m = model(dir.path(), b"abc");
        let err = require(&m).unwrap_err();
        assert!(
            format!("{err:#}").contains("no permission record"),
            "{err:#}"
        );
    }

    #[test]
    fn grant_then_require_round_trips_with_the_known_hash() {
        let dir = tempfile::tempdir().unwrap();
        let m = model(dir.path(), b"abc");
        grant(&m, Basis::Owner, "  audit my own fine-tune ", 42).unwrap();
        let r = require(&m).unwrap();
        // SHA-256("abc"), written by hand from the standard test vector.
        assert_eq!(
            r.model_sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            (r.basis, r.purpose.as_str(), r.recorded_at),
            (Basis::Owner, "audit my own fine-tune", 42)
        );
    }

    #[test]
    fn empty_purpose_is_refused_and_nothing_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let m = model(dir.path(), b"abc");
        assert!(grant(&m, Basis::Licence, "  ", 1).is_err());
        assert!(!record_path(&m).exists());
    }

    #[test]
    fn changed_model_invalidates_the_record() {
        let dir = tempfile::tempdir().unwrap();
        let m = model(dir.path(), b"abc");
        grant(&m, Basis::WrittenPermission, "interop study", 1).unwrap();
        std::fs::write(&m, b"abd").unwrap();
        let err = require(&m).unwrap_err();
        assert!(
            format!("{err:#}").contains("different model file"),
            "{err:#}"
        );
    }

    #[test]
    fn hostile_records_are_errors_not_panics() {
        let dir = tempfile::tempdir().unwrap();
        let m = model(dir.path(), b"abc");
        for body in [
            "",
            "{",
            "null",
            "[]",
            r#"{"version":1}"#,
            r#"{"version":2,"model_sha256":"x","basis":"owner","purpose":"p","recorded_at":1}"#,
            r#"{"version":1,"model_sha256":"x","basis":"owner","purpose":" ","recorded_at":1}"#,
            r#"{"version":1,"model_sha256":"x","basis":"pirate","purpose":"p","recorded_at":1}"#,
        ] {
            std::fs::write(record_path(&m), body).unwrap();
            assert!(require(&m).is_err(), "{body}");
        }
    }

    #[test]
    fn folders_and_missing_files_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        assert!(grant(dir.path(), Basis::Owner, "p", 1).is_err());
        assert!(require(&dir.path().join("nope.bin")).is_err());
    }
}
