// SPDX-License-Identifier: AGPL-3.0-only
//! Legacy HDF5 (`.h5`) Keras files: the architecture is a JSON string in the `model_config`
//! attribute, and the Keras rules (Lambda layers, unknown classes) are applied to it.
//!
//! This is not an HDF5 reader. The first `limit` bytes are read in one bounded buffer, and every
//! `{"class_name"` object is cut out by brace matching and parsed as JSON. Limits: attributes
//! stored compressed, chunked or in fragmented heaps are not raw JSON and are missed; a file
//! larger than the cap is searched only up to it, and the finding says so. A clean result means
//! no code-executing layer was found in the readable part of the file. It never means safe.

use std::path::Path;

use super::{read_limited, walk_keras_config, ScanReport, Severity};

/// HDF5 superblock signature at offset 0.
pub(super) const MAGIC: &[u8] = b"\x89HDF\r\n\x1a\n";
/// Largest file searched for `model_config`; a larger file is searched only up to this size.
pub(super) const MAX_BYTES: u64 = 64 << 20;
/// Keras writes the model object as `{"class_name": ...}`, so that is where a config starts.
const CONFIG_START: &[u8] = b"{\"class_name\"";

/// Scans the first `limit` bytes of an `.h5` file for Keras `model_config` objects.
pub(super) fn scan_h5(path: &Path, rel: &str, limit: u64, report: &mut ScanReport) {
    let data = match read_limited(path, limit.saturating_add(1)) {
        Ok(d) => d,
        Err(e) => {
            report.add(Severity::Medium, "unreadable", rel, format!("{e:#}"));
            return;
        }
    };
    if !data.starts_with(MAGIC) {
        report.add(
            Severity::Medium,
            "hdf5-malformed",
            rel,
            "file has an .h5 name but no HDF5 signature; not inspected as HDF5".to_string(),
        );
        return;
    }
    if data.len() as u64 > limit {
        report.add(
            Severity::Medium,
            "hdf5-not-inspected",
            rel,
            format!("larger than the {limit}-byte limit; only the first {limit} bytes were searched for model_config"),
        );
    }
    let data = &data[..data.len().min(limit as usize)];

    let mut seen = false;
    let mut parsed = false;
    let mut pos = MAGIC.len();
    while let Some(start) = find(data, CONFIG_START, pos) {
        seen = true;
        let Some(end) = object_end(data, start) else {
            report.add(
                Severity::Medium,
                "hdf5-not-inspected",
                rel,
                format!("JSON object at byte {start} is cut off before its end (truncated file or the size limit); not inspected"),
            );
            break;
        };
        match serde_json::from_slice::<serde_json::Value>(&data[start..end]) {
            Ok(json) => {
                parsed = true;
                let at = format!("{rel}!model_config");
                walk_keras_config(&json, 0, &at, "model_config", true, report);
            }
            Err(e) => report.add(
                Severity::Medium,
                "hdf5-not-inspected",
                rel,
                format!("JSON object at byte {start} looks like a Keras config but is not valid ({e}); not inspected"),
            ),
        }
        pos = end;
    }
    if parsed {
        report.add(
            Severity::Info,
            "hdf5-partial",
            rel,
            "model_config was read from the readable bytes; attributes stored compressed, chunked or in fragmented heaps are not searched, so this is not proof that the model is safe".to_string(),
        );
    } else if !seen {
        report.add(
            Severity::Info,
            "hdf5-not-inspected",
            rel,
            "no model_config JSON in the readable bytes (a weights-only file, or attributes stored compressed or chunked); not inspected".to_string(),
        );
    }
}

/// Offset of the first `pat` at or after `from`.
fn find(data: &[u8], pat: &[u8], from: usize) -> Option<usize> {
    data.get(from..)?
        .windows(pat.len())
        .position(|w| w == pat)
        .map(|i| i + from)
}

/// End (exclusive) of the JSON object that opens at `start`, found by brace matching outside
/// strings. None when the bytes end before it closes.
fn object_end(data: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in data.iter().enumerate().skip(start) {
        if in_string {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                _ => {}
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                _ => {}
            }
        }
    }
    None
}
