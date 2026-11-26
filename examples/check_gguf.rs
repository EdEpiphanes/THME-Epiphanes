// SPDX-License-Identifier: AGPL-3.0-only
//! Runs THME's GGUF structure checker on files: `cargo run --release --example check_gguf -- a.gguf b.gguf`.
//! Exits with code 1 when any file has an error finding (THME-2208).
use thme::exporter::gguf_check::{check_gguf, Severity};

fn main() {
    let mut failed = false;
    for path in std::env::args().skip(1) {
        match std::fs::read(&path) {
            Ok(bytes) => {
                let findings = check_gguf(&bytes);
                let errors = findings
                    .iter()
                    .filter(|f| f.severity == Severity::Error)
                    .count();
                println!(
                    "{path}: {} bytes, {errors} errors, {} warnings",
                    bytes.len(),
                    findings.len() - errors
                );
                for finding in &findings {
                    println!("  {finding}");
                }
                failed |= errors > 0;
            }
            Err(err) => {
                println!("{path}: cannot read: {err}");
                failed = true;
            }
        }
    }
    std::process::exit(i32::from(failed));
}
