// SPDX-License-Identifier: AGPL-3.0-only
//! Clean-Machine Standalone Verification Integration Test (Milestone v0.2.0.9).
//! Verifies that THME executes cleanly on ephemeral environments without
//! external dynamic dependencies, cache assumptions, or hardcoded paths.

use clap::error::ErrorKind;
use clap::Parser;
use thme::cli::Cli;
use thme::exporter::gguf_writer::GgufWriter;
use thme::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
use thme::parser::gguf::{GgmlType, GgufParser};
use thme::parser::ModelImporter;

#[test]
fn clean_machine_cli_parses_core_commands() {
    let help_cases = [
        vec!["thme", "--help"],
        vec!["thme", "--version"],
        vec!["thme", "inspect", "--help"],
        vec!["thme", "profile", "--help"],
        vec!["thme", "plan", "--help"],
        vec!["thme", "validate", "--help"],
        vec!["thme", "tweak", "--help"],
        vec!["thme", "export", "--help"],
        vec!["thme", "roofline", "--help"],
        vec!["thme", "speculate", "--help"],
        vec!["thme", "co-surgery", "--help"],
        vec!["thme", "interop", "--help"],
        vec!["thme", "mcp", "--help"],
    ];

    for args in &help_cases {
        let res = Cli::try_parse_from(args.clone());
        match res {
            Ok(_) => {}
            Err(e) => {
                assert!(
                    e.kind() == ErrorKind::DisplayHelp || e.kind() == ErrorKind::DisplayVersion,
                    "Unexpected parse failure for {:?}: {:?}",
                    args,
                    e
                );
            }
        }
    }
}

#[test]
fn clean_machine_standalone_export_and_inspection() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let source_path = temp_dir.path().join("source.gguf");
    let export_dest = temp_dir.path().join("clean_model.gguf");

    // 1. Create a minimal valid GGUF file
    let mut writer = GgufWriter::new();
    writer
        .add_string("general.architecture", "llama")
        .add_u32("llama.block_count", 1)
        .add_u32("llama.embedding_length", 4)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.attention.head_count_kv", 1)
        .add_u32("llama.context_length", 128)
        .add_tensor(
            "weights",
            vec![4],
            GgmlType::F16,
            vec![0x00, 0x3c, 0x00, 0xc0, 0x00, 0x42, 0x00, 0x34],
        );
    writer
        .write_to_file(&source_path)
        .expect("write source gguf");

    // 2. Parse from file to obtain a source-backed ModelIR
    let model = GgufParser::parse_from_file(&source_path).expect("parse source gguf");

    // 3. Export to clean path
    let cfg = ExportConfig {
        format: ExportFormat::GGUFv3,
        quant_type: ExportQuantType::FP16,
        output_path: export_dest.clone(),
        compute_checksum: true,
    };
    let rep = ModelExporter::export(&model, &cfg).expect("export to clean path");
    assert!(rep.exported_bytes > 0);
    assert!(export_dest.exists());

    // 4. Re-import zero-copy
    let imported = ModelImporter::import(&export_dest).expect("re-import exported GGUF");
    assert_eq!(imported.name, "clean_model");
    assert_eq!(imported.tensors.len(), model.tensors.len());
}
