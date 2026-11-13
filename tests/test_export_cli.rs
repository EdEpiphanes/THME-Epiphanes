// SPDX-License-Identifier: AGPL-3.0-only
use clap::Parser;
use thme::cli::{Cli, Commands};
use thme::exporter::gguf_writer::GgufWriter;
use thme::parser::gguf::{GgmlType, GgufParser};

#[test]
fn export_cli_parses_supported_types_and_preserves_interop_command() {
    for (value, expected) in [
        ("bitnet-b1.58", "TQ1_0"),
        ("bitnet", "TQ1_0"),
        ("q8_0", "Q8_0"),
        ("fp16", "FP16"),
    ] {
        let cli = Cli::try_parse_from([
            "thme",
            "export",
            "model.gguf",
            "--output",
            "out.gguf",
            "--quant",
            value,
        ])
        .expect("parse export arguments");
        let Commands::Export(args) = cli.command else {
            panic!("expected export command");
        };
        assert_eq!(format!("{:?}", args.quant), expected);
    }

    assert!(matches!(
        Cli::try_parse_from(["thme", "interop"])
            .expect("parse interop")
            .command,
        Commands::Interop(_)
    ));
    assert!(
        Cli::try_parse_from(["thme", "export", "m.gguf", "-o", "o.gguf", "--quant", "q4"]).is_err()
    );
}

#[test]
fn export_command_writes_source_backed_gguf() {
    let dir = tempfile::tempdir().expect("temp dir");
    let source_path = dir.path().join("source.gguf");
    let output_path = dir.path().join("out.gguf");
    let payload = [0x00, 0x3c, 0x00, 0xc0, 0x00, 0x42, 0x00, 0x34];
    let mut writer = GgufWriter::new();
    writer
        .add_string("general.architecture", "llama")
        .add_u32("llama.block_count", 1)
        .add_u32("llama.embedding_length", 4)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.attention.head_count_kv", 1)
        .add_u32("llama.context_length", 128)
        .add_tensor("weights", vec![4], GgmlType::F16, payload.to_vec());
    writer.write_to_file(&source_path).expect("write source");

    let model = GgufParser::parse_from_file(&source_path).expect("parse source");
    let cli = Cli::try_parse_from([
        "thme",
        "export",
        source_path.to_str().expect("utf8 source path"),
        "--output",
        output_path.to_str().expect("utf8 output path"),
        "--quant",
        "fp16",
    ])
    .expect("parse CLI");
    let Commands::Export(args) = cli.command else {
        panic!("expected export command");
    };
    thme::cli::export::execute(args).expect("export succeeds");

    let exported = GgufParser::parse_from_file(&output_path).expect("parse exported model");
    assert_eq!(exported.tensors.len(), model.tensors.len());
    assert!(
        std::fs::metadata(output_path)
            .expect("output metadata")
            .len()
            > 0
    );
}
