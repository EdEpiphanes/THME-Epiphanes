//! Independent consumer and optional runtime checks for exported model files.

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::tempdir;
use thme::exporter::safetensors_writer::SafeTensorsWriter;
use thme::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
use thme::ir::{ModelArchitecture, ModelIR};
use thme::parser::SafetensorsScanner;

fn safetensors_source(path: &Path) -> (ModelIR, Vec<u8>) {
    let payload = [0x00, 0x3c, 0x00, 0xc0, 0x00, 0x42, 0x00, 0x34].to_vec();
    let mut writer = SafeTensorsWriter::new();
    writer.add_tensor("weights", "F16", vec![4], payload.clone());
    writer
        .write_to_file(path)
        .expect("write source SafeTensors");
    let tensors = SafetensorsScanner::parse_metadata_from_file(path)
        .expect("parse source SafeTensors metadata");
    let model = ModelIR::new(
        "compatibility-fixture".into(),
        ModelArchitecture::Llama,
        1,
        4,
        1,
        1,
        4,
        128,
        tensors,
    );
    (model, payload)
}

#[test]
fn safetensors_export_loads_with_independent_safetensors_consumer() {
    let dir = tempdir().expect("create temporary directory");
    let (model, expected) = safetensors_source(&dir.path().join("source.safetensors"));
    let output = dir.path().join("export.safetensors");
    ModelExporter::export(
        &model,
        &ExportConfig {
            format: ExportFormat::SafeTensors,
            quant_type: ExportQuantType::FP16,
            output_path: output.clone(),
            compute_checksum: true,
        },
    )
    .expect("export SafeTensors");

    let bytes = std::fs::read(output).expect("read exported SafeTensors");
    let consumer = safetensors::SafeTensors::deserialize(&bytes)
        .expect("independent safetensors crate should accept exported file");
    let tensor = consumer.tensor("weights").expect("find exported tensor");
    assert_eq!(tensor.dtype(), safetensors::Dtype::F16);
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.data(), expected);
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn run(command: &mut Command, message: &str) {
    let output = command.output().expect(message);
    assert!(output.status.success(), "{message}");
}

#[test]
fn exported_gguf_runs_in_configured_llama_cpp_and_ollama() {
    let required = std::env::var("THME_REQUIRE_RUNTIME_COMPAT").as_deref() == Ok("1");
    let source = env_path("THME_COMPAT_MODEL");
    let llama = env_path("THME_LLAMA_CPP");
    let ollama = env_path("THME_OLLAMA");

    if source.is_none() || llama.is_none() || ollama.is_none() {
        assert!(
            !required,
            "runtime compatibility requires THME_COMPAT_MODEL, THME_LLAMA_CPP, and THME_OLLAMA"
        );
        eprintln!("runtime compatibility skipped: downstream runtimes/model are not configured");
        return;
    }

    let source = source.expect("checked above");
    let llama = llama.expect("checked above");
    let ollama = ollama.expect("checked above");
    let model = thme::parser::ModelImporter::import(&source)
        .expect("import configured F16 compatibility model");
    let dir = tempdir().expect("create temporary directory");
    let exported = dir.path().join("compatibility.gguf");
    ModelExporter::export(
        &model,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::FP16,
            output_path: exported.clone(),
            compute_checksum: false,
        },
    )
    .expect("export compatibility model to GGUF");

    run(
        Command::new(llama)
            .arg("-m")
            .arg(&exported)
            .arg("-p")
            .arg("Reply with one word.")
            .arg("-n")
            .arg("1"),
        "llama.cpp failed to load and execute the exported GGUF",
    );

    let model_name = format!("thme-compat-{}", std::process::id());
    let model_file = dir.path().join("Modelfile");
    let exported_path = exported
        .canonicalize()
        .expect("resolve exported model path")
        .to_string_lossy()
        .replace('\\', "/");
    std::fs::write(&model_file, format!("FROM {exported_path}\n")).expect("write Ollama Modelfile");

    run(
        Command::new(&ollama)
            .arg("create")
            .arg(&model_name)
            .arg("-f")
            .arg(&model_file),
        "Ollama failed to import the exported GGUF",
    );
    let execution = Command::new(&ollama)
        .arg("run")
        .arg(&model_name)
        .arg("Reply with one word.")
        .output()
        .expect("run exported GGUF in Ollama");
    let cleanup = Command::new(&ollama).arg("rm").arg(&model_name).output();
    assert!(
        execution.status.success(),
        "Ollama failed to execute exported GGUF"
    );
    assert!(cleanup.is_ok(), "failed to clean up temporary Ollama model");
}
