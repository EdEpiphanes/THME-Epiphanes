use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
use crate::parser::ModelImporter;

#[derive(Args, Debug)]
pub struct ExportArgs {
    /// Source GGUF file or SafeTensors model directory
    #[arg(value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Destination model artifact
    #[arg(short, long, value_name = "OUTPUT")]
    pub output: PathBuf,

    /// Container format: gguf or safetensors
    #[arg(long, default_value = "gguf")]
    pub format: ExportFormat,

    /// Weight type: bitnet-b1.58, q8_0, fp16, or per-tensor (each tensor in its selected format)
    #[arg(long, default_value = "fp16", value_parser = parse_quant_type)]
    pub quant: ExportQuantType,

    /// Recipe JSON to apply before exporting; implies --quant per-tensor, so the artifact holds
    /// exactly the formats `thme evaluate --recipe` measured
    #[arg(long, value_name = "RECIPE")]
    pub recipe: Option<PathBuf>,

    /// With --recipe: skip operations the exporter cannot apply instead of failing
    #[arg(long)]
    pub allow_partial: bool,

    /// Compute SHA-256 checksum after export
    #[arg(long)]
    pub checksum: bool,
}

fn parse_quant_type(value: &str) -> std::result::Result<ExportQuantType, String> {
    match value.to_ascii_lowercase().as_str() {
        // BitNet b1.58 weights use the interoperable GGML ternary block encoding.
        "bitnet" | "bitnet-b1.58" | "bitnet_b158" | "ternary" | "tq1" | "tq1_0" => {
            Ok(ExportQuantType::TQ1_0)
        }
        "q8" | "q8_0" | "int8" => Ok(ExportQuantType::Q8_0),
        "fp16" | "f16" => Ok(ExportQuantType::FP16),
        "per-tensor" | "per_tensor" | "recipe" | "selected" => Ok(ExportQuantType::PerTensor),
        other => Err(format!(
            "unsupported quantization '{other}'; choose bitnet-b1.58 (GGUF TQ1_0), q8_0, fp16, or per-tensor"
        )),
    }
}

pub fn execute(args: ExportArgs) -> Result<()> {
    let source = ModelImporter::import(&args.model)
        .with_context(|| format!("Failed to load source model from {:?}", args.model))?;
    let (model, quant) = match &args.recipe {
        None => (source, args.quant),
        Some(path) => {
            if !matches!(
                args.quant,
                ExportQuantType::FP16 | ExportQuantType::PerTensor
            ) {
                anyhow::bail!("--recipe exports each tensor in its recipe format; drop --quant or use --quant per-tensor");
            }
            let model = apply_recipe(&source, path, args.allow_partial)?;
            (model, ExportQuantType::PerTensor)
        }
    };
    let report = ModelExporter::export(
        &model,
        &ExportConfig {
            format: args.format,
            quant_type: quant,
            output_path: args.output.clone(),
            compute_checksum: args.checksum,
        },
    )?;

    println!(
        "Exported {} tensors to {:?} ({} bytes)",
        report.total_tensors, args.output, report.exported_bytes
    );
    if let Some(checksum) = report.sha256_checksum {
        println!("SHA-256: {checksum}");
    }
    Ok(())
}

/// Applies a recipe exactly as `thme evaluate --recipe` does, returning the model whose
/// tensors carry their selected formats.
pub fn apply_recipe(
    source: &crate::ir::ModelIR,
    path: &std::path::Path,
    allow_partial: bool,
) -> Result<crate::ir::ModelIR> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("Failed to read recipe {path:?}"))?;
    let recipe: crate::surgery::recipe::TweakRecipe =
        serde_json::from_str(&text).with_context(|| format!("Failed to parse recipe {path:?}"))?;
    let registry = crate::cli::evaluate::load_registry(&[])?;
    let prepared =
        crate::cli::evaluate::prepare_candidate(source, &recipe, &registry, allow_partial)?;
    for (format, count) in &prepared.counts {
        println!("  {format}: {count} tensors");
    }
    Ok(prepared.model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporter::SafeTensorsWriter;
    use crate::inference::{
        evaluate_perplexity, fake_quantize, read_tensor_f32, read_token_file, ArchConfig,
        PerplexityConfig, WeightPrecision,
    };
    use crate::ir::TensorFormat;

    /// THME-2201 done-means: exporting a recipe and re-importing the file reproduces the
    /// perplexity `thme evaluate --recipe` measured for the same recipe.
    #[test]
    fn exported_recipe_reproduces_the_measured_perplexity() {
        let dir = tempfile::tempdir().unwrap();
        let (model_dir, tokens_path) =
            crate::cli::evaluate::tests::synthetic_checkpoint(dir.path());
        let recipe = dir.path().join("recipe.json");
        std::fs::write(
            &recipe,
            r#"{"name":"mixed","author":"t","target_model":"tiny","version":"1","description":"","operations":[
                {"type":"SetPrecision","params":{"start_layer":0,"end_layer":1,"role":"MlpDown","format":"Q4_0"}},
                {"type":"SetPrecision","params":{"start_layer":0,"end_layer":1,"role":"AttentionQ","format":"Q8_0"}},
                {"type":"SetPrecision","params":{"start_layer":1,"end_layer":1,"role":"MlpUp","format":"FP16"}}]}"#,
        )
        .unwrap();

        let report = crate::cli::evaluate::evaluate(&crate::cli::evaluate::EvaluateArgs {
            model: model_dir.clone(),
            recipe: recipe.clone(),
            tokens: tokens_path.clone(),
            eval_context: 32,
            eval_max_tokens: None,
            max_ppl_delta: 100.0,
            report: dir.path().join("report.json"),
            allow_partial: false,
            algorithms: Vec::new(),
            artifact: None,
            calibration: None,
        })
        .unwrap();
        let measured = report["evaluation"]["variants"][1]["perplexity"]
            .as_f64()
            .unwrap();

        let source = ModelImporter::import(&model_dir).unwrap();
        let model = apply_recipe(&source, &recipe, false).unwrap();
        let out = dir.path().join("mixed.gguf");
        ModelExporter::export(
            &model,
            &ExportConfig {
                format: ExportFormat::GGUFv3,
                quant_type: ExportQuantType::PerTensor,
                output_path: out.clone(),
                compute_checksum: false,
            },
        )
        .unwrap();

        let exported = ModelImporter::import(&out).unwrap();
        let dtype = |name: &str| {
            exported
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.original_dtype.clone())
                .unwrap()
        };
        assert_eq!(dtype("model.layers.0.mlp.down_proj.weight"), "Q4_0");
        assert_eq!(dtype("model.layers.1.self_attn.q_proj.weight"), "Q8_0");
        assert_eq!(dtype("model.layers.1.mlp.up_proj.weight"), "F16");
        assert_eq!(dtype("model.layers.0.mlp.up_proj.weight"), "F32");
        assert_eq!(dtype("model.norm.weight"), "F32");

        // Every weight in the file equals what the measured path computed for the recipe.
        for tensor in &model.tensors {
            let mut expected = read_tensor_f32(tensor).unwrap();
            if tensor.shape.len() >= 2 {
                crate::inference::apply_selected_format(tensor, &mut expected).unwrap();
            }
            let file = exported
                .tensors
                .iter()
                .find(|t| t.name == tensor.name)
                .unwrap();
            assert_eq!(read_tensor_f32(file).unwrap(), expected, "{}", tensor.name);
        }

        let arch = ArchConfig::detect(&exported, &out).unwrap();
        let lm = arch.load(&exported, WeightPrecision::AsStored).unwrap();
        let tokens = read_token_file(&tokens_path).unwrap();
        let config = PerplexityConfig {
            context_len: 32,
            max_tokens: None,
        };
        let reloaded = evaluate_perplexity(&*lm, &tokens, &config, |_, _, _| {})
            .unwrap()
            .perplexity;
        assert!(
            (reloaded - measured).abs() <= 1e-9 * measured.max(1.0),
            "exported {reloaded} vs measured {measured}"
        );
    }

    #[test]
    fn k_quant_tensors_export_bit_exactly_and_bad_assignments_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let values: Vec<f32> = (0..512)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) / 300.0)
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut w = SafeTensorsWriter::new();
        for name in ["a.weight", "b.weight", "c.weight", "d.weight"] {
            w.add_tensor(name, "F32", vec![2, 256], bytes.clone());
        }
        w.add_tensor(
            "norm.weight",
            "F32",
            vec![4],
            1.5f32.to_le_bytes().repeat(4),
        );
        let src = dir.path().join("src.safetensors");
        w.write_to_file(&src).unwrap();

        let mut model = ModelImporter::import(&src).unwrap();
        let formats = [
            ("a.weight", TensorFormat::Q4_K),
            ("b.weight", TensorFormat::Q5_K),
            ("c.weight", TensorFormat::Q6_K),
            ("d.weight", TensorFormat::FP32),
        ];
        for tensor in &mut model.tensors {
            tensor.selected_format = formats
                .iter()
                .find(|(n, _)| *n == tensor.name)
                .map_or(TensorFormat::Q4_K, |(_, f)| f.clone()); // norm: 1-D, kept as stored
        }
        let out = dir.path().join("k.gguf");
        let config = ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::PerTensor,
            output_path: out.clone(),
            compute_checksum: false,
        };
        ModelExporter::export(&model, &config).unwrap();

        let exported = ModelImporter::import(&out).unwrap();
        for (name, format) in &formats {
            let tensor = exported.tensors.iter().find(|t| t.name == *name).unwrap();
            let mut expected = values.clone();
            fake_quantize(&mut expected, format).unwrap();
            assert_eq!(read_tensor_f32(tensor).unwrap(), expected, "{name}");
        }
        let norm = exported
            .tensors
            .iter()
            .find(|t| t.name == "norm.weight")
            .unwrap();
        assert_eq!(norm.original_dtype, "F32");
        assert_eq!(read_tensor_f32(norm).unwrap(), vec![1.5; 4]);

        // A K-quant on rows narrower than 256 values cannot be written as GGUF.
        let mut narrow = SafeTensorsWriter::new();
        narrow.add_tensor("x.weight", "F32", vec![4, 128], bytes.clone());
        let narrow_src = dir.path().join("narrow.safetensors");
        narrow.write_to_file(&narrow_src).unwrap();
        let mut narrow_model = ModelImporter::import(&narrow_src).unwrap();
        narrow_model.tensors[0].selected_format = TensorFormat::Q4_K;
        let error = ModelExporter::export(&narrow_model, &config).unwrap_err();
        assert!(
            format!("{error:#}").contains("divisible by 256"),
            "{error:#}"
        );

        narrow_model.tensors[0].selected_format = TensorFormat::Custom("thme.hadamard_int4".into());
        let error = ModelExporter::export(&narrow_model, &config).unwrap_err();
        assert!(format!("{error:#}").contains("BYOA"), "{error:#}");
    }
}
