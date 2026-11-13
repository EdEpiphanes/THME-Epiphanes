// SPDX-License-Identifier: AGPL-3.0-only
//! `thme evaluate`: run a surgery recipe through the measured quality gate.
//!
//! The recipe is applied to the model's IR, every targeted tensor is round-tripped through
//! its new format, and the reference forward pass measures perplexity on real tokens for the
//! source and the candidate. The report uses the same schema as the verification pipeline,
//! so `thme validate --report` and `scripts/generate_dossier.py` read it directly.
//!
//! Only operations the forward pass can execute are measured (today: `SetPrecision`). A
//! recipe with other operations is refused unless `--allow-partial` is given, in which case
//! the skipped operations are listed in the report.

use crate::quality::gate::{GateSpec, GateVerdict, Measure};
use anyhow::{bail, Context, Result};
use clap::Args;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::inference::{
    evaluate_perplexity, read_token_file, ArchConfig, PerplexityConfig, PerplexityReport,
    WeightPrecision, BYOA_METADATA_KEY,
};
use crate::ir::{ModelIR, TensorFormat};
use crate::parser::ModelImporter;
use crate::surgery::recipe::{SurgeryEngine, TweakOperation, TweakRecipe};
use crate::transform::byoa::AlgorithmRegistry;

#[derive(Args, Debug)]
pub struct EvaluateArgs {
    /// Model directory or file with on-disk weights (LLaMA family or GPT-NeoX)
    #[arg(value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Surgery recipe JSON (docs/schemas/surgery-recipe-v1.json)
    #[arg(long, value_name = "RECIPE_JSON")]
    pub recipe: PathBuf,

    /// Token file (little-endian u32) from scripts/prepare_eval_tokens.py
    #[arg(long, value_name = "TOKENS")]
    pub tokens: PathBuf,

    /// Tokens per forward pass
    #[arg(long, default_value_t = 1024)]
    pub eval_context: usize,

    /// Evaluate only this many leading corpus tokens (default: all)
    #[arg(long)]
    pub eval_max_tokens: Option<usize>,

    /// Maximum allowed perplexity increase (the default gate's absolute margin)
    #[arg(long, default_value_t = 0.45)]
    pub max_ppl_delta: f64,

    /// Quality gate file (JSON: name, version, criteria); replaces --max-ppl-delta
    #[arg(long, conflicts_with = "max_ppl_delta", value_name = "GATE_JSON")]
    pub gate: Option<PathBuf>,

    /// Where to write the measured report
    #[arg(long, default_value = "target/evaluate_report.json")]
    pub report: PathBuf,

    /// Measure the operations the forward pass supports and list the others in the report
    #[arg(long)]
    pub allow_partial: bool,

    /// BYOA algorithm definition files (one definition or an array) used by the recipe
    #[arg(long = "algorithms", value_name = "ALGORITHMS_JSON")]
    pub algorithms: Vec<PathBuf>,

    /// Keep the recipe's GGUF artifact here. Without it the artifact is written to a temporary
    /// file, measured and deleted; either way the report records its real size.
    #[arg(long, value_name = "GGUF")]
    pub artifact: Option<PathBuf>,

    /// Calibration statistics from `thme calibrate`, required by activation-aware algorithms
    /// such as `thme.awq_q4_0`. Collect them on tokens outside the evaluation split.
    #[arg(long, value_name = "CALIBRATION_JSON")]
    pub calibration: Option<PathBuf>,
}

/// Attaches each tensor's calibration statistics as metadata; returns how many were attached.
pub fn attach_calibration(model: &mut ModelIR, path: &Path) -> Result<usize> {
    use crate::transform::awq::{ChannelStats, CALIBRATION_METADATA_KEY};
    let file = crate::inference::CalibrationFile::load(path)?;
    let mut attached = 0;
    for tensor in &mut model.tensors {
        if let Some(stats) = file.get(&tensor.name) {
            let json = serde_json::to_string(&ChannelStats {
                mean_abs: stats.mean_abs.clone(),
                rms: stats.rms.clone(),
            })?;
            tensor
                .metadata
                .insert(CALIBRATION_METADATA_KEY.to_string(), json);
            attached += 1;
        }
    }
    Ok(attached)
}

/// Writes the candidate as a per-tensor GGUF and reports the file's real size (block scales,
/// metadata and alignment included). Candidates the exporter cannot write (BYOA algorithms,
/// K-quants on narrow rows) are reported as not measured, with the reason.
pub fn measure_artifact(candidate: &ModelIR, keep: Option<&Path>) -> Value {
    use crate::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
    let (path, temporary) = match keep {
        Some(path) => (path.to_path_buf(), false),
        None => {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let name = format!("thme-evaluate-{}-{nanos}.gguf", std::process::id());
            (std::env::temp_dir().join(name), true)
        }
    };
    let result = ModelExporter::export(
        candidate,
        &ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::PerTensor,
            output_path: path.clone(),
            compute_checksum: !temporary,
        },
    );
    let value = match result {
        Ok(report) => json!({
            "measured": true,
            "container": "GGUF v3, per-tensor formats",
            "artifact_bytes": report.exported_bytes,
            "file": if temporary {
                Value::Null
            } else {
                json!(path.file_name().map(|n| n.to_string_lossy().into_owned()))
            },
            "sha256": report.sha256_checksum,
            "note": "Size of the written file, including block scales, metadata and alignment.",
        }),
        Err(error) => json!({"measured": false, "reason": format!("{error:#}")}),
    };
    if temporary {
        let _ = std::fs::remove_file(&path);
    }
    value
}

/// Formats the measured path can round-trip.
fn measurable(format: &TensorFormat) -> bool {
    matches!(
        format,
        TensorFormat::FP32
            | TensorFormat::FP16
            | TensorFormat::BF16
            | TensorFormat::Q8_0
            | TensorFormat::Q4_0
            | TensorFormat::Q4_K
            | TensorFormat::Q5_K
            | TensorFormat::Q6_K
            | TensorFormat::Ternary
    )
}

fn operation_name(op: &TweakOperation) -> &'static str {
    match op {
        TweakOperation::SetPrecision { .. } => "SetPrecision",
        TweakOperation::ReplaceMixer { .. } => "ReplaceMixer",
        TweakOperation::ConfigureEarlyExit { .. } => "ConfigureEarlyExit",
        TweakOperation::PruneAttentionHeads { .. } => "PruneAttentionHeads",
        TweakOperation::QuantizeKvCache { .. } => "QuantizeKvCache",
        TweakOperation::ApplyCustomAlgorithm { .. } => "ApplyCustomAlgorithm",
        TweakOperation::RebuildArchitecture { .. } => "RebuildArchitecture",
        TweakOperation::ConvertParadigm { .. } => "ConvertParadigm",
        TweakOperation::ExtractDraftModel { .. } => "ExtractDraftModel",
    }
}

/// Splits a recipe into the part the forward pass can measure and the names of the rest.
pub fn measurable_part(recipe: &TweakRecipe) -> (TweakRecipe, Vec<&'static str>) {
    let mut kept = recipe.clone();
    let mut skipped = Vec::new();
    kept.operations.retain(|op| {
        let ok = matches!(
            op,
            TweakOperation::SetPrecision { .. } | TweakOperation::ApplyCustomAlgorithm { .. }
        );
        if !ok {
            skipped.push(operation_name(op));
        }
        ok
    });
    (kept, skipped)
}

/// Metrics a perplexity evaluation measures, for the quality gate.
pub(crate) const MEASURED_METRICS: [&str; 2] = ["perplexity", "cross_entropy_nats"];

/// Judges `candidate` against `baseline` on every metric a perplexity run measures.
pub(crate) fn judge_reports(
    gate: &GateSpec,
    baseline: &PerplexityReport,
    candidate: &PerplexityReport,
) -> GateVerdict {
    gate.judge(&[
        Measure::new("perplexity", baseline.perplexity, candidate.perplexity),
        Measure::new(
            "cross_entropy_nats",
            baseline.cross_entropy_nats,
            candidate.cross_entropy_nats,
        ),
    ])
}

/// One report row: a measured variant and, when it has a baseline, the gate's verdict on it.
pub(crate) fn variant_json(
    id: &str,
    description: &str,
    report: &PerplexityReport,
    base: Option<&PerplexityReport>,
    gate: &GateSpec,
) -> Value {
    let delta = base.map(|b| report.perplexity - b.perplexity);
    let accepted = base.map(|b| judge_reports(gate, b, report).accepted);
    json!({
        "id": id,
        "description": description,
        "perplexity": report.perplexity,
        "cross_entropy_nats": report.cross_entropy_nats,
        "scored_tokens": report.scored_tokens,
        "corpus_tokens_used": report.corpus_tokens_used,
        "windows": report.windows,
        "elapsed_secs": report.elapsed_secs,
        "tokens_per_sec": report.tokens_per_sec,
        "delta_ppl": delta,
        "accepted": accepted,
    })
}

/// A recipe applied to a model, ready for the measured forward pass.
pub struct PreparedCandidate {
    pub model: ModelIR,
    pub skipped: Vec<&'static str>,
    pub counts: BTreeMap<String, usize>,
    pub nominal_bits: f64,
    pub params: usize,
    pub relabelled: usize,
    pub operations_evaluated: usize,
}

/// Loads the built-in BYOA registry plus algorithm definition files (one definition or an array).
pub fn load_registry(files: &[PathBuf]) -> Result<AlgorithmRegistry> {
    let mut registry = AlgorithmRegistry::new();
    for path in files {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read algorithm file {path:?}"))?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("Failed to parse algorithm file {path:?}"))?;
        let items = match value {
            Value::Array(items) => items,
            single => vec![single],
        };
        for item in items {
            registry
                .import_from_json(&item.to_string())
                .map_err(|e| anyhow::anyhow!("Invalid algorithm in {path:?}: {e}"))?;
        }
    }
    Ok(registry)
}

/// Applies the measurable part of `recipe` to `source`. Untargeted tensors keep their stored
/// values; BYOA algorithms are resolved, contract-checked and attached to their tensors.
pub fn prepare_candidate(
    source: &ModelIR,
    recipe: &TweakRecipe,
    registry: &AlgorithmRegistry,
    allow_partial: bool,
) -> Result<PreparedCandidate> {
    let (measured_recipe, skipped) = measurable_part(recipe);
    if !skipped.is_empty() && !allow_partial {
        bail!(
            "Recipe contains operations the forward pass cannot run yet: {}. Re-run with --allow-partial to measure only its SetPrecision and ApplyCustomAlgorithm operations.",
            skipped.join(", ")
        );
    }
    if measured_recipe.operations.is_empty() {
        bail!("Recipe has no measurable operations (SetPrecision or ApplyCustomAlgorithm)");
    }
    // Untargeted tensors are marked FP32, which the measured path treats as "keep as stored".
    let mut model = source.clone();
    for tensor in &mut model.tensors {
        tensor.selected_format = TensorFormat::FP32;
    }
    let diff = SurgeryEngine::apply_recipe(&mut model, &measured_recipe);
    let (mut counts, mut nominal_bits, mut params) =
        (BTreeMap::<String, usize>::new(), 0.0f64, 0usize);
    for tensor in &mut model.tensors {
        let matrix = tensor.shape.len() >= 2;
        let stored_bits = 8.0 * tensor.byte_size as f64 / tensor.numel.max(1) as f64;
        let bits = match tensor.selected_format.clone() {
            TensorFormat::FP32 => stored_bits,
            _ if !matrix => stored_bits, // norms and biases stay as stored
            TensorFormat::Custom(id) => {
                let algorithm = registry.get(&id).with_context(|| {
                    format!("Recipe uses unknown algorithm '{id}'; register it with --algorithms <file.json>")
                })?;
                algorithm.measurable().map_err(|e| anyhow::anyhow!(e))?;
                tensor.metadata.insert(
                    BYOA_METADATA_KEY.to_string(),
                    algorithm
                        .to_json()
                        .context("Failed to serialize algorithm")?,
                );
                *counts.entry(format!("BYOA {id}")).or_default() += 1;
                f64::from(algorithm.effective_bits_per_param)
            }
            format => {
                if !measurable(&format) {
                    bail!(
                        "Tensor '{}' is assigned {}, which the measured path cannot round-trip yet (supported: FP16, BF16, Q8_0, Q6_K, Q5_K, Q4_K, Q4_0, Ternary, or a BYOA algorithm)",
                        tensor.name,
                        format.display_name()
                    );
                }
                *counts.entry(format.display_name().to_string()).or_default() += 1;
                format.bits_per_weight()
            }
        };
        nominal_bits += bits * tensor.numel as f64;
        params += tensor.numel;
    }
    Ok(PreparedCandidate {
        model,
        skipped,
        counts,
        nominal_bits,
        params,
        relabelled: diff.modified_tensors_count,
        operations_evaluated: measured_recipe.operations.len(),
    })
}

/// Runs the evaluation and returns the report (also written to `args.report`).
pub fn evaluate(args: &EvaluateArgs) -> Result<Value> {
    let text = std::fs::read_to_string(&args.recipe)
        .with_context(|| format!("Failed to read recipe {:?}", args.recipe))?;
    let recipe: TweakRecipe = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse recipe {:?}", args.recipe))?;
    let registry = load_registry(&args.algorithms)?;
    let source = ModelImporter::import(&args.model)
        .with_context(|| format!("Failed to import {:?}", args.model))?;
    if source.tensors.iter().any(|t| t.source.is_none()) {
        bail!("Model has tensors without on-disk payloads; evaluation needs real weights");
    }
    let gate = crate::quality::gate::resolve(args.gate.as_deref(), args.max_ppl_delta)?;
    gate.require_metrics(&MEASURED_METRICS, "thme evaluate")?;
    let PreparedCandidate {
        model: mut candidate,
        skipped,
        counts,
        nominal_bits,
        params,
        relabelled,
        operations_evaluated,
    } = prepare_candidate(&source, &recipe, &registry, args.allow_partial)?;
    let calibration = match &args.calibration {
        Some(path) => {
            let attached = attach_calibration(&mut candidate, path)?;
            println!("  calibration statistics attached to {attached} tensors");
            json!({
                "file": path.file_name().map(|n| n.to_string_lossy().into_owned()),
                "tensors_attached": attached,
            })
        }
        None => Value::Null,
    };
    let arch = ArchConfig::detect(&source, &args.model)?;
    let tokens = read_token_file(&args.tokens)?;
    let eval = PerplexityConfig {
        context_len: args.eval_context,
        max_tokens: args.eval_max_tokens,
    };
    println!(
        "Evaluating '{}' on {} ({})",
        recipe.name,
        source.name,
        arch.family()
    );
    let baseline_model = arch.load(&source, WeightPrecision::AsStored)?;
    let baseline = evaluate_perplexity(&*baseline_model, &tokens, &eval, |_, _, _| {})?;
    drop(baseline_model);
    println!("  source     PPL {:.4}", baseline.perplexity);
    let candidate_model = arch.load(&candidate, WeightPrecision::SelectedFormat)?;
    let measured = evaluate_perplexity(&*candidate_model, &tokens, &eval, |_, _, _| {})?;
    let delta = measured.perplexity - baseline.perplexity;
    let verdict = judge_reports(&gate, &baseline, &measured);
    let passed = verdict.accepted;
    println!(
        "  candidate  PPL {:.4}  Δ {delta:+.4}  {}",
        measured.perplexity,
        if passed { "PASSED" } else { "FAILED" }
    );
    if !skipped.is_empty() {
        println!("  NOT EVALUATED: {}", skipped.join(", "));
    }
    let artifact = measure_artifact(&candidate, args.artifact.as_deref());
    match artifact["artifact_bytes"].as_u64() {
        Some(bytes) => println!("  artifact   {bytes} bytes (GGUF, measured)"),
        None => println!(
            "  artifact   not written: {}",
            artifact["reason"].as_str().unwrap_or("unknown reason")
        ),
    }

    let corpus: Value = std::fs::read_to_string(args.tokens.with_extension("json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null);
    let report = json!({
        "thme_version": env!("CARGO_PKG_VERSION"),
        "generated_unix_secs": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "tool": "thme evaluate",
        "model_name": source.name,
        "architecture": arch.family(),
        "num_parameters": params,
        "source_payload_bytes": source.original_byte_size,
        "recipe": {
            "name": recipe.name,
            "relabelled_tensors": relabelled,
            "format_counts": counts,
            "estimated_bytes": (nominal_bits / 8.0) as u64,
            "estimated_bytes_note": "Computed from nominal bits per weight; export.artifact_bytes is the size of the written GGUF.",
            "operations_evaluated": operations_evaluated,
            "operations_not_evaluated": skipped,
        },
        "export": artifact,
        "calibration": calibration,
        "evaluation": {
            "measured": true,
            "corpus": corpus,
            "context_len": args.eval_context,
            "max_tokens": args.eval_max_tokens,
            "max_ppl_delta": gate.max_ppl_delta(),
            "gate": gate.to_report(),
            "gate_verdict": verdict,
            "gate_passed": passed,
            "variants": [
                variant_json("baseline", "Source weights exactly as stored on disk", &baseline, None, &gate),
                variant_json(
                    "recipe",
                    &format!("Recipe '{}': targeted tensors round-tripped through their assigned formats", recipe.name),
                    &measured,
                    Some(&baseline),
                    &gate,
                ),
            ],
        },
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            "backend": "THME reference CPU forward pass (f32, multithreaded)",
        },
    });
    write_report(&args.report, &report)?;
    Ok(report)
}

fn write_report(path: &Path, report: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)? + "\n")
        .with_context(|| format!("Failed to write {path:?}"))
}

pub fn execute(args: EvaluateArgs) -> Result<()> {
    let report = evaluate(&args)?;
    println!("Report written to {:?}", args.report);
    if report["evaluation"]["gate_passed"] != Value::Bool(true) {
        bail!(
            "Measured quality gate {} failed",
            report["evaluation"]["gate"]["id"].as_str().unwrap_or("?")
        );
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::exporter::SafeTensorsWriter;

    /// Writes a tiny LLaMA-shaped checkpoint with pseudo-random weights, plus a token file.
    pub(crate) fn synthetic_checkpoint(dir: &Path) -> (PathBuf, PathBuf) {
        let (d, kv, inter, vocab, layers) = (32usize, 16usize, 64usize, 64usize, 2usize);
        let mut state = 17u64;
        let mut tensor =
            |writer: &mut SafeTensorsWriter, name: String, shape: Vec<usize>, scale: f32| {
                let n: usize = shape.iter().product();
                let bytes = (0..n)
                    .flat_map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((((state >> 33) as f32 / 2_147_483_648.0) - 1.0) * scale).to_le_bytes()
                    })
                    .collect();
                writer.add_tensor(name, "F32", shape, bytes);
            };
        let mut w = SafeTensorsWriter::new();
        tensor(
            &mut w,
            "model.embed_tokens.weight".into(),
            vec![vocab, d],
            1.0,
        );
        for l in 0..layers {
            let p = format!("model.layers.{l}");
            tensor(
                &mut w,
                format!("{p}.self_attn.q_proj.weight"),
                vec![d, d],
                0.3,
            );
            tensor(
                &mut w,
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv, d],
                0.3,
            );
            tensor(
                &mut w,
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv, d],
                0.3,
            );
            tensor(
                &mut w,
                format!("{p}.self_attn.o_proj.weight"),
                vec![d, d],
                0.3,
            );
            tensor(
                &mut w,
                format!("{p}.mlp.gate_proj.weight"),
                vec![inter, d],
                0.3,
            );
            tensor(
                &mut w,
                format!("{p}.mlp.up_proj.weight"),
                vec![inter, d],
                0.3,
            );
            tensor(
                &mut w,
                format!("{p}.mlp.down_proj.weight"),
                vec![d, inter],
                0.3,
            );
            w.add_tensor(
                format!("{p}.input_layernorm.weight"),
                "F32",
                vec![d],
                1.0f32.to_le_bytes().repeat(d),
            );
            w.add_tensor(
                format!("{p}.post_attention_layernorm.weight"),
                "F32",
                vec![d],
                1.0f32.to_le_bytes().repeat(d),
            );
        }
        w.add_tensor(
            "model.norm.weight",
            "F32",
            vec![d],
            1.0f32.to_le_bytes().repeat(d),
        );
        let model_dir = dir.join("tiny");
        std::fs::create_dir_all(&model_dir).unwrap();
        w.write_to_file(model_dir.join("model.safetensors"))
            .unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            format!(r#"{{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":{d},"num_hidden_layers":{layers},"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":{inter},"vocab_size":{vocab},"rms_norm_eps":1e-5}}"#),
        )
        .unwrap();
        let tokens: Vec<u8> = (0..200u32)
            .flat_map(|i| ((i * 7) % vocab as u32).to_le_bytes())
            .collect();
        let token_path = dir.join("tokens.u32");
        std::fs::write(&token_path, tokens).unwrap();
        (model_dir, token_path)
    }

    fn write_recipe(dir: &Path, with_mixer: bool) -> PathBuf {
        let mut ops = vec![
            json!({"type": "SetPrecision", "params": {"start_layer": 0, "end_layer": 1, "role": "MlpDown", "format": "Q8_0"}}),
        ];
        if with_mixer {
            ops.push(json!({"type": "ReplaceMixer", "params": {"layer": 1, "mixer_type": "gated_deltanet"}}));
        }
        let recipe = json!({"name": "test", "author": "test", "target_model": "tiny", "version": "1", "description": "", "operations": ops});
        let path = dir.join(if with_mixer {
            "mixed.json"
        } else {
            "recipe.json"
        });
        std::fs::write(&path, recipe.to_string()).unwrap();
        path
    }

    fn args(
        model: PathBuf,
        recipe: PathBuf,
        tokens: PathBuf,
        report: PathBuf,
        allow_partial: bool,
    ) -> EvaluateArgs {
        EvaluateArgs {
            model,
            recipe,
            tokens,
            eval_context: 64,
            eval_max_tokens: None,
            max_ppl_delta: 0.45,
            report,
            allow_partial,
            algorithms: Vec::new(),
            artifact: None,
            calibration: None,
            gate: None,
        }
    }

    #[test]
    fn recipe_is_measured_end_to_end_and_report_is_gateable() {
        let dir = tempfile::tempdir().unwrap();
        let (model, tokens) = synthetic_checkpoint(dir.path());
        let recipe = write_recipe(dir.path(), false);
        let out = dir.path().join("report.json");
        let report = evaluate(&args(model, recipe, tokens, out.clone(), false)).unwrap();
        assert_eq!(report["evaluation"]["measured"], Value::Bool(true));
        assert_eq!(report["recipe"]["format_counts"]["Q8_0"], json!(2));
        let written: Value = serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap();
        let (gate, variants) = crate::cli::validate::read_measured_report(&written).unwrap();
        assert_eq!(gate.max_ppl_delta(), Some(0.45));
        assert_eq!(gate.id(), "thme-perplexity@v1");
        assert_eq!(report["evaluation"]["gate"]["hash"], json!(gate.hash()));
        assert_eq!(variants.len(), 2);
        assert!(variants
            .iter()
            .all(|v| v.perplexity.is_finite() && v.perplexity > 1.0));
        // THME-2209: the written GGUF's real size sits beside the nominal estimate.
        assert_eq!(report["export"]["measured"], json!(true));
        let artifact = report["export"]["artifact_bytes"].as_u64().unwrap();
        let estimated = report["recipe"]["estimated_bytes"].as_u64().unwrap();
        assert!(
            artifact > estimated,
            "file adds metadata: {artifact} vs {estimated}"
        );
        assert!(artifact < report["source_payload_bytes"].as_u64().unwrap() + 64 * 1024);
        assert_eq!(
            report["export"]["file"],
            Value::Null,
            "temporary artifact is not kept"
        );
    }

    #[test]
    fn unsupported_operations_are_refused_unless_partial_is_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let (model, tokens) = synthetic_checkpoint(dir.path());
        let recipe = write_recipe(dir.path(), true);
        let out = dir.path().join("report.json");
        let refused = evaluate(&args(
            model.clone(),
            recipe.clone(),
            tokens.clone(),
            out.clone(),
            false,
        ));
        assert!(refused.unwrap_err().to_string().contains("ReplaceMixer"));
        let report = evaluate(&args(model, recipe, tokens, out, true)).unwrap();
        assert_eq!(
            report["recipe"]["operations_not_evaluated"],
            json!(["ReplaceMixer"])
        );
    }

    /// THME-2204: an AWQ recipe runs through the measured path with calibration attached,
    /// and refuses to run without it.
    #[test]
    fn activation_aware_recipe_needs_and_uses_calibration() {
        let dir = tempfile::tempdir().unwrap();
        let (model, tokens) = synthetic_checkpoint(dir.path());
        let ir = ModelImporter::import(&model).unwrap();
        let lm = ArchConfig::detect(&ir, &model)
            .unwrap()
            .load(&ir, WeightPrecision::AsStored)
            .unwrap();
        let token_ids = read_token_file(&tokens).unwrap();
        let calib =
            crate::inference::collect_calibration(&*lm, &ir, &token_ids, 32, Some(96)).unwrap();
        let calib_path = dir.path().join("calibration.json");
        calib.save(&calib_path).unwrap();

        let recipe = write_custom_recipe(dir.path(), "thme.awq_q4_0");
        let out = dir.path().join("report.json");
        let refused = evaluate(&args(
            model.clone(),
            recipe.clone(),
            tokens.clone(),
            out.clone(),
            false,
        ));
        assert!(format!("{:#}", refused.unwrap_err()).contains("--calibration"));

        let mut with_calibration = args(model, recipe, tokens, out, false);
        with_calibration.calibration = Some(calib_path);
        let report = evaluate(&with_calibration).unwrap();
        assert_eq!(report["calibration"]["tensors_attached"], json!(14));
        let recipe_ppl = report["evaluation"]["variants"][1]["perplexity"]
            .as_f64()
            .unwrap();
        assert!(recipe_ppl.is_finite() && recipe_ppl > 1.0);
        assert_eq!(report["export"]["measured"], json!(false));
    }

    /// THME-3003: a gate file replaces the default; its identity, hash and verdict land in the
    /// report; a metric this tool cannot measure is refused before any work.
    #[test]
    fn gate_file_decides_and_is_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let (model, tokens) = synthetic_checkpoint(dir.path());
        let recipe = write_recipe(dir.path(), false);
        let gate_path = dir.path().join("gate.json");
        std::fs::write(
            &gate_path,
            r#"{"name": "strict-ce", "version": 2, "criteria": [
                {"metric": "cross_entropy_nats", "lower_is_better": true, "max_rel": 1e-12}]}"#,
        )
        .unwrap();
        let mut gated = args(
            model.clone(),
            recipe.clone(),
            tokens.clone(),
            dir.path().join("r.json"),
            false,
        );
        gated.gate = Some(gate_path.clone());
        let report = evaluate(&gated).unwrap();
        let evaluation = &report["evaluation"];
        assert_eq!(evaluation["gate"]["id"], json!("strict-ce@v2"));
        assert_eq!(
            evaluation["gate"]["hash"],
            json!(crate::quality::gate::GateSpec::load(&gate_path)
                .unwrap()
                .hash())
        );
        assert_eq!(
            evaluation["gate_verdict"]["outcomes"][0]["metric"],
            json!("cross_entropy_nats")
        );
        // Q8_0 changes the loss, so a near-zero relative margin rejects it.
        assert_eq!(evaluation["gate_passed"], json!(false));
        assert_eq!(evaluation["max_ppl_delta"], Value::Null);
        assert!(execute(gated).is_err());

        std::fs::write(
            &gate_path,
            r#"{"name": "acc", "version": 1, "criteria": [
                {"metric": "mmlu_accuracy", "lower_is_better": false, "max_abs": 1.0}]}"#,
        )
        .unwrap();
        let mut unmeasurable = args(model, recipe, tokens, dir.path().join("u.json"), false);
        unmeasurable.gate = Some(gate_path);
        let err = evaluate(&unmeasurable).unwrap_err();
        assert!(format!("{err:#}").contains("does not measure"));
    }

    fn write_custom_recipe(dir: &Path, algorithm: &str) -> PathBuf {
        let recipe = json!({
            "name": "byoa", "author": "test", "target_model": "tiny", "version": "1", "description": "",
            "operations": [{"type": "ApplyCustomAlgorithm", "params": {"start_layer": 0, "end_layer": 1, "role": "MlpUp", "algorithm_id": algorithm}}]
        });
        let path = dir.join(format!("{}.json", algorithm.replace('.', "_")));
        std::fs::write(&path, recipe.to_string()).unwrap();
        path
    }

    #[test]
    fn byoa_algorithms_are_measured_and_invalid_ones_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (model, tokens) = synthetic_checkpoint(dir.path());
        let out = dir.path().join("report.json");

        let rotated = write_custom_recipe(dir.path(), "thme.hadamard_ternary");
        let report = evaluate(&args(
            model.clone(),
            rotated,
            tokens.clone(),
            out.clone(),
            false,
        ))
        .unwrap();
        assert_eq!(report["export"]["measured"], json!(false));
        assert!(report["export"]["reason"]
            .as_str()
            .unwrap()
            .contains("BYOA"));
        assert_eq!(
            report["recipe"]["format_counts"]["BYOA thme.hadamard_ternary"],
            json!(2)
        );
        let ppl = report["evaluation"]["variants"][1]["perplexity"]
            .as_f64()
            .unwrap();
        assert!(ppl.is_finite());

        let mixer = write_custom_recipe(dir.path(), "thme.gated_deltanet");
        let err = evaluate(&args(
            model.clone(),
            mixer,
            tokens.clone(),
            out.clone(),
            false,
        ))
        .unwrap_err();
        assert!(err.to_string().contains("sequence mixer"), "{err}");

        let unknown = write_custom_recipe(dir.path(), "user.nope");
        let err = evaluate(&args(
            model.clone(),
            unknown,
            tokens.clone(),
            out.clone(),
            false,
        ))
        .unwrap_err();
        assert!(err.to_string().contains("--algorithms"), "{err}");

        // A user-supplied definition file makes the same id measurable.
        let definition = json!([{
            "id": "user.nope", "name": "User 3-level codebook", "author": "me", "version": "1",
            "category": "Quantization", "description": "", "effective_bits_per_param": 1.6,
            "compression_ratio": 10.0, "hardware_speedup_factor": 1.0, "target_roles": ["MlpUp"],
            "rule": {"type": "QuantizeCodebook", "params": {"levels": [-1.0, 0.0, 1.0], "scale_mode": "AbsMean"}}
        }]);
        let def_path = dir.path().join("algorithms.json");
        std::fs::write(&def_path, definition.to_string()).unwrap();
        let unknown = write_custom_recipe(dir.path(), "user.nope");
        let mut with_file = args(model, unknown, tokens, out, false);
        with_file.algorithms = vec![def_path];
        let report = evaluate(&with_file).unwrap();
        assert_eq!(
            report["recipe"]["format_counts"]["BYOA user.nope"],
            json!(2)
        );
    }
}
