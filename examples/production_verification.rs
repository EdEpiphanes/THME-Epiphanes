// SPDX-License-Identifier: AGPL-3.0-only
//! Production Open-Weights End-to-End Verification Pipeline (Milestone v0.2.0.1 / v0.2.0.2).
//!
//! Pipeline Stages:
//! 1. Ingestion / zero-copy memory mapping
//! 2. Layerwise topological sensitivity analysis (SVD decay, condition numbers)
//! 3. Heterogeneous precision surgery recipe (Ternary, Q8_0, FP16 labels on the IR)
//! 4. Binary export (GGUF v3, 32-byte aligned)
//! 5. Measured perplexity quality gate (real forward pass over a tokenized corpus)
//!
//! Stage 5 only runs when on-disk weights and a token file are supplied:
//!
//! ```text
//! py -3 scripts/prepare_eval_tokens.py
//! cargo run --release --example production_verification -- \
//!     --model-path models/SmolLM-135M \
//!     --tokens target/eval/wikitext2_test.SmolLM-135M.u32
//! ```
//!
//! Without them the gate is reported as NOT EVALUATED. No perplexity figure is
//! ever estimated or synthesized by this pipeline.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use thme::exporter::{ExportConfig, ExportFormat, ExportQuantType, ModelExporter};
use thme::inference::{
    evaluate_perplexity, read_token_file, ArchConfig, PerplexityConfig, PerplexityReport,
    WeightPrecision,
};
use thme::ir::rebuilder::{ArchitectureBlueprint, ModelRebuilder};
use thme::ir::{ModelIR, TensorFormat, TensorRole};
use thme::parser::ModelImporter;
use thme::profiling::SensitivityProfiler;
use thme::quality::evaluator::{EvaluationCorpus, ModelEvaluator, PerplexityGateDecision};
use thme::surgery::recipe::{SurgeryEngine, TweakOperation, TweakRecipe};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VerificationTarget {
    /// SmolLM-135M (30 layers, hidden dim 576, intermediate size 1536)
    Smollm135m,
    /// LLaMA-3.2-1B (16 layers, hidden dim 2048, intermediate size 8192)
    Llama321b,
}

#[derive(Parser, Debug)]
#[command(
    name = "production_verification",
    about = "THME Production Core Verification Pipeline for Open-Weights Models"
)]
pub struct VerificationArgs {
    /// Target model architecture for verification
    #[arg(long, value_enum, default_value_t = VerificationTarget::Smollm135m)]
    pub target: VerificationTarget,

    /// Optional path to genuine model weights file (SafeTensors or GGUF)
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Destination directory for exported artifacts
    #[arg(long, default_value = "target/verification_artifacts")]
    pub output_dir: PathBuf,

    /// Maximum allowable perplexity delta before gate failure
    #[arg(long, default_value_t = 0.45)]
    pub max_ppl_delta: f64,

    /// Dry run mode (profiling and recipe generation only, skip writing files)
    #[arg(long)]
    pub dry_run: bool,

    /// Token file (little-endian u32 ids) from scripts/prepare_eval_tokens.py; enables the measured gate
    #[arg(long)]
    pub tokens: Option<PathBuf>,

    /// Tokens per forward pass during perplexity evaluation
    #[arg(long, default_value_t = 1024)]
    pub eval_context: usize,

    /// Evaluate only this many leading corpus tokens (default: the whole corpus)
    #[arg(long)]
    pub eval_max_tokens: Option<usize>,

    /// Where to write the machine-readable verification report
    #[arg(long, default_value = "target/verification_report.json")]
    pub report_json: PathBuf,
}

/// One evaluated set of weights and its measured perplexity.
pub struct EvaluatedVariant {
    pub id: &'static str,
    pub description: String,
    pub report: PerplexityReport,
    /// `None` for the baseline itself.
    pub delta_ppl: Option<f64>,
    pub accepted: Option<bool>,
}

pub struct ProductionVerificationReport {
    pub target: VerificationTarget,
    /// `None` when no forward pass was run.
    pub gate_passed: Option<bool>,
    pub variants: Vec<EvaluatedVariant>,
}

pub fn run_verification(args: &VerificationArgs) -> Result<ProductionVerificationReport> {
    println!("════════════════════════════════════════════════════════════════");
    println!("  THME Production Core Verification Pipeline (v0.2.0)");
    println!("  Target: {:?}", args.target);
    println!("════════════════════════════════════════════════════════════════");

    // STAGE 1: INGESTION
    println!("\n[1/5] Ingesting model checkpoint...");
    let mut model = match &args.model_path {
        Some(path) => {
            println!("  Loading weights from {:?}", path);
            ModelImporter::import(path)
                .with_context(|| format!("Failed to import checkpoint from {path:?}"))?
        }
        None => {
            println!("  No on-disk weights provided; constructing architectural scaffold...");
            build_target_scaffold(args.target)
        }
    };
    let source_payload_bytes = model.original_byte_size;
    println!(
        "  Loaded model '{}' ({} layers, {} tensors, {:.2} MB source payload)",
        model.name,
        model.num_layers,
        model.tensors.len(),
        mib(source_payload_bytes)
    );

    // STAGE 2: TOPOLOGICAL SENSITIVITY PROFILING
    println!("\n[2/5] Running zero-copy topological sensitivity profiling...");
    SensitivityProfiler::profile_model(&mut model);
    println!("  Topological profiling complete. Condition numbers and SVD spectra cached.");

    // STAGE 3: HETEROGENEOUS PARETO SURGERY
    println!("\n[3/5] Applying heterogeneous Pareto surgery recipe...");
    let recipe = build_target_surgery_recipe(&model, args.target);
    let diff = SurgeryEngine::apply_recipe(&mut model, &recipe);
    let recipe_estimated_bytes = model.total_estimated_bytes();
    let mut format_counts: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &model.tensors {
        *format_counts
            .entry(tensor.selected_format.display_name().to_string())
            .or_default() += 1;
    }
    println!(
        "  Applied recipe '{}': relabelled {} tensors.",
        recipe.name, diff.modified_tensors_count
    );
    println!(
        "  Recipe footprint (estimated from bits/weight, not an artifact size): {:.2} MB",
        mib(recipe_estimated_bytes)
    );

    // STAGE 4: BINARY EXPORT
    let mut export_json = Value::Null;
    let mut exported_gguf: Option<PathBuf> = None;
    if !args.dry_run {
        println!("\n[4/5] Exporting binary artifact (GGUF v3)...");
        std::fs::create_dir_all(&args.output_dir)?;

        let gguf_dest = args.output_dir.join(format!("{}.gguf", model.name));
        let export_cfg = ExportConfig {
            format: ExportFormat::GGUFv3,
            quant_type: ExportQuantType::Q8_0,
            output_path: gguf_dest.clone(),
            compute_checksum: true,
        };
        let export_rep = ModelExporter::export(&model, &export_cfg)?;
        println!(
            "  Exported GGUF to {:?} ({} bytes, sha256={})",
            gguf_dest,
            export_rep.exported_bytes,
            export_rep.sha256_checksum.as_deref().unwrap_or("none")
        );
        println!(
            "  Note: the exporter writes every tensor as Q8_0; per-tensor recipe formats are not applied to the artifact."
        );
        export_json = json!({
            "container": "GGUF v3",
            "quant_type": "Q8_0 (uniform, all tensors)",
            "bytes": export_rep.exported_bytes,
            "sha256": export_rep.sha256_checksum,
        });
        exported_gguf = Some(gguf_dest);
    } else {
        println!("\n[4/5] Skipping file export (dry-run mode active).");
    }

    // STAGE 5: MEASURED QUALITY GATE
    println!("\n[5/5] Perplexity quality gate...");
    let has_weights = model.tensors.iter().all(|t| t.source.is_some());
    let mut variants = Vec::new();
    let mut gate_passed = None;
    let mut corpus_provenance = Value::Null;

    match (&args.tokens, &args.model_path) {
        (Some(token_path), Some(model_path)) if has_weights => {
            let tokens = read_token_file(token_path)?;
            corpus_provenance = read_provenance(token_path);
            let eval = PerplexityConfig {
                context_len: args.eval_context,
                max_tokens: args.eval_max_tokens,
            };
            let config = ArchConfig::detect(&model, model_path)?;
            println!(
                "  Measured forward pass: {} corpus tokens available, context {}, limit {}",
                tokens.len(),
                eval.context_len,
                eval.max_tokens
                    .map_or_else(|| "none".to_string(), |m| m.to_string())
            );

            let baseline = measure(
                "source weights, as stored",
                &model,
                &config,
                WeightPrecision::AsStored,
                &tokens,
                &eval,
            )?;
            let baseline_metrics = baseline.to_metrics(EvaluationCorpus::WikiText2);
            variants.push(EvaluatedVariant {
                id: "baseline",
                description: "Source weights exactly as stored on disk".to_string(),
                report: baseline,
                delta_ppl: None,
                accepted: None,
            });

            let mut candidates: Vec<(&'static str, String, PerplexityReport)> = Vec::new();
            if let Some(gguf) = &exported_gguf {
                let exported = ModelImporter::import(gguf)
                    .with_context(|| format!("Failed to re-import exported artifact {gguf:?}"))?;
                let report = measure(
                    "exported GGUF artifact",
                    &exported,
                    &config,
                    WeightPrecision::AsStored,
                    &tokens,
                    &eval,
                )?;
                candidates.push((
                    "exported_gguf",
                    "Exported GGUF artifact re-imported and decoded (uniform Q8_0)".to_string(),
                    report,
                ));
            }
            let report = measure(
                "surgery recipe applied to weights",
                &model,
                &config,
                WeightPrecision::SelectedFormat,
                &tokens,
                &eval,
            )?;
            candidates.push((
                "recipe",
                format!(
                    "Recipe '{}' applied in memory: each tensor round-tripped through its selected format",
                    recipe.name
                ),
                report,
            ));

            let mut all_accepted = true;
            for (id, description, report) in candidates {
                let metrics = report.to_metrics(EvaluationCorpus::WikiText2);
                let gate =
                    ModelEvaluator::assert_gate(&baseline_metrics, &metrics, args.max_ppl_delta);
                let accepted = matches!(gate.decision, PerplexityGateDecision::Accept);
                all_accepted &= accepted;
                variants.push(EvaluatedVariant {
                    id,
                    description,
                    delta_ppl: Some(report.perplexity - baseline_metrics.perplexity),
                    report,
                    accepted: Some(accepted),
                });
            }
            gate_passed = Some(all_accepted);

            println!("\n  Measured perplexity (WikiText-2 raw test):");
            for variant in &variants {
                let verdict = match variant.accepted {
                    None => "baseline".to_string(),
                    Some(true) => "PASSED [ACCEPT]".to_string(),
                    Some(false) => "FAILED [ROLLBACK]".to_string(),
                };
                let delta = variant
                    .delta_ppl
                    .map_or_else(String::new, |d| format!(" (Δ {d:+.4})"));
                println!(
                    "    {:<14} PPL {:>12.4}{}  {}",
                    variant.id, variant.report.perplexity, delta, verdict
                );
            }
        }
        _ => {
            println!("  NOT EVALUATED: no forward pass was run, so no perplexity is reported.");
            println!(
                "  Supply --model-path with on-disk weights and --tokens (see scripts/prepare_eval_tokens.py)."
            );
        }
    }

    let report_json = json!({
        "thme_version": env!("CARGO_PKG_VERSION"),
        "generated_unix_secs": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "target": format!("{:?}", args.target),
        "model_name": model.name,
        "num_parameters": model.num_parameters,
        "source_payload_bytes": source_payload_bytes,
        "recipe": {
            "name": recipe.name,
            "relabelled_tensors": diff.modified_tensors_count,
            "format_counts": format_counts,
            "estimated_bytes": recipe_estimated_bytes,
            "estimated_bytes_note": "Computed from nominal bits per weight; no artifact of this size is written.",
        },
        "export": export_json,
        "evaluation": {
            "measured": gate_passed.is_some(),
            "corpus": corpus_provenance,
            "context_len": args.eval_context,
            "max_tokens": args.eval_max_tokens,
            "max_ppl_delta": args.max_ppl_delta,
            "gate_passed": gate_passed,
            "variants": variants.iter().map(|v| json!({
                "id": v.id,
                "description": v.description,
                "perplexity": v.report.perplexity,
                "cross_entropy_nats": v.report.cross_entropy_nats,
                "scored_tokens": v.report.scored_tokens,
                "corpus_tokens_used": v.report.corpus_tokens_used,
                "windows": v.report.windows,
                "elapsed_secs": v.report.elapsed_secs,
                "tokens_per_sec": v.report.tokens_per_sec,
                "delta_ppl": v.delta_ppl,
                "accepted": v.accepted,
            })).collect::<Vec<_>>(),
        },
        "host": host_info(),
    });
    if let Some(parent) = args.report_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &args.report_json,
        serde_json::to_string_pretty(&report_json)? + "\n",
    )
    .with_context(|| format!("Failed to write report to {:?}", args.report_json))?;
    println!("\n  Report written to {:?}", args.report_json);

    Ok(ProductionVerificationReport {
        target: args.target,
        gate_passed,
        variants,
    })
}

fn measure(
    label: &str,
    model: &ModelIR,
    config: &ArchConfig,
    precision: WeightPrecision,
    tokens: &[u32],
    eval: &PerplexityConfig,
) -> Result<PerplexityReport> {
    println!("  Evaluating {label}...");
    let dense = config
        .load(model, precision)
        .with_context(|| format!("Failed to load weights for: {label}"))?;
    let report = evaluate_perplexity(&*dense, tokens, eval, |done, total, running| {
        if done == total || done % 16 == 0 {
            println!("    window {done}/{total}  running PPL {running:.4}");
        }
    })?;
    println!(
        "    PPL {:.4} over {} scored tokens ({:.1} tok/s, {:.1} s)",
        report.perplexity, report.scored_tokens, report.tokens_per_sec, report.elapsed_secs
    );
    Ok(report)
}

/// Loads the provenance sidecar written next to the token file, if present.
fn read_provenance(token_path: &Path) -> Value {
    std::fs::read_to_string(token_path.with_extension("json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null)
}

fn host_info() -> Value {
    let mut system = sysinfo::System::new();
    system.refresh_cpu_all();
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu": system.cpus().first().map(|c| c.brand().trim().to_string()),
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "backend": "THME reference CPU forward pass (f32, multithreaded)",
    })
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn build_target_scaffold(target: VerificationTarget) -> ModelIR {
    match target {
        VerificationTarget::Smollm135m => {
            let bp = ArchitectureBlueprint::new_hybrid_llama("SmolLM-135M", 30, 576, 3);
            ModelRebuilder::build_scaffold(&bp)
        }
        VerificationTarget::Llama321b => {
            let bp = ArchitectureBlueprint::new_hybrid_llama("LLaMA-3.2-1B", 16, 2048, 4);
            ModelRebuilder::build_scaffold(&bp)
        }
    }
}

fn build_target_surgery_recipe(model: &ModelIR, target: VerificationTarget) -> TweakRecipe {
    let num_layers = model.num_layers;
    let mut operations = Vec::new();

    // Preserve first and last layers in high precision (FP16/Q8_0)
    // Quantize intermediate feed-forward layers to Ternary (BitNet b1.58)
    if num_layers > 2 {
        operations.push(TweakOperation::SetPrecision {
            start_layer: 1,
            end_layer: num_layers - 2,
            role: Some(TensorRole::MlpDown),
            format: TensorFormat::Ternary,
        });
        operations.push(TweakOperation::SetPrecision {
            start_layer: 1,
            end_layer: num_layers - 2,
            role: Some(TensorRole::AttentionQ),
            format: TensorFormat::Q8_0,
        });
    }

    TweakRecipe {
        name: format!("Production-Pareto-{}", model.name),
        version: "0.2.0".to_string(),
        author: "THME Production Gate".to_string(),
        description: format!("Heterogeneous Pareto surgery profile for {:?}", target),
        target_model: model.name.clone(),
        operations,
        expected_speedup: None,
        expected_memory_savings_pct: None,
    }
}

fn main() -> Result<()> {
    let args = VerificationArgs::parse();
    let report = run_verification(&args)?;

    match report.gate_passed {
        Some(true) => println!("\nVerification successful for {:?}.", report.target),
        Some(false) => {
            let failed: Vec<&str> = report
                .variants
                .iter()
                .filter(|v| v.accepted == Some(false))
                .map(|v| v.id)
                .collect();
            anyhow::bail!(
                "Verification quality gate failed for [{}]: ΔPPL exceeds tolerance {:.4}",
                failed.join(", "),
                args.max_ppl_delta
            );
        }
        None => println!(
            "\nPipeline stages 1-4 completed for {:?}; quality gate not evaluated.",
            report.target
        ),
    }
    Ok(())
}
