// SPDX-License-Identifier: AGPL-3.0-only
//! Header-only run planner producing [`RunPlan`] with provenance labels (THME-3057).
//!
//! Derives [`ModelShape`] from [`ModelIR`] and delegates budget fitting to [`shrink_to_budget`].
//! Every decision currently carries [`Basis::Predicted`] because all figures come from header
//! arithmetic and nothing is measured on hardware.

use serde::{Deserialize, Serialize};

use crate::ir::model::ModelMetadataValue;
use crate::ir::{ModelArchitecture, ModelIR, TensorRole};

use super::run_memory::{run_memory, ModelShape, RunMemory};
use super::shrink::{shrink_to_budget, Fixed, RunSettings, ShrinkError, ShrinkLimits};

/// Provenance of a planned setting or memory figure.
///
/// Only `Predicted` exists right now: all numbers come from model-header arithmetic, with
/// nothing measured on hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Basis {
    Predicted,
}

impl Basis {
    /// Returns the canonical lowercase label for display and serialization.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Predicted => "predicted",
        }
    }
}

/// Names one planning decision and records its [`Basis`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BasisLabel {
    pub decision: String,
    pub basis: Basis,
}

/// Complete run plan for a single model, including settings, memory breakdown, and basis labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunPlan {
    pub model: String,
    pub settings: RunSettings,
    pub memory: RunMemory,
    pub budget_bytes: Option<u64>,
    pub basis: Vec<BasisLabel>,
}

impl RunPlan {
    /// Total predicted bytes across weights, KV cache, and single-token scratch space.
    pub const fn total_bytes(&self) -> u64 {
        self.memory.total()
    }
}

fn metadata_positive_usize(value: &ModelMetadataValue) -> Option<usize> {
    match value {
        ModelMetadataValue::Uint8(n) if *n > 0 => Some(*n as usize),
        ModelMetadataValue::Uint16(n) if *n > 0 => Some(*n as usize),
        ModelMetadataValue::Uint32(n) if *n > 0 => Some(*n as usize),
        ModelMetadataValue::Uint64(n) if *n > 0 => usize::try_from(*n).ok(),
        ModelMetadataValue::Int8(n) if *n > 0 => Some(*n as usize),
        ModelMetadataValue::Int16(n) if *n > 0 => Some(*n as usize),
        ModelMetadataValue::Int32(n) if *n > 0 => Some(*n as usize),
        ModelMetadataValue::Int64(n) if *n > 0 => usize::try_from(*n).ok(),
        _ => None,
    }
}

fn infer_intermediate_size(ir: &ModelIR) -> Option<usize> {
    for tensor in &ir.tensors {
        if !matches!(
            tensor.role,
            TensorRole::MlpGate | TensorRole::MlpUp | TensorRole::MlpDown
        ) {
            continue;
        }
        if let [rows, cols] = tensor.shape.as_slice() {
            if *rows == ir.hidden_size && *cols > 0 {
                return Some(*cols);
            }
            if *cols == ir.hidden_size && *rows > 0 {
                return Some(*rows);
            }
            if tensor.numel > 0 && tensor.numel.is_multiple_of(ir.hidden_size) {
                let dim = tensor.numel / ir.hidden_size;
                if dim > 0 {
                    return Some(dim);
                }
            }
        }
    }

    ir.source_gguf_metadata
        .iter()
        .filter(|(k, _)| *k == "feed_forward_length" || k.ends_with(".feed_forward_length"))
        .find_map(|(_, v)| metadata_positive_usize(v))
}

/// Derives a [`ModelShape`] from a [`ModelIR`]'s architecture and tensor metadata.
///
/// Returns `None` when any required dimension (`num_layers`, `hidden_size`,
/// `num_attention_heads`, `num_key_value_heads`, `vocab_size`, or `intermediate`) is missing
/// or zero.
pub fn shape_from_ir(ir: &ModelIR) -> Option<ModelShape> {
    if ir.num_layers == 0
        || ir.hidden_size == 0
        || ir.num_attention_heads == 0
        || ir.num_key_value_heads == 0
        || ir.vocab_size == 0
    {
        return None;
    }

    let intermediate = infer_intermediate_size(ir)?;
    let gated_mlp = if ir.tensors.iter().any(|t| t.role == TensorRole::MlpGate) {
        true
    } else if ir
        .tensors
        .iter()
        .any(|t| matches!(t.role, TensorRole::MlpUp | TensorRole::MlpDown))
    {
        false
    } else {
        matches!(
            ir.architecture,
            ModelArchitecture::Llama
                | ModelArchitecture::Mistral
                | ModelArchitecture::Qwen
                | ModelArchitecture::Gemma
                | ModelArchitecture::DeepSeek
        )
    };
    let tied_embeddings = !ir.tensors.iter().any(|t| t.role == TensorRole::OutputHead);

    Some(ModelShape {
        layers: ir.num_layers,
        hidden: ir.hidden_size,
        intermediate,
        heads: ir.num_attention_heads,
        kv_heads: ir.num_key_value_heads,
        vocab: ir.vocab_size,
        gated_mlp,
        tied_embeddings,
    })
}

fn predicted_basis() -> Vec<BasisLabel> {
    [
        "context",
        "kv",
        "weight_bytes_per_param",
        "memory.weights",
        "memory.kv",
        "memory.scratch",
        "memory.total",
    ]
    .into_iter()
    .map(|decision| BasisLabel {
        decision: decision.to_string(),
        basis: Basis::Predicted,
    })
    .collect()
}

/// Builds a [`RunPlan`] from a [`ModelShape`], shrinking settings when `budget_bytes` is given.
///
/// # Errors
///
/// Propagates [`ShrinkError`] if `budget_bytes` is provided and [`shrink_to_budget`] cannot fit
/// within the requested budget while honoring `fixed` and `limits`.
pub fn plan_run(
    shape: &ModelShape,
    model_name: impl Into<String>,
    start: RunSettings,
    fixed: Fixed,
    limits: ShrinkLimits,
    budget_bytes: Option<u64>,
) -> Result<RunPlan, ShrinkError> {
    let settings = match budget_bytes {
        Some(budget) => shrink_to_budget(shape, start, fixed, limits, budget)?,
        None => start,
    };
    let memory = run_memory(
        shape,
        settings.weight_bytes_per_param,
        settings.context,
        settings.kv,
    );

    Ok(RunPlan {
        model: model_name.into(),
        settings,
        memory,
        budget_bytes,
        basis: predicted_basis(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::TensorIR;
    use crate::profiling::run_memory::KvType;

    fn tiny_llama() -> ModelShape {
        ModelShape {
            layers: 2,
            hidden: 8,
            intermediate: 16,
            heads: 2,
            kv_heads: 1,
            vocab: 10,
            gated_mlp: true,
            tied_embeddings: false,
        }
    }

    #[test]
    fn plan_run_tiny_llama_no_budget_matches_hand_worked_literals() {
        // Tiny LLaMA at 4.0 bytes/param, F32 KV, context=5:
        //   weights = 1352 * 4 = 5408 B
        //   kv      = 2 * 2 * 5 * (4 * 4) = 320 B
        //   scratch = (8*4 + 16*2 + 5*2) * 4 = 74 * 4 = 296 B
        //   total   = 5408 + 320 + 296 = 6024 B
        let shape = tiny_llama();
        let start = RunSettings {
            context: 5,
            kv: KvType::F32,
            weight_bytes_per_param: 4.0,
        };
        let plan = plan_run(
            &shape,
            "tiny-llama",
            start,
            Fixed::default(),
            ShrinkLimits { context_floor: 1 },
            None,
        )
        .expect("unconstrained plan should succeed");

        assert_eq!(plan.model, "tiny-llama");
        assert_eq!(plan.settings, start);
        assert_eq!(plan.memory.weights, 5408);
        assert_eq!(plan.memory.kv, 320);
        assert_eq!(plan.memory.scratch, 296);
        assert_eq!(plan.total_bytes(), 6024);
        assert_eq!(plan.budget_bytes, None);
        assert!(!plan.basis.is_empty());
        assert!(plan.basis.iter().all(|b| b.basis == Basis::Predicted));
    }

    #[test]
    fn plan_run_tiny_llama_shrinks_kv_and_context_with_literal_budgets() {
        let shape = tiny_llama();
        let start = RunSettings {
            context: 5,
            kv: KvType::F32,
            weight_bytes_per_param: 4.0,
        };
        let limits = ShrinkLimits { context_floor: 1 };

        // Exact 6024 B budget keeps start unchanged.
        let exact = plan_run(
            &shape,
            "tiny-llama",
            start,
            Fixed::default(),
            limits,
            Some(6024),
        )
        .expect("6024 B should fit without shrinking");
        assert_eq!(exact.settings, start);
        assert_eq!(exact.total_bytes(), 6024);
        assert_eq!(exact.budget_bytes, Some(6024));

        // 5900 B shrinks KV F32 (6024 B) -> F16 (5408 + 160 + 296 = 5864 B).
        let shrunk_kv = plan_run(
            &shape,
            "tiny-llama",
            start,
            Fixed::default(),
            limits,
            Some(5900),
        )
        .expect("5900 B should fit at F16 KV");
        assert_eq!(shrunk_kv.settings.kv, KvType::F16);
        assert_eq!(shrunk_kv.settings.context, 5);
        assert_eq!(shrunk_kv.memory.weights, 5408);
        assert_eq!(shrunk_kv.memory.kv, 160);
        assert_eq!(shrunk_kv.memory.scratch, 296);
        assert_eq!(shrunk_kv.total_bytes(), 5864);

        // 5800 B with KV fixed at F32 halves context 5 -> 2 (5808 B) -> 1 (5736 B).
        let fixed_kv = Fixed {
            context: false,
            kv: true,
            weights: false,
        };
        let shrunk_ctx = plan_run(&shape, "tiny-llama", start, fixed_kv, limits, Some(5800))
            .expect("5800 B should fit at context 1");
        assert_eq!(shrunk_ctx.settings.kv, KvType::F32);
        assert_eq!(shrunk_ctx.settings.context, 1);
        assert_eq!(shrunk_ctx.memory.weights, 5408);
        assert_eq!(shrunk_ctx.memory.kv, 64);
        assert_eq!(shrunk_ctx.memory.scratch, 264);
        assert_eq!(shrunk_ctx.total_bytes(), 5736);

        // 5000 B with weights fixed cannot fit (minimum is Int4 KV, context=1 -> 5696 B).
        let fixed_weights = Fixed {
            context: false,
            kv: false,
            weights: true,
        };
        let err = plan_run(
            &shape,
            "tiny-llama",
            start,
            fixed_weights,
            limits,
            Some(5000),
        )
        .expect_err("5000 B with fixed 5408 B weights must fail");
        assert!(matches!(
            err,
            ShrinkError::DoesNotFit {
                needed: 5696,
                budget: 5000,
                ..
            }
        ));
    }

    #[test]
    fn shape_from_ir_handles_complete_and_missing_fields() {
        // Default ModelIR has no tensors and no feed_forward_length -> intermediate missing -> None.
        let empty = ModelIR::default();
        assert_eq!(shape_from_ir(&empty), None);

        // Zero dimension -> None.
        let zero_layers = ModelIR::new(
            "bad".into(),
            ModelArchitecture::Llama,
            0,
            8,
            2,
            1,
            10,
            16,
            vec![TensorIR::new(
                "model.layers.0.mlp.gate_proj.weight".into(),
                vec![16, 8],
                "F16".into(),
                256,
            )],
        );
        assert_eq!(shape_from_ir(&zero_layers), None);

        // Complete IR with gated MLP and tied embeddings (no lm_head tensor).
        let complete = ModelIR::new(
            "complete".into(),
            ModelArchitecture::Llama,
            2,
            8,
            2,
            1,
            10,
            16,
            vec![
                TensorIR::new(
                    "model.embed_tokens.weight".into(),
                    vec![10, 8],
                    "F16".into(),
                    160,
                ),
                TensorIR::new(
                    "model.layers.0.mlp.gate_proj.weight".into(),
                    vec![8, 16],
                    "F16".into(),
                    256,
                ),
            ],
        );
        assert_eq!(
            shape_from_ir(&complete),
            Some(ModelShape {
                layers: 2,
                hidden: 8,
                intermediate: 16,
                heads: 2,
                kv_heads: 1,
                vocab: 10,
                gated_mlp: true,
                tied_embeddings: true,
            })
        );
    }

    #[test]
    fn run_plan_serializes_to_json_with_predicted_basis() {
        let shape = tiny_llama();
        let plan = plan_run(
            &shape,
            "tiny-llama",
            RunSettings {
                context: 5,
                kv: KvType::F32,
                weight_bytes_per_param: 4.0,
            },
            Fixed::default(),
            ShrinkLimits { context_floor: 1 },
            Some(6024),
        )
        .expect("plan should succeed");

        let value = serde_json::to_value(&plan).expect("serialize RunPlan");
        assert_eq!(value["model"], "tiny-llama");
        assert_eq!(value["settings"]["context"], 5);
        assert_eq!(value["settings"]["kv"], "F32");
        assert_eq!(value["settings"]["weight_bytes_per_param"], 4.0);
        assert_eq!(value["memory"]["weights"], 5408);
        assert_eq!(value["memory"]["kv"], 320);
        assert_eq!(value["memory"]["scratch"], 296);
        assert_eq!(value["budget_bytes"], 6024);
        assert!(value["basis"]
            .as_array()
            .expect("basis array")
            .iter()
            .all(|entry| entry["basis"] == "predicted"));
    }
}
