// SPDX-License-Identifier: AGPL-3.0-only
use serde::{Deserialize, Serialize};

use crate::ir::ModelIR;
use crate::profiling::sensitivity::LayerSensitivityReport;

/// Structured prompt context payload for external Teacher LLMs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherPromptPayload {
    pub model_name: String,
    pub architecture: String,
    pub total_layers: usize,
    pub total_parameters: usize,
    pub top_sensitive_layers: Vec<LayerSensitivitySummary>,
    pub memory_footprint_mb: f64,
    pub target_reduction_ratio: f32,
    pub max_ppl_delta: f32,
    pub instruction: String,
    pub recipe_json_schema: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSensitivitySummary {
    pub layer_index: usize,
    pub average_sensitivity: f64,
    pub total_parameters: usize,
    pub recommended_format: String,
}

/// Prompt Synthesizer generating concise markdown and JSON context for Teacher agents.
pub struct PromptSynthesizer;

impl PromptSynthesizer {
    /// Formats layer sensitivities and model properties into structured Teacher prompt payload.
    pub fn build_payload(
        model: &ModelIR,
        reports: &[LayerSensitivityReport],
        target_reduction: f32,
        max_ppl_delta: f32,
    ) -> TeacherPromptPayload {
        let mut summaries: Vec<LayerSensitivitySummary> = reports
            .iter()
            .map(|r| LayerSensitivitySummary {
                layer_index: r.layer_index,
                average_sensitivity: r.average_sensitivity,
                total_parameters: r.total_parameters,
                recommended_format: r.recommended_format.display_name().to_string(),
            })
            .collect();

        // Sort descending by sensitivity to highlight critical layers first
        summaries.sort_by(|a, b| {
            b.average_sensitivity
                .partial_cmp(&a.average_sensitivity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        summaries.truncate(8); // Keep top 8 for prompt conciseness

        let schema = r#"{
  "name": "Teacher Surgery Recipe",
  "version": "1.0.0",
  "author": "Teacher LLM",
  "description": "Surgery recipe proposed by Teacher LLM",
  "target_model": "llama",
  "expected_memory_savings_pct": 30.0,
  "expected_speedup": 1.2,
  "operations": [
    {
      "type": "SetPrecision",
      "params": {
        "start_layer": 0,
        "end_layer": 4,
        "role": "MlpDown",
        "format": "Ternary"
      }
    }
  ]
}"#;

        TeacherPromptPayload {
            model_name: model.name.clone(),
            architecture: model.architecture.display_name().to_string(),
            total_layers: model.num_layers,
            total_parameters: model.num_parameters,
            top_sensitive_layers: summaries,
            memory_footprint_mb: model.total_estimated_bytes() as f64 / (1024.0 * 1024.0),
            target_reduction_ratio: target_reduction,
            max_ppl_delta,
            instruction: "Analyze the layer sensitivity profile. Propose a Pareto-optimal TweakRecipe JSON reducing parameters while keeping perplexity delta within budget.".to_string(),
            recipe_json_schema: schema.to_string(),
        }
    }

    /// Renders payload into compact Markdown suitable for agent system prompt or user query (<2k tokens).
    pub fn render_markdown(payload: &TeacherPromptPayload) -> String {
        let mut out = String::with_capacity(1500);
        out.push_str("# THME Teacher Co-Surgery Brief\n\n");
        out.push_str(&format!(
            "- **Model**: {} ({})\n",
            payload.model_name, payload.architecture
        ));
        out.push_str(&format!(
            "- **Total Layers**: {} | **Parameters**: {}\n",
            payload.total_layers, payload.total_parameters
        ));
        out.push_str(&format!(
            "- **Current Memory**: {:.2} MB\n",
            payload.memory_footprint_mb
        ));
        out.push_str(&format!(
            "- **Target Reduction**: {:.1}%\n",
            payload.target_reduction_ratio * 100.0
        ));
        out.push_str(&format!(
            "- **Max Allowed PPL Delta**: +{:.3}\n\n",
            payload.max_ppl_delta
        ));

        out.push_str("### Critical Sensitive Layers (Top K)\n");
        out.push_str("| Layer | Average Sensitivity | Parameters | Recommended Format |\n");
        out.push_str("|---|---|---|---|\n");
        for s in &payload.top_sensitive_layers {
            out.push_str(&format!(
                "| Layer {} | {:.4} | {} | {} |\n",
                s.layer_index, s.average_sensitivity, s.total_parameters, s.recommended_format
            ));
        }

        out.push_str("\n### Response Format Requirement\n");
        out.push_str("You must reply ONLY with valid JSON conforming to the following TweakRecipe schema:\n```json\n");
        out.push_str(&payload.recipe_json_schema);
        out.push_str("\n```\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::rebuilder::{ArchitectureBlueprint, ModelRebuilder};
    use crate::ir::TensorFormat;

    #[test]
    fn test_prompt_synthesizer_token_and_format_bounds() {
        let blueprint = ArchitectureBlueprint::new_hybrid_llama("llama-8b-mock", 1, 64, 1);
        let model = ModelRebuilder::build_scaffold(&blueprint);

        let reports = vec![LayerSensitivityReport {
            layer_index: 0,
            average_sensitivity: 0.78,
            total_parameters: 4096,
            total_bytes: 16384,
            recommended_format: TensorFormat::Ternary,
        }];

        let payload = PromptSynthesizer::build_payload(&model, &reports, 0.40, 0.12);
        let md = PromptSynthesizer::render_markdown(&payload);

        assert!(md.contains("THME Teacher Co-Surgery Brief"));
        assert!(md.contains("llama-8b-mock"));
        assert!(md.len() < 2500); // Guarantees compact context well within token budgets
    }
}
