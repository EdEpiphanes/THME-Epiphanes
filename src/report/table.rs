// SPDX-License-Identifier: AGPL-3.0-only
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Color, ContentArrangement, Table};

use crate::allocator::AllocationPlan;
use crate::ir::ModelIR;
use crate::profiling::{KvCacheProjection, LayerSensitivityReport};
use crate::quality::GateResult;

pub struct ReportPrinter;

impl ReportPrinter {
    /// Render Model Inspection Table
    pub fn print_model_summary(model: &ModelIR) {
        println!("\n=== THME Model Inspection Summary ===");
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);

        table.set_header(vec![
            Cell::new("Property").fg(Color::Cyan),
            Cell::new("Value").fg(Color::Yellow),
        ]);

        table.add_row(vec!["Model Name", &model.name]);
        table.add_row(vec!["Architecture", model.architecture.display_name()]);
        table.add_row(vec![
            "Total Parameters",
            &format!(
                "{:.2} B ({})",
                model.num_parameters as f64 / 1e9,
                model.num_parameters
            ),
        ]);
        table.add_row(vec![
            "Raw Weight Size",
            &format!(
                "{:.2} GB",
                model.original_byte_size as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
        ]);
        table.add_row(vec!["Hidden Size", &model.hidden_size.to_string()]);
        table.add_row(vec!["Number of Layers", &model.num_layers.to_string()]);
        table.add_row(vec![
            "Attention Heads (Q)",
            &model.num_attention_heads.to_string(),
        ]);
        table.add_row(vec![
            "Key-Value Heads (KV)",
            &model.num_key_value_heads.to_string(),
        ]);
        table.add_row(vec!["Head Dimension", &model.head_dim.to_string()]);
        table.add_row(vec!["Vocabulary Size", &model.vocab_size.to_string()]);
        table.add_row(vec![
            "Max Context Length",
            &model.max_position_embeddings.to_string(),
        ]);
        table.add_row(vec!["Total Tensors", &model.tensors.len().to_string()]);

        println!("{table}");
    }

    /// Render Layer Sensitivity & Recommended Precision Table
    pub fn print_layer_sensitivity(layers: &[LayerSensitivityReport]) {
        println!("\n=== Topological Layer Sensitivity Analysis ===");
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);

        table.set_header(vec![
            Cell::new("Layer").fg(Color::Cyan),
            Cell::new("Parameters").fg(Color::Cyan),
            Cell::new("Original MB").fg(Color::Cyan),
            Cell::new("Sensitivity Score").fg(Color::Cyan),
            Cell::new("Recommended Precision").fg(Color::Green),
        ]);

        for l in layers {
            let sens_color = if l.average_sensitivity > 0.6 {
                Color::Red
            } else if l.average_sensitivity > 0.35 {
                Color::Yellow
            } else {
                Color::Green
            };

            table.add_row(vec![
                Cell::new(format!("Layer {:02}", l.layer_index)),
                Cell::new(format!("{:.2} M", l.total_parameters as f64 / 1e6)),
                Cell::new(format!(
                    "{:.1} MB",
                    l.total_bytes as f64 / (1024.0 * 1024.0)
                )),
                Cell::new(format!("{:.3}", l.average_sensitivity)).fg(sens_color),
                Cell::new(l.recommended_format.display_name()).fg(Color::Cyan),
            ]);
        }

        println!("{table}");
    }

    /// Render KV Cache Projections Table
    pub fn print_kv_cache(projections: &[KvCacheProjection]) {
        println!("\n=== Autoregressive KV-Cache Memory Projections ===");
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);

        table.set_header(vec![
            Cell::new("Context Tokens").fg(Color::Cyan),
            Cell::new("Batch Size").fg(Color::Cyan),
            Cell::new("FP16 Memory").fg(Color::Yellow),
            Cell::new("FP8 Memory").fg(Color::Green),
            Cell::new("Q4 Memory (Est.)").fg(Color::Magenta),
        ]);

        for p in projections {
            table.add_row(vec![
                Cell::new(p.context_length.to_string()),
                Cell::new(p.batch_size.to_string()),
                Cell::new(format!("{:.2} MB", p.mb_fp16())),
                Cell::new(format!("{:.2} MB ({:.1}% savings)", p.mb_fp8(), 50.0)),
                Cell::new(format!("{:.2} MB ({:.1}% savings)", p.mb_q4(), 75.0)),
            ]);
        }

        println!("{table}");
    }

    /// Render Heterogeneous Allocation Plan
    pub fn print_allocation_plan(plan: &AllocationPlan) {
        println!("\n=== Heterogeneous Precision Allocation Plan ===");
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);

        table.set_header(vec![
            Cell::new("Metric").fg(Color::Cyan),
            Cell::new("Value").fg(Color::Yellow),
        ]);

        table.add_row(vec!["Target Model", &plan.model_name]);
        table.add_row(vec![
            "Original Size",
            &format!("{:.2} GB", plan.original_size_gb),
        ]);
        table.add_row(vec![
            "Planned Compressed Size",
            &format!("{:.2} GB", plan.planned_size_gb),
        ]);
        table.add_row(vec![
            "Compression Ratio",
            &format!("{:.2}x", plan.compression_ratio),
        ]);
        table.add_row(vec![
            "Estimated Quality Retention",
            &format!("{:.1}%", plan.estimated_quality_retention_pct),
        ]);

        println!("{table}");
    }

    /// Render Quality Gate Decision
    pub fn print_gate_result(result: &GateResult) {
        println!("\n=== Quality Gate Verification ===");
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS);

        let status_cell = match result.decision {
            crate::quality::GateDecision::Accept => Cell::new("ACCEPTED").fg(Color::Green),
            crate::quality::GateDecision::Reject => Cell::new("REJECTED").fg(Color::Red),
        };

        table.add_row(vec![Cell::new("Gate Decision"), status_cell]);
        table.add_row(vec![
            Cell::new("Measured Degradation"),
            Cell::new(format!("{:.3} pts", result.degradation)),
        ]);
        table.add_row(vec![
            Cell::new("Threshold Tolerance"),
            Cell::new(format!("{:.3} pts", result.max_allowed_degradation)),
        ]);
        table.add_row(vec![Cell::new("Summary"), Cell::new(&result.summary)]);

        println!("{table}");
    }
}
