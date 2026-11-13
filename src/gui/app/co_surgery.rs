// SPDX-License-Identifier: AGPL-3.0-only
//! Studio tab: TAB 9: Closed-Loop Teacher-Student Co-Surgery Dashboard (v0.1.9)

use super::*;

impl THMEApp {
    /// TAB 9: Closed-Loop Teacher-Student Co-Surgery Dashboard (v0.1.9)
    pub(super) fn render_tab_cosurgery(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();

        ui.heading(
            egui::RichText::new("CLOSED-LOOP TEACHER-STUDENT CO-SURGERY DASHBOARD")
                .size(13.0)
                .color(p.text_main),
        );
        ui.label(
            egui::RichText::new(
                "Autonomous co-surgery loop: layer topological profiling, Teacher prompt synthesis, 3D Pareto frontier optimization, self-healing relaxation, and anti-degradation rollback watchdog.",
            )
            .size(10.0)
            .color(p.text_muted),
        );
        ui.label(
            egui::RichText::new(
                "SIMULATED: perplexity figures in this tab come from the analytic proxy. For measured figures run `thme co-surgery <MODEL_PATH> --tokens <TOKENS>`.",
            )
            .size(10.0)
            .strong()
            .color(p.warning),
        );
        ui.add_space(8.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Top Section: Budget & Configuration Controls + Action Trigger
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Optimization Budget:").strong().color(p.text_main));
                    ui.add_space(8.0);
                    ui.label("Iterations:");
                    ui.add(egui::Slider::new(&mut self.cosurgery_budget_iterations, 1..=30));
                    ui.add_space(8.0);
                    ui.label("Max Δ-PPL:");
                    ui.add(egui::Slider::new(&mut self.cosurgery_max_ppl_delta, 0.01..=1.00).step_by(0.01));
                    ui.add_space(8.0);
                    ui.label("Target Reduction:");
                    let mut red_pct = (self.cosurgery_target_reduction * 100.0).round() as u32;
                    if ui.add(egui::Slider::new(&mut red_pct, 10..=90).suffix("%")).changed() {
                        self.cosurgery_target_reduction = red_pct as f64 / 100.0;
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(egui::RichText::new("> Run Autonomous Co-Surgery").color(p.text_main)).clicked() {
                            let mut model_ir = self.model().source_model.clone().unwrap_or_else(|| self.model().to_model_ir());
                            if model_ir.tensors.is_empty() {
                                let bp = crate::ir::rebuilder::ArchitectureBlueprint::new_hybrid_llama("scaffold", 2, 128, 1);
                                model_ir = crate::ir::ModelRebuilder::build_scaffold(&bp);
                            }
                            let budget = crate::surgery::CoSurgeryBudget {
                                max_iterations: self.cosurgery_budget_iterations,
                                max_ppl_delta: self.cosurgery_max_ppl_delta,
                                target_reduction_ratio: self.cosurgery_target_reduction,
                                max_consecutive_rollbacks: 3,
                            };
                            let mut controller = crate::surgery::CoSurgeryController::new(model_ir, budget);
                            let report = controller.run_to_completion();

                            self.cosurgery_log.clear();
                            self.cosurgery_log.push(format!("Initialized co-surgery run: budget {} iters, max_ppl_delta {:.3}", self.cosurgery_budget_iterations, self.cosurgery_max_ppl_delta));
                            for cp in &report.history {
                                self.cosurgery_log.push(format!("Iter {:>2}: PPL {:.4} (Δ {:+.4}) | Size {:.1}% red | {}", cp.iteration, cp.candidate_ppl, cp.ppl_delta, cp.reduction_ratio * 100.0, cp.recipe.name));
                            }
                            self.cosurgery_log.push(format!("Finalized: {} commits, {} rollbacks, {} relaxations | Final PPL {:.4}", report.successful_commits, report.total_rollbacks, report.total_relaxations, report.final_ppl));
                            let model_name = self.model().name.clone();
                            self.log_audit("INFO", "Co-Surgery", &format!("Completed co-surgery: {} commits, {:.1}% reduction", report.successful_commits, report.final_reduction_ratio * 100.0), &model_name);
                            self.cosurgery_report = Some(report);
                        }
                    });
                });
            });

            ui.add_space(8.0);

            // KPI Dashboard Ribbon
            if let Some(report) = &self.cosurgery_report {
                ui.columns(5, |cols| {
                    render_stat_box(&mut cols[0], "ITERATIONS / COMMITS", format!("{} / {}", report.total_iterations, report.successful_commits), p.success, &p);
                    render_stat_box(&mut cols[1], "ROLLBACKS / RELAX", format!("{} / {}", report.total_rollbacks, report.total_relaxations), p.warning, &p);
                    render_stat_box(&mut cols[2], "BASELINE PPL", format!("{:.4}", report.baseline_ppl), p.text_main, &p);
                    let delta_color = if crate::quality::gate::GateSpec::perplexity(self.cosurgery_max_ppl_delta).accepts_perplexity(report.baseline_ppl, report.final_ppl) { p.success } else { p.danger };
                    render_stat_box(&mut cols[3], "FINAL PPL (Δ)", format!("{:.4} ({:+.3})", report.final_ppl, report.final_ppl - report.baseline_ppl), delta_color, &p);
                    render_stat_box(&mut cols[4], "SIZE REDUCTION", format!("{:.1}%", report.final_reduction_ratio * 100.0), p.accent, &p);
                });
            } else {
                ui.columns(5, |cols| {
                    render_stat_box(&mut cols[0], "ITERATIONS / COMMITS", "0 / 0".into(), p.text_muted, &p);
                    render_stat_box(&mut cols[1], "ROLLBACKS / RELAX", "0 / 0".into(), p.text_muted, &p);
                    render_stat_box(&mut cols[2], "BASELINE PPL", "5.8400".into(), p.text_muted, &p);
                    render_stat_box(&mut cols[3], "FINAL PPL (Δ)", "5.8400 (+0.000)".into(), p.text_muted, &p);
                    render_stat_box(&mut cols[4], "SIZE REDUCTION", "0.0%".into(), p.text_muted, &p);
                });
            }

            ui.add_space(10.0);

            // Two-column display: Iteration Timeline / Checkpoint History & Telemetry Log
            ui.columns(2, |cols| {
                // Column 0: Checkpoint History & Pareto Points
                cols[0].group(|ui| {
                    ui.label(egui::RichText::new("Pareto Frontier & Checkpoint History").strong().color(p.text_main));
                    ui.separator();
                    egui::ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                        if let Some(report) = &self.cosurgery_report {
                            if report.history.is_empty() {
                                ui.label(egui::RichText::new("No committed checkpoints yet.").size(10.0).color(p.text_muted));
                            } else {
                                egui::Grid::new("cosurgery_history_grid").striped(true).min_col_width(50.0).show(ui, |ui| {
                                    ui.label(egui::RichText::new("Iter").strong().size(9.0));
                                    ui.label(egui::RichText::new("Candidate PPL").strong().size(9.0));
                                    ui.label(egui::RichText::new("Δ-PPL").strong().size(9.0));
                                    ui.label(egui::RichText::new("Reduction").strong().size(9.0));
                                    ui.label(egui::RichText::new("Recipe Name").strong().size(9.0));
                                    ui.end_row();

                                    for cp in &report.history {
                                        ui.label(egui::RichText::new(format!("#{}", cp.iteration)).size(9.0));
                                        ui.label(egui::RichText::new(format!("{:.4}", cp.candidate_ppl)).size(9.0).color(p.accent));
                                        let delta_color = if crate::quality::gate::GateSpec::perplexity(self.cosurgery_max_ppl_delta).accepts_perplexity(0.0, cp.ppl_delta) { p.success } else { p.warning };
                                        ui.label(egui::RichText::new(format!("{:+.4}", cp.ppl_delta)).size(9.0).color(delta_color));
                                        ui.label(egui::RichText::new(format!("{:.1}%", cp.reduction_ratio * 100.0)).size(9.0).color(p.success));
                                        ui.label(egui::RichText::new(&cp.recipe.name).size(9.0).color(p.text_muted));
                                        ui.end_row();
                                    }
                                });
                            }
                        } else {
                            ui.label(egui::RichText::new("Run co-surgery to populate Pareto checkpoints.").size(10.0).color(p.text_muted));
                        }
                    });
                });

                // Column 1: Co-Surgery Telemetry Stream & Watchdog Log
                cols[1].group(|ui| {
                    ui.label(egui::RichText::new("Telemetry Stream & Watchdog Log").strong().color(p.text_main));
                    ui.separator();
                    egui::ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                        if self.cosurgery_log.is_empty() {
                            ui.label(egui::RichText::new("Waiting for optimization telemetry...").size(10.0).color(p.text_muted));
                        } else {
                            for line in &self.cosurgery_log {
                                ui.label(egui::RichText::new(line).monospace().size(9.0).color(p.secondary));
                            }
                        }
                    });
                });
            });
        });
    }
}
