//! Studio tab: TAB 0: Master Dashboard (All tabs unified in one master cockpit)

use super::*;

impl THMEApp {
    /// TAB 0: Master Dashboard (All tabs unified in one master cockpit)
    pub(super) fn render_tab_dashboard(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.columns(4, |cols| {
                // Card 1: Models & Active Architecture
                cols[0].vertical(|ui| {
                    manga::glass_frame(&p, p.border)
                        .inner_margin(egui::Margin::same(12))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("MODELS REPOSITORY").strong().color(p.primary));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Manage ->").clicked() {
                                        self.active_tab = 1;
                                    }
                                });
                            });
                            ui.separator();
                            let m = self.model();
                            ui.label(egui::RichText::new(f!("Active: {}", m.name)).strong().color(p.text_main));
                            ui.label(egui::RichText::new(f!("Format: {} | Arch: {}", m.format, m.architecture)).size(10.0).color(p.text_muted));
                            ui.label(egui::RichText::new(f!("Layers: {} | Params: {}M", m.layers.len(), m.total_parameters / 1_000_000)).size(10.0).color(p.text_muted));
                            ui.label(egui::RichText::new(f!("Tensors Loaded: {}", m.tensors.len())).size(10.0).color(p.success));
                        });
                });

                // Card 2: Matrix & Distillation Plan Status
                cols[1].vertical(|ui| {
                    manga::glass_frame(&p, p.border)
                        .inner_margin(egui::Margin::same(12))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("MATRIX ENGINE").strong().color(p.success));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Engine ->").clicked() {
                                        self.active_tab = 3;
                                    }
                                });
                            });
                            ui.separator();
                            if let Some(summary) = &self.plan_summary {
                                ui.label(egui::RichText::new(f!("Compression: {:.2}x", summary.overall_compression_ratio)).strong().color(p.primary));
                                ui.label(egui::RichText::new(f!("{:.1} MB -> {:.1} MB", summary.original_storage_mb, summary.optimized_storage_mb)).size(10.0).color(p.text_muted));
                                ui.label(egui::RichText::new(f!("Distortion: {:.4} | Qual: {:.1}%", summary.average_distortion_snr, summary.mean_quality_score * 100.0)).size(10.0).color(p.warning));
                                ui.label(egui::RichText::new(f!("BYOA Custom: {} | SVD: {} | BitNet: {}", summary.count_custom, summary.count_svd, summary.count_bitnet)).size(9.0).color(p.text_muted));
                            } else {
                                ui.label(egui::RichText::new("Auto-Decider plan pending.").color(p.text_muted));
                            }
                        });
                });

                // Card 3: REA-LLM & EWHA Telemetry
                cols[2].vertical(|ui| {
                    manga::glass_frame(&p, p.border)
                        .inner_margin(egui::Margin::same(12))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("REA-LLM / EWHA").strong().color(p.accent));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Metrics ->").clicked() {
                                        self.active_tab = 4;
                                    }
                                });
                            });
                            ui.separator();
                            let (_, total_traffic_mb, f_bar, joules, _) = self.model().compute_costs();
                            ui.label(egui::RichText::new(f!("Depth Fraction: {:.1}%", f_bar * 100.0)).strong().color(p.accent));
                            ui.label(egui::RichText::new(f!("Traffic W: {:.2} MB/tok", total_traffic_mb)).size(10.0).color(p.text_main));
                            ui.label(egui::RichText::new(f!("Energy E: {:.2} mJ/tok", joules * 1000.0)).size(10.0).color(p.success));
                            ui.label(egui::RichText::new(f!("Checkpoint: L{} (80/20 Exit)", self.checkpoint_layer_idx)).size(9.0).color(p.text_muted));
                        });
                });

                // Card 4: Benchmarks & Live Telemetry
                cols[3].vertical(|ui| {
                    manga::glass_frame(&p, p.border)
                        .inner_margin(egui::Margin::same(12))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("FLOW TELEMETRY").strong().color(p.secondary));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Flow ->").clicked() {
                                        self.active_tab = 4;
                                    }
                                });
                            });
                            ui.separator();
                            let snap = self.simulator.snapshot();
                            ui.label(egui::RichText::new(f!("Live: {:.1} tok/s", snap.current_tok_per_sec)).strong().color(p.primary));
                            ui.label(egui::RichText::new(f!("Stage: {}", snap.active_stage.display_name())).size(10.0).color(p.text_muted));
                            ui.label(egui::RichText::new(f!("Progress: {}/{} tokens", snap.tokens_generated, snap.target_tokens)).size(10.0).color(p.text_muted));
                            ui.label(egui::RichText::new(f!("KV Cache: {:.2} MB", snap.kv_cache_mb)).size(10.0).color(p.warning));
                        });
                });
            });

            ui.add_space(8.0);

            // Dashboard Quick Action Bar
            ui.horizontal(|ui| {
                if ui.button(egui::RichText::new("Run Auto-Decider For All Parameters").strong().color(p.text_main)).clicked() {
                    self.run_auto_decider_internal();
                    self.status_msg = "Evaluated all parameters and generated Pareto-optimal matrix & distillation plan!".to_string();
                }

                if ui.button(egui::RichText::new("Execute Global Transformation Plan").strong().color(p.text_main)).clicked() {
                    self.execute_global_transformation_plan();
                }

                if ui.button("Snapshot Checkpoint").clicked() {
                    self.create_checkpoint("Manual Snapshot", "Saved state from Master Dashboard");
                    self.status_msg = "Snapshot checkpoint created.".to_string();
                }

                if ui.button("Run Live Benchmark").clicked() {
                    self.active_tab = 4;
                    self.simulator.start();
                }
            });

            ui.add_space(8.0);

            // Live Neural Network Stage Telemetry & Manifold Map
            ui.columns(2, |cols| {
                cols[0].vertical(|ui| {
                    ui.heading(egui::RichText::new("LIVE TOKEN FLOW TELEMETRY").size(12.0).color(p.text_main));
                    ui.label(egui::RichText::new("Real-time telemetry through 13 discrete Transformer execution stages.").size(9.0).color(p.text_muted));
                    ui.add_space(4.0);

                    let snap = self.simulator.snapshot();
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 160.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 6.0, p.bg_card);
                    ui.painter().rect_stroke(rect, 6.0, Stroke::new(1.0_f32, p.border), egui::StrokeKind::Outside);

                    let num_layers = snap.total_layers.max(1);
                    let start_x = rect.min.x + 28.0;
                    let spacing_x = (rect.width() - 56.0) / (num_layers + 2) as f32;
                    let cy = rect.center().y;

                    // Backbone connecting line
                    ui.painter().line_segment(
                        [Pos2::new(start_x, cy), Pos2::new(start_x + (num_layers + 1) as f32 * spacing_x, cy)],
                        Stroke::new(2.0_f32, p.border),
                    );

                    for l in 0..num_layers {
                        let lx = start_x + (l + 1) as f32 * spacing_x;
                        let is_cur = snap.status == crate::profiling::TokenFlowStatus::Generating && snap.active_layer == l;
                        let is_cp = l == self.checkpoint_layer_idx;
                        let l_rect = Rect::from_center_size(Pos2::new(lx, cy), Vec2::new(24.0, 36.0));

                        let col = if is_cur { p.primary } else if is_cp { p.success } else { p.bg_input };
                        ui.painter().rect_filled(l_rect, 4.0, col);
                        if is_cur {
                            ui.painter().rect_stroke(l_rect, 4.0, Stroke::new(2.0_f32, p.text_main), egui::StrokeKind::Outside);
                        }

                        ui.painter().text(
                            Pos2::new(lx, cy),
                            egui::Align2::CENTER_CENTER,
                            f!("L{}", l),
                            egui::FontId::proportional(8.0),
                            if p.is_light() { p.text_main } else { Color32::WHITE },
                        );
                    }

                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(f!("Generated Stream: {}", snap.token_stream.join(" "))).monospace().size(9.0).color(p.text_muted));
                });

                cols[1].vertical(|ui| {
                    ui.heading(egui::RichText::new("LAYER MANIFOLD SENSITIVITY & ALLOCATION").size(12.0).color(p.text_main));
                    ui.label(egui::RichText::new("Parabolic sensitivity curve and precision bit-width distribution across layers.").size(9.0).color(p.text_muted));
                    ui.add_space(4.0);

                    theme::draw_waterfall_chart(ui, &self.model().layers, &p);
                });
            });

            ui.add_space(8.0);

            // Audit Trail Snippet and Checkpoints
            ui.columns(2, |cols| {
                cols[0].vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("RECENT HISTORY CHECKPOINTS").strong().color(p.text_main));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("All Checkpoints ->").clicked() {
                                self.active_tab = 5;
                            }
                        });
                    });
                    ui.separator();
                    for cp in self.history.iter().rev().take(3) {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(f!("#{} [{}]", cp.id, cp.timestamp)).monospace().size(9.0).color(p.primary));
                            ui.label(egui::RichText::new(&cp.label).strong().size(10.0));
                            ui.label(egui::RichText::new(f!("{:.1} MB ({:.1}x)", cp.storage_mb, cp.compression_ratio)).size(9.0).color(p.success));
                        });
                    }
                });

                cols[1].vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("LIVE AUDIT EVENT STREAM").strong().color(p.text_main));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Full Audit Log ->").clicked() {
                                self.active_tab = 6;
                            }
                        });
                    });
                    ui.separator();
                    for ev in self.audit_events.iter().rev().take(3) {
                        ui.horizontal(|ui| {
                            let badge_col = match ev.level.as_str() {
                                "OPTIMIZE" => p.success,
                                "DISTILL" => p.primary,
                                "SURGERY" => p.secondary,
                                "WARN" => p.danger,
                                _ => p.accent,
                            };
                            ui.label(egui::RichText::new(f!("[{}]", ev.level)).strong().size(9.0).color(badge_col));
                            ui.label(egui::RichText::new(&ev.message).size(9.0).color(p.text_muted));
                        });
                    }
                });
            });
        });
    }
}
