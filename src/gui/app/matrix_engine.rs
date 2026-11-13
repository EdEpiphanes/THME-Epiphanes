//! Studio tab: TAB 3: Distillation & Matrix Engine (All algorithms + Automated Parameter Decider)

use super::*;

impl THMEApp {
    /// TAB 3: Distillation & Matrix Engine (All algorithms + Automated Parameter Decider)
    pub(super) fn render_tab_matrix_distill(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();

        ui.heading(
            egui::RichText::new("DISTILLATION & MATRIX TRANSFORMATION ENGINE")
                .size(13.0)
                .color(p.text_main),
        );
        ui.label(
            egui::RichText::new(
                "Implements Truncated SVD, Walsh-Hadamard rotation, SmoothQuant, BitNet b1.58, 2:4 Sparsity, and Knowledge Distillation. The engine evaluates every parameter and assigns the optimal transformation.",
            )
            .size(10.0)
            .color(p.text_muted),
        );
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui
                .button(
                    egui::RichText::new("Run Automated Parameter Decider")
                        .strong()
                        .color(p.text_main),
                )
                .clicked()
            {
                self.run_auto_decider_internal();
                self.status_msg =
                    "Automated parameter decision engine completed analysis!".to_string();
            }

            if ui
                .button(
                    egui::RichText::new("Execute Global Transformation Plan")
                        .strong()
                        .color(p.text_main),
                )
                .clicked()
            {
                self.execute_global_transformation_plan();
            }

            ui.add_space(12.0);
            ui.label("Distillation Temp (τ):");
            ui.add(egui::Slider::new(&mut self.distill_temperature, 1.0..=5.0).text("Temperature"));

            ui.add_space(8.0);
            ui.label("Quality Gate (ε):");
            if ui
                .add(egui::Slider::new(&mut self.quality_tolerance, 0.01..=0.08).text("Tolerance"))
                .changed()
            {
                self.run_auto_decider_internal();
            }
        });

        ui.add_space(8.0);

        if let Some(summary) = &self.plan_summary {
            ui.columns(5, |cols| {
                render_scorecard_box(
                    &mut cols[0],
                    "COMPRESSION",
                    f!("{:.2}x Smaller", summary.overall_compression_ratio),
                    f!(
                        "{:.1} MB Saved",
                        summary.original_storage_mb - summary.optimized_storage_mb
                    ),
                    p.primary,
                    &p,
                );
                render_scorecard_box(
                    &mut cols[1],
                    "MEAN DISTORTION",
                    f!("{:.4} SNR", summary.average_distortion_snr),
                    "Under Gate ε".to_string(),
                    p.success,
                    &p,
                );
                render_scorecard_box(
                    &mut cols[2],
                    "QUALITY SCORE",
                    f!("{:.1}%", summary.mean_quality_score * 100.0),
                    "High Fidelity".to_string(),
                    p.warning,
                    &p,
                );
                render_scorecard_box(
                    &mut cols[3],
                    "SVD & HADAMARD",
                    f!("{} / {}", summary.count_svd, summary.count_hadamard),
                    "Low-Rank & Rotated".to_string(),
                    p.secondary,
                    &p,
                );
                render_scorecard_box(
                    &mut cols[4],
                    "BITNET & BYOA",
                    f!("{} / {}", summary.count_bitnet, summary.count_custom),
                    "Ternary & Custom".to_string(),
                    p.danger,
                    &p,
                );
            });
        }

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Decided Parameter Transformations & Mathematical Rationale:")
                .strong()
                .color(p.text_main),
        );

        egui::ScrollArea::vertical()
            .max_height(280.0)
            .show(ui, |ui| {
                egui::Grid::new("decisions_grid")
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Layer")
                                .strong()
                                .size(9.0)
                                .color(p.text_main),
                        );
                        ui.label(
                            egui::RichText::new("Parameter")
                                .strong()
                                .size(9.0)
                                .color(p.text_main),
                        );
                        ui.label(
                            egui::RichText::new("Role")
                                .strong()
                                .size(9.0)
                                .color(p.text_main),
                        );
                        ui.label(
                            egui::RichText::new("Selected Algorithm")
                                .strong()
                                .size(9.0)
                                .color(p.text_main),
                        );
                        ui.label(
                            egui::RichText::new("Bits/Param")
                                .strong()
                                .size(9.0)
                                .color(p.text_main),
                        );
                        ui.label(
                            egui::RichText::new("Ratio")
                                .strong()
                                .size(9.0)
                                .color(p.text_main),
                        );
                        ui.label(
                            egui::RichText::new("Distortion")
                                .strong()
                                .size(9.0)
                                .color(p.text_main),
                        );
                        ui.label(
                            egui::RichText::new("Rationale")
                                .strong()
                                .size(9.0)
                                .color(p.text_main),
                        );
                        ui.end_row();

                        for dec in &self.parameter_decisions {
                            ui.label(f!("L{}", dec.layer_index));
                            ui.label(egui::RichText::new(&dec.tensor_name).monospace().size(8.0));
                            ui.label(&dec.section_role);
                            ui.label(
                                egui::RichText::new(dec.algorithm.name())
                                    .color(p.primary)
                                    .strong()
                                    .size(9.0),
                            );
                            ui.label(f!("{:.2}", dec.algorithm.effective_bits_per_param()));
                            ui.label(f!("{:.1}x", dec.compression_ratio));
                            ui.label(f!("{:.4}", dec.estimated_distortion_snr));
                            ui.label(
                                egui::RichText::new(&dec.rationale)
                                    .size(8.0)
                                    .color(p.text_muted),
                            );
                            ui.end_row();
                        }
                    });
            });

        ui.add_space(10.0);
        manga::glass_frame(&p, p.border)
            .inner_margin(egui::Margin::same(12))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("BRING YOUR OWN ALGORITHM (BYOA) REGISTRY").strong().color(p.primary));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Design Custom Algorithm").clicked() {
                            self.byoa_designer_open = !self.byoa_designer_open;
                        }
                        if ui.button("Import (.json)").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("JSON", &["json"])
                                .pick_file()
                            {
                                if let Ok(content) = std::fs::read_to_string(&path) {
                                    match crate::transform::byoa::AlgorithmDefinition::from_json(&content) {
                                        Ok(def) => {
                                            let id = def.id.clone();
                                            self.algorithm_registry.register(def);
                                            self.selected_byoa_id = id.clone();
                                            self.status_msg = format!("Imported custom algorithm: {}", id);
                                            self.log_audit("INFO", "BYOA_Import", &format!("Imported algorithm {}", id), "BYOA");
                                            self.run_auto_decider_internal();
                                        }
                                        Err(e) => {
                                            self.status_msg = format!("Failed to parse algorithm JSON: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                        if ui.button("Export Selected (.json)").clicked() {
                            if let Some(alg) = self.algorithm_registry.get(&self.selected_byoa_id) {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("JSON", &["json"])
                                    .set_file_name(format!("{}.json", alg.id))
                                    .save_file()
                                {
                                    if let Ok(json) = alg.to_json() {
                                        let _ = std::fs::write(path, json);
                                        self.status_msg = format!("Exported algorithm '{}' to JSON file!", alg.id);
                                    }
                                }
                            }
                        }
                    });
                });
                ui.label(egui::RichText::new("Register custom quantization, linear transformations, sparsity masks, and sequence mixers. Evaluated dynamically by the Auto-Decider.").size(9.0).color(p.text_muted));

                if self.byoa_designer_open {
                    ui.add_space(8.0);
                    egui::Frame::NONE.fill(manga::glass_fill(p.bg_input, &p)).stroke(Stroke::new(1.0_f32, p.primary)).corner_radius(8)
                        .inner_margin(egui::Margin::same(10))
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Custom Algorithm Designer").strong().color(p.primary));
                            ui.horizontal(|ui| {
                                ui.label("ID:");
                                ui.text_edit_singleline(&mut self.new_byoa_id);
                                ui.label("Name:");
                                ui.text_edit_singleline(&mut self.new_byoa_name);
                                ui.label("Author:");
                                ui.text_edit_singleline(&mut self.new_byoa_author);
                            });
                            ui.horizontal(|ui| {
                                ui.label("Category:");
                                egui::ComboBox::from_id_salt("byoa_designer_cat")
                                    .selected_text(match self.new_byoa_category_idx {
                                        0 => "Quantization",
                                        1 => "LinearTransform",
                                        2 => "SparsityMask",
                                        _ => "SequenceMixer",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut self.new_byoa_category_idx, 0, "Quantization");
                                        ui.selectable_value(&mut self.new_byoa_category_idx, 1, "LinearTransform");
                                        ui.selectable_value(&mut self.new_byoa_category_idx, 2, "SparsityMask");
                                        ui.selectable_value(&mut self.new_byoa_category_idx, 3, "SequenceMixer");
                                    });
                                ui.label("Bits/Param:");
                                ui.add(egui::Slider::new(&mut self.new_byoa_bits, 0.5..=16.0).step_by(0.1));
                            });
                            ui.horizontal(|ui| {
                                if ui.button(egui::RichText::new("Register Algorithm").strong().color(p.text_main)).clicked()
                                    && !self.new_byoa_id.is_empty()
                                    && !self.new_byoa_name.is_empty()
                                {
                                    let cat = match self.new_byoa_category_idx {
                                            0 => crate::transform::byoa::AlgorithmCategory::Quantization,
                                            1 => crate::transform::byoa::AlgorithmCategory::LinearTransformation,
                                            2 => crate::transform::byoa::AlgorithmCategory::SparsityPruning,
                                            _ => crate::transform::byoa::AlgorithmCategory::SequenceMixer,
                                        };
                                        let rule = match self.new_byoa_category_idx {
                                            0 => crate::transform::byoa::TransformationRule::AbsmeanTernary {
                                                scale_gamma: 0.0,
                                                zero_band: 0.05,
                                            },
                                            1 => crate::transform::byoa::TransformationRule::OrthogonalRotation {
                                                block_size: 64,
                                                basis: "Hadamard".to_string(),
                                            },
                                            2 => crate::transform::byoa::TransformationRule::StructuredSparsity {
                                                m: 2,
                                                n: 4,
                                            },
                                            _ => crate::transform::byoa::TransformationRule::AffineProject {
                                                target_dim: 1024,
                                                rank: 64,
                                            },
                                        };
                                        let alg = crate::transform::byoa::AlgorithmDefinition {
                                            id: self.new_byoa_id.clone(),
                                            name: self.new_byoa_name.clone(),
                                            author: self.new_byoa_author.clone(),
                                            version: "1.0.0".to_string(),
                                            category: cat,
                                            description: format!("Custom {} algorithm", self.new_byoa_name),
                                            effective_bits_per_param: self.new_byoa_bits,
                                            compression_ratio: 16.0 / self.new_byoa_bits.max(0.5),
                                            hardware_speedup_factor: 16.0 / self.new_byoa_bits.max(0.5),
                                            target_roles: vec![
                                                crate::ir::TensorRole::AttentionQ,
                                                crate::ir::TensorRole::AttentionK,
                                                crate::ir::TensorRole::AttentionV,
                                                crate::ir::TensorRole::MlpUp,
                                                crate::ir::TensorRole::MlpDown,
                                            ],
                                            rule,
                                        };
                                        self.selected_byoa_id = alg.id.clone();
                                        self.algorithm_registry.register(alg);
                                        self.byoa_designer_open = false;
                                        self.status_msg = format!("Registered new custom algorithm '{}'!", self.new_byoa_id);
                                        self.run_auto_decider_internal();
                                    }
                                    if ui.button("Cancel").clicked() {
                                        self.byoa_designer_open = false;
                                    }
                            });
                        });
                }

                ui.add_space(6.0);
                egui::Grid::new("byoa_registry_grid")
                    .striped(true)
                    .spacing([14.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("ID").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("Name").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("Category").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("Bits/Param").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("Speedup").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("Author").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("Action").strong().size(9.0).color(p.text_main));
                        ui.end_row();

                        for alg in self.algorithm_registry.list() {
                            let is_sel = alg.id == self.selected_byoa_id;
                            ui.label(egui::RichText::new(&alg.id).monospace().size(8.0).color(if is_sel { p.primary } else { p.text_main }));
                            ui.label(egui::RichText::new(&alg.name).strong().size(9.0));
                            ui.label(egui::RichText::new(alg.category.display_name()).size(8.0).color(p.warning));
                            ui.label(egui::RichText::new(format!("{:.1}", alg.effective_bits_per_param)).size(8.0));
                            ui.label(egui::RichText::new(format!("{:.1}x", alg.hardware_speedup_factor)).size(8.0).color(p.success));
                            ui.label(egui::RichText::new(&alg.author).size(8.0).color(p.text_muted));

                            if ui.small_button(if is_sel { "Selected" } else { "Select" }).clicked() {
                                self.selected_byoa_id = alg.id.clone();
                            }
                            ui.end_row();
                        }
                    });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button(egui::RichText::new("Live Test Selected Algorithm On 8x8 Weight Slice").strong().color(p.text_main)).clicked() {
                        if let Some(alg) = self.algorithm_registry.get(&self.selected_byoa_id) {
                            let mut sample = if let Some(t) = self.model().tensors.first() {
                                if t.sample_weights.len() >= 64 {
                                    t.sample_weights[0..64].to_vec()
                                } else {
                                    vec![0.05; 64]
                                }
                            } else {
                                vec![0.05; 64]
                            };

                            let orig_norm: f32 = sample.iter().map(|w| w * w).sum::<f32>().sqrt();
                            let _ = alg.apply_to_weights(&mut sample);
                            let mut mse = 0.0;
                            let mut zeros = 0;
                            for (i, &w) in sample.iter().enumerate() {
                                if w.abs() < 1e-6 { zeros += 1; }
                                let orig = if let Some(t) = self.model().tensors.first() {
                                    if t.sample_weights.len() > i { t.sample_weights[i] } else { 0.05 }
                                } else { 0.05 };
                                mse += (w - orig).powi(2);
                            }
                            mse /= sample.len() as f32;
                            let sparsity = zeros as f32 / sample.len() as f32;

                            self.byoa_test_status = format!(
                                "Tested '{}': MSE={:.6}, Sparsity={:.1}%, OrigNorm={:.3}",
                                alg.name, mse, sparsity * 100.0, orig_norm
                            );
                        } else {
                            self.byoa_test_status = "No algorithm selected.".to_string();
                        }
                    }
                    if !self.byoa_test_status.is_empty() {
                        ui.label(egui::RichText::new(&self.byoa_test_status).strong().color(p.success).size(10.0));
                    }
                });
            });
    }
}
