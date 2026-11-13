//! Studio tab: TAB 1: Models Manager (Loaded models, GGUF/SafeTensors importer, tensor inventory)

use super::*;

impl THMEApp {
    /// TAB 1: Models Manager (Loaded models, GGUF/SafeTensors importer, tensor inventory)
    pub(super) fn render_tab_models(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();
        let mut switch_idx = None;

        ui.columns(2, |cols| {
            cols[0].vertical(|ui| {
                ui.heading(egui::RichText::new("LOADED MODELS REPOSITORY").size(13.0).color(p.text_main));
                ui.label(egui::RichText::new("Manage active model slots, import local files or directories, and switch architectures.").size(10.0).color(p.text_muted));
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    if ui.button("Generate Synthetic Model").clicked() {
                        let m = UniversalModel::create_synthetic(12, 2048);
                        let name = m.name.clone();
                        self.models.push(m);
                        self.active_model_idx = self.models.len() - 1;
                        self.auto_scan_and_benchmark();
                        self.run_auto_decider_internal();
                        self.log_audit("INFO", "ModelCreated", "Generated 12-layer synthetic model", &name);
                    }

                    if ui.button("Rebuild Architecture From Scratch").clicked() {
                        self.rebuilder_open = !self.rebuilder_open;
                    }

                    if ui.button("Universal Paradigm Converter (8 AI Models)").clicked() {
                        self.paradigm_converter_open = !self.paradigm_converter_open;
                    }
                });

                if self.rebuilder_open {
                    ui.add_space(6.0);
                    manga::glass_frame(&p, p.primary)
                        .inner_margin(egui::Margin::same(10))
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("REBUILD ARCHITECTURE FROM SCRATCH").strong().color(p.primary));
                            ui.label(egui::RichText::new("Define topological layer sequences, assign early-exit depth, and optionally remap open weights.").size(9.0).color(p.text_muted));
                            ui.add_space(4.0);

                            ui.horizontal(|ui| {
                                ui.label("Model Name:");
                                ui.text_edit_singleline(&mut self.rebuild_model_name);
                            });

                            ui.horizontal(|ui| {
                                ui.label("Layers:");
                                ui.add(egui::Slider::new(&mut self.rebuild_num_layers, 4..=48).text("L"));
                                ui.label("Dim:");
                                egui::ComboBox::from_id_salt("rebuild_dim_select")
                                    .selected_text(format!("{}", self.rebuild_hidden_dim))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut self.rebuild_hidden_dim, 1024, "1024");
                                        ui.selectable_value(&mut self.rebuild_hidden_dim, 2048, "2048");
                                        ui.selectable_value(&mut self.rebuild_hidden_dim, 4096, "4096");
                                    });
                            });

                            ui.horizontal(|ui| {
                                ui.label("Exit Checkpoint:");
                                let max_cp = self.rebuild_num_layers.saturating_sub(1);
                                ui.add(egui::Slider::new(&mut self.rebuild_checkpoint_layer, 1..=max_cp).text("Idx"));
                            });

                            ui.checkbox(&mut self.rebuild_remap_open_weights, "Transfer / Remap Open Weights From Active Model");

                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                if ui.button(egui::RichText::new("Assemble & Activate New Architecture").strong().color(p.text_main)).clicked() {
                                    let bp = crate::ir::ArchitectureBlueprint::new_hybrid_llama(
                                        &self.rebuild_model_name,
                                        self.rebuild_num_layers,
                                        self.rebuild_hidden_dim,
                                        self.rebuild_checkpoint_layer,
                                    );

                                    let new_ir = if self.rebuild_remap_open_weights && !self.model().tensors.is_empty() {
                                        let source_ir = self.model().to_model_ir();
                                        crate::ir::ModelRebuilder::rebuild_from_open_weights(&bp, &source_ir, crate::ir::RemapStrategy::LowRankProject)
                                    } else {
                                        crate::ir::ModelRebuilder::build_scaffold(&bp)
                                    };

                                    let mut new_m = UniversalModel::from_model_ir(new_ir, "CustomScaffold");
                                    new_m.update_exit_routing(self.exit_ratio_pct, self.rebuild_checkpoint_layer);
                                    let name = new_m.name.clone();
                                    self.models.push(new_m);
                                    self.active_model_idx = self.models.len() - 1;
                                    self.rebuilder_open = false;
                                    self.auto_scan_and_benchmark();
                                    self.run_auto_decider_internal();
                                    self.log_audit("INFO", "ArchitectureRebuilt", "Assembled and activated custom architecture", &name);
                                    self.status_msg = format!("Assembled custom model '{}' ({} layers) successfully!", name, self.rebuild_num_layers);
                                }

                                if ui.button("Cancel").clicked() {
                                    self.rebuilder_open = false;
                                }
                            });
                        });
                }

                if self.paradigm_converter_open {
                    ui.add_space(6.0);
                    manga::glass_frame(&p, p.secondary)
                        .inner_margin(egui::Margin::same(10))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("UNIVERSAL AI MODEL PARADIGM CONVERTER (8 SPECIALIZED MODELS)")
                                    .strong()
                                    .color(p.secondary),
                            );
                            ui.label(
                                egui::RichText::new(
                                    "Cross-convert any loaded model topology into LLM, LCM, LAM, MoE, VLM, SLM, MLM, or SAM with mathematical manifold restructuring."
                                )
                                .size(9.0)
                                .color(p.text_muted),
                            );
                            ui.add_space(6.0);

                            let source_ir = self.model().to_model_ir();
                            let detected_source = crate::ir::ParadigmConverter::detect_paradigm(&source_ir);

                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("Source Paradigm (Detected):").strong().color(p.text_main));
                                ui.label(
                                    egui::RichText::new(format!("{} ({})", detected_source.display_name(), detected_source.code()))
                                        .strong()
                                        .color(p.primary),
                                );
                            });

                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("Select Target Paradigm:").strong().color(p.text_main));

                            let paradigms = crate::ir::SpecializedModelParadigm::all();
                            ui.horizontal_wrapped(|ui| {
                                for (idx, paradigm) in paradigms.iter().enumerate() {
                                    let is_selected = self.selected_target_paradigm_idx == idx;
                                    let btn_text = format!("{} ({})", paradigm.display_name(), paradigm.code());
                                    if ui.selectable_label(is_selected, btn_text).clicked() {
                                        self.selected_target_paradigm_idx = idx;
                                    }
                                }
                            });

                            let target_paradigm = paradigms[self.selected_target_paradigm_idx.min(paradigms.len() - 1)];

                            ui.add_space(6.0);
                            egui::Frame::NONE.fill(manga::glass_fill(p.bg_input, &p)).stroke(Stroke::new(1.0_f32, p.border)).corner_radius(8)
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(format!("Target Specs: {}", target_paradigm.description()))
                                            .size(10.0)
                                            .color(p.text_main),
                                    );
                                    let comps = target_paradigm.primary_components().join(" -> ");
                                    ui.label(egui::RichText::new(format!("Pipeline: {}", comps)).size(9.0).color(p.text_muted));
                                });

                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("Paradigm Configuration Parameters:").strong().color(p.text_main));

                            match target_paradigm {
                                crate::ir::SpecializedModelParadigm::MoE => {
                                    ui.horizontal(|ui| {
                                        ui.label("Number of Experts:");
                                        ui.selectable_value(&mut self.paradigm_num_experts, 2, "2 Experts");
                                        ui.selectable_value(&mut self.paradigm_num_experts, 4, "4 Experts");
                                        ui.selectable_value(&mut self.paradigm_num_experts, 8, "8 Experts");
                                        ui.selectable_value(&mut self.paradigm_num_experts, 16, "16 Experts");
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Top-K Routing:");
                                        ui.selectable_value(&mut self.paradigm_top_k, 1, "Top-1");
                                        ui.selectable_value(&mut self.paradigm_top_k, 2, "Top-2");
                                        ui.selectable_value(&mut self.paradigm_top_k, 4, "Top-4");
                                    });
                                }
                                crate::ir::SpecializedModelParadigm::SLM => {
                                    ui.horizontal(|ui| {
                                        ui.label("Target Hidden Dimension:");
                                        ui.selectable_value(&mut self.paradigm_target_hidden, 512, "512");
                                        ui.selectable_value(&mut self.paradigm_target_hidden, 1024, "1024");
                                        ui.selectable_value(&mut self.paradigm_target_hidden, 2048, "2048");
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Target Pruned Layers:");
                                        ui.selectable_value(&mut self.paradigm_target_layers, 4, "4 Layers");
                                        ui.selectable_value(&mut self.paradigm_target_layers, 8, "8 Layers");
                                        ui.selectable_value(&mut self.paradigm_target_layers, 12, "12 Layers");
                                    });
                                    ui.checkbox(&mut self.paradigm_apply_bitnet, "Apply BitNet b1.58 Ternary Discretization ({-1, 0, +1})");
                                }
                                _ => {
                                    ui.label(
                                        egui::RichText::new("Standard auto-scaling manifold adapters will be synthesized.")
                                            .size(9.5)
                                            .color(p.text_muted),
                                    );
                                }
                            }

                            ui.add_space(8.0);
                            ui.horizontal(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new(format!("Execute Conversion to {}", target_paradigm.code()))
                                            .strong()
                                            .color(p.text_main),
                                    )
                                    .clicked()
                                {
                                    let mut config = crate::ir::ParadigmConversionConfig::new(target_paradigm);
                                    config.num_experts = self.paradigm_num_experts;
                                    config.top_k = self.paradigm_top_k;
                                    config.target_hidden_size = Some(self.paradigm_target_hidden);
                                    config.target_layers = Some(self.paradigm_target_layers);
                                    config.apply_bitnet = self.paradigm_apply_bitnet;

                                    match crate::ir::ParadigmConverter::convert(&source_ir, &config) {
                                        Ok((converted_ir, report)) => {
                                            let new_m = UniversalModel::from_model_ir(converted_ir, &format!("{}-Converted", target_paradigm.code()));
                                            let name = new_m.name.clone();
                                            self.models.push(new_m);
                                            self.active_model_idx = self.models.len() - 1;
                                            self.paradigm_last_report = Some(report.format_summary());
                                            self.auto_scan_and_benchmark();
                                            self.run_auto_decider_internal();
                                            self.log_audit("INFO", "ParadigmConverted", &format!("Converted model topology to {}", target_paradigm.code()), &name);
                                            self.status_msg = format!("Successfully converted model to {} paradigm!", target_paradigm.code());
                                        }
                                        Err(e) => {
                                            self.status_msg = format!("Paradigm conversion failed: {}", e);
                                        }
                                    }
                                }

                                if ui.button("Close").clicked() {
                                    self.paradigm_converter_open = false;
                                }
                            });

                            if let Some(ref report) = self.paradigm_last_report {
                                ui.add_space(8.0);
                                manga::glass_deep(&p, p.success)
                                    .inner_margin(egui::Margin::same(8))
                                    .show(ui, |ui| {
                                        ui.label(egui::RichText::new("CONVERSION MANIFOLD REPORT").strong().color(p.success));
                                        for line in report.lines() {
                                            ui.label(egui::RichText::new(line).size(9.0).monospace().color(p.text_main));
                                        }
                                    });
                            }
                        });
                }

                ui.add_space(8.0);
                ui.label(egui::RichText::new("Active Models List:").strong().color(p.text_main));

                for i in 0..self.models.len() {
                    let is_active = i == self.active_model_idx;
                    let (name, format, arch, num_layers) = {
                        let m = &self.models[i];
                        (m.name.clone(), m.format.clone(), m.architecture.clone(), m.num_layers)
                    };

                    egui::Frame::NONE
                        .fill(if is_active { manga::glass_fill(p.bg_input, &p) } else { manga::glass_fill(p.bg_card, &p) })
                        .stroke(Stroke::new(1.0_f32, if is_active { p.primary } else { p.border }))
                        .inner_margin(egui::Margin::same(10))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.label(egui::RichText::new(&name).strong().color(if is_active { p.primary } else { p.text_main }));
                                    ui.label(egui::RichText::new(f!("Format: {} | Arch: {} | Layers: {}", format, arch, num_layers)).size(9.0).color(p.text_muted));
                                });
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if !is_active {
                                        if ui.button("Select").clicked() {
                                            switch_idx = Some(i);
                                        }
                                    } else {
                                        ui.label(egui::RichText::new("ACTIVE").color(p.success).strong().size(9.0));
                                    }
                                });
                            });
                        });
                    ui.add_space(4.0);
                }
            });

            cols[1].vertical(|ui| {
                ui.heading(egui::RichText::new("TENSOR REGISTRY & INVENTORY").size(13.0).color(p.text_main));
                ui.label(egui::RichText::new("Detailed registry of all weights, shapes, and layer affiliations.").size(10.0).color(p.text_muted));
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    ui.label("Search:");
                    ui.text_edit_singleline(&mut self.weight_search_query);
                    if ui.button("Clear").clicked() {
                        self.weight_search_query.clear();
                    }
                });

                ui.add_space(4.0);
                egui::ScrollArea::vertical().max_height(450.0).show(ui, |ui| {
                    egui::Grid::new("tensors_table")
                        .striped(true)
                        .spacing([12.0, 4.0])
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Tensor Name").strong().size(9.0).color(p.text_main));
                            ui.label(egui::RichText::new("Shape").strong().size(9.0).color(p.text_main));
                            ui.label(egui::RichText::new("DType").strong().size(9.0).color(p.text_main));
                            ui.label(egui::RichText::new("Elements").strong().size(9.0).color(p.text_main));
                            ui.label(egui::RichText::new("Role").strong().size(9.0).color(p.text_main));
                            ui.end_row();

                            let q = self.weight_search_query.to_lowercase();
                            for t in self.model().tensors.iter().filter(|t| q.is_empty() || t.name.to_lowercase().contains(&q)) {
                                ui.label(egui::RichText::new(&t.name).size(8.0).monospace());
                                ui.label(egui::RichText::new(f!("{:?}", t.shape)).size(8.0).monospace());
                                ui.label(egui::RichText::new(&t.dtype).size(8.0).color(p.primary));
                                ui.label(egui::RichText::new(f!("{}", t.num_elements)).size(8.0));
                                ui.label(egui::RichText::new(&t.role).size(8.0).color(p.warning));
                                ui.end_row();
                            }
                        });
                });
            });
        });

        if let Some(idx) = switch_idx {
            self.active_model_idx = idx;
            self.auto_scan_and_benchmark();
            self.run_auto_decider_internal();
        }
    }
}
