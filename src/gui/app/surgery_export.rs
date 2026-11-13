// SPDX-License-Identifier: AGPL-3.0-only
//! Studio tab: TAB 7: Export & Surgery (Teacher LLM JSON recipes and model exporter)

use super::*;

impl THMEApp {
    /// TAB 7: Export & Surgery (Teacher LLM JSON recipes and model exporter)
    pub(super) fn render_tab_export(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();

        ui.columns(2, |cols| {
            cols[0].vertical(|ui| {
                ui.heading(egui::RichText::new("EXPORT RE-STRUCTURED & TUNED MODEL").size(13.0).color(p.text_main));
                ui.label(egui::RichText::new("Ship the optimized, ultra-tuned and restructured LLM artifact matching research paper expectations.").size(10.0).color(p.text_muted));
                ui.add_space(12.0);

                ui.horizontal(|ui| {
                    ui.label("Target Format:");
                    egui::ComboBox::from_id_salt("export_fmt_cb")
                        .selected_text(&self.export_format)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.export_format, "SafeTensors".to_string(), "SafeTensors (.safetensors)");
                            ui.selectable_value(&mut self.export_format, "GGUF".to_string(), "GGUF (.gguf)");
                        });
                });

                if self.export_format == "GGUF" {
                    ui.horizontal(|ui| {
                        ui.label("Weight Format:");
                        egui::ComboBox::from_id_salt("export_quant_cb")
                            .selected_text(match self.export_quant {
                                crate::exporter::ExportQuantType::FP16 => "FP16",
                                crate::exporter::ExportQuantType::Q8_0 => "Q8_0",
                                crate::exporter::ExportQuantType::TQ1_0 => "TQ1_0 (ternary)",
                                crate::exporter::ExportQuantType::BitNetB158 => "Unsupported BitNet legacy",
                                crate::exporter::ExportQuantType::PerTensor => "Per tensor (selected formats)",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.export_quant, crate::exporter::ExportQuantType::FP16, "FP16 (same-dtype)");
                                ui.selectable_value(&mut self.export_quant, crate::exporter::ExportQuantType::Q8_0, "Q8_0");
                                ui.selectable_value(&mut self.export_quant, crate::exporter::ExportQuantType::TQ1_0, "TQ1_0 (ternary)");
                            });
                    });
                } else {
                    self.export_quant = crate::exporter::ExportQuantType::FP16;
                    ui.label(egui::RichText::new("SafeTensors export preserves same-dtype F16 weights.").size(9.0).color(p.text_muted));
                }

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Export Destination Path:");
                    ui.text_edit_singleline(&mut self.export_path);
                    if ui.button("Browse...").clicked() {
                        if let Some(dest) = rfd::FileDialog::new()
                            .add_filter("SafeTensors", &["safetensors"])
                            .add_filter("GGUF", &["gguf"])
                            .save_file()
                        {
                            self.export_path = dest.to_string_lossy().to_string();
                        }
                    }
                });

                if let Some(source) = &self.model().source_model {
                    ui.label(egui::RichText::new(format!(
                        "Source weights loaded: {}. Export writes source weights; GUI-only surgery settings are not applied to tensor data.",
                        source.name
                    )).size(9.0).color(p.text_muted));
                } else {
                    ui.label(egui::RichText::new("Load a file-backed GGUF or SafeTensors model to export real tensor weights.").size(9.0).color(p.danger));
                }

                ui.checkbox(&mut self.export_checksum, "Compute SHA-256 checksum");

                ui.add_space(16.0);
                if ui.button(egui::RichText::new("Export Source Model").strong().color(p.text_main)).clicked() {
                    let source_model = self.model().source_model.clone();
                    let output_path = std::path::PathBuf::from(&self.export_path);
                    let output_name = output_path.file_name().and_then(|name| name.to_str()).unwrap_or("output").to_string();
                    let config = crate::exporter::ExportConfig {
                        format: if self.export_format == "GGUF" {
                            crate::exporter::ExportFormat::GGUFv3
                        } else {
                            crate::exporter::ExportFormat::SafeTensors
                        },
                        quant_type: self.export_quant,
                        output_path,
                        compute_checksum: self.export_checksum,
                    };

                    if let Some(source_model) = source_model {
                        match crate::exporter::ModelExporter::export(&source_model, &config) {
                            Ok(report) => {
                                let checksum = report.sha256_checksum.as_deref().map(|value| format!(" SHA-256 {value}")).unwrap_or_default();
                                let message = format!("Exported {} tensors to {output_name} ({} bytes).{checksum}", report.total_tensors, report.exported_bytes);
                                self.log_audit("INFO", "Export", &message, &source_model.name);
                                self.status_msg = message;
                            }
                            Err(err) => {
                                self.status_msg = format!("Export failed: {err:#}");
                            }
                        }
                    } else {
                        self.status_msg = "Export failed: load a file-backed model before exporting tensor weights".to_string();
                    }
                }

                ui.add_space(16.0);
                manga::glass_frame(&p, p.primary)
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("PYTORCH & CUDA ECOSYSTEM BRIDGE").strong().color(p.primary).size(11.0));
                        ui.label(egui::RichText::new("Generate standalone PyTorch/CUDA fine-tuning scripts and Hugging Face configs.").size(9.0).color(p.text_muted));
                        ui.add_space(6.0);

                        if ui.button("Generate PyTorch CUDA Trainer (train_cuda.py)").clicked() {
                            let config = crate::interop::PyTorchInteropConfig {
                                model_id_or_path: self.model().name.clone(),
                                ..Default::default()
                            };
                            let script = crate::interop::PyTorchScriptGenerator::generate_training_script(&config, None);
                            if let Some(dest) = rfd::FileDialog::new()
                                .add_filter("Python Script", &["py"])
                                .set_file_name("train_cuda.py")
                                .save_file()
                            {
                                let _ = std::fs::write(&dest, script);
                                self.status_msg = format!("Generated PyTorch CUDA script: {:?}", dest);
                                self.log_audit("INFO", "Interop", "Generated PyTorch CUDA training script", "PyTorchBridge");
                            }
                        }

                        if ui.button("Export Hugging Face config.json").clicked() {
                            let model_ir = crate::ir::ModelIR::new(
                                self.model().name.clone(),
                                crate::ir::ModelArchitecture::Llama,
                                self.model().layers.len(),
                                self.model().hidden_size,
                                32,
                                32,
                                32000,
                                4096,
                                Vec::new(),
                            );
                            let hf_cfg = crate::interop::HuggingFaceConfigGenerator::generate_config_json(&model_ir);
                            if let Some(dest) = rfd::FileDialog::new()
                                .add_filter("JSON Config", &["json"])
                                .set_file_name("config.json")
                                .save_file()
                            {
                                if let Ok(json_str) = serde_json::to_string_pretty(&hf_cfg) {
                                    let _ = std::fs::write(&dest, json_str);
                                    self.status_msg = format!("Exported Hugging Face config: {:?}", dest);
                                    self.log_audit("INFO", "Interop", "Exported Hugging Face config.json", "PyTorchBridge");
                                }
                            }
                        }
                    });
            });

            cols[1].vertical(|ui| {
                ui.heading(egui::RichText::new("TEACHER LLM SURGERY & RESTRUCTURING RECIPES").size(13.0).color(p.text_main));
                ui.label(egui::RichText::new("Execute declarative JSON mutation recipes from automated agents and Teacher LLMs.").size(10.0).color(p.text_muted));
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    if ui.button("Load Default Recipe").clicked() {
                        let rec = crate::surgery::TweakRecipe::default();
                        self.tweak_recipe_text = serde_json::to_string_pretty(&rec).unwrap_or_default();
                        self.status_msg = "Loaded default Teacher LLM restructuring recipe.".to_string();
                    }

                    if ui.button("Load Recipe File...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().add_filter("JSON Recipe", &["json"]).pick_file() {
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                self.tweak_recipe_text = content;
                                self.status_msg = f!("Loaded recipe from: {:?}", path);
                            }
                        }
                    }

                    if ui.button("Save Recipe File...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().add_filter("JSON Recipe", &["json"]).save_file() {
                            let _ = std::fs::write(&path, &self.tweak_recipe_text);
                            self.status_msg = f!("Saved recipe to: {:?}", path);
                        }
                    }
                });

                ui.add_space(6.0);
                if ui.button(egui::RichText::new("Apply Teacher LLM Surgery Recipe").strong().color(p.text_main)).clicked() {
                    match serde_json::from_str::<crate::surgery::TweakRecipe>(&self.tweak_recipe_text) {
                        Ok(recipe) => {
                            let mut model_ir = crate::ir::ModelIR::new(
                                self.model().name.clone(),
                                crate::ir::ModelArchitecture::Llama,
                                self.model().layers.len(),
                                self.model().hidden_size,
                                32,
                                32,
                                32000,
                                4096,
                                Vec::new(),
                            );
                            let diff = crate::surgery::SurgeryEngine::apply_recipe(&mut model_ir, &recipe);
                            let m_name = self.model().name.clone();
                            self.log_audit("SURGERY", "RecipeApplied", &format!("Applied recipe '{}': Saved {:.1}% VRAM", diff.recipe_name, diff.savings_percent), &m_name);
                            self.status_msg = f!("Applied recipe '{}': Saved {:.1} MB ({:.1}%)!", diff.recipe_name, diff.memory_saved_bytes as f32 / 1024.0 / 1024.0, diff.savings_percent);
                            self.surgery_diff = Some(diff);
                        }
                        Err(e) => {
                            self.status_msg = f!("Recipe JSON parsing error: {}", e);
                        }
                    }
                }

                if let Some(diff) = &self.surgery_diff {
                    ui.add_space(8.0);
                    ui.columns(3, |sub| {
                        render_stat_box(&mut sub[0], "MODIFIED TENSORS", f!("{}", diff.modified_tensors_count), p.primary, &p);
                        render_stat_box(&mut sub[1], "STORAGE SAVED", f!("{:.1}%", diff.savings_percent), p.success, &p);
                        render_stat_box(&mut sub[2], "EST. SPEEDUP", f!("{:.2}x", diff.estimated_speedup), p.warning, &p);
                    });
                }

                ui.add_space(8.0);
                ui.label(egui::RichText::new("Recipe JSON Specification:").strong().size(10.0).color(p.text_main));
                egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.tweak_recipe_text)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .desired_rows(12),
                    );
                });
            });
        });
    }
}
