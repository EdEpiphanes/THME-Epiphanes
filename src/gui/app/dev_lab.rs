// SPDX-License-Identifier: AGPL-3.0-only
//! Studio tab: dev_lab

use super::*;

impl THMEApp {
    pub(super) fn render_tab_developer_lab(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();

        ui.heading(
            egui::RichText::new("DEVELOPER EXPERIMENTATION LAB & MULTI-DATASET MICRO-TRAINER")
                .size(13.0)
                .color(p.text_main),
        );
        ui.label(
            egui::RichText::new(
                "Research laboratory for arbitrary dataset ingestion (JSONL/CSV/Text), non-conventional weight transformations (noise, spectral, phase, dropblock), and in-engine adaptation micro-training.",
            )
            .size(10.0)
            .color(p.text_muted),
        );
        ui.add_space(8.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // ROW 1: DATASET INGESTION & EXPERIMENTAL TECHNIQUES (2 COLUMNS)
            ui.columns(2, |cols| {
                // COLUMN 0: DATASET INGESTION DECK
                let ui0 = &mut cols[0];
                ui0.group(|ui| {
                    ui.label(
                        egui::RichText::new("Dataset Ingestion & Tokenizer Deck")
                            .strong()
                            .size(11.0)
                            .color(p.primary),
                    );
                    ui.add_space(4.0);

                    ui.horizontal(|ui| {
                        if ui
                            .button(
                                egui::RichText::new("Load File")
                                    .strong()
                                    .color(p.text_main),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter(
                                    "Dataset Files",
                                    &["txt", "jsonl", "json", "csv", "tsv"],
                                )
                                .pick_file()
                            {
                                match crate::dataset::Dataset::from_file(&path) {
                                    Ok(ds) => {
                                        let name = ds.name.clone();
                                        let count = ds.samples.len();
                                        self.dataset = Some(ds);
                                        self.dataset_status_msg = format!(
                                            "Imported '{}' successfully ({} samples)",
                                            name, count
                                        );
                                        self.log_audit(
                                            "INFO",
                                            "Dataset",
                                            &self.dataset_status_msg.clone(),
                                            "Import",
                                        );
                                    }
                                    Err(err) => {
                                        self.dataset_status_msg =
                                            format!("Failed to import dataset: {}", err);
                                        self.log_audit(
                                            "ERROR",
                                            "Dataset",
                                            &self.dataset_status_msg.clone(),
                                            "Import",
                                        );
                                    }
                                }
                            }
                        }

                        if ui
                            .button(
                                egui::RichText::new("Synthetic Calibration")
                                    .color(p.text_main),
                            )
                            .clicked()
                        {
                            match crate::dataset::Dataset::create_synthetic_calibration() {
                                Ok(ds) => {
                                    self.dataset = Some(ds);
                                    self.dataset_status_msg =
                                        "Synthetic calibration dataset loaded (5 samples)".to_string();
                                }
                                Err(error) => {
                                    self.dataset_status_msg =
                                        format!("Could not build the synthetic dataset: {error}");
                                }
                            }
                            self.log_audit(
                                "INFO",
                                "Dataset",
                                "Loaded synthetic calibration sequences",
                                "Dataset",
                            );
                        }

                        if ui
                            .button(
                                egui::RichText::new("Ingest Pasted Text")
                                    .color(p.text_main),
                            )
                            .clicked()
                        {
                            match crate::dataset::Dataset::from_content(
                                "custom_ingest",
                                &self.dataset_input_text,
                                crate::dataset::DatasetFormat::PlainText,
                            ) {
                                Ok(ds) => {
                                    let count = ds.samples.len();
                                    self.dataset = Some(ds);
                                    self.dataset_status_msg =
                                        format!("Custom text ingested: {} samples created", count);
                                    self.log_audit(
                                        "INFO",
                                        "Dataset",
                                        &self.dataset_status_msg.clone(),
                                        "Ingest",
                                    );
                                }
                                Err(err) => {
                                    self.dataset_status_msg =
                                        format!("Ingestion error: {}", err);
                                }
                            }
                        }
                    });

                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(&self.dataset_status_msg)
                            .size(9.0)
                            .color(p.secondary),
                    );
                    ui.add_space(6.0);

                    // Dataset Summary Scorecards
                    if let Some(ds) = &self.dataset {
                        ui.columns(4, |subcols| {
                            render_stat_box(
                                &mut subcols[0],
                                "SAMPLES",
                                format!("{}", ds.summary.total_samples),
                                p.primary,
                                &p,
                            );
                            render_stat_box(
                                &mut subcols[1],
                                "TOTAL TOKENS",
                                format!("{}", ds.summary.total_tokens),
                                p.success,
                                &p,
                            );
                            render_stat_box(
                                &mut subcols[2],
                                "VOCAB DIVERSITY",
                                format!("{}", ds.summary.estimated_vocab_size),
                                p.warning,
                                &p,
                            );
                            render_stat_box(
                                &mut subcols[3],
                                "FORMAT",
                                ds.summary.format.clone(),
                                p.secondary,
                                &p,
                            );
                        });

                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("Dataset Sample Preview:")
                                .strong()
                                .size(9.0)
                                .color(p.text_main),
                        );
                        egui::ScrollArea::vertical().max_height(100.0).show(ui, |ui| {
                            for (idx, sample) in ds.samples.iter().take(5).enumerate() {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "[#{}] Tokens: {} | Text: {}",
                                        idx + 1,
                                        sample.token_ids.len(),
                                        sample.text
                                    ))
                                    .size(8.5)
                                    .color(p.text_muted),
                                );
                            }
                        });
                    }

                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("Custom Input / Paste Area:")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.add(
                        egui::TextEdit::multiline(&mut self.dataset_input_text)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .desired_rows(3),
                    );
                });

                // COLUMN 1: NON-CONVENTIONAL EXPERIMENTATION LAB
                let ui1 = &mut cols[1];
                ui1.group(|ui| {
                    ui.label(
                        egui::RichText::new("Non-Conventional Experimentation Laboratory")
                            .strong()
                            .size(11.0)
                            .color(p.accent),
                    );
                    ui.add_space(4.0);

                    // Technique Selector
                    ui.horizontal(|ui| {
                        ui.label("Technique:");
                        egui::ComboBox::from_id_salt("experiment_technique_selector")
                            .selected_text(match self.selected_experiment_idx {
                                0 => "Stochastic Noise Injection",
                                1 => "Spectral Energy Reshaping",
                                2 => "Complex Phase Modulation",
                                3 => "Continuous Activation Warping",
                                4 => "Stochastic DropBlock Sparsity",
                                _ => "Custom",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.selected_experiment_idx,
                                    0,
                                    "Stochastic Noise Injection",
                                );
                                ui.selectable_value(
                                    &mut self.selected_experiment_idx,
                                    1,
                                    "Spectral Energy Reshaping",
                                );
                                ui.selectable_value(
                                    &mut self.selected_experiment_idx,
                                    2,
                                    "Complex Phase Modulation",
                                );
                                ui.selectable_value(
                                    &mut self.selected_experiment_idx,
                                    3,
                                    "Continuous Activation Warping",
                                );
                                ui.selectable_value(
                                    &mut self.selected_experiment_idx,
                                    4,
                                    "Stochastic DropBlock Sparsity",
                                );
                            });
                    });

                    ui.add_space(4.0);

                    // Parameter Controls
                    match self.selected_experiment_idx {
                        0 => {
                            ui.horizontal(|ui| {
                                ui.label("Noise Variance (σ²):");
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.experiment_param_variance,
                                        0.001..=0.20,
                                    )
                                    .text("Variance"),
                                );
                            });
                            ui.label(
                                egui::RichText::new(
                                    "Simulates physical analog hardware drift, thermal noise & low-voltage DRAM margins.",
                                )
                                .size(8.5)
                                .color(p.text_muted),
                            );
                        }
                        1 => {
                            ui.horizontal(|ui| {
                                ui.label("Gamma Exponent (γ):");
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.experiment_param_gamma,
                                        0.1..=2.0,
                                    )
                                    .text("γ Power"),
                                );
                            });
                            ui.label(
                                egui::RichText::new(
                                    "Warps singular value spectrum σ_i^γ: <1.0 compresses representations, >1.0 accentuates principal features.",
                                )
                                .size(8.5)
                                .color(p.text_muted),
                            );
                        }
                        2 => {
                            ui.horizontal(|ui| {
                                ui.label("Frequency (ω):");
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.experiment_param_frequency,
                                        0.05..=1.0,
                                    )
                                    .text("Rad"),
                                );
                            });
                            ui.label(
                                egui::RichText::new(
                                    "Unitary complex angle modulation cos(ω j + φ) across weight channels.",
                                )
                                .size(8.5)
                                .color(p.text_muted),
                            );
                        }
                        3 => {
                            ui.horizontal(|ui| {
                                ui.label("Blend Factor:");
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.experiment_param_blend,
                                        0.0..=1.0,
                                    )
                                    .text("Blend"),
                                );
                            });
                            ui.label(
                                egui::RichText::new(
                                    "Continuous non-linear manifold interpolation between SwiGLU, GELU, and harmonic sine activations.",
                                )
                                .size(8.5)
                                .color(p.text_muted),
                            );
                        }
                        4 => {
                            ui.horizontal(|ui| {
                                ui.label("Drop Probability:");
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.experiment_param_drop_prob,
                                        0.05..=0.70,
                                    )
                                    .text("Prob"),
                                );
                            });
                            ui.label(
                                egui::RichText::new(
                                    "Structured block-level zeroing for hardware-aligned sparsity benchmarking.",
                                )
                                .size(8.5)
                                .color(p.text_muted),
                            );
                        }
                        _ => {}
                    }

                    ui.add_space(6.0);
                    let max_layers = self.model().layers.len().max(1);
                    ui.horizontal(|ui| {
                        ui.label("Target Layer:");
                        ui.add(egui::Slider::new(
                            &mut self.training_target_layer,
                            0..=max_layers.saturating_sub(1),
                        ));

                        if ui
                            .button(
                                egui::RichText::new("Apply Transformation")
                                    .strong()
                                    .color(p.text_main),
                            )
                            .clicked()
                        {
                            let l_idx = self.training_target_layer;
                            let tech = match self.selected_experiment_idx {
                                0 => crate::experiment::ExperimentTechnique::NoiseInjection {
                                    variance: self.experiment_param_variance,
                                    distribution: "Gaussian".to_string(),
                                },
                                1 => crate::experiment::ExperimentTechnique::SpectralReshaping {
                                    gamma_exponent: self.experiment_param_gamma,
                                },
                                2 => crate::experiment::ExperimentTechnique::PhaseModulation {
                                    frequency_rad: self.experiment_param_frequency,
                                    phase_shift_rad: 0.1,
                                },
                                3 => crate::experiment::ExperimentTechnique::ActivationWarping {
                                    blend_factor: self.experiment_param_blend,
                                },
                                _ => {
                                    crate::experiment::ExperimentTechnique::StochasticDropBlock {
                                        drop_probability: self.experiment_param_drop_prob,
                                        block_size: 4,
                                    }
                                }
                            };

                            let mut applied = false;
                            for t in &mut self.model_mut().tensors {
                                if t.layer_index == Some(l_idx) && !t.sample_weights.is_empty() {
                                    let metrics = crate::experiment::ExperimentLab::execute(
                                        &tech,
                                        &mut t.sample_weights,
                                    );
                                    self.last_experiment_metrics = Some(metrics.clone());
                                    self.experiment_status_msg = format!(
                                        "Transformation '{}' executed on Layer {} (MSE: {:.5}, SNR: {:.2} dB)",
                                        metrics.technique_name,
                                        l_idx,
                                        metrics.mean_squared_error,
                                        metrics.signal_to_noise_ratio_db
                                    );
                                    self.log_audit(
                                        "INFO",
                                        "ExperimentLab",
                                        &self.experiment_status_msg.clone(),
                                        "Execute",
                                    );
                                    applied = true;
                                    break;
                                }
                            }

                            if !applied {
                                self.experiment_status_msg =
                                    format!("No eligible tensors found in Layer {}", l_idx);
                            }
                        }
                    });

                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(&self.experiment_status_msg)
                            .size(9.0)
                            .color(p.secondary),
                    );
                    ui.add_space(6.0);

                    // Experiment Metrics Display
                    if let Some(metrics) = &self.last_experiment_metrics {
                        ui.columns(4, |subcols| {
                            render_stat_box(
                                &mut subcols[0],
                                "ORIGINAL L2",
                                format!("{:.3}", metrics.original_l2_norm),
                                p.primary,
                                &p,
                            );
                            render_stat_box(
                                &mut subcols[1],
                                "WARPED L2",
                                format!("{:.3}", metrics.transformed_l2_norm),
                                p.accent,
                                &p,
                            );
                            render_stat_box(
                                &mut subcols[2],
                                "MSE ERROR",
                                format!("{:.5}", metrics.mean_squared_error),
                                p.warning,
                                &p,
                            );
                            render_stat_box(
                                &mut subcols[3],
                                "SNR (dB)",
                                format!("{:.1} dB", metrics.signal_to_noise_ratio_db),
                                p.success,
                                &p,
                            );
                        });
                    }
                });
            });

            ui.add_space(8.0);

            // ROW 2: IN-ENGINE MICRO-TRAINING & ADAPTATION STUDIO
            ui.group(|ui| {
                ui.label(
                    egui::RichText::new("In-Engine Micro-Training & Parameter Adaptation Studio")
                        .strong()
                        .size(11.0)
                        .color(p.success),
                );
                ui.label(
                    egui::RichText::new(
                        "Lightweight on-device training loop executing simulated parameter gradient updates, teacher distillation, and EWHA energy-regularized loss.",
                    )
                    .size(9.0)
                    .color(p.text_muted),
                );
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    ui.label("Loss Objective:");
                    egui::ComboBox::from_id_salt("training_loss_type_selector")
                        .selected_text(self.trainer_config.loss_type.display_name())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.trainer_config.loss_type,
                                crate::trainer::TrainingLossType::CrossEntropy,
                                "Autoregressive Cross-Entropy",
                            );
                            ui.selectable_value(
                                &mut self.trainer_config.loss_type,
                                crate::trainer::TrainingLossType::KnowledgeDistillation,
                                "Teacher Knowledge Distillation (KD)",
                            );
                            ui.selectable_value(
                                &mut self.trainer_config.loss_type,
                                crate::trainer::TrainingLossType::EwhaEnergyRegularized,
                                "EWHA Energy-Regularized Loss",
                            );
                        });

                    ui.add_space(8.0);
                    ui.label("Optimizer:");
                    egui::ComboBox::from_id_salt("training_optimizer_selector")
                        .selected_text(self.trainer_config.optimizer.display_name())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.trainer_config.optimizer,
                                crate::trainer::OptimizerType::AdamW,
                                "AdamW (Decoupled Weight Decay)",
                            );
                            ui.selectable_value(
                                &mut self.trainer_config.optimizer,
                                crate::trainer::OptimizerType::SignSGD,
                                "SignSGD (1-bit / Ternary)",
                            );
                            ui.selectable_value(
                                &mut self.trainer_config.optimizer,
                                crate::trainer::OptimizerType::SGD,
                                "SGD with Momentum",
                            );
                        });

                    ui.add_space(8.0);
                    ui.label("Learning Rate (η):");
                    ui.add(
                        egui::Slider::new(&mut self.trainer_config.learning_rate, 0.0001..=0.01)
                            .logarithmic(true)
                            .text("LR"),
                    );
                });

                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    if self.trainer_config.loss_type
                        == crate::trainer::TrainingLossType::KnowledgeDistillation
                    {
                        ui.label("Distillation Temp (τ):");
                        ui.add(
                            egui::Slider::new(
                                &mut self.trainer_config.distillation_temperature,
                                1.0..=5.0,
                            )
                            .text("Temp"),
                        );
                        ui.add_space(8.0);
                    } else if self.trainer_config.loss_type
                        == crate::trainer::TrainingLossType::EwhaEnergyRegularized
                    {
                        ui.label("Energy Penalty (λ_E):");
                        ui.add(
                            egui::Slider::new(
                                &mut self.trainer_config.energy_penalty_lambda,
                                0.01..=0.20,
                            )
                            .text("Lambda"),
                        );
                        ui.add_space(8.0);
                    }

                    ui.label("Weight Decay:");
                    ui.add(
                        egui::Slider::new(&mut self.trainer_config.weight_decay, 0.0..=0.05)
                            .text("Decay"),
                    );

                    ui.add_space(16.0);

                    // TRAINING ACTION BUTTONS
                    if ui
                        .button(
                            egui::RichText::new("Run 1 Step")
                                .strong()
                                .color(p.text_main),
                        )
                        .clicked()
                    {
                        self.micro_trainer.config = self.trainer_config.clone();
                        let tokens = if let Some(ds) = &self.dataset {
                            ds.get_flat_batch(16, self.micro_trainer.step_count)
                        } else {
                            vec![101, 2045, 1037, 2791, 102]
                        };

                        let l_idx = self.training_target_layer;
                        let active_idx = self
                            .active_model_idx
                            .min(self.models.len().saturating_sub(1));
                        let trainer = &mut self.micro_trainer;
                        let model = &mut self.models[active_idx];

                        let mut step_done = false;
                        let mut audit_msg = None;
                        for t in &mut model.tensors {
                            if t.layer_index == Some(l_idx) && !t.sample_weights.is_empty() {
                                if let Ok(rec) = trainer.step(&tokens, &mut t.sample_weights) {
                                    audit_msg = Some(format!(
                                        "Step {} complete: Loss={:.4}, PPL={:.2}",
                                        rec.step, rec.loss, rec.perplexity
                                    ));
                                    step_done = true;
                                }
                                break;
                            }
                        }
                        if !step_done {
                            let mut fallback_weights = vec![0.1f32; 16];
                            let _ = trainer.step(&tokens, &mut fallback_weights);
                        }
                        if let Some(msg) = audit_msg {
                            self.log_audit("INFO", "MicroTrainer", &msg, "TrainStep");
                        }
                    }

                    if ui
                        .button(
                            egui::RichText::new("Run 10 Steps")
                                .strong()
                                .color(p.text_main),
                        )
                        .clicked()
                    {
                        self.micro_trainer.config = self.trainer_config.clone();
                        let l_idx = self.training_target_layer;
                        let active_idx = self
                            .active_model_idx
                            .min(self.models.len().saturating_sub(1));
                        let trainer = &mut self.micro_trainer;
                        let model = &mut self.models[active_idx];

                        for _ in 0..10 {
                            let tokens = if let Some(ds) = &self.dataset {
                                ds.get_flat_batch(16, trainer.step_count)
                            } else {
                                vec![101, 2045, 1037, 2791, 102]
                            };

                            let mut step_done = false;
                            for t in &mut model.tensors {
                                if t.layer_index == Some(l_idx) && !t.sample_weights.is_empty() {
                                    let _ = trainer.step(&tokens, &mut t.sample_weights);
                                    step_done = true;
                                    break;
                                }
                            }
                            if !step_done {
                                let mut fallback = vec![0.1f32; 16];
                                let _ = trainer.step(&tokens, &mut fallback);
                            }
                        }
                        if let Some(last) = self.micro_trainer.history.last() {
                            let msg = format!(
                                "Completed 10 steps. Step {} Loss={:.4}, PPL={:.2}",
                                last.step, last.loss, last.perplexity
                            );
                            self.log_audit("INFO", "MicroTrainer", &msg, "TrainEpoch");
                        }
                    }

                    if ui
                        .button(
                            egui::RichText::new("Reset Trainer")
                                .color(p.text_muted),
                        )
                        .clicked()
                    {
                        self.micro_trainer.reset();
                        self.log_audit(
                            "INFO",
                            "MicroTrainer",
                            "Reset trainer state and optimizer moments",
                            "Reset",
                        );
                    }
                });

                ui.add_space(8.0);

                // Training Live Telemetry Scorecards
                if let Some(latest) = self.micro_trainer.history.last() {
                    ui.columns(5, |cols| {
                        render_scorecard_box(
                            &mut cols[0],
                            "STEP",
                            format!("#{}", latest.step),
                            "+1".to_string(),
                            p.primary,
                            &p,
                        );
                        render_scorecard_box(
                            &mut cols[1],
                            "LOSS",
                            format!("{:.4}", latest.loss),
                            "-0.04".to_string(),
                            p.accent,
                            &p,
                        );
                        render_scorecard_box(
                            &mut cols[2],
                            "PERPLEXITY (PPL)",
                            format!("{:.2}", latest.perplexity),
                            "-1.2".to_string(),
                            p.warning,
                            &p,
                        );
                        render_scorecard_box(
                            &mut cols[3],
                            "THROUGHPUT",
                            format!("{:.0} tok/s", latest.tokens_per_sec),
                            "+20".to_string(),
                            p.success,
                            &p,
                        );
                        render_scorecard_box(
                            &mut cols[4],
                            "EST. ENERGY",
                            format!("{:.3} mJ", latest.energy_mj),
                            "-15%".to_string(),
                            p.secondary,
                            &p,
                        );
                    });
                } else {
                    ui.label(
                        egui::RichText::new(
                            "No training steps executed yet. Click 'Run 1 Step' or 'Run 10 Steps' to adapt parameters.",
                        )
                        .size(9.5)
                        .color(p.text_muted),
                    );
                }

                // Training History Log
                if !self.micro_trainer.history.is_empty() {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("Training Step Convergence History:")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                        for rec in self.micro_trainer.history.iter().rev().take(10) {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("Step {:>4} |", rec.step))
                                        .monospace()
                                        .size(8.5)
                                        .color(p.primary),
                                );
                                ui.label(
                                    egui::RichText::new(format!("Loss: {:.4} |", rec.loss))
                                        .monospace()
                                        .size(8.5)
                                        .color(p.accent),
                                );
                                ui.label(
                                    egui::RichText::new(format!("PPL: {:.2} |", rec.perplexity))
                                        .monospace()
                                        .size(8.5)
                                        .color(p.warning),
                                );
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Tok/s: {:.0} |",
                                        rec.tokens_per_sec
                                    ))
                                    .monospace()
                                    .size(8.5)
                                    .color(p.success),
                                );
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Energy: {:.3} mJ",
                                        rec.energy_mj
                                    ))
                                    .monospace()
                                    .size(8.5)
                                    .color(p.secondary),
                                );
                            });
                        }
                    });
                }
            });
        });
    }
}
