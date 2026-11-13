//! Studio tab: TAB 4: Full Benchmarks (Live Flow simulator, batch scaling, context length scaling)

use super::*;

impl THMEApp {
    /// TAB 4: Full Benchmarks (Live Flow simulator, batch scaling, context length scaling)
    pub(super) fn render_tab_full_benchmarks(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();
        let snap = self.simulator.snapshot();

        ui.columns(2, |cols| {
            cols[0].vertical(|ui| {
                ui.heading(egui::RichText::new("LIVE TOKEN & PROCESS FLOW BENCHMARKER").size(13.0).color(p.text_main));
                ui.label(egui::RichText::new("Simulates real-time token generation, memory bandwidth saturation, and layer execution waterfall.").size(10.0).color(p.text_muted));
                ui.add_space(8.0);

                ui.label("Evaluation Prompt:");
                ui.text_edit_singleline(&mut self.baseline_prompt);

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("Max Tokens:");
                    if ui.add(egui::DragValue::new(&mut self.baseline_tokens).range(8..=256)).changed() {
                        self.simulator.target_tokens = self.baseline_tokens;
                    }
                    ui.add_space(8.0);
                    ui.label("Temp:");
                    ui.add(egui::DragValue::new(&mut self.baseline_temp).speed(0.05).range(0.1..=1.5));
                    ui.add_space(8.0);
                    ui.label("Speed:");
                    ui.add(egui::Slider::new(&mut self.sim_speed, 0.5..=10.0).text("x"));
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    match snap.status {
                        crate::profiling::TokenFlowStatus::Idle | crate::profiling::TokenFlowStatus::Completed => {
                            if ui.button(egui::RichText::new("Start Live Flow Benchmark").strong().color(p.text_main)).clicked() {
                                self.simulator = crate::profiling::LiveFlowSimulator::new(
                                    &self.baseline_prompt,
                                    self.baseline_tokens,
                                    self.model().layers.len(),
                                    self.model().hidden_size,
                                    self.checkpoint_layer_idx,
                                    self.exit_ratio_pct / 100.0,
                                );
                                self.simulator.start();
                                let m_name = self.model().name.clone();
                                self.log_audit("BENCHMARK", "TokenFlow", "Started live token flow benchmark", &m_name);
                                self.status_msg = "Live token flow simulation started!".to_string();
                            }
                        }
                        crate::profiling::TokenFlowStatus::Generating | crate::profiling::TokenFlowStatus::Prefill => {
                            if ui.button(egui::RichText::new("Pause").strong().color(p.text_main)).clicked() {
                                self.simulator.pause();
                            }
                        }
                        crate::profiling::TokenFlowStatus::Paused => {
                            if ui.button(egui::RichText::new("Resume").strong().color(p.text_main)).clicked() {
                                self.simulator.resume();
                            }
                        }
                    }

                    if ui.button("Step Pipeline").clicked() {
                        self.simulator.step_pipeline();
                    }

                    if ui.button("Reset").clicked() {
                        self.simulator.reset();
                        self.status_msg = "Simulator reset to idle state.".to_string();
                    }
                });

                ui.add_space(10.0);
                ui.columns(2, |sub| {
                    render_stat_box(&mut sub[0], "THROUGHPUT", f!("{:.1} tok/s", snap.current_tok_per_sec), p.primary, &p);
                    render_stat_box(&mut sub[1], "TIME/TOKEN", f!("{:.1} ms", snap.time_per_token_ms), p.text_main, &p);
                });
                ui.add_space(4.0);
                ui.columns(2, |sub| {
                    render_stat_box(&mut sub[0], "KV CACHE FOOTPRINT", f!("{:.2} MB", snap.kv_cache_mb), p.warning, &p);
                    render_stat_box(&mut sub[1], "MEMORY BANDWIDTH", f!("{:.1} GB/s", snap.memory_bandwidth_gbps), p.success, &p);
                });

                ui.add_space(10.0);
                ui.label(egui::RichText::new("Live Stream Terminal:").strong().color(p.text_main));
                manga::glass_frame(&p, p.border)
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        let stream_text = if snap.token_stream.is_empty() {
                            "[Press Start to initiate live token generation flow...]".to_string()
                        } else {
                            format!("{}: {}", self.baseline_prompt, snap.token_stream.join(" "))
                        };
                        ui.label(egui::RichText::new(stream_text).monospace().size(9.0).color(p.text_main));
                    });
            });

            cols[1].vertical(|ui| {
                ui.heading(egui::RichText::new("BATCH & CONTEXT SCALABILITY STRESS SUITE").size(13.0).color(p.text_main));
                ui.label(egui::RichText::new("Analytical scaling curves across batch dimensions and extended sequence contexts.").size(10.0).color(p.text_muted));
                ui.add_space(8.0);

                ui.label(egui::RichText::new("Batch Size Latency Scaling:").strong().color(p.text_main));
                egui::Grid::new("batch_grid").striped(true).spacing([16.0, 4.0]).show(ui, |ui| {
                    ui.label(egui::RichText::new("Batch Size").strong().size(9.0).color(p.text_main));
                    ui.label(egui::RichText::new("Total Throughput").strong().size(9.0).color(p.text_main));
                    ui.label(egui::RichText::new("Latency/Token").strong().size(9.0).color(p.text_main));
                    ui.label(egui::RichText::new("VRAM Traffic").strong().size(9.0).color(p.text_main));
                    ui.end_row();

                    for &b in &[1, 4, 8, 16, 32, 64] {
                        let tok_s = snap.current_tok_per_sec * (b as f32).powf(0.65);
                        let lat_ms = (1000.0 / tok_s) * b as f32;
                        let traffic = (snap.memory_bandwidth_gbps * (b as f32).powf(0.5)).min(900.0);
                        ui.label(f!("B = {}", b));
                        ui.label(f!("{:.1} tok/s", tok_s));
                        ui.label(f!("{:.2} ms", lat_ms));
                        ui.label(f!("{:.1} GB/s", traffic));
                        ui.end_row();
                    }
                });

                ui.add_space(12.0);
                ui.label(egui::RichText::new("Context Length KV Cache Growth:").strong().color(p.text_main));
                egui::Grid::new("context_grid").striped(true).spacing([16.0, 4.0]).show(ui, |ui| {
                    ui.label(egui::RichText::new("Sequence Length").strong().size(9.0).color(p.text_main));
                    ui.label(egui::RichText::new("KV Cache (FP16)").strong().size(9.0).color(p.text_main));
                    ui.label(egui::RichText::new("KV Cache (Tuned)").strong().size(9.0).color(p.text_main));
                    ui.label(egui::RichText::new("Savings").strong().size(9.0).color(p.text_main));
                    ui.end_row();

                    let num_layers = self.model().layers.len();
                    for &ctx_len in &[512, 1024, 2048, 4096, 8192] {
                        let fp16_mb = (2.0 * 1.0 * 8.0 * ctx_len as f32 * 128.0 * 2.0 * num_layers as f32) / (1024.0 * 1024.0);
                        let tuned_mb = fp16_mb * 0.30;
                        ui.label(f!("{} tokens", ctx_len));
                        ui.label(f!("{:.2} MB", fp16_mb));
                        ui.label(f!("{:.2} MB", tuned_mb));
                        ui.label(egui::RichText::new("70.0%").color(p.success).strong());
                        ui.end_row();
                    }
                });
            });
        });

        ui.add_space(10.0);
        manga::glass_frame(&p, p.border)
            .inner_margin(egui::Margin::same(12))
            .show(ui, |ui| {
                ui.heading(
                    egui::RichText::new("REA-LLM: EXECUTION-WEIGHTED HETEROGENEOUS ALLOCATION (EWHA) TELEMETRY")
                        .size(12.0)
                        .color(p.primary),
                );
                ui.label(
                    egui::RichText::new(
                        "Execution reach profile q_l under 80/20 routing, realized per-token memory traffic W_bytes/token, per-token energy E_token, and (α, β, γ) representation family mix."
                    )
                    .size(9.0)
                    .color(p.text_muted),
                );
                ui.add_space(8.0);

                let (_, traffic_mb, f_bar, joules, throughput) = self.model().compute_costs();

                ui.columns(4, |sub| {
                    render_stat_box(&mut sub[0], "DEPTH FRACTION (f̄)", format!("{:.1}%", f_bar * 100.0), p.accent, &p);
                    render_stat_box(&mut sub[1], "TRAFFIC (W_bytes/tok)", format!("{:.2} MB", traffic_mb), p.primary, &p);
                    render_stat_box(&mut sub[2], "ENERGY (E_token)", format!("{:.2} mJ", joules * 1000.0), p.success, &p);
                    render_stat_box(&mut sub[3], "EST. THROUGHPUT", format!("{:.1} tok/s", throughput), p.secondary, &p);
                });

                ui.add_space(8.0);
                ui.label(egui::RichText::new("Layerwise Reach & Representation Family Breakdown:").strong().size(10.0).color(p.text_main));

                egui::Grid::new("rea_layer_grid")
                    .striped(true)
                    .spacing([14.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Layer").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("Reach q_l").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("α (Binary)").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("β (Ternary)").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("γ (Phase)").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("Layer Type").strong().size(9.0).color(p.text_main));
                        ui.label(egui::RichText::new("Checkpoint").strong().size(9.0).color(p.text_main));
                        ui.end_row();

                        for l in &self.model().layers {
                            ui.label(f!("L{}", l.layer_index));
                            ui.label(egui::RichText::new(f!("{:.0}%", l.reach_probability * 100.0)).strong().color(if l.reach_probability > 0.5 { p.primary } else { p.text_muted }));
                            ui.label(f!("{:.0}%", l.alpha_binary * 100.0));
                            ui.label(f!("{:.0}%", l.beta_ternary * 100.0));
                            ui.label(f!("{:.0}%", l.gamma_phase * 100.0));
                            ui.label(egui::RichText::new(&l.layer_type).size(8.0).color(p.warning));
                            ui.label(if l.is_exit_checkpoint { egui::RichText::new("EXIT").color(p.success).strong() } else { egui::RichText::new("-").color(p.text_muted) });
                            ui.end_row();
                        }
                    });
            });

        ui.add_space(10.0);
        manga::glass_frame(&p, p.accent)
            .inner_margin(egui::Margin::same(12))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("SPECULATIVE DECODING & MULTI-TOKEN DRAFT SYNTHESIZER (v0.1.6)")
                            .strong()
                            .size(11.0)
                            .color(p.accent),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "Estimated Speedup: {:.2}x | Acc: {:.1}%",
                                self.spec_speedup_val, self.spec_acceptance_val
                            ))
                            .strong()
                            .color(p.success),
                        );
                    });
                });
                ui.label(
                    egui::RichText::new(
                        "Extract lightweight prefix draft sub-manifolds and accelerate autoregressive inference via speculative rejection sampling."
                    )
                    .size(9.0)
                    .color(p.text_muted),
                );

                ui.add_space(8.0);
                ui.columns(4, |sub| {
                    render_stat_box(&mut sub[0], "SPEEDUP MULTIPLIER", format!("{:.2}x", self.spec_speedup_val), p.success, &p);
                    render_stat_box(&mut sub[1], "ACCEPTANCE RATE (α)", format!("{:.1}%", self.spec_acceptance_val), p.primary, &p);
                    render_stat_box(&mut sub[2], "TOKENS / TARGET PASS", format!("{:.2}", self.spec_tokens_per_pass), p.secondary, &p);
                    render_stat_box(&mut sub[3], "MEMORY TRAFFIC SAVED", format!("{:.1}%", self.spec_traffic_savings), p.warning, &p);
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Draft Layers:");
                    let max_layers = self.model().layers.len().saturating_sub(1).max(1);
                    ui.add(egui::Slider::new(&mut self.spec_draft_layers, 1..=max_layers).text("Layers"));

                    ui.add_space(10.0);
                    ui.label("Horizon (γ):");
                    ui.add(egui::Slider::new(&mut self.spec_lookahead_horizon, 1..=10).text("Tokens"));

                    ui.add_space(10.0);
                    ui.label("Temp (τ):");
                    ui.add(egui::Slider::new(&mut self.spec_temperature, 0.0..=1.5).text(""));

                    ui.add_space(10.0);
                    ui.checkbox(&mut self.spec_apply_bitnet, "BitNet b1.58 Ternary Draft");
                    ui.checkbox(&mut self.spec_use_tree, "Tree Speculation (Medusa)");
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button(egui::RichText::new("Simulate Speculative Forward Pass").strong().color(p.text_main)).clicked() {
                        let source_ir = self.model().to_model_ir();
                        let draft_cfg = crate::speculative::DraftExtractionConfig {
                            strategy: if self.spec_apply_bitnet {
                                crate::speculative::DraftExtractionStrategy::TernaryBitNet
                            } else {
                                crate::speculative::DraftExtractionStrategy::PrefixLayers
                            },
                            num_layers: self.spec_draft_layers,
                            target_hidden_size: None,
                            apply_bitnet: self.spec_apply_bitnet,
                            stride_step: 4,
                        };

                        if let Ok((_draft, synth_rep)) = crate::speculative::DraftSynthesizer::extract_draft_model(&source_ir, &draft_cfg) {
                            let spec_cfg = crate::speculative::SpeculativeExecutionConfig {
                                lookahead_horizon: self.spec_lookahead_horizon,
                                temperature: self.spec_temperature,
                                strategy: crate::speculative::SpeculativeVerificationStrategy::RejectionSampling,
                                base_draft_fidelity: 0.82,
                                latency_cost_ratio: synth_rep.latency_cost_ratio,
                            };

                            let tel = crate::speculative::SpeculativeEngine::simulate_speculative_run(&spec_cfg, 25, 12.0);
                            self.spec_speedup_val = tel.effective_speedup_multiplier;
                            self.spec_acceptance_val = tel.mean_acceptance_rate * 100.0;
                            self.spec_tokens_per_pass = tel.mean_tokens_per_pass;
                            self.spec_traffic_savings = tel.memory_traffic_savings_pct;

                            let (opt_g, opt_s) = crate::speculative::SpeculativeEngine::find_optimal_horizon(0.82, synth_rep.latency_cost_ratio);

                            let tree_str = if self.spec_use_tree {
                                let tree = crate::speculative::SpeculativeTreeTopology::build_canonical(&[3, 2, 1]);
                                let (_, acc, rate) = tree.evaluate_tree_acceptance(0.82);
                                format!("\n- Multi-Branch Tree Speculation: {} candidate nodes, {} accepted tokens ({:.1}% branch acc)", tree.total_nodes, acc, rate * 100.0)
                            } else {
                                String::new()
                            };

                            let summary = format!(
                                "{}\n------------------------------------------------------------\n{}\n- Optimal Horizon: gamma = {} (Est. Speedup: {:.2}x){}",
                                synth_rep.format_summary(),
                                tel.format_summary(),
                                opt_g,
                                opt_s,
                                tree_str
                            );
                            let m_name = self.model().name.clone();
                            self.spec_last_report = Some(summary);
                            self.log_audit("BENCHMARK", "SpeculationSimulated", &format!("Speculative speedup: {:.2}x at gamma={}", tel.effective_speedup_multiplier, self.spec_lookahead_horizon), &m_name);
                            self.status_msg = format!("Speculative simulation evaluated: {:.2}x effective speedup!", tel.effective_speedup_multiplier);
                        }
                    }

                    if ui.button("Synthesize & Register Draft Model Instance").clicked() {
                        let source_ir = self.model().to_model_ir();
                        let draft_cfg = crate::speculative::DraftExtractionConfig {
                            strategy: if self.spec_apply_bitnet {
                                crate::speculative::DraftExtractionStrategy::TernaryBitNet
                            } else {
                                crate::speculative::DraftExtractionStrategy::PrefixLayers
                            },
                            num_layers: self.spec_draft_layers,
                            target_hidden_size: None,
                            apply_bitnet: self.spec_apply_bitnet,
                            stride_step: 4,
                        };

                        if let Ok((draft_ir, synth_rep)) = crate::speculative::DraftSynthesizer::extract_draft_model(&source_ir, &draft_cfg) {
                            let new_m = UniversalModel::from_model_ir(draft_ir, "SynthesizedDraft");
                            let name = new_m.name.clone();
                            self.models.push(new_m);
                            self.active_model_idx = self.models.len() - 1;
                            self.auto_scan_and_benchmark();
                            self.run_auto_decider_internal();
                            self.spec_last_report = Some(synth_rep.format_summary());
                            self.log_audit("SURGERY", "DraftExtracted", &format!("Extracted {}-layer draft model", self.spec_draft_layers), &name);
                            self.status_msg = format!("Registered draft model '{}' ({} layers) successfully!", name, self.spec_draft_layers);
                        }
                    }
                });

                if let Some(ref report) = self.spec_last_report {
                    ui.add_space(8.0);
                    manga::glass_deep(&p, p.accent)
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("SPECULATIVE SYNTHESIS & TELEMETRY REPORT").strong().color(p.accent));
                            for line in report.lines() {
                                ui.label(egui::RichText::new(line).size(9.0).monospace().color(p.text_main));
                            }
                        });
                }
            });

        ui.add_space(10.0);
        manga::glass_frame(&p, p.primary)
            .inner_margin(egui::Margin::same(12))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(
                        egui::RichText::new("ROOFLINE & MICRO-KERNEL PROFILER INTERACTIVE DECK (v0.1.7)")
                            .size(12.0)
                            .color(p.primary),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "Knee: {:.1} FLOP/B | {:.1}% Memory-Bound",
                                self.roofline_knee_intensity, self.roofline_mem_bound_pct
                            ))
                            .strong()
                            .color(p.warning),
                        );
                    });
                });
                ui.label(
                    egui::RichText::new(
                        "Profile layer arithmetic intensity against empirical hardware ceilings, cache hierarchies, and thermal throttling."
                    )
                    .size(9.0)
                    .color(p.text_muted),
                );

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Hardware Target:");
                    egui::ComboBox::from_id_salt("roofline_hw_combo")
                        .selected_text(match self.roofline_hardware_idx {
                            0 => "NVIDIA H100 SXM5 (989 TFLOPS | 3.35 TB/s)",
                            1 => "NVIDIA RTX 4090 (165 TFLOPS | 1.01 TB/s)",
                            2 => "Apple M3 Max (32 TFLOPS | 400 GB/s)",
                            3 => "AMD MI300X (1307 TFLOPS | 5.30 TB/s)",
                            _ => "Intel Xeon Emerald Rapids (179 TFLOPS | 560 GB/s)",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.roofline_hardware_idx, 0, "NVIDIA H100 SXM5");
                            ui.selectable_value(&mut self.roofline_hardware_idx, 1, "NVIDIA RTX 4090");
                            ui.selectable_value(&mut self.roofline_hardware_idx, 2, "Apple M3 Max");
                            ui.selectable_value(&mut self.roofline_hardware_idx, 3, "AMD MI300X");
                            ui.selectable_value(&mut self.roofline_hardware_idx, 4, "Intel Xeon Emerald Rapids");
                        });

                    ui.add_space(10.0);
                    ui.label("Batch:");
                    ui.add(egui::DragValue::new(&mut self.roofline_batch_size).range(1..=64));

                    ui.add_space(10.0);
                    ui.label("Seq Len:");
                    ui.add(egui::DragValue::new(&mut self.roofline_seq_len).range(128..=8192));

                    ui.add_space(10.0);
                    ui.checkbox(&mut self.roofline_flash_attn, "FlashAttention-2");
                    ui.checkbox(&mut self.roofline_thermal_sim, "Thermal Sim");

                    if self.roofline_thermal_sim {
                        ui.add(egui::Slider::new(&mut self.roofline_temp_c, 20.0..=105.0).text("°C"));
                    }
                });

                ui.add_space(8.0);
                ui.columns(4, |sub| {
                    render_stat_box(&mut sub[0], "COMPUTE CEILING", format!("{:.1} TFLOPS", self.roofline_peak_tflops), p.primary, &p);
                    render_stat_box(&mut sub[1], "MEMORY BANDWIDTH", format!("{:.2} TB/s", self.roofline_bandwidth_tbps), p.accent, &p);
                    render_stat_box(&mut sub[2], "MACHINE BALANCE KNEE", format!("{:.1} FLOP/B", self.roofline_knee_intensity), p.warning, &p);
                    render_stat_box(&mut sub[3], "BOUND DISTRIBUTION", format!("{:.0}% Mem / {:.0}% Comp", self.roofline_mem_bound_pct, self.roofline_comp_bound_pct), p.secondary, &p);
                });

                ui.add_space(8.0);
                // Custom Painter Interactive Roofline Graph
                let (rect, _resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 160.0), egui::Sense::hover());
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, p.bg_dark);
                painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, p.border), egui::StrokeKind::Outside);

                // Coordinate math and rendering for Roofline Curve
                let origin = Pos2::new(rect.min.x + 50.0, rect.max.y - 30.0);
                let top_right = Pos2::new(rect.max.x - 20.0, rect.min.y + 20.0);
                let knee_ratio = (self.roofline_knee_intensity.log10() / 4.0).clamp(0.2, 0.8) as f32;
                let knee_x = origin.x + (top_right.x - origin.x) * knee_ratio;
                let ceiling_y = top_right.y + 15.0;

                // Draw axes
                painter.line_segment([origin, Pos2::new(top_right.x, origin.y)], Stroke::new(1.0_f32, p.border));
                painter.line_segment([origin, Pos2::new(origin.x, top_right.y)], Stroke::new(1.0_f32, p.border));

                // Draw Sloped Memory-Bound Line & Horizontal Compute-Bound Line
                painter.line_segment([origin, Pos2::new(knee_x, ceiling_y)], Stroke::new(2.0_f32, p.warning));
                painter.line_segment([Pos2::new(knee_x, ceiling_y), Pos2::new(top_right.x, ceiling_y)], Stroke::new(2.0_f32, p.primary));

                // Knee marker
                painter.circle_filled(Pos2::new(knee_x, ceiling_y), 4.0, p.danger);
                painter.text(Pos2::new(knee_x, ceiling_y - 12.0), egui::Align2::CENTER_BOTTOM, format!("Knee ({:.0} FLOP/B)", self.roofline_knee_intensity), egui::FontId::monospace(9.0), p.text_main);

                // Scatter Points: Memory-bound (projections) and Compute-bound (attention/swiglu)
                let mem_pt = Pos2::new(origin.x + (knee_x - origin.x) * 0.25, origin.y - (origin.y - ceiling_y) * 0.25);
                painter.circle_filled(mem_pt, 5.0, p.danger);
                painter.text(Pos2::new(mem_pt.x + 8.0, mem_pt.y), egui::Align2::LEFT_CENTER, "RMSNorm & Proj (Mem-Bound)", egui::FontId::monospace(8.0), p.text_muted);

                let comp_pt = Pos2::new(knee_x + (top_right.x - knee_x) * 0.35, ceiling_y);
                painter.circle_filled(comp_pt, 5.0, p.success);
                painter.text(Pos2::new(comp_pt.x + 8.0, comp_pt.y), egui::Align2::LEFT_CENTER, "Attn & SwiGLU (Comp-Bound)", egui::FontId::monospace(8.0), p.text_muted);

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button(egui::RichText::new("Run Kernel Roofline Profiler").strong().color(p.text_main)).clicked() {
                        let hw = match self.roofline_hardware_idx {
                            0 => crate::profiling::hardware::HardwareSpec::predefined_h100(),
                            1 => crate::profiling::hardware::HardwareSpec::predefined_rtx4090(),
                            2 => crate::profiling::hardware::HardwareSpec::predefined_apple_m3_max(),
                            3 => crate::profiling::hardware::HardwareSpec::predefined_mi300x(),
                            _ => crate::profiling::hardware::HardwareSpec::predefined_intel_emerald_rapids(),
                        };

                        let thermal_factor = if self.roofline_thermal_sim {
                            crate::profiling::thermal::DynamicThermalModel::calculate_throttling_factor(
                                self.roofline_temp_c,
                                hw.tdp_watts,
                            )
                        } else {
                            1.0
                        };

                        let model_ir = self.model().to_model_ir();
                        if let Ok(report) = crate::profiling::roofline::RooflineModel::profile_model(
                            &model_ir,
                            &hw,
                            self.roofline_batch_size,
                            self.roofline_seq_len,
                            self.roofline_flash_attn,
                            thermal_factor,
                        ) {
                            self.roofline_peak_tflops = report.peak_compute_tflops;
                            self.roofline_bandwidth_tbps = report.memory_bandwidth_gbps / 1000.0;
                            self.roofline_knee_intensity = report.knee_intensity_flop_per_byte;

                            let mut mem_count = 0;
                            let mut comp_count = 0;
                            for pt in &report.points {
                                match pt.classification {
                                    crate::profiling::BoundClassification::MemoryBound { .. } => mem_count += 1,
                                    crate::profiling::BoundClassification::ComputeBound { .. } => comp_count += 1,
                                }
                            }
                            let total = mem_count + comp_count;
                            self.roofline_mem_bound_pct = if total > 0 { (mem_count as f64 / total as f64) * 100.0 } else { 0.0 };
                            self.roofline_comp_bound_pct = if total > 0 { (comp_count as f64 / total as f64) * 100.0 } else { 0.0 };

                            let cache_sim = crate::profiling::cache_sim::CacheSimulator4Tier::new(hw.clone());
                            let working_set_bytes = (model_ir.hidden_size * model_ir.hidden_size * 2 * 4).max(16_000_000);
                            let cache_summary = cache_sim.simulate_layer_working_set(working_set_bytes);
                            self.roofline_est_latency_ms = cache_summary.effective_latency_ns * 1e-6 * total as f64 + (self.roofline_batch_size as f64 * 0.05);

                            let ascii_rep = report.format_ascii_report(&hw, report.peak_compute_tflops, report.knee_intensity_flop_per_byte);
                            self.roofline_last_report = Some(ascii_rep);

                            let m_name = self.model().name.clone();
                            self.log_audit("PROFILER", "RooflineProfiled", &format!("Hardware: {}, Batch: {}, Knee: {:.1} FLOP/B", hw.name, self.roofline_batch_size, self.roofline_knee_intensity), &m_name);
                            self.status_msg = format!("Roofline analysis completed for {}!", hw.name);
                        }
                    }

                    if self.roofline_last_report.is_some() && ui.button("Clear Report").clicked() {
                        self.roofline_last_report = None;
                    }
                });

                if let Some(ref report) = self.roofline_last_report {
                    ui.add_space(8.0);
                    manga::glass_deep(&p, p.primary)
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("EMPIRICAL ROOFLINE & KERNEL PROFILE REPORT").strong().color(p.primary));
                            for line in report.lines() {
                                ui.label(egui::RichText::new(line).size(9.0).monospace().color(p.text_main));
                            }
                        });
                }
            });
    }
}
