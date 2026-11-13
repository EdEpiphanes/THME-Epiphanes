//! Studio tab: TAB 2: Layer Tweaker & Sections (drill-down into parameters, weights, and sections)

use super::*;

impl THMEApp {
    /// TAB 2: Layer Tweaker & Sections (drill-down into parameters, weights, and sections)
    pub(super) fn render_tab_layer_tweaker(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();
        let max_l = self.model().layers.len().saturating_sub(1);
        let cur_l = self.selected_layer_idx.min(max_l);

        // Deconstruct the layer into discrete architectural components
        let decomp = self.model().deconstruct_layer(cur_l);

        // Extract layer values to local variables to prevent double borrows
        let mut l_mixer = self.model().layers[cur_l].layer_type.clone();
        let mut is_cp = self.model().layers[cur_l].is_exit_checkpoint;
        let mut a_pct = (self.model().layers[cur_l].alpha_binary * 100.0).round() as i32;
        let mut b_pct = (self.model().layers[cur_l].beta_ternary * 100.0).round() as i32;
        let mut layer_modified = false;
        let mut auto_select_clicked = false;
        let mut apply_subblock_algo = false;
        let mut focused_tensor_name = None;

        ui.heading(
            egui::RichText::new("LAYER ANATOMY, DECONSTRUCTION & DATAFLOW ROUTING")
                .size(13.0)
                .color(p.text_main),
        );
        ui.label(
            egui::RichText::new(
                "Deconstructs every parameter, matrix shape, active routing path, and sequence mixer inside the layer. Customize architectures with hybrids, tri-brids, and sub-block BYOA transformations.",
            )
            .size(10.0)
            .color(p.text_muted),
        );
        ui.add_space(6.0);

        // Layer Navigation Bar & Cockpit
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Selected Layer:")
                    .strong()
                    .color(p.text_main),
            );
            ui.add(
                egui::Slider::new(&mut self.selected_layer_idx, 0..=max_l)
                    .text(f!("Layer {}", cur_l)),
            );
            ui.add_space(16.0);
            ui.label(
                egui::RichText::new(f!("Topology: {}", decomp.topology_style.display_name()))
                    .strong()
                    .color(p.primary),
            );
            ui.separator();
            ui.label(
                egui::RichText::new(f!(
                    "Params: {:.2}M",
                    decomp.total_parameters as f32 / 1_000_000.0
                ))
                .size(10.0)
                .color(p.text_main),
            );
            ui.label(
                egui::RichText::new(f!(
                    "Storage: {:.1} MB",
                    decomp.total_bytes as f32 / (1024.0 * 1024.0)
                ))
                .size(10.0)
                .color(p.text_muted),
            );
            ui.label(
                egui::RichText::new(f!(
                    "Active Traffic: {:.2} MB/tok (q={:.0}%)",
                    decomp.active_bytes_per_token as f32 / (1024.0 * 1024.0),
                    decomp.reach_probability * 100.0
                ))
                .size(10.0)
                .color(p.success),
            );
        });

        ui.add_space(8.0);

        // 1. VISUAL LAYER ANATOMY & DATAFLOW ROUTING MAP
        manga::glass_frame(&p, p.border)
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(f!(
                            "LAYER {} COMPONENT DECONSTRUCTION & FORWARD DATAFLOW MAP",
                            cur_l
                        ))
                        .strong()
                        .color(p.primary),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let route_badge = if decomp.is_exit_checkpoint {
                            egui::RichText::new(
                                "[CP] Adaptive Checkpoint (80% Exit Head / 20% Pass)",
                            )
                            .color(p.warning)
                            .strong()
                        } else {
                            egui::RichText::new(f!(
                                "[SEQ] 100% Sequential Pass to Layer {}",
                                cur_l + 1
                            ))
                            .color(p.success)
                            .strong()
                        };
                        ui.label(route_badge);
                    });
                });
                ui.separator();

                egui::ScrollArea::horizontal()
                    .max_height(210.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Sub-Block 1: Input RMSNorm
                            if let Some(norm) = &decomp.input_norm {
                                let is_sel = self.selected_subblock
                                    == Some(crate::ir::SubBlockCategory::InputNorm);
                                egui::Frame::NONE
                                    .fill(manga::glass_fill(p.bg_input, &p))
                                    .stroke(Stroke::new(
                                        if is_sel { 2.0_f32 } else { 1.0_f32 },
                                        if is_sel { p.primary } else { p.border },
                                    ))
                                    .inner_margin(egui::Margin::same(8))
                                    .show(ui, |ui| {
                                        ui.vertical(|ui| {
                                            ui.set_width(170.0);
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    egui::RichText::new("1. INPUT NORM")
                                                        .strong()
                                                        .color(p.primary)
                                                        .size(9.0),
                                                );
                                                if ui
                                                    .small_button(if is_sel {
                                                        "Active"
                                                    } else {
                                                        "Select"
                                                    })
                                                    .clicked()
                                                {
                                                    self.selected_subblock = Some(
                                                        crate::ir::SubBlockCategory::InputNorm,
                                                    );
                                                }
                                            });
                                            ui.label(
                                                egui::RichText::new("RMSNorm (Variance Scale)")
                                                    .size(8.0)
                                                    .color(p.text_muted),
                                            );
                                            ui.separator();
                                            for t in &norm.tensors {
                                                ui.horizontal(|ui| {
                                                    ui.label(
                                                        egui::RichText::new("γ (Scale)")
                                                            .size(8.0)
                                                            .color(p.text_main),
                                                    );
                                                    ui.label(
                                                        egui::RichText::new(f!("{:?}", t.shape))
                                                            .size(7.0)
                                                            .monospace(),
                                                    );
                                                    if ui.small_button("View").clicked() {
                                                        focused_tensor_name = Some(t.name.clone());
                                                        self.selected_subblock = Some(
                                                            crate::ir::SubBlockCategory::InputNorm,
                                                        );
                                                    }
                                                });
                                            }
                                            ui.add_space(4.0);
                                            ui.label(
                                                egui::RichText::new("Routing Target:")
                                                    .size(7.0)
                                                    .color(p.text_muted),
                                            );
                                            ui.label(
                                                egui::RichText::new("-> Sequence Mixer")
                                                    .size(8.0)
                                                    .color(p.secondary),
                                            );
                                        });
                                    });

                                ui.label(egui::RichText::new("-->").size(14.0).color(p.primary));
                            }

                            // Sub-Block 2: Sequence Mixer
                            let is_sel_mix = self.selected_subblock
                                == Some(crate::ir::SubBlockCategory::SequenceMixer);
                            egui::Frame::NONE
                                .fill(manga::glass_fill(p.bg_input, &p))
                                .stroke(Stroke::new(
                                    if is_sel_mix { 2.0_f32 } else { 1.0_f32 },
                                    if is_sel_mix { p.primary } else { p.border },
                                ))
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    ui.vertical(|ui| {
                                        ui.set_width(220.0);
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                egui::RichText::new(f!(
                                                    "2. MIXER [{}]",
                                                    l_mixer.to_uppercase()
                                                ))
                                                .strong()
                                                .color(p.primary)
                                                .size(9.0),
                                            );
                                            if ui
                                                .small_button(if is_sel_mix {
                                                    "Active"
                                                } else {
                                                    "Select"
                                                })
                                                .clicked()
                                            {
                                                self.selected_subblock = Some(
                                                    crate::ir::SubBlockCategory::SequenceMixer,
                                                );
                                            }
                                            ui.with_layout(
                                                egui::Layout::right_to_left(egui::Align::Center),
                                                |ui| {
                                                    ui.label(
                                                        egui::RichText::new(f!(
                                                            "{:.1}M",
                                                            decomp.mixer_block.total_parameters
                                                                as f32
                                                                / 1_000_000.0
                                                        ))
                                                        .size(8.0)
                                                        .color(p.text_muted),
                                                    );
                                                },
                                            );
                                        });
                                        ui.label(
                                            egui::RichText::new("Token sequence correlation")
                                                .size(8.0)
                                                .color(p.text_muted),
                                        );
                                        ui.separator();
                                        for t in &decomp.mixer_block.tensors {
                                            ui.horizontal(|ui| {
                                                let role_str = match t.role {
                                                    crate::ir::TensorRole::AttentionQ => "W_q",
                                                    crate::ir::TensorRole::AttentionK => "W_k",
                                                    crate::ir::TensorRole::AttentionV => "W_v",
                                                    crate::ir::TensorRole::AttentionOut => "W_o",
                                                    _ => "W_mix",
                                                };
                                                ui.label(
                                                    egui::RichText::new(role_str)
                                                        .strong()
                                                        .size(8.0)
                                                        .color(p.primary),
                                                );
                                                ui.label(
                                                    egui::RichText::new(f!("{:?}", t.shape))
                                                        .size(7.0)
                                                        .monospace(),
                                                );
                                                ui.label(
                                                    egui::RichText::new(&t.selected_format)
                                                        .size(7.0)
                                                        .color(p.success),
                                                );
                                                if ui.small_button("View").clicked() {
                                                    focused_tensor_name = Some(t.name.clone());
                                                    self.selected_subblock = Some(
                                                        crate::ir::SubBlockCategory::SequenceMixer,
                                                    );
                                                }
                                            });
                                        }
                                        ui.add_space(4.0);
                                        ui.label(
                                            egui::RichText::new("Routing Target:")
                                                .size(7.0)
                                                .color(p.text_muted),
                                        );
                                        ui.label(
                                            egui::RichText::new("-> Residual Highway 1 (Add)")
                                                .size(8.0)
                                                .color(p.secondary),
                                        );
                                    });
                                });

                            ui.label(egui::RichText::new("-->").size(14.0).color(p.primary));

                            // Sub-Block 3: Residual Highway 1 (Add)
                            manga::glass_frame(&p, p.border)
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    ui.vertical(|ui| {
                                        ui.set_width(120.0);
                                        ui.label(
                                            egui::RichText::new("3. RESIDUAL 1")
                                                .strong()
                                                .color(p.accent)
                                                .size(9.0),
                                        );
                                        ui.label(
                                            egui::RichText::new("Identity Add")
                                                .size(8.0)
                                                .color(p.text_muted),
                                        );
                                        ui.separator();
                                        ui.label(
                                            egui::RichText::new("x = x + Mixer(x)")
                                                .size(8.0)
                                                .monospace(),
                                        );
                                        ui.label(
                                            egui::RichText::new("Zero latency skip")
                                                .size(7.0)
                                                .color(p.text_muted),
                                        );
                                    });
                                });

                            ui.label(egui::RichText::new("-->").size(14.0).color(p.primary));

                            // Sub-Block 4: Post-Mixer RMSNorm
                            if let Some(post_norm) = &decomp.post_mixer_norm {
                                let is_sel_post = self.selected_subblock
                                    == Some(crate::ir::SubBlockCategory::PostMixerNorm);
                                egui::Frame::NONE
                                    .fill(manga::glass_fill(p.bg_input, &p))
                                    .stroke(Stroke::new(
                                        if is_sel_post { 2.0_f32 } else { 1.0_f32 },
                                        if is_sel_post { p.primary } else { p.border },
                                    ))
                                    .inner_margin(egui::Margin::same(8))
                                    .show(ui, |ui| {
                                        ui.vertical(|ui| {
                                            ui.set_width(160.0);
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    egui::RichText::new("4. POST-MIX NORM")
                                                        .strong()
                                                        .color(p.primary)
                                                        .size(9.0),
                                                );
                                                if ui
                                                    .small_button(if is_sel_post {
                                                        "Active"
                                                    } else {
                                                        "Select"
                                                    })
                                                    .clicked()
                                                {
                                                    self.selected_subblock = Some(
                                                        crate::ir::SubBlockCategory::PostMixerNorm,
                                                    );
                                                }
                                            });
                                            ui.label(
                                                egui::RichText::new("RMSNorm (Variance Scale)")
                                                    .size(8.0)
                                                    .color(p.text_muted),
                                            );
                                            ui.separator();
                                            for t in &post_norm.tensors {
                                                ui.horizontal(|ui| {
                                                    ui.label(
                                                        egui::RichText::new("γ (Scale)").size(8.0),
                                                    );
                                                    ui.label(
                                                        egui::RichText::new(f!("{:?}", t.shape))
                                                            .size(7.0)
                                                            .monospace(),
                                                    );
                                                    if ui.small_button("View").clicked() {
                                                        focused_tensor_name = Some(t.name.clone());
                                                        self.selected_subblock = Some(
                                                        crate::ir::SubBlockCategory::PostMixerNorm,
                                                    );
                                                    }
                                                });
                                            }
                                            ui.add_space(4.0);
                                            ui.label(
                                                egui::RichText::new("Routing Target:")
                                                    .size(7.0)
                                                    .color(p.text_muted),
                                            );
                                            ui.label(
                                                egui::RichText::new("-> SwiGLU MLP")
                                                    .size(8.0)
                                                    .color(p.secondary),
                                            );
                                        });
                                    });

                                ui.label(egui::RichText::new("-->").size(14.0).color(p.primary));
                            }

                            // Sub-Block 5: Feed-Forward Network (SwiGLU MLP)
                            if let Some(ffn) = &decomp.ffn_block {
                                let is_sel_ffn = self.selected_subblock
                                    == Some(crate::ir::SubBlockCategory::FeedForwardMlp);
                                egui::Frame::NONE
                                    .fill(manga::glass_fill(p.bg_input, &p))
                                    .stroke(Stroke::new(
                                        if is_sel_ffn { 2.0_f32 } else { 1.0_f32 },
                                        if is_sel_ffn { p.primary } else { p.border },
                                    ))
                                    .inner_margin(egui::Margin::same(8))
                                    .show(ui, |ui| {
                                        ui.vertical(|ui| {
                                            ui.set_width(220.0);
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    egui::RichText::new("5. SWIGLU MLP")
                                                        .strong()
                                                        .color(p.primary)
                                                        .size(9.0),
                                                );
                                                if ui
                                                    .small_button(if is_sel_ffn {
                                                        "Active"
                                                    } else {
                                                        "Select"
                                                    })
                                                    .clicked()
                                                {
                                                    self.selected_subblock = Some(
                                                        crate::ir::SubBlockCategory::FeedForwardMlp,
                                                    );
                                                }
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        ui.label(
                                                            egui::RichText::new(f!(
                                                                "{:.1}M",
                                                                ffn.total_parameters as f32
                                                                    / 1_000_000.0
                                                            ))
                                                            .size(8.0)
                                                            .color(p.text_muted),
                                                        );
                                                    },
                                                );
                                            });
                                            ui.label(
                                                egui::RichText::new("Non-linear channel expansion")
                                                    .size(8.0)
                                                    .color(p.text_muted),
                                            );
                                            ui.separator();
                                            for t in &ffn.tensors {
                                                ui.horizontal(|ui| {
                                                    let role_str = match t.role {
                                                        crate::ir::TensorRole::MlpGate => "W_gate",
                                                        crate::ir::TensorRole::MlpUp => "W_up",
                                                        crate::ir::TensorRole::MlpDown => "W_down",
                                                        _ => "W_mlp",
                                                    };
                                                    ui.label(
                                                        egui::RichText::new(role_str)
                                                            .strong()
                                                            .size(8.0)
                                                            .color(p.warning),
                                                    );
                                                    ui.label(
                                                        egui::RichText::new(f!("{:?}", t.shape))
                                                            .size(7.0)
                                                            .monospace(),
                                                    );
                                                    ui.label(
                                                        egui::RichText::new(&t.selected_format)
                                                            .size(7.0)
                                                            .color(p.success),
                                                    );
                                                    if ui.small_button("View").clicked() {
                                                        focused_tensor_name = Some(t.name.clone());
                                                        self.selected_subblock = Some(
                                                        crate::ir::SubBlockCategory::FeedForwardMlp,
                                                    );
                                                    }
                                                });
                                            }
                                            ui.add_space(4.0);
                                            ui.label(
                                                egui::RichText::new("Routing Target:")
                                                    .size(7.0)
                                                    .color(p.text_muted),
                                            );
                                            ui.label(
                                                egui::RichText::new("-> Residual Highway 2 (+)")
                                                    .size(8.0)
                                                    .color(p.secondary),
                                            );
                                        });
                                    });

                                ui.label(egui::RichText::new("-->").size(14.0).color(p.primary));
                            }

                            // Sub-Block 6: Residual Highway 2 (+ Add)
                            manga::glass_frame(&p, p.border)
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    ui.vertical(|ui| {
                                        ui.set_width(120.0);
                                        ui.label(
                                            egui::RichText::new("6. RESIDUAL 2")
                                                .strong()
                                                .color(p.accent)
                                                .size(9.0),
                                        );
                                        ui.label(
                                            egui::RichText::new("+ Identity Add")
                                                .size(8.0)
                                                .color(p.text_muted),
                                        );
                                        ui.separator();
                                        ui.label(
                                            egui::RichText::new("x = x + MLP(x)")
                                                .size(8.0)
                                                .monospace(),
                                        );
                                        ui.label(
                                            egui::RichText::new("Output collector")
                                                .size(7.0)
                                                .color(p.text_muted),
                                        );
                                    });
                                });

                            ui.label(egui::RichText::new("-->").size(14.0).color(p.primary));

                            // Sub-Block 7: Routing Gateway
                            egui::Frame::NONE
                                .fill(if decomp.is_exit_checkpoint {
                                    p.bg_input
                                } else {
                                    p.bg_card
                                })
                                .stroke(Stroke::new(
                                    1.0_f32,
                                    if decomp.is_exit_checkpoint {
                                        p.warning
                                    } else {
                                        p.border
                                    },
                                ))
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    ui.set_width(170.0);
                                    ui.label(
                                        egui::RichText::new("7. ROUTING GATEWAY")
                                            .strong()
                                            .color(p.primary)
                                            .size(9.0),
                                    );
                                    ui.label(
                                        egui::RichText::new(if decomp.is_exit_checkpoint {
                                            "Checkpoint Branch"
                                        } else {
                                            "Sequential Forward"
                                        })
                                        .size(8.0)
                                        .color(p.text_muted),
                                    );
                                    ui.separator();
                                    ui.label(
                                        egui::RichText::new(&decomp.routing_block.routing_target)
                                            .size(8.0)
                                            .color(p.success),
                                    );
                                });
                        });
                    });
            });

        ui.add_space(8.0);

        // 2. TWO-COLUMN SPLIT: HYBRID ARCHITECTURE RESTRUCTURER vs DEEP PARAMETER INSPECTOR
        ui.columns(2, |cols| {
            // Column 0: Hybrid / Tri-brid Architecture Restructurer & Sub-Block Converter
            cols[0].vertical(|ui| {
                ui.heading(
                    egui::RichText::new("HYBRID & TRI-BRID ARCHITECTURE RESTRUCTURER")
                        .size(12.0)
                        .color(p.text_main),
                );
                ui.label(
                    egui::RichText::new(
                        "Reconfigure this layer's topology into hybrid or tri-brid mixers and convert sub-components with custom algorithms.",
                    )
                    .size(9.0)
                    .color(p.text_muted),
                );
                ui.add_space(6.0);

                manga::glass_frame(&p, p.border)
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Layer Mixer:");
                            let prev_mixer = l_mixer.clone();
                            egui::ComboBox::from_id_salt(f!("layer_tweaker_mixer_{}", cur_l))
                                .selected_text(&l_mixer)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut l_mixer, "gqa_attention".to_string(), "GQA Attention (Monolithic)");
                                    ui.selectable_value(&mut l_mixer, "gated_deltanet".to_string(), "Gated DeltaNet (Associative)");
                                    ui.selectable_value(&mut l_mixer, "phase_recurrence".to_string(), "Phase Recurrence (Unitary)");
                                    ui.selectable_value(&mut l_mixer, "sliding_window".to_string(), "Sliding Window Attention");
                                    ui.selectable_value(&mut l_mixer, "mamba2".to_string(), "Mamba-2 SSM (State Space)");
                                });
                            if l_mixer != prev_mixer {
                                layer_modified = true;
                            }
                        });

                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("Topology Style Selection:")
                                .strong()
                                .size(9.0),
                        );
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("topology_mode_select")
                                .selected_text(match self.hybrid_topology_mode {
                                    0 => "Standard Monolithic",
                                    1 => "Hybrid: GQA + Gated DeltaNet",
                                    2 => "Tri-brid: GQA + DeltaNet + Phase",
                                    _ => "Mamba-2 SSM Hybrid",
                                })
                                .show_ui(ui, |ui| {
                                    if ui.selectable_value(&mut self.hybrid_topology_mode, 0, "Standard Monolithic").clicked() {
                                        a_pct = 0;
                                        b_pct = 100;
                                        l_mixer = "gqa_attention".to_string();
                                        layer_modified = true;
                                    }
                                    if ui.selectable_value(&mut self.hybrid_topology_mode, 1, "Hybrid: GQA + Gated DeltaNet").clicked() {
                                        a_pct = 20;
                                        b_pct = 60;
                                        l_mixer = "gated_deltanet".to_string();
                                        layer_modified = true;
                                    }
                                    if ui.selectable_value(&mut self.hybrid_topology_mode, 2, "Tri-brid: GQA + DeltaNet + Phase").clicked() {
                                        a_pct = 25;
                                        b_pct = 45;
                                        l_mixer = "gated_deltanet".to_string();
                                        layer_modified = true;
                                    }
                                    if ui.selectable_value(&mut self.hybrid_topology_mode, 3, "Mamba-2 SSM Hybrid").clicked() {
                                        a_pct = 15;
                                        b_pct = 70;
                                        l_mixer = "mamba2".to_string();
                                        layer_modified = true;
                                    }
                                });
                        });

                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("Channel Representation Ratios (α / β / γ):")
                                .strong()
                                .size(9.0),
                        );
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("α (Binary):").color(p.primary));
                            if ui.add(egui::DragValue::new(&mut a_pct).range(0..=100).suffix("%")).changed() {
                                a_pct = a_pct.clamp(0, 100);
                                if a_pct + b_pct > 100 { b_pct = 100 - a_pct; }
                                layer_modified = true;
                            }
                            ui.label(egui::RichText::new("β (Ternary):").color(p.success));
                            if ui.add(egui::DragValue::new(&mut b_pct).range(0..=100).suffix("%")).changed() {
                                b_pct = b_pct.clamp(0, 100);
                                if a_pct + b_pct > 100 { a_pct = 100 - b_pct; }
                                layer_modified = true;
                            }
                            let g_pct = (100 - a_pct - b_pct).max(0);
                            ui.label(egui::RichText::new(f!("γ (Phase): {}%", g_pct)).color(p.warning));
                        });

                        ui.separator();
                        ui.label(egui::RichText::new("Sub-Block Converter & Custom BYOA Algorithm:").strong().size(9.0).color(p.primary));
                        ui.horizontal(|ui| {
                            ui.label("Target:");
                            egui::ComboBox::from_id_salt("subblock_target_select")
                                .selected_text(match self.subblock_convert_target {
                                    0 => "Sequence Mixer Sub-Block",
                                    1 => "Feed-Forward MLP Sub-Block",
                                    _ => "All Layer Sub-Blocks",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.subblock_convert_target, 0, "Sequence Mixer Sub-Block");
                                    ui.selectable_value(&mut self.subblock_convert_target, 1, "Feed-Forward MLP Sub-Block");
                                    ui.selectable_value(&mut self.subblock_convert_target, 2, "All Layer Sub-Blocks");
                                });
                        });

                        ui.horizontal(|ui| {
                            ui.label("Algorithm:");
                            egui::ComboBox::from_id_salt("subblock_algo_select")
                                .selected_text(&self.subblock_target_algo_id)
                                .show_ui(ui, |ui| {
                                    for alg in self.algorithm_registry.list() {
                                        ui.selectable_value(&mut self.subblock_target_algo_id, alg.id.clone(), format!("{} [{:.1}b]", alg.name, alg.effective_bits_per_param));
                                    }
                                });
                        });

                        if ui.button(egui::RichText::new("Apply Algorithm to Selected Sub-Block").strong().color(p.text_main)).clicked() {
                            apply_subblock_algo = true;
                        }

                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut is_cp, "Early-Exit Routing Checkpoint").changed() {
                                layer_modified = true;
                            }
                        });

                        ui.add_space(4.0);
                        if ui.button(egui::RichText::new("Auto-Select Pareto Optimal Configuration").strong().color(p.text_main)).clicked() {
                            auto_select_clicked = true;
                        }
                    });
            });

            // Column 1: Deep Parameter Inspector & 8×8 Numerical Matrix Slice
            cols[1].vertical(|ui| {
                ui.heading(
                    egui::RichText::new("FOCUSED PARAMETER INSPECTOR & MATRIX SLICE")
                        .size(12.0)
                        .color(p.text_main),
                );
                ui.label(
                    egui::RichText::new(
                        "Drill down into individual parameter tensors, statistics, and 8×8 weight heatmaps.",
                    )
                    .size(9.0)
                    .color(p.text_muted),
                );
                ui.add_space(6.0);

                // Find tensors for this layer
                let layer_tensors: Vec<(usize, String, String, Vec<usize>)> = self
                    .model()
                    .tensors
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.layer_index == Some(cur_l))
                    .map(|(idx, t)| (idx, t.name.clone(), t.role.clone(), t.shape.clone()))
                    .collect();

                if let Some(target_name) = focused_tensor_name {
                    if let Some((idx, _, _, _)) = layer_tensors.iter().find(|(_, name, _, _)| name == &target_name) {
                        self.selected_tensor_idx = *idx;
                    }
                }

                if !layer_tensors.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label("Section Filter:");
                        egui::ComboBox::from_id_salt("layer_sec_filter_cb")
                            .selected_text(&self.selected_section_filter)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.selected_section_filter, "All Sections".to_string(), "All Sections");
                                ui.selectable_value(&mut self.selected_section_filter, "Attention".to_string(), "Attention (Q/K/V/O)");
                                ui.selectable_value(&mut self.selected_section_filter, "Mlp".to_string(), "MLP (Gate/Up/Down)");
                                ui.selectable_value(&mut self.selected_section_filter, "LayerNorm".to_string(), "LayerNorm");
                            });
                    });

                    let filtered: Vec<&(usize, String, String, Vec<usize>)> = layer_tensors
                        .iter()
                        .filter(|(_, _, role, _)| {
                            if self.selected_section_filter == "All Sections" {
                                true
                            } else {
                                role.contains(&self.selected_section_filter)
                            }
                        })
                        .collect();

                    if let Some(first) = filtered.first() {
                        if !filtered.iter().any(|(idx, _, _, _)| *idx == self.selected_tensor_idx) {
                            self.selected_tensor_idx = first.0;
                        }
                    }

                    ui.horizontal(|ui| {
                        ui.label("Parameter:");
                        let cur_name = layer_tensors
                            .iter()
                            .find(|(idx, _, _, _)| *idx == self.selected_tensor_idx)
                            .map(|(_, name, _, _)| name.clone())
                            .unwrap_or_default();

                        egui::ComboBox::from_id_salt("param_select_combobox")
                            .selected_text(&cur_name)
                            .width(260.0)
                            .show_ui(ui, |ui| {
                                for (idx, _, role, shape) in &filtered {
                                    ui.selectable_value(&mut self.selected_tensor_idx, *idx, format!("{} [{:?}]", role, shape));
                                }
                            });
                    });

                    if let Some(t) = self.model().tensors.get(self.selected_tensor_idx) {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(f!("Shape: {:?} | DType: {} | Elements: {}", t.shape, t.dtype, t.num_elements))
                                .monospace()
                                .size(9.0)
                                .color(p.primary),
                        );

                        ui.columns(2, |sub| {
                            render_stat_box(&mut sub[0], "RANGE (MIN..MAX)", f!("{:.4} .. {:.4}", t.min_val, t.max_val), p.text_main, &p);
                            render_stat_box(&mut sub[1], "MOMENTS (μ ± σ)", f!("{:.4} ± {:.4}", t.mean_val, t.std_val), p.warning, &p);
                        });
                        ui.columns(2, |sub| {
                            render_stat_box(&mut sub[0], "SPARSITY", f!("{:.2}%", t.sparsity * 100.0), p.success, &p);
                            render_stat_box(&mut sub[1], "OUTLIERS (>3σ)", f!("{:.3}%", t.outlier_ratio * 100.0), p.danger, &p);
                        });

                        ui.add_space(6.0);
                        ui.label(egui::RichText::new("Interactive 8×8 Numerical Matrix Slice:").strong().size(9.0).color(p.text_main));
                        let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 110.0), egui::Sense::hover());
                        ui.painter().rect_filled(rect, 4.0, p.bg_input);
                        ui.painter().rect_stroke(rect, 4.0, Stroke::new(1.0_f32, p.border), egui::StrokeKind::Outside);

                        let cell_w = (rect.width() - 16.0) / 8.0;
                        let cell_h = (rect.height() - 16.0) / 8.0;

                        for r in 0..8 {
                            for c in 0..8 {
                                let idx = (r * 8 + c) % t.sample_weights.len().max(1);
                                let val = t.sample_weights.get(idx).copied().unwrap_or(0.0);
                                let norm_v = (val / 0.15).clamp(-1.0, 1.0);

                                let cell_col = if norm_v > 0.0 {
                                    Color32::from_rgb(
                                        (p.bg_card.r() as f32 + (p.primary.r() as f32 - p.bg_card.r() as f32) * norm_v) as u8,
                                        (p.bg_card.g() as f32 + (p.primary.g() as f32 - p.bg_card.g() as f32) * norm_v) as u8,
                                        (p.bg_card.b() as f32 + (p.primary.b() as f32 - p.bg_card.b() as f32) * norm_v) as u8,
                                    )
                                } else {
                                    Color32::from_rgb(
                                        (p.bg_card.r() as f32 + (p.danger.r() as f32 - p.bg_card.r() as f32) * norm_v.abs()) as u8,
                                        (p.bg_card.g() as f32 + (p.danger.g() as f32 - p.bg_card.g() as f32) * norm_v.abs()) as u8,
                                        (p.bg_card.b() as f32 + (p.danger.b() as f32 - p.bg_card.b() as f32) * norm_v.abs()) as u8,
                                    )
                                };

                                let cell_rect = Rect::from_min_size(
                                    Pos2::new(rect.min.x + 8.0 + (c as f32 * cell_w), rect.min.y + 8.0 + (r as f32 * cell_h)),
                                    Vec2::new(cell_w - 2.0, cell_h - 2.0),
                                );
                                ui.painter().rect_filled(cell_rect, 2.0, cell_col);
                                ui.painter().text(
                                    cell_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    f!("{:.2}", val),
                                    egui::FontId::monospace(7.0),
                                    Color32::WHITE,
                                );
                            }
                        }

                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label("Scale:");
                            ui.add(egui::Slider::new(&mut self.layer_scale_factor, 0.5..=2.0).text("x"));
                            ui.label("Bias:");
                            ui.add(egui::Slider::new(&mut self.layer_bias_offset, -0.1..=0.1).text("+"));
                            ui.label("Prune:");
                            ui.add(egui::Slider::new(&mut self.layer_pruning_sparsity, 0.0..=0.8).text("0s"));
                        });
                    }
                }
            });
        });

        // Write back modifications after UI closures have completed
        if layer_modified {
            let layer = &mut self.model_mut().layers[cur_l];
            layer.layer_type = l_mixer;
            layer.is_exit_checkpoint = is_cp;
            layer.alpha_binary = a_pct as f32 / 100.0;
            layer.beta_ternary = b_pct as f32 / 100.0;
            layer.gamma_phase = (100 - a_pct - b_pct).max(0) as f32 / 100.0;
        }

        if apply_subblock_algo {
            let alg_opt = self
                .algorithm_registry
                .get(&self.subblock_target_algo_id)
                .cloned();
            if let Some(alg) = alg_opt {
                let target_cat = self.subblock_convert_target;
                let mut modified_count = 0;
                for t in &mut self.model_mut().tensors {
                    if t.layer_index == Some(cur_l) {
                        let role = crate::ir::TensorRole::detect(&t.name);
                        let should_apply = match target_cat {
                            0 => matches!(
                                role,
                                crate::ir::TensorRole::AttentionQ
                                    | crate::ir::TensorRole::AttentionK
                                    | crate::ir::TensorRole::AttentionV
                                    | crate::ir::TensorRole::AttentionOut
                            ),
                            1 => matches!(
                                role,
                                crate::ir::TensorRole::MlpGate
                                    | crate::ir::TensorRole::MlpUp
                                    | crate::ir::TensorRole::MlpDown
                            ),
                            _ => true,
                        };
                        if should_apply {
                            let _ = alg.apply_to_weights(&mut t.sample_weights);
                            modified_count += 1;
                        }
                    }
                }
                self.status_msg = format!(
                    "Applied '{}' to {} parameters in Layer {}!",
                    alg.name, modified_count, cur_l
                );
                self.log_audit(
                    "OPTIMIZE",
                    "SubBlockTransform",
                    &format!("Applied {} to Layer {}", alg.name, cur_l),
                    &format!("Layer {}", cur_l),
                );
                self.run_auto_decider_internal();
            }
        }

        if auto_select_clicked {
            let num_layers = self.model().layers.len();
            let depth_frac = cur_l as f32 / num_layers.max(1) as f32;

            let layer = &mut self.model_mut().layers[cur_l];
            layer.is_optimized = true;
            if depth_frac < 0.25 {
                layer.alpha_binary = 0.15;
                layer.beta_ternary = 0.70;
                layer.gamma_phase = 0.15;
                layer.layer_type = "sliding_window".to_string();
            } else if depth_frac < 0.75 {
                layer.alpha_binary = 0.25;
                layer.beta_ternary = 0.50;
                layer.gamma_phase = 0.25;
                layer.layer_type = "gated_deltanet".to_string();
            } else {
                layer.alpha_binary = 0.40;
                layer.beta_ternary = 0.35;
                layer.gamma_phase = 0.25;
                layer.layer_type = "phase_recurrence".to_string();
            }
            let target_str = format!("Layer {}", cur_l);
            self.log_audit(
                "OPTIMIZE",
                "LayerTweak",
                &format!("Auto-selected optimal algorithm for Layer {}", cur_l),
                &target_str,
            );
            self.status_msg = f!("Auto-selected optimal parameters for Layer {}!", cur_l);
        }
    }
}
