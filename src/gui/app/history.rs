// SPDX-License-Identifier: AGPL-3.0-only
//! Studio tab: TAB 5: History & Checkpoints (Timeline of optimization checkpoints, diffs, rollback)

use super::*;

impl THMEApp {
    /// TAB 5: History & Checkpoints (Timeline of optimization checkpoints, diffs, rollback)
    pub(super) fn render_tab_history(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();
        let mut restore_idx = None;

        ui.heading(
            egui::RichText::new("OPTIMIZATION HISTORY & ROLLBACK CHECKPOINTS")
                .size(13.0)
                .color(p.text_main),
        );
        ui.label(egui::RichText::new("Every tuning session, manual tweak, and distillation pass records an immutable snapshot.").size(10.0).color(p.text_muted));
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui.button("Create Manual Snapshot Checkpoint").clicked() {
                self.create_checkpoint("Manual Snapshot", "Saved state from History tab");
                self.status_msg = "Manual checkpoint recorded.".to_string();
            }

            if ui.button("Export History Report (JSON)").clicked() {
                if let Ok(json) = serde_json::to_string_pretty(&self.history) {
                    let _ = std::fs::write("history_report.json", json);
                    self.status_msg = "Exported history_report.json successfully!".to_string();
                }
            }
        });

        ui.add_space(8.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("history_checkpoints_grid")
                .striped(true)
                .spacing([14.0, 6.0])
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("ID")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.label(
                        egui::RichText::new("Timestamp")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.label(
                        egui::RichText::new("Label")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.label(
                        egui::RichText::new("Storage")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.label(
                        egui::RichText::new("Throughput")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.label(
                        egui::RichText::new("Compression")
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
                        egui::RichText::new("Details")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.label(
                        egui::RichText::new("Action")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.end_row();

                    for (idx, cp) in self.history.iter().enumerate() {
                        ui.label(f!("#{}", cp.id));
                        ui.label(egui::RichText::new(&cp.timestamp).monospace().size(8.0));
                        ui.label(egui::RichText::new(&cp.label).strong().color(p.primary));
                        ui.label(f!("{:.1} MB", cp.storage_mb));
                        ui.label(f!("{:.1} tok/s", cp.throughput_tok_s));
                        ui.label(
                            egui::RichText::new(f!("{:.2}x", cp.compression_ratio))
                                .color(p.success),
                        );
                        ui.label(f!("{:.4}", cp.avg_distortion));
                        ui.label(
                            egui::RichText::new(&cp.details)
                                .size(8.0)
                                .color(p.text_muted),
                        );

                        if ui.button("<- Restore").clicked() {
                            restore_idx = Some(idx);
                        }
                        ui.end_row();
                    }
                });
        });

        if let Some(idx) = restore_idx {
            self.restore_checkpoint(idx);
        }
    }
}
