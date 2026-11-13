// SPDX-License-Identifier: AGPL-3.0-only
//! Studio tab: TAB 6: Audit Log (Structured filterable event stream)

use super::*;

impl THMEApp {
    /// TAB 6: Audit Log (Structured filterable event stream)
    pub(super) fn render_tab_audit_log(&mut self, ui: &mut egui::Ui) {
        let p = self.palette();

        ui.heading(
            egui::RichText::new("STRUCTURED AUDIT LOG STREAM")
                .size(13.0)
                .color(p.text_main),
        );
        ui.label(
            egui::RichText::new(
                "Complete chronological compliance and operation audit trail for all operations.",
            )
            .size(10.0)
            .color(p.text_muted),
        );
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("Filter Level:");
            egui::ComboBox::from_id_salt("audit_level_cb")
                .selected_text(&self.audit_filter_level)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.audit_filter_level,
                        "ALL".to_string(),
                        "ALL Levels",
                    );
                    ui.selectable_value(
                        &mut self.audit_filter_level,
                        "OPTIMIZE".to_string(),
                        "OPTIMIZE",
                    );
                    ui.selectable_value(
                        &mut self.audit_filter_level,
                        "DISTILL".to_string(),
                        "DISTILL",
                    );
                    ui.selectable_value(
                        &mut self.audit_filter_level,
                        "SURGERY".to_string(),
                        "SURGERY",
                    );
                    ui.selectable_value(
                        &mut self.audit_filter_level,
                        "BENCHMARK".to_string(),
                        "BENCHMARK",
                    );
                    ui.selectable_value(&mut self.audit_filter_level, "INFO".to_string(), "INFO");
                    ui.selectable_value(&mut self.audit_filter_level, "WARN".to_string(), "WARN");
                });

            ui.add_space(12.0);
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.audit_search_query);
            if ui.button("Clear").clicked() {
                self.audit_search_query.clear();
            }

            ui.add_space(12.0);
            if ui.button("Export CSV").clicked() {
                let mut csv = "ID,Timestamp,Level,Category,Target,Message\n".to_string();
                for ev in &self.audit_events {
                    csv.push_str(&format!(
                        "{},{},{},{},{},\"{}\"\n",
                        ev.id, ev.timestamp, ev.level, ev.category, ev.target, ev.message
                    ));
                }
                let _ = std::fs::write("audit_log.csv", csv);
                self.status_msg = "Audit log exported to audit_log.csv".to_string();
            }
        });

        ui.add_space(8.0);

        let q = self.audit_search_query.to_lowercase();
        let filtered: Vec<&AuditEvent> = self
            .audit_events
            .iter()
            .filter(|ev| {
                let level_match =
                    self.audit_filter_level == "ALL" || ev.level == self.audit_filter_level;
                let text_match = q.is_empty()
                    || ev.message.to_lowercase().contains(&q)
                    || ev.target.to_lowercase().contains(&q)
                    || ev.category.to_lowercase().contains(&q);
                level_match && text_match
            })
            .collect();

        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("audit_log_grid")
                .striped(true)
                .spacing([12.0, 4.0])
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
                        egui::RichText::new("Level")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.label(
                        egui::RichText::new("Category")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.label(
                        egui::RichText::new("Target")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.label(
                        egui::RichText::new("Message")
                            .strong()
                            .size(9.0)
                            .color(p.text_main),
                    );
                    ui.end_row();

                    for ev in filtered.iter().rev() {
                        let badge_col = match ev.level.as_str() {
                            "OPTIMIZE" => p.success,
                            "DISTILL" => p.primary,
                            "SURGERY" => p.secondary,
                            "BENCHMARK" => p.warning,
                            "WARN" => p.danger,
                            _ => p.accent,
                        };

                        ui.label(f!("#{}", ev.id));
                        ui.label(egui::RichText::new(&ev.timestamp).monospace().size(8.0));
                        ui.label(
                            egui::RichText::new(&ev.level)
                                .strong()
                                .size(9.0)
                                .color(badge_col),
                        );
                        ui.label(&ev.category);
                        ui.label(egui::RichText::new(&ev.target).monospace().size(8.0));
                        ui.label(
                            egui::RichText::new(&ev.message)
                                .size(9.0)
                                .color(p.text_main),
                        );
                        ui.end_row();
                    }
                });
        });
    }
}
