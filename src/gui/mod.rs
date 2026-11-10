pub mod app;

pub use app::THMEApp;

use eframe::egui;

/// Launch the THME / LLM-Tweaker native desktop GUI
pub fn launch_gui() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1380.0, 880.0])
            .with_min_inner_size([1100.0, 700.0])
            .with_title("THME v2.0 // Universal LLM Re-structurer, Optimizer & Tuner"),
        ..Default::default()
    };

    eframe::run_native(
        "THME Studio",
        options,
        Box::new(|cc| Ok(Box::new(THMEApp::new(cc)))),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run GUI: {}", e))
}
