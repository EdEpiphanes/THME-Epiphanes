// SPDX-License-Identifier: AGPL-3.0-only
//! Studio "Welcome" window (roadmap v0.3.0.7): what `thme doctor` and `thme discover` find on
//! this machine, and what to try next. Both run on a background thread; the window only reads
//! the result. Hardware facts are checked (read here) or assumed (taken from the build). Nothing
//! shown is a quality or speed claim.

use super::provenance::{self, Evidence};
use super::theme::ThemePalette;
use super::*;
use crate::discover::DiscoveredModel;
use crate::doctor::{self, HostFacts};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};

/// Model rows listed in the window; the rest are only counted.
const MAX_ROWS: usize = doctor::MAX_SHOWN_MODELS;

/// Where the check stands.
#[derive(Debug, Default, PartialEq)]
pub(super) enum Phase {
    #[default]
    Idle,
    Looking,
    Ready(Box<HostFacts>),
    Error(String),
}

/// The Welcome window's state. The transitions are pure; `refresh` and `poll` only move the
/// background check in and out of them.
#[derive(Default)]
pub(super) struct WelcomePanel {
    /// Whether the window is drawn.
    pub(super) open: bool,
    phase: Phase,
    /// Outcome of the last Open: a refusal or an import error, as text.
    message: Option<String>,
    /// Where the background check reports; `None` when no check runs.
    results: Option<Receiver<HostFacts>>,
}

/// A button press, applied after the window has been drawn.
pub(super) enum Action {
    Refresh,
    Open(PathBuf),
}

impl WelcomePanel {
    pub(super) fn phase(&self) -> &Phase {
        &self.phase
    }

    pub(super) fn facts(&self) -> Option<&HostFacts> {
        match &self.phase {
            Phase::Ready(facts) => Some(facts),
            _ => None,
        }
    }

    pub(super) fn is_idle(&self) -> bool {
        self.phase == Phase::Idle
    }

    pub(super) fn is_looking(&self) -> bool {
        self.phase == Phase::Looking
    }

    pub(super) fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    /// The model rows shown: the first ten, and none until a check is ready.
    pub(super) fn visible_models(&self) -> &[DiscoveredModel] {
        match self.facts() {
            Some(facts) => facts
                .models
                .get(..MAX_ROWS.min(facts.models.len()))
                .unwrap_or_default(),
            None => &[],
        }
    }

    /// Moves to `Looking` unless a check is already running. Returns whether one should start.
    pub(super) fn begin(&mut self) -> bool {
        if self.is_looking() {
            return false;
        }
        self.phase = Phase::Looking;
        true
    }

    /// Replaces the results with a finished check.
    pub(super) fn finish(&mut self, facts: HostFacts) {
        self.phase = Phase::Ready(Box::new(facts));
    }

    pub(super) fn fail(&mut self, reason: String) {
        self.phase = Phase::Error(reason);
    }

    /// A successful Open closes the window; a refusal or import error stays on screen as text.
    pub(super) fn record_open(&mut self, outcome: Result<(), String>) {
        match outcome {
            Ok(()) => {
                self.open = false;
                self.message = Some("Model opened in the Studio.".to_string());
            }
            Err(err) => self.message = Some(err),
        }
    }

    /// Starts the background check unless one is running. Both collections run in it.
    pub(super) fn refresh(&mut self) {
        if !self.begin() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("thme-welcome".to_string())
            .spawn(move || {
                let _ = tx.send(doctor::collect_host_facts());
            });
        match spawned {
            Ok(_) => self.results = Some(rx),
            Err(err) => self.fail(format!("cannot start the check: {err}")),
        }
    }

    /// Takes the background result once it has arrived.
    pub(super) fn poll(&mut self) {
        let Some(rx) = self.results.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(facts) => {
                self.results = None;
                self.finish(facts);
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.results = None;
                self.fail("the check stopped before it reported".to_string());
            }
        }
    }
}

/// The note under the model rows when the list is cut short; `None` when every row is shown.
fn rows_note(total: usize) -> Option<String> {
    (total > MAX_ROWS).then(|| format!("Showing the first {MAX_ROWS} of {total}."))
}

impl THMEApp {
    /// Draws the Welcome window while it is open. Its first draw starts the check.
    pub(super) fn render_welcome(&mut self, ctx: &egui::Context) {
        self.welcome.poll();
        if self.welcome.is_looking() {
            ctx.request_repaint();
        }
        if !self.welcome.open {
            return;
        }
        if self.welcome.is_idle() {
            self.welcome.refresh();
        }
        let p = self.palette();
        let mut open = self.welcome.open;
        let mut actions = Vec::new();
        egui::Window::new("Welcome to THME")
            .open(&mut open)
            .collapsible(false)
            .default_width(560.0)
            .show(ctx, |ui| draw_panel(ui, &self.welcome, &p, &mut actions));
        self.welcome.open = open;
        for action in actions {
            match action {
                Action::Refresh => self.welcome.refresh(),
                Action::Open(path) => {
                    let outcome = self.open_model_file(&path);
                    self.welcome.record_open(outcome);
                }
            }
        }
    }

    /// The header button: shows the window and checks again, unless a check is running.
    pub(super) fn show_welcome(&mut self) {
        self.welcome.open = true;
        self.welcome.refresh();
    }
}

fn draw_panel(
    ui: &mut egui::Ui,
    panel: &WelcomePanel,
    p: &ThemePalette,
    actions: &mut Vec<Action>,
) {
    ui.label(
        egui::RichText::new("First run: what this machine can do")
            .strong()
            .color(p.text_main),
    );
    ui.label(
        egui::RichText::new(
            "Read-only: nothing is downloaded, loaded or written. Nothing here is a quality or speed claim.",
        )
        .size(10.0)
        .color(p.text_muted),
    );
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!panel.is_looking(), egui::Button::new("Refresh"))
            .clicked()
        {
            actions.push(Action::Refresh);
        }
    });
    if let Some(message) = panel.message() {
        ui.label(egui::RichText::new(message).color(p.warning));
    }
    ui.separator();

    egui::ScrollArea::vertical()
        .max_height(480.0)
        .show(ui, |ui| match panel.phase() {
            Phase::Idle | Phase::Looking => {
                ui.label("Looking...");
            }
            Phase::Error(reason) => {
                ui.label(format!("Could not check this machine: {reason}"));
            }
            Phase::Ready(facts) => {
                // provenance: welcome-hardware
                let note = provenance::labelled("Checked facts", Evidence::Measured);
                ui.label(format!(
                    "{note}. Assumed facts come from the build, not a measurement."
                ));
                for line in doctor::fact_lines(facts) {
                    ui.label(line);
                }

                ui.add_space(8.0);
                // provenance: welcome-models
                let found = provenance::labelled(
                    format!("Models found: {}", facts.models.len()),
                    Evidence::Measured,
                );
                ui.label(found);
                if let Some(note) = rows_note(facts.models.len()) {
                    ui.label(note);
                }
                egui::Grid::new("welcome_models")
                    .striped(true)
                    .show(ui, |ui| {
                        for model in panel.visible_models() {
                            ui.label(model.name.as_str());
                            ui.label(model.source.label());
                            ui.label(model.format.label());
                            ui.label(format!("{} bytes", model.size_bytes));
                            if ui.button("Open").clicked() {
                                actions.push(Action::Open(model.path.clone()));
                            }
                            ui.end_row();
                        }
                    });

                ui.add_space(8.0);
                ui.label(egui::RichText::new("Next").strong());
                for step in doctor::next_steps(facts) {
                    ui.label(format!("- {step}"));
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::{DiscoveredModel, Format, Source};
    use crate::doctor::Fact;
    use std::path::PathBuf;

    fn model(i: usize) -> DiscoveredModel {
        DiscoveredModel {
            name: format!("m{i}"),
            source: Source::Folder,
            format: Format::Gguf,
            size_bytes: 1024,
            path: PathBuf::from("models").join(format!("m{i}.gguf")),
        }
    }

    fn facts(count: usize, gpu: Option<&str>) -> HostFacts {
        HostFacts {
            os: Fact::Assumed("windows".to_string()),
            arch: Fact::Assumed("x86_64".to_string()),
            memory_total_bytes: Fact::Checked(32 << 30),
            memory_available_bytes: Fact::Checked(16 << 30),
            disk_free_bytes: Fact::Unknown,
            gpu_adapter: gpu.map(str::to_string),
            cpu_threads: Fact::Checked(12),
            models: (0..count).map(model).collect(),
        }
    }

    #[test]
    fn starts_idle_then_looking_then_ready() {
        let mut p = WelcomePanel::default();
        assert_eq!(p.phase(), &Phase::Idle);
        assert!(p.begin());
        assert!(p.is_looking());
        p.finish(facts(2, None));
        assert!(matches!(p.phase(), Phase::Ready(f) if f.models.len() == 2));
    }

    #[test]
    fn a_second_refresh_while_looking_is_refused() {
        let mut p = WelcomePanel::default();
        assert!(p.begin());
        assert!(!p.begin());
        assert!(p.is_looking());
    }

    #[test]
    fn refresh_replaces_the_results_it_looked_at() {
        let mut p = WelcomePanel::default();
        p.begin();
        p.finish(facts(3, None));
        assert!(p.begin());
        assert!(
            p.visible_models().is_empty(),
            "old rows are hidden while looking"
        );
        p.finish(facts(1, Some("Test GPU")));
        let shown = p.facts().expect("ready");
        assert_eq!(shown.models.len(), 1);
        assert_eq!(shown.gpu_adapter.as_deref(), Some("Test GPU"));
    }

    #[test]
    fn a_failed_check_is_an_error_and_refresh_can_retry() {
        let mut p = WelcomePanel::default();
        p.begin();
        p.fail("the check stopped".to_string());
        assert_eq!(p.phase(), &Phase::Error("the check stopped".to_string()));
        assert!(p.begin());
    }

    #[test]
    fn at_most_ten_rows_and_the_count_is_shown() {
        let mut p = WelcomePanel::default();
        p.begin();
        p.finish(facts(12, None));
        assert_eq!(p.visible_models().len(), 10);
        assert_eq!(
            rows_note(12).as_deref(),
            Some("Showing the first 10 of 12.")
        );
    }

    #[test]
    fn a_short_list_is_shown_whole_without_a_note() {
        let mut p = WelcomePanel::default();
        p.begin();
        p.finish(facts(2, None));
        assert_eq!(p.visible_models().len(), 2);
        assert_eq!(rows_note(2), None);
        assert_eq!(rows_note(10), None);
    }

    #[test]
    fn refused_open_sets_a_visible_message_and_keeps_the_panel_open() {
        let mut p = WelcomePanel {
            open: true,
            ..Default::default()
        };
        p.record_open(Err("Failed to import model: refusing to load x".to_string()));
        assert!(p.open);
        assert!(p.message().is_some_and(|m| m.contains("refusing to load")));
    }

    #[test]
    fn a_real_refused_model_gives_the_refusal_text() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("extra.pkl"),
            b"cos\nsystem\n(S'echo fixture'\ntR.",
        )
        .unwrap();
        let outcome = UniversalModel::load_universal(dir.path()).map(|_| ());
        assert!(outcome.is_err());
        let mut p = WelcomePanel {
            open: true,
            ..Default::default()
        };
        p.record_open(outcome);
        assert!(p.open);
        assert!(p.message().is_some_and(|m| m.contains("refusing to load")));
    }

    #[test]
    fn a_successful_open_closes_the_panel() {
        let mut p = WelcomePanel {
            open: true,
            ..Default::default()
        };
        p.record_open(Ok(()));
        assert!(!p.open);
        assert!(p.message().is_some());
    }

    #[test]
    fn poll_takes_the_background_result() {
        let mut p = WelcomePanel::default();
        p.begin();
        let (tx, rx) = mpsc::channel();
        p.results = Some(rx);
        tx.send(facts(1, None)).unwrap();
        p.poll();
        assert!(matches!(p.phase(), Phase::Ready(_)));
        assert!(p.results.is_none());
    }

    #[test]
    fn poll_reports_a_check_that_stopped_early() {
        let mut p = WelcomePanel::default();
        p.begin();
        let (tx, rx) = mpsc::channel::<HostFacts>();
        p.results = Some(rx);
        drop(tx);
        p.poll();
        assert!(matches!(p.phase(), Phase::Error(_)));
    }
}
