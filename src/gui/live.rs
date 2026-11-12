//! Live 3D model view: what every layer and tensor of a running model is doing.
//!
//! A background worker runs the reference forward pass with the probe enabled and streams
//! one frame per evaluation window. The view draws each applied layer as a slab, each
//! linear tensor as a row of channel-group tiles lit by measured output magnitude and
//! tinted by its format family, token flow between layers as particles whose strength is
//! the measured residual update, and the router between layers. Anomalies (non-finite
//! values, exploding residuals, dead MLP units) turn tiles red and are listed beside the
//! view. egui has no 3D, so this is a small software renderer: orbit camera, perspective
//! projection, painter's-algorithm depth sort.

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use super::theme::ThemePalette;
use crate::inference::{
    read_token_file, synthetic_demo, ArchConfig, CausalLm, ForwardTrace, WeightPrecision,
};
use crate::ir::{ModelIR, TensorFormat, TensorRole};
use crate::parser::ModelImporter;

/// Representation families a tensor can belong to. Binary and phase families are part of
/// the THME research design but have no implementation yet; they appear in the legend
/// marked as such so the view never implies they are running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Dense,
    Int8,
    Int4,
    Ternary,
    Binary,
    Phase,
}

impl Family {
    pub fn of(format: &TensorFormat) -> Self {
        match format {
            TensorFormat::FP32
            | TensorFormat::FP16
            | TensorFormat::BF16
            | TensorFormat::Custom(_) => Self::Dense,
            TensorFormat::INT8 | TensorFormat::Q8_0 => Self::Int8,
            TensorFormat::Q4_0 | TensorFormat::Q4_K | TensorFormat::Q2_K => Self::Int4,
            TensorFormat::Ternary => Self::Ternary,
            TensorFormat::Bit1_5 => Self::Binary,
        }
    }

    fn from_dtype(dtype: &str) -> Self {
        match dtype {
            "Q8_0" | "Q8_1" | "Q8_K" | "I8" => Self::Int8,
            d if d.starts_with("Q4")
                || d.starts_with("Q5")
                || d.starts_with("Q2")
                || d.starts_with("Q3")
                || d.starts_with("IQ") =>
            {
                Self::Int4
            }
            "TQ1_0" | "TQ2_0" => Self::Ternary,
            _ => Self::Dense,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense (FP32/FP16/BF16)",
            Self::Int8 => "8-bit blocks",
            Self::Int4 => "4-bit and below",
            Self::Ternary => "Ternary {-1, 0, +1}",
            Self::Binary => "Binary (not implemented)",
            Self::Phase => "Phase (not implemented)",
        }
    }

    /// Categorical hues validated for dark surfaces; identity is also carried by the legend.
    fn color(self) -> Color32 {
        match self {
            Self::Dense => Color32::from_rgb(0x39, 0x87, 0xe5),
            Self::Int8 => Color32::from_rgb(0x19, 0x9e, 0x70),
            Self::Int4 => Color32::from_rgb(0xc9, 0x85, 0x00),
            Self::Ternary => Color32::from_rgb(0xd9, 0x59, 0x26),
            Self::Binary => Color32::from_rgb(0x90, 0x85, 0xe9),
            Self::Phase => Color32::from_rgb(0xd5, 0x51, 0x81),
        }
    }
}

const ANOMALY: Color32 = Color32::from_rgb(0xd0, 0x3b, 0x3b);
const ALL_FAMILIES: [Family; 6] = [
    Family::Dense,
    Family::Int8,
    Family::Int4,
    Family::Ternary,
    Family::Binary,
    Family::Phase,
];

/// Per stored layer: `(tensor short name, family)`.
pub type Composition = Vec<Vec<(String, Family)>>;

pub struct LiveFrame {
    pub window: usize,
    pub total: Option<usize>,
    pub running_ppl: f64,
    pub tokens_per_sec: f64,
    pub trace: ForwardTrace,
}

pub enum LiveMessage {
    Started {
        label: String,
        composition: Composition,
        perplexity_meaningful: bool,
    },
    Frame(Box<LiveFrame>),
    Finished(String),
    Failed(String),
}

enum Source {
    Demo,
    Model {
        dir: PathBuf,
        tokens: Option<PathBuf>,
    },
}

struct Session {
    rx: Receiver<LiveMessage>,
    stop: Arc<AtomicBool>,
}

#[derive(Clone, Copy)]
struct Hover {
    step: usize,
    tensor: usize,
    bin: usize,
}

pub struct LiveView {
    model_dir: String,
    tokens_path: String,
    context: usize,
    session: Option<Session>,
    status: String,
    label: String,
    perplexity_meaningful: bool,
    composition: Composition,
    latest: Option<Box<LiveFrame>>,
    ppl_history: VecDeque<f64>,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    started_demo: bool,
    epoch: Instant,
}

impl Default for LiveView {
    fn default() -> Self {
        Self {
            model_dir: "models/Pythia-14M".to_string(),
            tokens_path: String::new(),
            context: 256,
            session: None,
            status: "Idle".to_string(),
            label: String::new(),
            perplexity_meaningful: false,
            composition: Vec::new(),
            latest: None,
            ppl_history: VecDeque::new(),
            yaw: 0.65,
            pitch: 0.38,
            zoom: 1.0,
            started_demo: false,
            epoch: Instant::now(),
        }
    }
}

impl LiveView {
    fn start(&mut self, ctx: &egui::Context, source: Source) {
        self.stop();
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let ctx = ctx.clone();
        let context = self.context;
        std::thread::spawn(move || {
            let result = run_worker(source, &ctx, &tx, &worker_stop, context);
            let message = match result {
                Ok(summary) => LiveMessage::Finished(summary),
                Err(error) => LiveMessage::Failed(format!("{error:#}")),
            };
            let _ = tx.send(message);
            ctx.request_repaint();
        });
        self.session = Some(Session { rx, stop });
        self.latest = None;
        self.ppl_history.clear();
        self.status = "Loading...".to_string();
    }

    fn stop(&mut self) {
        if let Some(session) = self.session.take() {
            session.stop.store(true, Ordering::Relaxed);
        }
    }

    fn poll(&mut self) {
        let Some(session) = &self.session else {
            return;
        };
        let mut finished = false;
        while let Ok(message) = session.rx.try_recv() {
            match message {
                LiveMessage::Started {
                    label,
                    composition,
                    perplexity_meaningful,
                } => {
                    self.label = label;
                    self.composition = composition;
                    self.perplexity_meaningful = perplexity_meaningful;
                    self.status = "Running".to_string();
                }
                LiveMessage::Frame(frame) => {
                    self.ppl_history.push_back(frame.running_ppl);
                    if self.ppl_history.len() > 240 {
                        self.ppl_history.pop_front();
                    }
                    self.latest = Some(frame);
                }
                LiveMessage::Finished(summary) => {
                    self.status = summary;
                    finished = true;
                }
                LiveMessage::Failed(error) => {
                    self.status = format!("Error: {error}");
                    finished = true;
                }
            }
        }
        if finished {
            self.session = None;
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, p: &ThemePalette) {
        let ctx = ui.ctx().clone();
        if !self.started_demo && self.session.is_none() && self.latest.is_none() {
            self.started_demo = true;
            self.start(&ctx, Source::Demo);
        }
        self.poll();
        ctx.request_repaint();

        ui.heading(
            egui::RichText::new("LIVE 3D MODEL VIEW")
                .size(13.0)
                .color(p.text_main),
        );
        ui.label(
            egui::RichText::new(
                "Each slab is one applied layer; each row is a linear tensor split into output-channel groups, lit by the measured output magnitude of the current window and tinted by format family. Particles carry the residual stream between layers, brighter where a layer rewrites it more. Red marks anomalies. Drag to orbit, scroll to zoom, hover a tile to inspect it.",
            )
            .size(10.0)
            .color(p.text_muted),
        );
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Model folder:");
            ui.add(egui::TextEdit::singleline(&mut self.model_dir).desired_width(220.0));
            ui.label("Token file (optional):");
            ui.add(egui::TextEdit::singleline(&mut self.tokens_path).desired_width(260.0));
            ui.add(egui::Slider::new(&mut self.context, 32..=1024).text("tokens/window"));
            if ui.button("Run model").clicked() {
                let tokens = (!self.tokens_path.trim().is_empty())
                    .then(|| PathBuf::from(self.tokens_path.trim()));
                self.start(
                    &ctx,
                    Source::Model {
                        dir: PathBuf::from(self.model_dir.trim()),
                        tokens,
                    },
                );
            }
            if ui.button("Run synthetic demo").clicked() {
                self.start(&ctx, Source::Demo);
            }
            if self.session.is_some() && ui.button("Stop").clicked() {
                self.stop();
                self.status = "Stopped".to_string();
            }
        });
        ui.add_space(4.0);

        let side = 300.0;
        let available = ui.available_size();
        ui.horizontal(|ui| {
            let view_size = Vec2::new(
                (available.x - side - 12.0).max(300.0),
                available.y.max(360.0),
            );
            self.draw_scene(ui, view_size, p);
            ui.vertical(|ui| {
                ui.set_width(side);
                self.draw_side_panel(ui, p);
            });
        });
    }

    fn draw_side_panel(&self, ui: &mut egui::Ui, p: &ThemePalette) {
        let strong = |t: &str| egui::RichText::new(t).strong().color(p.text_main);
        let muted = |t: String| egui::RichText::new(t).size(10.0).color(p.text_muted);
        ui.label(strong("Source"));
        ui.label(muted(if self.label.is_empty() {
            "-".to_string()
        } else {
            self.label.clone()
        }));
        ui.label(muted(format!("Status: {}", self.status)));
        ui.separator();
        if let Some(frame) = &self.latest {
            let ppl = if self.perplexity_meaningful {
                format!("{:.3}", frame.running_ppl)
            } else {
                format!("{:.3} (not meaningful for this source)", frame.running_ppl)
            };
            ui.label(strong("Running perplexity"));
            ui.label(egui::RichText::new(ppl).size(16.0).color(p.text_main));
            let windows = match frame.total {
                Some(total) => format!("window {}/{}", frame.window, total),
                None => format!("window {}", frame.window),
            };
            ui.label(muted(format!(
                "{windows} · {} tokens · {:.0} tokens/s",
                frame.trace.tokens, frame.tokens_per_sec
            )));
            ui.add_space(4.0);
            ui.label(strong("Residual update per layer (sqrt scale)"));
            draw_update_bars(ui, &frame.trace, p);
            ui.add_space(4.0);
            let anomalies = frame.trace.anomalies();
            ui.label(strong(&format!("Anomalies ({})", anomalies.len())));
            if anomalies.is_empty() {
                ui.label(muted("None in this window".to_string()));
            }
            for anomaly in anomalies.iter().take(8) {
                ui.label(
                    egui::RichText::new(format!("! {anomaly}"))
                        .size(10.0)
                        .color(ANOMALY),
                );
            }
        } else {
            ui.label(muted("Waiting for the first window...".to_string()));
        }
        ui.separator();
        ui.label(strong("Format families"));
        for family in ALL_FAMILIES {
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                ui.painter()
                    .rect_filled(rect, egui::CornerRadius::same(2), family.color());
                ui.label(muted(family.label().to_string()));
            });
        }
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
            ui.painter()
                .rect_filled(rect, egui::CornerRadius::same(2), ANOMALY);
            ui.label(muted("Anomaly in this window".to_string()));
        });
        ui.add_space(4.0);
        ui.label(muted(
            "Router: layers run in sequence. Adaptive routing and looping between layers are planned (v0.2.3); a schedule that repeats a layer is drawn as a loop."
                .to_string(),
        ));
    }

    fn draw_scene(&mut self, ui: &mut egui::Ui, size: Vec2, p: &ThemePalette) {
        let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, egui::CornerRadius::same(6), p.bg_dark);
        painter.rect_stroke(
            rect,
            egui::CornerRadius::same(6),
            Stroke::new(1.0_f32, p.border),
            egui::StrokeKind::Inside,
        );

        let drag = response.drag_delta();
        self.yaw += drag.x * 0.008;
        self.pitch = (self.pitch + drag.y * 0.008).clamp(-1.2, 1.2);
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.3, 4.0);
        }

        let Some(frame) = &self.latest else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Waiting for the first window...",
                egui::FontId::proportional(13.0),
                p.text_muted,
            );
            return;
        };
        let trace = &frame.trace;
        let steps = trace.layers.len();
        if steps == 0 {
            return;
        }
        let rows = trace
            .layers
            .iter()
            .map(|l| l.tensors.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let spacing = 1.25f32;
        let row_h = 0.42f32;
        let half_w = 2.0f32;
        let camera = Camera {
            yaw: self.yaw,
            pitch: self.pitch,
            dist: (3.5 + steps as f32 * spacing * 0.75) / self.zoom,
            rect,
        };
        let z_of = |step: usize| (step as f32 - (steps as f32 - 1.0) / 2.0) * spacing;
        let y_of = |row: usize| ((rows as f32 - 1.0) / 2.0 - row as f32) * row_h;

        // Normalize activity per tensor role across all applied layers.
        let mut peak: Vec<(&str, f32)> = Vec::new();
        for layer in &trace.layers {
            for tensor in &layer.tensors {
                let max = tensor.bins.iter().cloned().fold(0.0f32, f32::max);
                match peak.iter_mut().find(|(n, _)| *n == tensor.name) {
                    Some(entry) => entry.1 = entry.1.max(max),
                    None => peak.push((tensor.name, max)),
                }
            }
        }
        let peak_of = |name: &str| {
            peak.iter()
                .find(|(n, _)| *n == name)
                .map_or(1.0, |(_, v)| v.max(1e-12))
        };

        let mut quads: Vec<(f32, [Pos2; 4], Color32, Option<Hover>)> = Vec::new();
        for (step, layer) in trace.layers.iter().enumerate() {
            let z = z_of(step);
            let composition = self.composition.get(layer.layer);
            // Slab backplate.
            let pad = 0.12;
            let top = y_of(0) + row_h / 2.0 + pad;
            let bottom = y_of(rows - 1) - row_h / 2.0 - pad;
            if let Some((depth, pts)) = camera.quad([
                [-half_w - pad, top, z],
                [half_w + pad, top, z],
                [half_w + pad, bottom, z],
                [-half_w - pad, bottom, z],
            ]) {
                quads.push((depth + 0.01, pts, with_alpha(p.bg_card, 200), None));
            }
            for (row, tensor) in layer.tensors.iter().enumerate() {
                let family = composition
                    .and_then(|c| c.iter().find(|(n, _)| n == tensor.name))
                    .map_or(Family::Dense, |(_, f)| *f);
                let base = family.color();
                let dim = lerp(p.bg_dark, base, 0.18);
                let y = y_of(row);
                let bins = tensor.bins.len().max(1);
                let w = 2.0 * half_w / bins as f32;
                let scale = peak_of(tensor.name);
                for (bin, &value) in tensor.bins.iter().enumerate() {
                    let x0 = -half_w + bin as f32 * w + 0.01;
                    let x1 = x0 + w - 0.02;
                    let color = if tensor.nonfinite > 0 {
                        ANOMALY
                    } else {
                        let t = (value / scale).clamp(0.0, 1.0).sqrt();
                        lerp(dim, base, 0.2 + 0.8 * t)
                    };
                    if let Some((depth, pts)) = camera.quad([
                        [x0, y + row_h * 0.38, z],
                        [x1, y + row_h * 0.38, z],
                        [x1, y - row_h * 0.38, z],
                        [x0, y - row_h * 0.38, z],
                    ]) {
                        quads.push((
                            depth,
                            pts,
                            color,
                            Some(Hover {
                                step,
                                tensor: row,
                                bin,
                            }),
                        ));
                    }
                }
            }
        }
        quads.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, pts, color, _) in &quads {
            painter.add(egui::Shape::convex_polygon(
                pts.to_vec(),
                *color,
                Stroke::NONE,
            ));
        }

        // Layer labels and routers between layers.
        let max_update = trace
            .layers
            .iter()
            .map(|l| l.update_ratio)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        for (step, layer) in trace.layers.iter().enumerate() {
            let z = z_of(step);
            if let Some((pos, _)) = camera.project([-half_w - 0.2, y_of(0) + row_h, z]) {
                let anomalous = layer.tensors.iter().any(|t| t.nonfinite > 0)
                    || layer.mlp_dead_fraction > 0.95
                    || !layer.resid_rms_out.is_finite();
                painter.text(
                    pos,
                    egui::Align2::RIGHT_BOTTOM,
                    format!("L{}", layer.layer),
                    egui::FontId::monospace(10.0),
                    if anomalous { ANOMALY } else { p.text_muted },
                );
            }
            if step + 1 < steps {
                let next = &trace.layers[step + 1];
                let zm = z + spacing / 2.0;
                let router = [-half_w - 0.45, 0.0, zm];
                if let (Some((c, _)), true) = (camera.project(router), true) {
                    let r = 5.0;
                    let diamond = vec![
                        c + Vec2::new(0.0, -r),
                        c + Vec2::new(r, 0.0),
                        c + Vec2::new(0.0, r),
                        c + Vec2::new(-r, 0.0),
                    ];
                    painter.add(egui::Shape::convex_polygon(
                        diamond,
                        p.text_muted,
                        Stroke::NONE,
                    ));
                }
                if next.layer <= layer.layer {
                    // A schedule that revisits a layer: draw the loop back.
                    let pts: Vec<Pos2> = (0..=12)
                        .filter_map(|i| {
                            let s = i as f32 / 12.0;
                            let zz = z_of(step + 1) + (z - z_of(step + 1)) * s;
                            let xx = -half_w - 0.45 - 0.6 * (std::f32::consts::PI * s).sin();
                            camera.project([xx, 0.0, zz]).map(|(p, _)| p)
                        })
                        .collect();
                    painter.add(egui::Shape::line(pts, Stroke::new(2.0_f32, p.warning)));
                }
                // Token flow: particles from this layer into the next, strength = measured update.
                let strength = (next.update_ratio / max_update).clamp(0.0, 1.0).sqrt();
                let time = self.epoch.elapsed().as_secs_f32();
                for j in 0..14 {
                    let h1 = hash01(j * 2 + 1 + step * 31);
                    let h2 = hash01(j * 2 + 2 + step * 31);
                    let phase = (time * 0.55 + j as f32 / 14.0).fract();
                    let world = [
                        -half_w + 0.2 + h1 * (2.0 * half_w - 0.4),
                        y_of(rows - 1) + h2 * (y_of(0) - y_of(rows - 1)),
                        z + phase * spacing,
                    ];
                    if let Some((pos, _)) = camera.project(world) {
                        let alpha = (60.0 + 195.0 * strength) as u8;
                        painter.circle_filled(
                            pos,
                            1.5 + 2.5 * strength,
                            with_alpha(p.accent, alpha),
                        );
                    }
                }
            }
        }

        // Hover inspection: the front-most tile under the pointer.
        if let Some(pointer) = response.hover_pos() {
            let hit = quads
                .iter()
                .rev()
                .find(|(_, pts, _, info)| info.is_some() && inside(pointer, pts))
                .and_then(|(_, _, _, info)| *info);
            if let Some(h) = hit {
                if let Some(layer) = trace.layers.get(h.step) {
                    if let Some(tensor) = layer.tensors.get(h.tensor) {
                        let bins = tensor.bins.len().max(1);
                        let lo = h.bin * tensor.out_dim / bins;
                        let hi = ((h.bin + 1) * tensor.out_dim / bins).max(lo + 1);
                        let family = self
                            .composition
                            .get(layer.layer)
                            .and_then(|c| c.iter().find(|(n, _)| n == tensor.name))
                            .map_or(Family::Dense, |(_, f)| *f);
                        let lines = [
                            format!("Layer {} · tensor {} · {}", layer.layer, tensor.name, family.label()),
                            format!("Output channels {lo}..{hi} of {}", tensor.out_dim),
                            format!("Mean |output| here: {:.4}", tensor.bins.get(h.bin).copied().unwrap_or(0.0)),
                            format!("Tensor mean |output| {:.4}, max {:.4}", tensor.mean_abs, tensor.max_abs),
                            format!(
                                "Layer update {:.3} · residual rms {:.3} -> {:.3} · MLP inactive {:.0}%",
                                layer.update_ratio,
                                layer.resid_rms_in,
                                layer.resid_rms_out,
                                100.0 * layer.mlp_dead_fraction
                            ),
                        ];
                        draw_tooltip(&painter, rect, pointer, &lines, p);
                    }
                }
            }
        }
    }
}

impl Drop for LiveView {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run_worker(
    source: Source,
    ctx: &egui::Context,
    tx: &Sender<LiveMessage>,
    stop: &AtomicBool,
    context: usize,
) -> anyhow::Result<String> {
    let context = context.max(2);
    let (model, composition, label, tokens, meaningful, repeat): (
        Box<dyn CausalLm>,
        Composition,
        String,
        Vec<u32>,
        bool,
        bool,
    ) = match source {
        Source::Demo => {
            let (model, composition) = synthetic_demo(7);
            let composition = composition
                .into_iter()
                .map(|layer| {
                    layer
                        .into_iter()
                        .map(|(name, format)| (name.to_string(), Family::of(&format)))
                        .collect()
                })
                .collect();
            let tokens = pseudo_random_tokens(model.vocab_size(), 4096, 11);
            (
                Box::new(model),
                composition,
                "Synthetic demo: 8 layers with pseudo-random weights and a mixed format composition. Activity is real computation; predictions are meaningless.".to_string(),
                tokens,
                false,
                true,
            )
        }
        Source::Model { dir, tokens } => {
            let ir = ModelImporter::import(&dir)
                .map_err(|e| anyhow::anyhow!("Failed to import {dir:?}: {e:#}"))?;
            let arch = ArchConfig::detect(&ir, &dir)?;
            let model = arch.load(&ir, WeightPrecision::AsStored)?;
            let composition = composition_from_ir(&ir);
            let (tokens, meaningful, what) = match tokens {
                Some(path) => (read_token_file(&path)?, true, "evaluation tokens"),
                None => (
                    pseudo_random_tokens(model.vocab_size(), 8192, 3),
                    false,
                    "random tokens (no token file given; perplexity is not meaningful)",
                ),
            };
            let label = format!(
                "{} ({}), {} layers, {what}",
                ir.name,
                arch.family(),
                model.num_layers()
            );
            (model, composition, label, tokens, meaningful, false)
        }
    };
    let _ = tx.send(LiveMessage::Started {
        label,
        composition,
        perplexity_meaningful: meaningful,
    });
    let schedule: Vec<usize> = (0..model.num_layers()).collect();
    let windows: Vec<&[u32]> = tokens.chunks(context).filter(|w| w.len() >= 2).collect();
    if windows.is_empty() {
        anyhow::bail!("Not enough tokens for a single window");
    }
    let (mut total_nll, mut scored, mut index) = (0.0f64, 0usize, 0usize);
    let started = Instant::now();
    loop {
        for window in &windows {
            if stop.load(Ordering::Relaxed) {
                return Ok("Stopped".to_string());
            }
            let (exit, trace) = model.forward_traced(window, &schedule)?;
            total_nll += exit.nll.iter().map(|&v| f64::from(v)).sum::<f64>();
            scored += exit.nll.len();
            index += 1;
            let frame = LiveFrame {
                window: index,
                total: (!repeat).then_some(windows.len()),
                running_ppl: (total_nll / scored.max(1) as f64).exp(),
                tokens_per_sec: scored as f64 / started.elapsed().as_secs_f64().max(1e-9),
                trace,
            };
            if tx.send(LiveMessage::Frame(Box::new(frame))).is_err() {
                return Ok("Viewer closed".to_string());
            }
            ctx.request_repaint();
            if repeat {
                std::thread::sleep(std::time::Duration::from_millis(60));
            }
        }
        if !repeat {
            return Ok(format!(
                "Finished: {index} windows, perplexity {:.4}",
                (total_nll / scored.max(1) as f64).exp()
            ));
        }
    }
}

fn composition_from_ir(ir: &ModelIR) -> Composition {
    let mut layers: Composition = vec![Vec::new(); ir.num_layers];
    for tensor in &ir.tensors {
        let (Some(layer), true) = (tensor.layer_index, tensor.shape.len() == 2) else {
            continue;
        };
        let name = match tensor.role {
            TensorRole::AttentionQ => "q",
            TensorRole::AttentionK => "k",
            TensorRole::AttentionV => "v",
            TensorRole::AttentionOut => "o",
            TensorRole::AttentionQkv => "qkv",
            TensorRole::MlpGate => "gate",
            TensorRole::MlpUp => "up",
            TensorRole::MlpDown => "down",
            _ => continue,
        };
        if let Some(slot) = layers.get_mut(layer) {
            slot.push((name.to_string(), Family::from_dtype(&tensor.original_dtype)));
        }
    }
    layers
}

fn pseudo_random_tokens(vocab: usize, count: usize, seed: u64) -> Vec<u32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % vocab.max(1) as u64) as u32
        })
        .collect()
}

struct Camera {
    yaw: f32,
    pitch: f32,
    dist: f32,
    rect: Rect,
}

impl Camera {
    /// Projects a world point; returns the screen position and camera-space depth.
    fn project(&self, p: [f32; 3]) -> Option<(Pos2, f32)> {
        let (sy, cy) = self.yaw.sin_cos();
        let x1 = p[0] * cy - p[2] * sy;
        let z1 = p[0] * sy + p[2] * cy;
        let (sp, cp) = self.pitch.sin_cos();
        let y2 = p[1] * cp - z1 * sp;
        let z2 = p[1] * sp + z1 * cp + self.dist;
        if z2 < 0.2 {
            return None;
        }
        let f = self.rect.height() * 1.15;
        Some((
            self.rect.center() + Vec2::new(x1 * f / z2, -y2 * f / z2),
            z2,
        ))
    }

    fn quad(&self, corners: [[f32; 3]; 4]) -> Option<(f32, [Pos2; 4])> {
        let mut pts = [Pos2::ZERO; 4];
        let mut depth = 0.0;
        for (slot, corner) in pts.iter_mut().zip(corners) {
            let (pos, z) = self.project(corner)?;
            *slot = pos;
            depth += z / 4.0;
        }
        Some((depth, pts))
    }
}

fn inside(point: Pos2, pts: &[Pos2; 4]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = pts[i];
        let b = pts[(i + 1) % 4];
        let cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
        if cross != 0.0 {
            if sign != 0.0 && cross.signum() != sign {
                return false;
            }
            sign = cross.signum();
        }
    }
    true
}

fn draw_update_bars(ui: &mut egui::Ui, trace: &ForwardTrace, p: &ThemePalette) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 70.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.line_segment(
        [rect.left_bottom(), rect.right_bottom()],
        Stroke::new(1.0_f32, p.border),
    );
    let n = trace.layers.len().max(1);
    let max = trace
        .layers
        .iter()
        .map(|l| l.update_ratio)
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let slot = rect.width() / n as f32;
    for (i, layer) in trace.layers.iter().enumerate() {
        let h = (rect.height() - 4.0) * (layer.update_ratio / max).clamp(0.0, 1.0).sqrt();
        let x = rect.left() + i as f32 * slot + slot * 0.2;
        let bar = Rect::from_min_max(
            Pos2::new(x, rect.bottom() - h),
            Pos2::new(x + (slot * 0.6).max(1.0), rect.bottom()),
        );
        painter.rect_filled(bar, egui::CornerRadius::same(1), p.primary);
    }
}

fn draw_tooltip(
    painter: &egui::Painter,
    area: Rect,
    pointer: Pos2,
    lines: &[String],
    p: &ThemePalette,
) {
    let font = egui::FontId::proportional(11.0);
    let width = 330.0;
    let height = 8.0 + lines.len() as f32 * 15.0;
    let mut origin = pointer + Vec2::new(14.0, 14.0);
    if origin.x + width > area.right() {
        origin.x = pointer.x - width - 14.0;
    }
    if origin.y + height > area.bottom() {
        origin.y = pointer.y - height - 14.0;
    }
    let rect = Rect::from_min_size(origin, Vec2::new(width, height));
    painter.rect_filled(rect, egui::CornerRadius::same(4), p.bg_card);
    painter.rect_stroke(
        rect,
        egui::CornerRadius::same(4),
        Stroke::new(1.0_f32, p.border),
        egui::StrokeKind::Inside,
    );
    for (i, line) in lines.iter().enumerate() {
        painter.text(
            origin + Vec2::new(8.0, 4.0 + i as f32 * 15.0),
            egui::Align2::LEFT_TOP,
            line,
            font.clone(),
            if i == 0 { p.text_main } else { p.text_muted },
        );
    }
}

fn lerp(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

fn with_alpha(c: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha)
}

fn hash01(i: usize) -> f32 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    (x % 10_000) as f32 / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_cover_formats_and_dtypes() {
        assert_eq!(Family::of(&TensorFormat::Ternary), Family::Ternary);
        assert_eq!(Family::of(&TensorFormat::Q8_0), Family::Int8);
        assert_eq!(Family::of(&TensorFormat::BF16), Family::Dense);
        assert_eq!(Family::from_dtype("Q4_K"), Family::Int4);
        assert_eq!(Family::from_dtype("TQ1_0"), Family::Ternary);
        assert_eq!(Family::from_dtype("F16"), Family::Dense);
    }

    #[test]
    fn camera_projects_points_in_front_and_point_in_quad_works() {
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            dist: 5.0,
            rect: Rect::from_min_size(Pos2::ZERO, Vec2::new(200.0, 100.0)),
        };
        let (center, depth) = camera.project([0.0, 0.0, 0.0]).unwrap();
        assert_eq!(center, Pos2::new(100.0, 50.0));
        assert_eq!(depth, 5.0);
        assert!(camera.project([0.0, 0.0, -10.0]).is_none());
        let (_, quad) = camera
            .quad([
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, -1.0, 0.0],
            ])
            .unwrap();
        assert!(inside(center, &quad));
        assert!(!inside(Pos2::new(0.0, 0.0), &quad));
    }

    #[test]
    fn demo_worker_streams_frames_until_stopped() {
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let handle =
            std::thread::spawn(move || run_worker(Source::Demo, &ctx, &tx, &worker_stop, 64));
        let mut frames = 0;
        while frames < 2 {
            match rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(LiveMessage::Frame(frame)) => {
                    assert_eq!(frame.trace.layers.len(), 8);
                    frames += 1;
                }
                Ok(_) => {}
                Err(error) => panic!("no frame from demo worker: {error}"),
            }
        }
        stop.store(true, Ordering::Relaxed);
        assert_eq!(handle.join().unwrap().unwrap(), "Stopped");
    }
}
