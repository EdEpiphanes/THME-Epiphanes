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

use super::manga::{self, calabi_yau_patches, ManifoldPatch};
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
            TensorFormat::INT8 | TensorFormat::Q8_0 | TensorFormat::Q5_K | TensorFormat::Q6_K => {
                Self::Int8
            }
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
            Self::Int8 => "5 to 8-bit blocks",
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

/// Anomaly marker: red in colour themes, solid ink in the black-and-white panel theme.
fn anomaly(p: &ThemePalette) -> Color32 {
    if p.is_mono() {
        manga::ink(p)
    } else {
        ANOMALY
    }
}

impl Family {
    /// Family colour for the active theme. Black-and-white themes use distinct grey tones
    /// (and, in the manifold, screentone densities) instead of hue.
    fn color_for(self, p: &ThemePalette) -> Color32 {
        if !p.is_mono() {
            return self.color();
        }
        Color32::from_gray(match self {
            Self::Dense => 46,
            Self::Int8 => 92,
            Self::Int4 => 132,
            Self::Ternary => 168,
            Self::Binary => 196,
            Self::Phase => 220,
        })
    }

    /// Screentone dot radius used for this family on the manifold in mono themes.
    fn tone_radius(self) -> f32 {
        match self {
            Self::Dense => 0.0,
            Self::Int8 => 0.7,
            Self::Int4 => 1.0,
            Self::Ternary => 1.3,
            Self::Binary => 1.6,
            Self::Phase => 1.9,
        }
    }
}

/// The family most tensors of a layer use (Dense when unknown).
fn dominant_family(entries: Option<&Vec<(String, Family)>>) -> Family {
    let Some(entries) = entries else {
        return Family::Dense;
    };
    let mut best = (Family::Dense, 0usize);
    for family in ALL_FAMILIES {
        let count = entries.iter().filter(|(_, f)| *f == family).count();
        if count > best.1 {
            best = (family, count);
        }
    }
    best.0
}

/// True when the screen quad turns the same way at every corner.
fn is_convex(pts: &[Pos2; 4]) -> bool {
    let mut sign = 0.0f32;
    for k in 0..4 {
        let (a, b, c) = (pts[k], pts[(k + 1) % 4], pts[(k + 2) % 4]);
        let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        if cross.abs() > 1e-3 {
            if sign != 0.0 && cross.signum() != sign {
                return false;
            }
            sign = cross.signum();
        }
    }
    true
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / len, v[1] / len, v[2] / len]
}

struct Face {
    depth: f32,
    pts: [Pos2; 4],
    fill: Color32,
    dots: f32,
    glint: bool,
    center: Pos2,
    patch: usize,
}

/// Draws the Calabi-Yau manifold: manga cel shading (three tones), ink linework,
/// screentone in shadow, a glass rim light and specular glints, and token particles.
#[allow(clippy::too_many_arguments)]
fn draw_manifold(
    painter: &egui::Painter,
    pointer: Option<Pos2>,
    camera: &Camera,
    patches: &[ManifoldPatch],
    trace: &ForwardTrace,
    composition: &Composition,
    p: &ThemePalette,
    time: f32,
) {
    let n_layers = trace.layers.len();
    if n_layers == 0 || patches.is_empty() {
        return;
    }
    let mono = p.is_mono();
    let ink = manga::ink(p);
    let paper = if p.is_light() {
        Color32::WHITE
    } else {
        Color32::from_gray(245)
    };
    let max_update = trace
        .layers
        .iter()
        .map(|l| l.update_ratio)
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let light = normalize3([-0.45, 0.75, -0.55]);
    let scale = 1.32f32;

    let mut faces: Vec<Face> = Vec::with_capacity(patches.len() * 128);
    for (index, patch) in patches.iter().enumerate() {
        let layer = &trace.layers[index % n_layers];
        let activity = (layer.update_ratio / max_update).clamp(0.0, 1.0).sqrt();
        let anomalous =
            layer.tensors.iter().any(|t| t.nonfinite > 0) || !layer.resid_rms_out.is_finite();
        let family = dominant_family(composition.get(layer.layer));
        let base = if anomalous {
            anomaly(p)
        } else {
            family.color_for(p)
        };
        // Classic manga in black-and-white themes: paper highlights, grey tone by family and
        // activity, dark shadow. Colour themes keep the family hue.
        let (lit, mid, shade) = if mono && !anomalous {
            (
                paper,
                manga::mix(paper, base, 0.18 + 0.32 * activity),
                manga::mix(paper, base, 0.72),
            )
        } else {
            (
                manga::mix(base, paper, 0.45 + 0.25 * activity),
                manga::mix(
                    manga::mix(base, p.bg_dark, 0.35),
                    base,
                    0.35 + 0.65 * activity,
                ),
                manga::mix(base, ink, 0.62),
            )
        };
        for j in 0..patch.nv - 1 {
            for i in 0..patch.nu - 1 {
                let at = |i: usize, j: usize| {
                    let q = patch.points[j * patch.nu + i];
                    camera.view([q[0] * scale, q[2] * scale, q[1] * scale])
                };
                let v = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
                if v.iter().any(|q| q[2] < 0.25) {
                    continue;
                }
                let mut n = normalize3(cross3(sub3(v[1], v[0]), sub3(v[3], v[0])));
                if n[2] > 0.0 {
                    n = [-n[0], -n[1], -n[2]];
                }
                let diffuse = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).max(0.0);
                let facing = (-n[2]).clamp(0.0, 1.0);
                let pts = v.map(|q| camera.screen(q));
                // Skip slivers: near the manifold's pinch points faces collapse, and stroking a
                // near-degenerate polygon makes egui's miter joins shoot out as long spikes.
                let area = 0.5
                    * ((pts[0].x * pts[1].y - pts[1].x * pts[0].y)
                        + (pts[1].x * pts[2].y - pts[2].x * pts[1].y)
                        + (pts[2].x * pts[3].y - pts[3].x * pts[2].y)
                        + (pts[3].x * pts[0].y - pts[0].x * pts[3].y))
                        .abs();
                let perimeter: f32 = (0..4).map(|k| pts[k].distance(pts[(k + 1) % 4])).sum();
                if area < 2.0 || area < 0.02 * perimeter * perimeter {
                    continue;
                }
                let center = Pos2::new(
                    (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4.0,
                    (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4.0,
                );
                // Cel shading: three flat tones, plus a glass rim where the surface turns away.
                let (mut fill, mut dots) = if diffuse > 0.72 {
                    (lit, 0.0)
                } else if diffuse > 0.32 {
                    (mid, if mono { family.tone_radius() } else { 0.0 })
                } else {
                    (
                        shade,
                        1.1 + if mono {
                            family.tone_radius() * 0.5
                        } else {
                            0.0
                        },
                    )
                };
                if facing < 0.22 {
                    fill = manga::mix(fill, paper, 0.55);
                    dots = 0.0;
                }
                faces.push(Face {
                    depth: (v[0][2] + v[1][2] + v[2][2] + v[3][2]) / 4.0,
                    pts,
                    fill: manga::with_alpha(fill, 236),
                    dots,
                    glint: diffuse > 0.95 && facing > 0.6,
                    center,
                    patch: index,
                });
            }
        }
    }
    faces.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    let line = Stroke::new(0.8_f32, manga::with_alpha(ink, 190));
    for face in &faces {
        // Fill without a polygon stroke, then ink the edges as plain segments: segments have
        // no miter joins, so collapsed faces cannot throw spikes across the view.
        if is_convex(&face.pts) {
            painter.add(egui::Shape::convex_polygon(
                face.pts.to_vec(),
                face.fill,
                Stroke::NONE,
            ));
        } else {
            for tri in [
                [face.pts[0], face.pts[1], face.pts[2]],
                [face.pts[0], face.pts[2], face.pts[3]],
            ] {
                painter.add(egui::Shape::convex_polygon(
                    tri.to_vec(),
                    face.fill,
                    Stroke::NONE,
                ));
            }
        }
        for k in 0..4 {
            painter.line_segment([face.pts[k], face.pts[(k + 1) % 4]], line);
        }
        if face.dots > 0.0 {
            painter.circle_filled(face.center, face.dots, manga::with_alpha(ink, 210));
        }
        if face.glint {
            painter.circle_filled(face.center, 1.8, manga::with_alpha(Color32::WHITE, 235));
        }
    }

    // Token particles riding each patch, brighter on layers that rewrite the stream more.
    let glow = if mono { ink } else { p.accent };
    for (index, patch) in patches.iter().enumerate() {
        let layer = &trace.layers[index % n_layers];
        let activity = (layer.update_ratio / max_update).clamp(0.0, 1.0).sqrt();
        for k in 0..2 {
            let phase = (time * 0.32 + hash01(index * 5 + k)).fract();
            let fu = hash01(index * 11 + k + 3) * (patch.nu - 1) as f32;
            let fv = phase * (patch.nv - 1) as f32;
            let (i, j) = (fu as usize, fv as usize);
            let (i1, j1) = ((i + 1).min(patch.nu - 1), (j + 1).min(patch.nv - 1));
            let (tu, tv) = (fu.fract(), fv.fract());
            let q = |i: usize, j: usize| patch.points[j * patch.nu + i];
            let blend = |a: [f32; 3], b: [f32; 3], t: f32| {
                [
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ]
            };
            let w = blend(
                blend(q(i, j), q(i1, j), tu),
                blend(q(i, j1), q(i1, j1), tu),
                tv,
            );
            if let Some((pos, _)) = camera.project([w[0] * scale, w[2] * scale, w[1] * scale]) {
                painter.circle_filled(
                    pos,
                    5.0,
                    manga::with_alpha(glow, (40.0 + 90.0 * activity) as u8),
                );
                painter.circle_filled(pos, 2.0, manga::with_alpha(paper, 240));
                painter.circle_stroke(pos, 2.0, Stroke::new(0.8_f32, ink));
            }
        }
    }

    // Hover: the front-most patch face under the pointer shows its layer.
    if let Some(pointer) = pointer {
        if let Some(face) = faces.iter().rev().find(|f| inside(pointer, &f.pts)) {
            let patch = &patches[face.patch];
            let layer = &trace.layers[face.patch % n_layers];
            let family = dominant_family(composition.get(layer.layer));
            let lines = [
                format!(
                    "Patch ({}, {}) -> layer {} · {}",
                    patch.k1,
                    patch.k2,
                    layer.layer,
                    family.label()
                ),
                format!(
                    "Residual update {:.3} (max this window {:.3})",
                    layer.update_ratio, max_update
                ),
                format!(
                    "Residual rms {:.3} -> {:.3} · MLP inactive {:.0}%",
                    layer.resid_rms_in,
                    layer.resid_rms_out,
                    100.0 * layer.mlp_dead_fraction
                ),
                "Geometry is illustrative; colour and light are measured.".to_string(),
            ];
            draw_tooltip(painter, camera.rect, pointer, &lines, p);
        }
    }
}
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
        recipe: Option<PathBuf>,
        algorithms: Option<PathBuf>,
    },
}

struct Session {
    rx: Receiver<LiveMessage>,
    stop: Arc<AtomicBool>,
}

/// How the live trace is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    /// Calabi-Yau manifold: each patch is a layer, lit by its measured residual update.
    Manifold,
    /// One slab per layer, one tile row per tensor (channel groups).
    Slabs,
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
    recipe_path: String,
    algorithms_path: String,
    gate: Option<Receiver<Result<Vec<String>, String>>>,
    gate_lines: Vec<String>,
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
    mode: ViewMode,
    manifold: Vec<ManifoldPatch>,
}

impl Default for LiveView {
    fn default() -> Self {
        Self {
            model_dir: "models/Pythia-14M".to_string(),
            tokens_path: String::new(),
            recipe_path: String::new(),
            algorithms_path: String::new(),
            gate: None,
            gate_lines: Vec::new(),
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
            // THME_GUI_LIVE_MODE=slabs opens the slab view (used for documentation captures).
            mode: match std::env::var("THME_GUI_LIVE_MODE").as_deref() {
                Ok("slabs") => ViewMode::Slabs,
                _ => ViewMode::Manifold,
            },
            manifold: calabi_yau_patches(5, 14, 9, 0.85),
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

    fn start_gate(&mut self, ctx: &egui::Context) {
        let (Some(recipe), Some(tokens)) = (
            optional_path(&self.recipe_path),
            optional_path(&self.tokens_path),
        ) else {
            self.gate_lines = vec!["Error: the gate needs a recipe and a token file".to_string()];
            return;
        };
        let args = crate::cli::evaluate::EvaluateArgs {
            model: PathBuf::from(self.model_dir.trim()),
            recipe,
            tokens,
            eval_context: self.context.max(64),
            eval_max_tokens: Some(16_384),
            max_ppl_delta: 0.45,
            report: PathBuf::from("target/studio_gate_report.json"),
            allow_partial: true,
            algorithms: optional_path(&self.algorithms_path).into_iter().collect(),
            artifact: None,
            calibration: None,
            gate: None,
        };
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = crate::cli::evaluate::evaluate(&args)
                .map(|report| gate_summary(&report, &args.report))
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
            ctx.request_repaint();
        });
        self.gate = Some(rx);
        self.gate_lines = vec!["Measuring source and recipe (first 16,384 tokens)...".to_string()];
    }

    fn poll_gate(&mut self) {
        let Some(rx) = &self.gate else { return };
        if let Ok(result) = rx.try_recv() {
            self.gate_lines = match result {
                Ok(lines) => lines,
                Err(error) => vec![format!("Error: {error}")],
            };
            self.gate = None;
        }
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
        self.poll_gate();
        ctx.request_repaint();

        ui.horizontal(|ui| {
            ui.heading(
                egui::RichText::new("LIVE 3D MODEL VIEW")
                    .size(15.0)
                    .color(p.text_main),
            );
            ui.add_space(12.0);
            ui.selectable_value(&mut self.mode, ViewMode::Manifold, "MANIFOLD");
            ui.selectable_value(&mut self.mode, ViewMode::Slabs, "LAYER SLABS");
        });
        let caption = match self.mode {
            ViewMode::Manifold => "A Calabi-Yau quintic cross-section with each of its 25 patches bound to a layer (cyclically). Patch colour is the layer's format family; its light is the measured residual update of the current window; anomalies are marked red (solid ink in black-and-white themes). Token particles flow across the patches. The geometry is illustrative; the colours and light are measured. Drag to orbit, scroll to zoom, hover a patch to inspect its layer.",
            ViewMode::Slabs => "Each slab is one applied layer; each row is a linear tensor split into output-channel groups, lit by the measured output magnitude of the current window and tinted by format family. Particles carry the residual stream between layers, brighter where a layer rewrites it more. Red marks anomalies. Drag to orbit, scroll to zoom, hover a tile to inspect it.",
        };
        ui.label(egui::RichText::new(caption).size(10.0).color(p.text_muted));
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Model folder:");
            ui.add(egui::TextEdit::singleline(&mut self.model_dir).desired_width(220.0));
            ui.label("Token file (optional):");
            ui.add(egui::TextEdit::singleline(&mut self.tokens_path).desired_width(260.0));
            ui.add(egui::Slider::new(&mut self.context, 32..=1024).text("tokens/window"));
            if ui.button("Run model").clicked() {
                self.start(
                    &ctx,
                    Source::Model {
                        dir: PathBuf::from(self.model_dir.trim()),
                        tokens: optional_path(&self.tokens_path),
                        recipe: optional_path(&self.recipe_path),
                        algorithms: optional_path(&self.algorithms_path),
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
        ui.horizontal_wrapped(|ui| {
            ui.label("Recipe (optional):");
            ui.add(egui::TextEdit::singleline(&mut self.recipe_path).desired_width(260.0));
            ui.label("BYOA algorithms (optional):");
            ui.add(egui::TextEdit::singleline(&mut self.algorithms_path).desired_width(220.0));
            let running = self.gate.is_some();
            if ui
                .add_enabled(!running, egui::Button::new(if running { "Gate running..." } else { "Run gate" }))
                .on_hover_text("Measure the recipe against the source with the reference forward pass (thme evaluate)")
                .clicked()
            {
                self.start_gate(&ctx);
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
            for item in anomalies.iter().take(8) {
                ui.label(
                    egui::RichText::new(format!("! {item}"))
                        .size(10.0)
                        .color(anomaly(p)),
                );
            }
        } else {
            ui.label(muted("Waiting for the first window...".to_string()));
        }
        ui.separator();
        if !self.gate_lines.is_empty() {
            ui.label(strong("Recipe gate (measured)"));
            for line in &self.gate_lines {
                let color = if line.starts_with("FAILED") || line.starts_with("Error") {
                    anomaly(p)
                } else {
                    p.text_muted
                };
                ui.label(egui::RichText::new(line).size(10.0).color(color));
            }
            ui.separator();
        }
        ui.label(strong("Format families"));
        for family in ALL_FAMILIES {
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                ui.painter()
                    .rect_filled(rect, egui::CornerRadius::same(2), family.color_for(p));
                ui.label(muted(family.label().to_string()));
            });
        }
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
            ui.painter()
                .rect_filled(rect, egui::CornerRadius::same(2), anomaly(p));
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
        let radius = egui::CornerRadius::same(manga::radius(p));
        painter.rect_filled(rect, radius, manga::glass_fill(p.bg_dark, p));
        manga::glass_sheen(&painter, rect, f32::from(manga::radius(p)), p);
        painter.rect_stroke(
            rect,
            radius,
            Stroke::new(2.0_f32, p.border),
            egui::StrokeKind::Inside,
        );

        {
            // Manga screentone: halftone dots deepening toward the bottom of the panel.
            let tone = with_alpha(p.text_main, if p.is_light() { 46 } else { 26 });
            let start = rect.top() + rect.height() * 0.55;
            let mut y = start;
            while y < rect.bottom() - 4.0 {
                let r = 2.4 * (y - start) / (rect.bottom() - start);
                let mut x = rect.left() + 6.0;
                while x < rect.right() - 4.0 {
                    painter.circle_filled(Pos2::new(x, y), r, tone);
                    x += 9.0;
                }
                y += 9.0;
            }
        }

        let drag = response.drag_delta();
        self.yaw += drag.x * 0.008;
        self.pitch = (self.pitch + drag.y * 0.008).clamp(-1.2, 1.2);
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.3, 4.0);
        }
        if self.mode == ViewMode::Manifold {
            if !response.dragged() {
                // Slow turntable spin while nobody is holding the camera.
                self.yaw += ui.input(|i| i.stable_dt).min(0.1) * 0.22;
            }
            // 4D rotation: the surface lives in C^2 (four real dimensions); sweeping the angle
            // that mixes Im z1 and Im z2 into the third axis turns it through the hidden ones.
            let t = self.epoch.elapsed().as_secs_f32();
            self.manifold = calabi_yau_patches(5, 14, 9, 0.85 + 0.55 * (t * 0.17).sin());
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
        if self.mode == ViewMode::Manifold {
            let camera = Camera {
                yaw: self.yaw,
                pitch: self.pitch,
                dist: 6.0 / self.zoom,
                rect,
            };
            draw_manifold(
                &painter,
                response.hover_pos(),
                &camera,
                &self.manifold,
                &frame.trace,
                &self.composition,
                p,
                self.epoch.elapsed().as_secs_f32(),
            );
            return;
        }
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
                let base = family.color_for(p);
                let dim = lerp(p.bg_dark, base, 0.18);
                let y = y_of(row);
                let bins = tensor.bins.len().max(1);
                let w = 2.0 * half_w / bins as f32;
                let scale = peak_of(tensor.name);
                for (bin, &value) in tensor.bins.iter().enumerate() {
                    let x0 = -half_w + bin as f32 * w + 0.01;
                    let x1 = x0 + w - 0.02;
                    let color = if tensor.nonfinite > 0 {
                        anomaly(p)
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
                    if anomalous { anomaly(p) } else { p.text_muted },
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
        Source::Model {
            dir,
            tokens,
            recipe,
            algorithms,
        } => {
            let ir = ModelImporter::import(&dir)
                .map_err(|e| anyhow::anyhow!("Failed to import {dir:?}: {e:#}"))?;
            let arch = ArchConfig::detect(&ir, &dir)?;
            let (model, composition, recipe_note) = match recipe {
                Some(path) => {
                    let text = std::fs::read_to_string(&path)
                        .map_err(|e| anyhow::anyhow!("Failed to read recipe {path:?}: {e}"))?;
                    let recipe: crate::surgery::recipe::TweakRecipe =
                        serde_json::from_str(&text)
                            .map_err(|e| anyhow::anyhow!("Failed to parse recipe {path:?}: {e}"))?;
                    let registry = crate::cli::evaluate::load_registry(
                        &algorithms.into_iter().collect::<Vec<_>>(),
                    )?;
                    let prepared =
                        crate::cli::evaluate::prepare_candidate(&ir, &recipe, &registry, true)?;
                    let model = arch.load(&prepared.model, WeightPrecision::SelectedFormat)?;
                    let skipped = if prepared.skipped.is_empty() {
                        String::new()
                    } else {
                        format!("; not applied: {}", prepared.skipped.join(", "))
                    };
                    let note = format!(
                        ", recipe '{}' applied to {} tensors{skipped}",
                        recipe.name, prepared.relabelled
                    );
                    (model, composition_from_candidate(&prepared.model), note)
                }
                None => (
                    arch.load(&ir, WeightPrecision::AsStored)?,
                    composition_from_ir(&ir),
                    String::new(),
                ),
            };
            let (tokens, meaningful, what) = match tokens {
                Some(path) => (read_token_file(&path)?, true, "evaluation tokens"),
                None => (
                    pseudo_random_tokens(model.vocab_size(), 8192, 3),
                    false,
                    "random tokens (no token file given; perplexity is not meaningful)",
                ),
            };
            let label = format!(
                "{} ({}), {} layers, {what}{recipe_note}",
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

/// Composition of a recipe-applied model: assigned formats, BYOA algorithms by their declared
/// bits, and stored formats for untouched tensors.
fn composition_from_candidate(ir: &ModelIR) -> Composition {
    let mut layers = composition_from_ir(ir);
    for tensor in &ir.tensors {
        let (Some(layer), true) = (tensor.layer_index, tensor.shape.len() == 2) else {
            continue;
        };
        let family = match &tensor.selected_format {
            TensorFormat::FP32 => continue, // untouched: keep the stored family
            TensorFormat::Custom(_) => {
                let bits = tensor
                    .metadata
                    .get(crate::inference::BYOA_METADATA_KEY)
                    .and_then(|json| {
                        crate::transform::byoa::AlgorithmDefinition::from_json(json).ok()
                    })
                    .map_or(16.0, |a| a.effective_bits_per_param);
                match bits {
                    b if b <= 1.2 => Family::Binary,
                    b if b <= 2.0 => Family::Ternary,
                    b if b <= 4.5 => Family::Int4,
                    b if b <= 8.5 => Family::Int8,
                    _ => Family::Dense,
                }
            }
            other => Family::of(other),
        };
        let name = role_short_name(tensor.role);
        if let (Some(slot), Some(name)) = (layers.get_mut(layer), name) {
            if let Some(entry) = slot.iter_mut().find(|(n, _)| n == name) {
                entry.1 = family;
            }
        }
    }
    layers
}

fn role_short_name(role: TensorRole) -> Option<&'static str> {
    Some(match role {
        TensorRole::AttentionQ => "q",
        TensorRole::AttentionK => "k",
        TensorRole::AttentionV => "v",
        TensorRole::AttentionOut => "o",
        TensorRole::AttentionQkv => "qkv",
        TensorRole::MlpGate => "gate",
        TensorRole::MlpUp => "up",
        TensorRole::MlpDown => "down",
        _ => return None,
    })
}

fn optional_path(text: &str) -> Option<PathBuf> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
}

/// Short lines for the side panel from a `thme evaluate` report.
fn gate_summary(report: &serde_json::Value, path: &std::path::Path) -> Vec<String> {
    let variants = report["evaluation"]["variants"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let ppl = |i: usize| {
        variants
            .get(i)
            .and_then(|v| v["perplexity"].as_f64())
            .unwrap_or(f64::NAN)
    };
    let delta = ppl(1) - ppl(0);
    let passed = report["evaluation"]["gate_passed"]
        .as_bool()
        .unwrap_or(false);
    let mut lines = vec![
        format!(
            "{} Δ PPL {delta:+.4}",
            if passed { "PASSED" } else { "FAILED" }
        ),
        format!("Source {:.4} -> recipe {:.4}", ppl(0), ppl(1)),
    ];
    if let Some(skipped) = report["recipe"]["operations_not_evaluated"].as_array() {
        if !skipped.is_empty() {
            let names: Vec<&str> = skipped.iter().filter_map(|s| s.as_str()).collect();
            lines.push(format!("Not evaluated: {}", names.join(", ")));
        }
    }
    lines.push(format!("Report: {}", path.display()));
    lines
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
    /// World point to camera space (x right, y up, z = depth away from the eye).
    fn view(&self, p: [f32; 3]) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let x1 = p[0] * cy - p[2] * sy;
        let z1 = p[0] * sy + p[2] * cy;
        let (sp, cp) = self.pitch.sin_cos();
        let y2 = p[1] * cp - z1 * sp;
        let z2 = p[1] * sp + z1 * cp + self.dist;
        [x1, y2, z2]
    }

    /// Camera-space point to screen (caller ensures depth > 0.2).
    fn screen(&self, v: [f32; 3]) -> Pos2 {
        let f = self.rect.height() * 1.15;
        self.rect.center() + Vec2::new(v[0] * f / v[2], -v[1] * f / v[2])
    }

    /// Projects a world point; returns the screen position and camera-space depth.
    fn project(&self, p: [f32; 3]) -> Option<(Pos2, f32)> {
        let v = self.view(p);
        if v[2] < 0.2 {
            return None;
        }
        Some((self.screen(v), v[2]))
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

    #[test]
    fn recipe_runs_show_their_composition_and_gate_summary_reads_reports() {
        let dir = tempfile::tempdir().unwrap();
        let (model, tokens) = crate::cli::evaluate::tests::synthetic_checkpoint(dir.path());
        let recipe = dir.path().join("recipe.json");
        std::fs::write(
            &recipe,
            r#"{"name":"r","author":"t","target_model":"tiny","version":"1","description":"","operations":[
                {"type":"SetPrecision","params":{"start_layer":0,"end_layer":1,"role":"MlpDown","format":"Q4_0"}},
                {"type":"ApplyCustomAlgorithm","params":{"start_layer":1,"end_layer":1,"role":"MlpUp","algorithm_id":"thme.hadamard_ternary"}}]}"#,
        )
        .unwrap();
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::channel();
        let stop = AtomicBool::new(false);
        let source = Source::Model {
            dir: model,
            tokens: Some(tokens),
            recipe: Some(recipe),
            algorithms: None,
        };
        let summary = run_worker(source, &ctx, &tx, &stop, 32).unwrap();
        assert!(summary.starts_with("Finished"));
        let mut composition = None;
        let mut frames = 0;
        while let Ok(message) = rx.try_recv() {
            match message {
                LiveMessage::Started {
                    composition: c,
                    label,
                    ..
                } => {
                    assert!(label.contains("recipe 'r'"), "{label}");
                    composition = Some(c);
                }
                LiveMessage::Frame(_) => frames += 1,
                _ => {}
            }
        }
        assert!(frames > 0);
        let composition = composition.unwrap();
        let family = |layer: usize, name: &str| {
            composition[layer]
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, f)| *f)
        };
        assert_eq!(family(0, "down"), Some(Family::Int4));
        assert_eq!(family(1, "up"), Some(Family::Ternary));
        assert_eq!(family(0, "q"), Some(Family::Dense));

        let report = serde_json::json!({
            "evaluation": {"gate_passed": false, "variants": [{"perplexity": 10.0}, {"perplexity": 12.5}]},
            "recipe": {"operations_not_evaluated": ["ReplaceMixer"]}
        });
        let lines = gate_summary(&report, std::path::Path::new("r.json"));
        assert!(lines[0].starts_with("FAILED"));
        assert!(lines.iter().any(|l| l.contains("ReplaceMixer")));
    }
}
