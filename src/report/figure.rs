// SPDX-License-Identifier: AGPL-3.0-only
//! `thme figure`: a model drawn as an isometric figure with the Rust anatomy kit
//! (`engine::iso`) in the Epiphanes manga style: a panel with an ink frame, a hard print shadow
//! and screentone. The model is a staircase of parts trays:
//!
//! - the embedding stands at the front left, then one step per transformer layer rising toward
//!   the back right, then the tensors outside the layers (final norm, output head); thin ink
//!   arrows along the floor show the forward pass;
//! - one block per tensor: footprint area proportional to its stored bytes, height proportional
//!   to its measured bits per weight (stored bytes x 8 / weights), so a quantized tensor sits
//!   visibly lower than an FP16 one;
//! - the lit block (or the lit layer) is ink glass with an ink splash. The dark theme (Manga
//!   Night) splashes blood-red ink; the light theme splashes black ink.
//!
//! Every dimension is read from the model file; the figure invents no number.

use std::fmt::Write as _;

use crate::engine::iso::{
    depth_of, dots_svg, fit_projection, inline_styles, inset_plan, iso, line_svg, path_of,
    plan_outline, slab_of, solid_svg, Flat, Palette, Plan, Point3, Projection, Scene, SolidStyle,
    Theme, Tone, AZIMUTH, ELEVATION,
};
use crate::ir::model::ModelIR;
use crate::ir::tensor::{TensorIR, TensorRole};

const W: f64 = 720.0;
const H: f64 = 440.0;
/// Gap between blocks and at the edge of a tray, in world units.
const GAP: f64 = 2.0;
/// Floor distance between two steps; the flow arrow runs across it.
const STEP_GAP: f64 = 16.0;
/// Height of one step of the staircase.
const RISE: f64 = 14.0;
/// Height of the bench under the staircase.
const BENCH_H: f64 = 6.0;
/// Thickness of a tray plate.
const PLATE: f64 = 1.2;
/// World units of block height per bit per weight.
const BITS_SCALE: f64 = 1.6;
/// Frames per tensor in the forward animation; a model with more tensors than
/// [`FORWARD_MAX_FRAMES`] lights consecutive tensors together so the animation stays that long.
pub const FORWARD_STEPS: usize = 4;
pub const FORWARD_MAX_FRAMES: usize = 120;
/// Phase of the glass sweep on every lit tensor of a layers frame: the band sits mid-tensor.
const LAYER_GLASS_PHASE: f64 = 0.5;
/// Phase of the layers splash: fully grown and not yet fading.
const LAYER_SPLASH_PHASE: f64 = 0.7;
/// Opacity of the tensors outside the lit layer in a layers frame.
const CALM_OPACITY: &str = "0.45";
/// Scale of the core of a two-tone splash inside its outer edge.
const CORE_SCALE: f64 = 0.72;
/// Droplets drawn on a two-tone splash.
const CORE_DROPS: usize = 9;
const SANS: &str = "'Chakra Petch',ui-sans-serif,system-ui,sans-serif";

/// One block to draw. `index` is the tensor's index in the model.
struct Block<'a> {
    index: usize,
    tensor: &'a TensorIR,
    plan: Plan,
    /// Height of the block's floor: the top of its tray plate.
    z: f64,
    height: f64,
}

/// One step of the staircase: the tensors of one layer, or one tensor outside the layers.
struct Tray {
    /// The layer index; `None` for a tensor outside the layers.
    layer: Option<usize>,
    plan: Plan,
    /// Height of the plate's underside.
    step: f64,
}

struct Layout<'a> {
    blocks: Vec<Block<'a>>,
    /// In forward-pass order: the first is the front left, the last the back right.
    trays: Vec<Tray>,
    base: Plan,
}

fn measured_bits(t: &TensorIR) -> f64 {
    if t.numel == 0 {
        0.0
    } else {
        t.byte_size as f64 * 8.0 / t.numel as f64
    }
}

fn role_rank(role: &TensorRole) -> usize {
    use TensorRole::*;
    match role {
        LayerNorm => 0,
        AttentionQ => 1,
        AttentionQkv => 1,
        AttentionK => 2,
        AttentionV => 3,
        AttentionOut => 4,
        MlpGate => 5,
        MlpUp => 6,
        MlpDown => 7,
        _ => 8,
    }
}

/// Lays out the staircase. Groups follow the forward pass: the tensors of one layer share a step,
/// each tensor outside the layers gets its own. Each step is one row further back and one step
/// higher than the last, so the embedding stands lowest at the front and the head highest at the
/// back. The trays are centred on x = 0.
fn layout(model: &ModelIR) -> Layout<'_> {
    let max_bytes = model
        .tensors
        .iter()
        .map(|t| t.byte_size)
        .max()
        .unwrap_or(1)
        .max(1) as f64;
    // The largest tensor gets a side of 60 units; areas stay proportional to bytes.
    let k = 60.0 / max_bytes.sqrt();
    let side = |t: &TensorIR| (k * (t.byte_size as f64).sqrt()).max(1.2);
    let height = |t: &TensorIR| (measured_bits(t) * BITS_SCALE).max(1.0);

    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in forward_order(model) {
        let layer = model.tensors[i].layer_index;
        match groups.last_mut() {
            Some(g) if layer.is_some() && model.tensors[g[0]].layer_index == layer => g.push(i),
            _ => groups.push(vec![i]),
        }
    }

    let mut blocks = Vec::new();
    let mut trays = Vec::new();
    let (mut min_x, mut max_x, mut min_y) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut front = 0.0;
    for (step, group) in groups.iter().enumerate() {
        let sizes: Vec<f64> = group.iter().map(|&i| side(&model.tensors[i])).collect();
        let width = sizes.iter().copied().fold(0.0, f64::max) + GAP * 2.0;
        let depth = sizes.iter().sum::<f64>() + GAP * (group.len() as f64 + 1.0);
        let (x0, y0) = (-width / 2.0, front - depth);
        let floor = BENCH_H + step as f64 * RISE;
        let mut y = y0 + GAP;
        for (&i, &s) in group.iter().zip(&sizes) {
            let t = &model.tensors[i];
            blocks.push(Block {
                index: i,
                tensor: t,
                plan: Plan {
                    x: x0 + (width - s) / 2.0,
                    y,
                    w: s,
                    d: s,
                    r: (s * 0.06).min(1.5),
                },
                z: floor + PLATE,
                height: height(t),
            });
            y += s + GAP;
        }
        trays.push(Tray {
            layer: model.tensors[group[0]].layer_index,
            plan: Plan {
                x: x0,
                y: y0,
                w: width,
                d: depth,
                r: 1.0,
            },
            step: floor,
        });
        min_x = min_x.min(x0);
        max_x = max_x.max(x0 + width);
        min_y = min_y.min(y0);
        front = y0 - STEP_GAP;
    }
    let base = Plan {
        x: min_x - 8.0,
        y: min_y - 8.0,
        w: max_x - min_x + 16.0,
        d: -min_y + 16.0,
        r: 4.0,
    };
    Layout {
        blocks,
        trays,
        base,
    }
}

/// The camera that fits the bench and every block into the frame.
fn fit(lay: &Layout) -> Projection {
    let base = &lay.base;
    let mut pts: Vec<Point3> = vec![
        (base.x, base.y, -4.0),
        (base.x + base.w, base.y + base.d, -4.0),
        (base.x, base.y + base.d, 0.0),
        (base.x + base.w, base.y, 0.0),
    ];
    for b in &lay.blocks {
        pts.push((b.plan.x, b.plan.y, b.z + b.height));
        pts.push((b.plan.x + b.plan.w, b.plan.y + b.plan.d, b.z + b.height));
    }
    fit_projection(&pts, W, H, 26.0, AZIMUTH, ELEVATION)
}

/// The colours of one theme: the manga panel and the ink that goes with it. Red is the spot
/// colour of the dark theme (Manga Night) only.
struct Manga {
    ground: &'static str,
    panel: &'static str,
    outline: &'static str,
    shadow: &'static str,
    shadow_opacity: &'static str,
    text: &'static str,
    muted: &'static str,
    screen: &'static str,
    screen_opacity: &'static str,
    glass: &'static str,
    sheen: &'static str,
    splash: &'static str,
    splash_edge: &'static str,
}

impl Manga {
    fn of(theme: Theme) -> Self {
        match theme {
            Theme::Light => Self {
                ground: "#F6F6F4",
                panel: "#FFFFFF",
                outline: "#0B0B0B",
                shadow: "#0B0B0B",
                shadow_opacity: "1",
                text: "#0B0B0B",
                muted: "#4A4A4A",
                screen: "#0B0B0B",
                screen_opacity: "0.2",
                glass: "#0B0B0B",
                sheen: "#FFFFFF",
                splash: "#0B0B0B",
                splash_edge: "#0B0B0B",
            },
            Theme::Dark => Self {
                ground: "#0B0B0B",
                panel: "#141414",
                outline: "#F6F6F4",
                shadow: "#C8102E",
                shadow_opacity: "0.6",
                text: "#F6F6F4",
                muted: "#A3A3A3",
                screen: "#F6F6F4",
                screen_opacity: "0.12",
                glass: "#D4D4D4",
                sheen: "#FFFFFF",
                splash: "#C8102E",
                splash_edge: "#3A0610",
            },
        }
    }
}

/// What the card around a figure says.
struct Plate {
    title: String,
    hint: String,
    readout: String,
    /// Lines of caption under the panel.
    caption: Vec<String>,
}

fn esc(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// A Chakra Petch label: `x`, `y`, text anchor, size, weight, fill and text.
fn label(x: f64, y: f64, anchor: &str, size: f64, weight: u16, fill: &str, text: &str) -> String {
    format!(
        "<text x=\"{x}\" y=\"{y}\" font-family=\"{SANS}\" font-size=\"{size}\" font-weight=\"{weight}\" letter-spacing=\"0.8\" text-anchor=\"{anchor}\" fill=\"{fill}\">{}</text>",
        esc(text)
    )
}

/// Screentone: dots on a 10 px grid whose radius grows toward the top-right corner of the panel.
/// The band starts below the header, so no dot sits behind text.
fn screentone(x0: f64, y0: f64, pw: f64, top: f64, m: &Manga) -> String {
    const GRID: f64 = 10.0;
    const REACH: f64 = 300.0;
    const MAX_R: f64 = 2.2;
    let (cx, cy) = (x0 + pw, y0);
    let mut out = format!(
        "<g fill=\"{}\" fill-opacity=\"{}\">",
        m.screen, m.screen_opacity
    );
    let mut y = y0 + top + GRID / 2.0;
    while y < y0 + top + REACH {
        let mut x = cx - GRID / 2.0;
        while x > x0 {
            let d = ((cx - x).powi(2) + (cy - y).powi(2)).sqrt();
            let r = MAX_R * (1.0 - d / REACH);
            if r > 0.3 {
                let _ = write!(out, "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"{r:.2}\"/>");
            }
            x -= GRID;
        }
        y += GRID;
    }
    out.push_str("</g>");
    out
}

/// The figure on its manga panel as one self-contained SVG in `theme`: an ink frame (paper in
/// the dark theme), a hard print shadow 4 px down and right, screentone in the top-right corner,
/// the wordmark and the model name in the header, the readout in the footer, captions below.
/// Every colour is resolved in Rust; no stylesheet.
fn plate_svg(
    width: f64,
    height: f64,
    name: &str,
    body: &str,
    plate: &Plate,
    theme: Theme,
) -> String {
    let m = Manga::of(theme);
    let pal = Palette::of(theme);
    let (margin, pad, top, foot) = (24.0, 20.0, 44.0, 40.0);
    let (pw, ph) = (width + pad * 2.0, height + top + foot);
    let (x1, y1) = (margin + pw, margin + ph);
    let captions = plate.caption.len() as f64;
    let (tw, th) = (pw + margin * 2.0, y1 + 28.0 + 20.0 * captions);
    let mut s = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {tw} {th}\" width=\"{tw}\" height=\"{th}\" role=\"img\" aria-label=\"{}\">",
        esc(name)
    );
    let _ = write!(
        s,
        "<rect width=\"{tw}\" height=\"{th}\" fill=\"{}\"/>",
        m.ground
    );
    // The hard print shadow: the panel again, 4 px down and right, no blur.
    let _ = write!(
        s,
        "<rect x=\"{}\" y=\"{}\" width=\"{pw}\" height=\"{ph}\" fill=\"{}\" fill-opacity=\"{}\"/>",
        margin + 4.0,
        margin + 4.0,
        m.shadow,
        m.shadow_opacity
    );
    let _ = write!(
        s,
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"2.5\"/>",
        margin + 1.25,
        margin + 1.25,
        pw - 2.5,
        ph - 2.5,
        m.panel,
        m.outline
    );
    s += &screentone(margin, margin, pw, top, &m);
    // Header: the wordmark as a solid tag, a rule, the model name.
    let _ = write!(
        s,
        "<rect x=\"{}\" y=\"{}\" width=\"180\" height=\"20\" fill=\"{}\"/>",
        margin + pad,
        margin + 12.0,
        m.outline
    );
    let _ = write!(
        s,
        "<text x=\"{}\" y=\"{}\" font-family=\"{SANS}\" font-size=\"12\" font-weight=\"700\" letter-spacing=\"1.6\" fill=\"{}\">EPIPHANES / THME</text>",
        margin + pad + 10.0,
        margin + 26.5,
        m.panel
    );
    let _ = write!(
        s,
        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
        margin + pad,
        margin + top - 4.0,
        x1 - pad,
        margin + top - 4.0,
        m.outline
    );
    s += &label(
        x1 - pad,
        margin + 27.0,
        "end",
        13.0,
        700,
        m.text,
        &plate.title.to_uppercase(),
    );
    // Footer: the hint and the live readout.
    s += &label(
        margin + pad,
        y1 - 16.0,
        "start",
        11.0,
        500,
        m.muted,
        &plate.hint.to_uppercase(),
    );
    s += &label(
        x1 - pad,
        y1 - 16.0,
        "end",
        12.5,
        500,
        m.text,
        &plate.readout,
    );
    let _ = write!(
        s,
        "<g transform=\"translate({} {})\">{}</g>",
        margin + pad,
        margin + top,
        inline_styles(body, &pal)
    );
    for (i, line) in plate.caption.iter().enumerate() {
        s += &label(
            margin,
            y1 + 22.0 + 20.0 * i as f64,
            "start",
            13.0,
            500,
            m.muted,
            line,
        );
    }
    s.push_str("</svg>\n");
    s
}

/// Phase and glass colours of one lit block.
struct Glass<'a> {
    id: String,
    phase: f64,
    ink: &'a str,
    sheen: &'a str,
}

/// The screen box (min x, min y, max x, max y) of a plan between two heights.
fn box_of(plan: &Plan, z0: f64, z1: f64, p: &Projection) -> (f64, f64, f64, f64) {
    let (x0, y0, x1, y1) = (plan.x, plan.y, plan.x + plan.w, plan.y + plan.d);
    let mut bx = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
        for z in [z0, z1] {
            let (sx, sy) = iso((x, y, z), p);
            bx = (bx.0.min(sx), bx.1.min(sy), bx.2.max(sx), bx.3.max(sy));
        }
    }
    bx
}

/// A lit block as ink glass: ink over its silhouette `fill`, then a bright band sweeping
/// diagonally across its screen box `bx` with the phase.
fn glass(bx: (f64, f64, f64, f64), fill: &str, g: &Glass) -> String {
    let (x0, y0, x1, y1) = bx;
    let (w, hh) = (x1 - x0, y1 - y0);
    let c = -0.4 + g.phase * 1.8;
    let (gx, gy) = (x0 + c * w, y0 + c * hh);
    let half = 0.22 * w.max(hh);
    format!(
        "<defs><linearGradient id=\"{id}\" gradientUnits=\"userSpaceOnUse\" x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"><stop offset=\"0\" stop-color=\"{s}\" stop-opacity=\"0\"/><stop offset=\"0.5\" stop-color=\"{s}\" stop-opacity=\"0.7\"/><stop offset=\"1\" stop-color=\"{s}\" stop-opacity=\"0\"/></linearGradient></defs><path d=\"{fill}\" fill=\"{ink}\" fill-opacity=\"0.9\"/><path d=\"{fill}\" fill=\"url(#{id})\"/><path d=\"{fill}\" fill=\"none\" stroke=\"{s}\" stroke-opacity=\"0.5\" stroke-width=\"0.6\"/>",
        gx - half,
        gy - half,
        gx + half,
        gy + half,
        id = g.id,
        s = g.sheen,
        ink = g.ink,
    )
}

/// `lit_layer` lights every tensor of one layer (the layers animation); it overrides the
/// highlight and the largest-tensor accent. Lit tensors in either animation carry the glass;
/// the layers animation calms the rest and puts one splash under the lit layer.
fn scene_with(
    model: &ModelIR,
    highlight: Option<&Highlight>,
    lit_layer: Option<usize>,
    theme: Theme,
) -> String {
    let lay = layout(model);
    let p = fit(&lay);
    let m = Manga::of(theme);
    let base = &lay.base;
    let mut s = Scene::new();

    // Feet under the bench corners.
    for (fx, fy) in [
        (base.x + 6.0, base.y + 6.0),
        (base.x + base.w - 12.0, base.y + 6.0),
        (base.x + 6.0, base.y + base.d - 12.0),
        (base.x + base.w - 12.0, base.y + base.d - 12.0),
    ] {
        let foot = slab_of(
            &Plan {
                x: fx,
                y: fy,
                w: 6.0,
                d: 6.0,
                r: 3.0,
            },
            -3.0,
            3.0,
            &p,
            8,
            0.0,
        );
        s.add(
            -1e6,
            solid_svg(
                &foot,
                &SolidStyle {
                    tone: Tone::Lo,
                    ..Default::default()
                },
            ),
        );
    }
    // The bench: a groove, corner screws.
    let mut inner = line_svg(
        &plan_outline(&inset_plan(base, 3.5), BENCH_H, &p, 8),
        Tone::Faint,
    );
    let screws: Vec<Flat> = [
        (base.x + 2.0, base.y + 2.0),
        (base.x + base.w - 2.0, base.y + 2.0),
        (base.x + 2.0, base.y + base.d - 2.0),
        (base.x + base.w - 2.0, base.y + base.d - 2.0),
    ]
    .iter()
    .map(|&(x, y)| iso((x, y, BENCH_H), &p))
    .collect();
    inner += &dots_svg(&screws, 0.55, Tone::Mid);
    s.add(
        -9e5,
        solid_svg(
            &slab_of(base, 0.0, BENCH_H, &p, 8, 0.8),
            &SolidStyle {
                tone: Tone::Lo,
                inner,
                ..Default::default()
            },
        ),
    );

    // The staircase: a riser under each step, then its plate. A plate's depth is its back
    // corner, so every block standing on it is painted after it.
    for tray in &lay.trays {
        let plan = &tray.plan;
        if tray.step > BENCH_H {
            let riser = slab_of(plan, BENCH_H, tray.step - BENCH_H, &p, 6, 0.4);
            s.add(
                depth_of((plan.x, plan.y, BENCH_H), &p),
                solid_svg(
                    &riser,
                    &SolidStyle {
                        tone: Tone::Lo,
                        ..Default::default()
                    },
                ),
            );
        }
        let plate = slab_of(plan, tray.step, PLATE, &p, 6, 0.4);
        let groove = line_svg(
            &plan_outline(&inset_plan(plan, 0.8), tray.step + PLATE, &p, 6),
            Tone::Faint,
        );
        s.add(
            depth_of((plan.x, plan.y, tray.step), &p),
            solid_svg(
                &plate,
                &SolidStyle {
                    tone: Tone::Mid,
                    inner: groove,
                    ..Default::default()
                },
            ),
        );
    }

    // Flow arrows: from the back edge of each step across the bench to the front of the next.
    for pair in lay.trays.windows(2) {
        let (a, b) = (&pair[0].plan, &pair[1].plan);
        let z = BENCH_H + 0.3;
        let (start, tip) = ((0.0, a.y - 2.0, z), (0.0, b.y + b.d + 2.0, z));
        let wings = [(tip.0 - 1.6, tip.1 + 2.6, z), (tip.0 + 1.6, tip.1 + 2.6, z)];
        let d = format!(
            "{}{}",
            path_of(&[iso(start, &p), iso(tip, &p)], false),
            path_of(&[iso(wings[0], &p), iso(tip, &p), iso(wings[1], &p)], false)
        );
        s.add(
            depth_of((0.0, (a.y + b.y + b.d) / 2.0, z), &p),
            line_svg(&d, Tone::Hi),
        );
    }

    // Blocks, painted by the depth of their nearest bottom corner. Without a highlight the
    // largest tensor is the lit accent; with one, the highlighted tensors are.
    let largest = lay
        .blocks
        .iter()
        .map(|b| b.tensor.byte_size)
        .max()
        .unwrap_or(0);
    let phase = highlight.map_or(LAYER_GLASS_PHASE, |h| h.phase);
    let mut lit_done = false;
    for b in &lay.blocks {
        let solid = slab_of(&b.plan, b.z, b.height, &p, 4, (b.plan.w * 0.04).min(0.6));
        let lit = match (lit_layer, highlight) {
            (Some(l), _) => b.tensor.layer_index == Some(l),
            (None, Some(h)) => h.lit.contains(&b.index),
            (None, None) => !lit_done && b.tensor.byte_size == largest,
        };
        lit_done |= lit;
        let style = SolidStyle {
            tone: if lit { Tone::Hi } else { Tone::Mid },
            lit,
            ..Default::default()
        };
        let near = (b.plan.x + b.plan.w, b.plan.y + b.plan.d, b.z);
        let depth = depth_of(near, &p);
        let bx = box_of(&b.plan, b.z, b.z + b.height, &p);
        if let Some(h) = highlight.filter(|h| h.active == b.index) {
            let (cx, cy) = iso(
                (b.plan.x + b.plan.w / 2.0, b.plan.y + b.plan.d / 2.0, b.z),
                &p,
            );
            let r = ((bx.2 - bx.0).max(bx.3 - bx.1) * 1.3).clamp(42.0, 120.0);
            s.add(depth - 1e-6, splash_svg((cx, cy), r, b.index, h.phase, &m));
        }
        let mut svg = solid_svg(&solid, &style);
        if lit && (highlight.is_some() || lit_layer.is_some()) {
            let g = Glass {
                id: format!("glass-{}", b.index),
                phase,
                ink: m.glass,
                sheen: m.sheen,
            };
            svg.push_str(&glass(bx, &solid.fill, &g));
        } else if lit_layer.is_some() {
            svg = format!("<g opacity=\"{CALM_OPACITY}\">{svg}</g>");
        }
        s.add(depth, svg);
    }

    // One splash under the lit layer, on its step.
    if let Some(l) = lit_layer {
        if let Some(tray) = lay.trays.iter().find(|t| t.layer == Some(l)) {
            let pl = &tray.plan;
            let z = tray.step + PLATE;
            let (cx, cy) = iso((pl.x + pl.w / 2.0, pl.y + pl.d / 2.0, z), &p);
            let bx = box_of(pl, z, z, &p);
            let r = ((bx.2 - bx.0).max(bx.3 - bx.1) * 1.3).clamp(42.0, 120.0);
            s.add(
                depth_of((pl.x, pl.y, tray.step), &p) + 1e-6,
                splash_svg((cx, cy), r, l, LAYER_SPLASH_PHASE, &m),
            );
        }
    }
    s.render()
}

/// The figure on its card as one self-contained SVG file in `theme`: every colour resolved in
/// Rust, no stylesheet.
pub fn model_figure_svg(model: &ModelIR, theme: Theme) -> String {
    let total: usize = model.tensors.iter().map(|t| t.byte_size).sum();
    let plate = Plate {
        title: model.name.clone(),
        hint: "Model anatomy".into(),
        readout: format!(
            "{} tensors \u{b7} {:.1} MiB",
            model.tensors.len(),
            total as f64 / (1u64 << 20) as f64
        ),
        caption: vec![
            "Embedding at the front left, one step per layer rising to the output head at the back right.".into(),
            "One block per tensor: footprint follows stored bytes, height follows measured bits per weight.".into(),
            "The lit block is the largest tensor. Every dimension is read from the model file.".into(),
        ],
    };
    plate_svg(
        W,
        H,
        &format!("{} drawn as a parts tray", model.name),
        &scene_with(model, None, None, theme),
        &plate,
        theme,
    )
}

/// What to light in one animation frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Highlight {
    /// Index into the model's tensors of the tensor the splash is thrown from (the first lit).
    pub active: usize,
    /// Indices into the model's tensors of every lit tensor (one, or a run of consecutive ones).
    pub lit: Vec<usize>,
    /// 0 at the start of this frame's turn, 1 at the end.
    pub phase: f64,
}

/// Tensor indices in forward-pass order: tensors outside the layers that come first (the
/// embedding), then each layer's tensors in role order, then the rest (final norm, head).
pub fn forward_order(model: &ModelIR) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..model.tensors.len()).collect();
    let key = |i: &usize| {
        let t = &model.tensors[*i];
        let stage = match (t.layer_index, &t.role) {
            (None, TensorRole::Embedding) => 0,
            (Some(_), _) => 1,
            _ => 2,
        };
        (
            stage,
            t.layer_index.unwrap_or(0),
            role_rank(&t.role),
            t.name.clone(),
        )
    };
    idx.sort_by_key(key);
    idx
}

/// The forward-pass animation frames. Up to [`FORWARD_MAX_FRAMES`] tensors, every tensor gets
/// [`FORWARD_STEPS`] frames. A model with more tensors lights consecutive tensors together, in
/// forward order, with as many frames per run as the cap allows, so the GIF stays under the cap.
pub fn forward_steps(model: &ModelIR) -> Vec<Highlight> {
    let order = forward_order(model);
    let mut out = Vec::new();
    if order.len() <= FORWARD_MAX_FRAMES {
        for &active in &order {
            for step in 0..FORWARD_STEPS {
                out.push(Highlight {
                    active,
                    lit: vec![active],
                    phase: (step as f64 + 0.5) / FORWARD_STEPS as f64,
                });
            }
        }
    } else {
        let run = order.len().div_ceil(FORWARD_MAX_FRAMES);
        let runs: Vec<&[usize]> = order.chunks(run).collect();
        let steps = (FORWARD_MAX_FRAMES / runs.len()).max(1);
        for chunk in runs {
            let Some(&active) = chunk.first() else {
                continue;
            };
            for step in 0..steps {
                out.push(Highlight {
                    active,
                    lit: chunk.to_vec(),
                    phase: (step as f64 + 0.5) / steps as f64,
                });
            }
        }
    }
    out
}

/// One flung droplet of an ink splash: an ellipse stretched along its flight, rotated `deg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InkDrop {
    pub x: f64,
    pub y: f64,
    pub rx: f64,
    pub ry: f64,
    pub deg: f64,
}

/// A manga ink splash in the caller's 2D coordinates: an organic blot with tendrils (closed
/// quadratic curve through `blot`, see [`InkSplash::outline`]), flung droplets and tapered speed
/// lines (triangles). Shared by the SVG figure and the Studio's live view.
#[derive(Debug, Clone, PartialEq)]
pub struct InkSplash {
    /// Opacity: 1 while the splash grows, fading to 0 over the last 30% of the phase.
    pub fade: f64,
    pub blot: Vec<(f64, f64)>,
    pub drops: Vec<InkDrop>,
    pub lines: Vec<[(f64, f64); 3]>,
}

/// Ink splash geometry around (`cx`, `cy`) with radius `r`, growing with `phase` (0..1).
/// `squash` foreshortens the vertical axis (1 = none), `unit` scales droplet and line sizes
/// (1 = the figure's pixels). Seeded by `seed`, so every frame is reproducible.
pub fn ink_splash_geometry(
    cx: f64,
    cy: f64,
    r: f64,
    squash: f64,
    unit: f64,
    seed: usize,
    phase: f64,
) -> InkSplash {
    use crate::bench::stats::SplitMix64;
    use std::f64::consts::TAU;
    let t = phase.clamp(0.0, 1.0);
    let grow = 1.0 - (1.0 - t).powi(3);
    let fade = if t < 0.7 {
        1.0
    } else {
        (1.0 - (t - 0.7) / 0.3).max(0.0)
    };
    let at = |a: f64, k: f64| (cx + k * a.cos(), cy + k * squash * a.sin());
    let mut rng = SplitMix64::new(0x9e37_79b9 ^ (seed as u64).wrapping_mul(0x2545_f491));

    // Blot: jittered points around the centre, a few pushed out into tendrils.
    let n = 26;
    let blot = (0..n)
        .map(|i| {
            let a = i as f64 / n as f64 * TAU + rng.unit() * 0.12;
            let tendril = if rng.unit() < 0.22 {
                1.0 + 0.9 * rng.unit()
            } else {
                0.0
            };
            let k = r * grow * (0.42 + 0.22 * rng.unit() + 0.38 * tendril);
            at(a, k)
        })
        .collect();

    // Droplets flung outward; the farther ones are smaller and stretched along their flight.
    let drops = (0..22)
        .map(|_| {
            let a = rng.unit() * TAU;
            let reach = 0.7 + 1.1 * rng.unit();
            let (x, y) = at(a, r * reach * grow);
            let size = (1.4 + 4.2 * rng.unit()) * (1.25 - 0.45 * reach / 1.8) * unit;
            let stretch = 1.0 + 1.6 * (reach - 0.7) * grow;
            let deg = (a.sin() * squash).atan2(a.cos()).to_degrees();
            InkDrop {
                x,
                y,
                rx: size * stretch,
                ry: size,
                deg,
            }
        })
        .collect();

    // Speed lines: tapered wedges radiating from the blot, thick at the base, sharp at the tip.
    let lines = (0..16)
        .map(|_| {
            let a = rng.unit() * TAU;
            let (r0, r1) = (
                r * (0.55 + 0.15 * rng.unit()),
                r * (1.05 + 0.75 * rng.unit()) * (0.5 + 0.5 * grow),
            );
            let half = (0.8 + 1.8 * rng.unit()) * (0.6 + 0.4 * grow) * unit;
            let (bx, by) = at(a, r0);
            let tip = at(a, r1);
            let (nx, ny) = (-(a.sin() * squash), a.cos());
            let len = (nx * nx + ny * ny).sqrt().max(1e-9);
            let (nx, ny) = (nx / len * half, ny / len * half);
            [(bx + nx, by + ny), (bx - nx, by - ny), tip]
        })
        .collect();
    InkSplash {
        fade,
        blot,
        drops,
        lines,
    }
}

impl InkSplash {
    /// The blot as a closed polygon: each quadratic segment (between midpoints of neighbouring
    /// blot points, controlled by the point itself) sampled `steps` times.
    pub fn outline(&self, steps: usize) -> Vec<(f64, f64)> {
        let n = self.blot.len();
        let mid = |a: (f64, f64), b: (f64, f64)| ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
        let mut out = Vec::with_capacity(n * steps.max(1));
        for i in 0..n {
            let (p0, c, p1) = (
                mid(self.blot[(i + n - 1) % n], self.blot[i]),
                self.blot[i],
                mid(self.blot[i], self.blot[(i + 1) % n]),
            );
            for s in 0..steps.max(1) {
                let t = s as f64 / steps.max(1) as f64;
                let u = 1.0 - t;
                out.push((
                    u * u * p0.0 + 2.0 * u * t * c.0 + t * t * p1.0,
                    u * u * p0.1 + 2.0 * u * t * c.1 + t * t * p1.1,
                ));
            }
        }
        out
    }
}

/// The closed quadratic path through a blot's points (midpoints as the on-curve points).
fn blot_d(pts: &[Flat]) -> String {
    let n = pts.len();
    let mid = |a: Flat, b: Flat| ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
    let start = mid(pts[n - 1], pts[0]);
    let mut d = format!("M{:.1} {:.1}", start.0, start.1);
    for i in 0..n {
        let m = mid(pts[i], pts[(i + 1) % n]);
        d += &format!("Q{:.1} {:.1} {:.1} {:.1}", pts[i].0, pts[i].1, m.0, m.1);
    }
    d
}

/// An ink splash centred on `centre` with radius `r`, as SVG. The light theme is one black
/// blot; the dark theme is blood-red: a near-black red edge with a deep red core (`CORE_SCALE`
/// of the blot) and a few red droplets.
fn splash_svg(centre: Flat, r: f64, seed: usize, phase: f64, m: &Manga) -> String {
    let g = ink_splash_geometry(centre.0, centre.1, r, 0.55, 1.0, seed, phase);
    let two_tone = m.splash != m.splash_edge;
    let mut out = format!(
        "<g fill=\"{}\" stroke=\"none\" opacity=\"{:.3}\">",
        m.splash_edge, g.fade
    );
    out += &format!("<path d=\"{}Z\"/>", blot_d(&g.blot));
    if two_tone {
        let core: Vec<Flat> = g
            .blot
            .iter()
            .map(|&(x, y)| {
                (
                    centre.0 + (x - centre.0) * CORE_SCALE,
                    centre.1 + (y - centre.1) * CORE_SCALE,
                )
            })
            .collect();
        out += &format!("<path fill=\"{}\" d=\"{}Z\"/>", m.splash, blot_d(&core));
    }
    let drops = g
        .drops
        .iter()
        .take(if two_tone { CORE_DROPS } else { usize::MAX });
    let fill = if two_tone {
        format!(" fill=\"{}\"", m.splash)
    } else {
        String::new()
    };
    for InkDrop { x, y, rx, ry, deg } in drops {
        out += &format!(
            "<ellipse{fill} cx=\"{x:.1}\" cy=\"{y:.1}\" rx=\"{rx:.2}\" ry=\"{ry:.2}\" transform=\"rotate({deg:.1} {x:.1} {y:.1})\"/>"
        );
    }
    for [(ax, ay), (bx, by), (tx, ty)] in &g.lines {
        out += &format!("<path d=\"M{ax:.1} {ay:.1}L{bx:.1} {by:.1}L{tx:.1} {ty:.1}Z\"/>");
    }
    out.push_str("</g>");
    out
}

/// Bytes as KiB, MiB or GiB with one decimal.
fn human_bytes(n: usize) -> String {
    let n = n as f64;
    if n >= (1u64 << 30) as f64 {
        format!("{:.1} GiB", n / (1u64 << 30) as f64)
    } else if n >= (1u64 << 20) as f64 {
        format!("{:.1} MiB", n / (1u64 << 20) as f64)
    } else {
        format!("{:.1} KiB", n / 1024.0)
    }
}

/// One forward-pass frame on its card: the lit tensors glow as ink glass with an ink splash,
/// and the readout names them.
pub fn model_figure_frame(model: &ModelIR, theme: Theme, highlight: &Highlight) -> String {
    let lit: Vec<&TensorIR> = highlight
        .lit
        .iter()
        .filter_map(|&i| model.tensors.get(i))
        .collect();
    let bytes: usize = lit.iter().map(|t| t.byte_size).sum();
    let readout = match (lit.first(), lit.last()) {
        (Some(a), Some(b)) if lit.len() > 1 => {
            format!("{} to {} \u{b7} {}", a.name, b.name, human_bytes(bytes))
        }
        (Some(a), _) => format!("{} \u{b7} {}", a.name, human_bytes(bytes)),
        _ => String::new(),
    };
    let plate = Plate {
        title: model.name.clone(),
        hint: "Forward pass".into(),
        readout,
        caption: vec![
            "Each tensor lights up in forward-pass order: the embedding, each layer's projections, then".into(),
            "the output. Block sizes follow stored bytes and measured bits per weight, as on the static figure.".into(),
        ],
    };
    plate_svg(
        W,
        H,
        &format!("{} forward pass", model.name),
        &scene_with(model, Some(highlight), None, theme),
        &plate,
        theme,
    )
}

/// The layer indices that occur among the model's tensors, sorted (the real structure, not
/// `num_layers`).
pub fn layer_indices(model: &ModelIR) -> Vec<usize> {
    let set: std::collections::BTreeSet<usize> =
        model.tensors.iter().filter_map(|t| t.layer_index).collect();
    set.into_iter().collect()
}

/// One layers-animation frame on its card: every tensor of layer `layer` is lit, with one
/// splash under the layer, and the readout names the layer with its tensor count and stored
/// bytes.
pub fn model_figure_layer_frame(model: &ModelIR, theme: Theme, layer: usize) -> String {
    let members = || {
        model
            .tensors
            .iter()
            .filter(|t| t.layer_index == Some(layer))
    };
    let bytes: usize = members().map(|t| t.byte_size).sum();
    let plate = Plate {
        title: format!("{} layers", model.name),
        hint: "Layers".into(),
        readout: format!(
            "Layer {} of {} \u{b7} {} tensors \u{b7} {}",
            layer,
            layer_indices(model).len(),
            members().count(),
            human_bytes(bytes)
        ),
        caption: vec![
            "Each frame lights one layer in turn: every tensor of that layer, from the model's real layer structure.".into(),
            "Block sizes follow stored bytes and measured bits per weight, as on the static figure.".into(),
        ],
    };
    plate_svg(
        W,
        H,
        &format!("{} layers in turn", model.name),
        &scene_with(model, None, Some(layer), theme),
        &plate,
        theme,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ModelImporter;

    /// A small synthetic checkpoint written for the test (the mock under examples/ is generated
    /// locally and never committed, so tests cannot rely on it).
    fn mock() -> ModelIR {
        let dir = tempfile::tempdir().unwrap();
        let (model_dir, _) = crate::cli::evaluate::tests::synthetic_checkpoint(dir.path());
        ModelImporter::import(&model_dir).unwrap()
    }

    /// The model with its tensors repeated (renamed) to `n` tensors; layers keep their index.
    fn with_tensors(m: &ModelIR, n: usize) -> ModelIR {
        let mut out = m.clone();
        out.tensors = (0..n)
            .map(|i| {
                let mut t = m.tensors[i % m.tensors.len()].clone();
                t.name = format!("{}#{i}", t.name);
                t
            })
            .collect();
        out
    }

    fn hl(active: usize, phase: f64) -> Highlight {
        Highlight {
            active,
            lit: vec![active],
            phase,
        }
    }

    #[test]
    fn one_block_per_tensor_and_one_step_per_group() {
        let m = mock();
        let lay = layout(&m);
        assert_eq!(lay.blocks.len(), m.tensors.len());
        let layer_trays = lay.trays.iter().filter(|t| t.layer.is_some()).count();
        assert_eq!(layer_trays, m.num_layers);
        let body = scene_with(&m, None, None, Theme::Light);
        // Feet, bench, one riser per step above the bench, one plate per step, one solid per block.
        let risers = lay.trays.iter().filter(|t| t.step > BENCH_H).count();
        assert_eq!(
            body.matches("class=\"iso-solid\"").count(),
            4 + 1 + risers + lay.trays.len() + lay.blocks.len()
        );
        assert_eq!(
            body.matches("data-lit=\"\"").count(),
            1,
            "exactly one lit accent"
        );
        let svg = model_figure_svg(&m, Theme::Dark);
        assert!(
            !svg.contains("<style") && !svg.contains("class="),
            "no stylesheet needed"
        );
    }

    #[test]
    fn areas_follow_bytes_and_heights_follow_bits() {
        let m = mock();
        let lay = layout(&m);
        let area = |b: &Block| b.plan.w * b.plan.d;
        let (a, b) = (
            &lay.blocks[0],
            lay.blocks
                .iter()
                .find(|x| {
                    x.tensor.byte_size > 0 && x.tensor.byte_size != lay.blocks[0].tensor.byte_size
                })
                .unwrap(),
        );
        let ratio_area = area(a) / area(b);
        let ratio_bytes = a.tensor.byte_size as f64 / b.tensor.byte_size as f64;
        assert!(
            (ratio_area / ratio_bytes - 1.0).abs() < 1e-9,
            "area {ratio_area} vs bytes {ratio_bytes}"
        );
        let loose = lay
            .blocks
            .iter()
            .find(|x| x.tensor.layer_index.is_none())
            .unwrap();
        assert!((loose.height - (measured_bits(loose.tensor) * BITS_SCALE).max(1.0)).abs() < 1e-9);
    }

    /// Layer n+1 sits higher and further back than layer n; the embedding starts at the front
    /// left and the output head ends at the back right.
    #[test]
    fn staircase_rises_and_recedes_in_forward_order() {
        let m = mock();
        let lay = layout(&m);
        let p = fit(&lay);
        let centre = |t: &Tray| {
            let pl = &t.plan;
            (pl.x + pl.w / 2.0, pl.y + pl.d / 2.0, t.step)
        };
        let layers: Vec<&Tray> = lay.trays.iter().filter(|t| t.layer.is_some()).collect();
        assert!(layers.len() >= 2);
        for pair in layers.windows(2) {
            let (a, b) = (centre(pair[0]), centre(pair[1]));
            assert!(b.2 > a.2, "layer n+1 is higher");
            assert!(b.1 < a.1, "layer n+1 is further back");
            assert!(iso(b, &p).1 < iso(a, &p).1, "and higher on the page");
        }
        for pair in lay.trays.windows(2) {
            let (a, b) = (iso(centre(&pair[0]), &p), iso(centre(&pair[1]), &p));
            assert!(
                b.0 > a.0 && b.1 < a.1,
                "each step goes right and up the page"
            );
        }
        let first = &lay.trays[0];
        assert!(first.layer.is_none(), "the embedding starts the staircase");
        assert!(
            lay.trays.last().unwrap().layer.is_none(),
            "the head ends it"
        );
    }

    #[test]
    fn frames_light_tensors_in_forward_order() {
        let m = mock();
        let order = forward_order(&m);
        assert_eq!(order.len(), m.tensors.len());
        assert_eq!(m.tensors[order[0]].role, TensorRole::Embedding);
        let layers: Vec<usize> = order
            .iter()
            .filter_map(|&i| m.tensors[i].layer_index)
            .collect();
        assert!(
            layers.windows(2).all(|w| w[0] <= w[1]),
            "layers in order: {layers:?}"
        );
        let a = model_figure_frame(&m, Theme::Light, &hl(order[3], 0.3));
        assert_eq!(
            a,
            model_figure_frame(&m, Theme::Light, &hl(order[3], 0.3)),
            "deterministic"
        );
        assert_ne!(
            a,
            model_figure_frame(&m, Theme::Light, &hl(order[3], 0.8)),
            "the phase moves the effect"
        );
        assert!(a.contains(&m.tensors[order[3]].name) && a.contains("url(#glass-"));
        assert!(!a.contains("class="), "frames need no stylesheet either");
    }

    /// Pins the dark forward frames byte for byte, so refactors of the glass and splash geometry
    /// (shared with the Studio) cannot change the committed figure GIF.
    #[test]
    fn frames_are_byte_stable() {
        let m = mock();
        let order = forward_order(&m);
        let fnv = |s: &str| {
            s.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            })
        };
        let hashes: Vec<u64> = [(3, 0.1), (3, 0.5), (7, 0.9)]
            .iter()
            .map(|&(i, phase)| fnv(&model_figure_frame(&m, Theme::Dark, &hl(order[i], phase))))
            .collect();
        assert_eq!(hashes, vec![HASH_A, HASH_B, HASH_C]);
    }

    /// Pinned by `frames_are_byte_stable`; regenerate only when the figure is meant to change.
    const HASH_A: u64 = 12263347357770414038;
    const HASH_B: u64 = 3658038394791952048;
    const HASH_C: u64 = 7995440245341394307;

    #[test]
    fn deterministic_and_fits_the_frame() {
        let m = mock();
        assert_eq!(
            model_figure_svg(&m, Theme::Light),
            model_figure_svg(&m, Theme::Light)
        );
        let lay = layout(&m);
        let p = fit(&lay);
        for b in &lay.blocks {
            for z in [b.z, b.z + b.height] {
                let (x, y) = iso((b.plan.x, b.plan.y, z), &p);
                assert!((0.0..=W).contains(&x) && (0.0..=H).contains(&y));
            }
        }
        let svg = model_figure_svg(&m, Theme::Dark);
        assert!(svg.contains("tensors \u{b7}"));
    }

    #[test]
    fn card_is_a_manga_panel_with_the_wordmark_and_no_kit_label() {
        let m = mock();
        for theme in [Theme::Light, Theme::Dark] {
            let svg = model_figure_svg(&m, theme);
            assert!(svg.contains("EPIPHANES / THME"), "wordmark");
            assert!(svg.contains(&m.name) || svg.contains(&m.name.to_uppercase()));
            assert!(
                !svg.contains(">Fig ") && !svg.contains("Fig 1"),
                "no kit label"
            );
            assert!(svg.contains("Chakra Petch"), "Chakra Petch captions");
        }
    }

    #[test]
    fn layer_indices_are_the_real_layers() {
        let m = mock();
        assert_eq!(layer_indices(&m), (0..m.num_layers).collect::<Vec<_>>());
    }

    #[test]
    fn each_layer_frame_lights_exactly_its_tensors() {
        let m = mock();
        let count = layer_indices(&m).len();
        for l in layer_indices(&m) {
            let members: Vec<&TensorIR> = m
                .tensors
                .iter()
                .filter(|t| t.layer_index == Some(l))
                .collect();
            let raw = scene_with(&m, None, Some(l), Theme::Light);
            assert_eq!(
                raw.matches("data-lit=\"\"").count(),
                members.len(),
                "layer {l}"
            );
            let bytes: usize = members.iter().map(|t| t.byte_size).sum();
            let frame = model_figure_layer_frame(&m, Theme::Light, l);
            assert!(
                frame.contains(&format!("Layer {l} of {count}")),
                "layer {l}"
            );
            assert!(frame.contains(&human_bytes(bytes)), "layer {l} bytes");
        }
    }

    #[test]
    fn lit_layer_tensors_carry_the_glass_and_the_rest_do_not() {
        let m = mock();
        for theme in [Theme::Light, Theme::Dark] {
            for l in layer_indices(&m) {
                let members = m
                    .tensors
                    .iter()
                    .filter(|t| t.layer_index == Some(l))
                    .count();
                let raw = scene_with(&m, None, Some(l), theme);
                assert_eq!(raw.matches("url(#glass-").count(), members, "layer {l}");
                assert_eq!(
                    raw.matches("<linearGradient id=\"glass").count(),
                    members,
                    "one glass gradient per lit tensor, all ids unique"
                );
                let frame = model_figure_layer_frame(&m, theme, l);
                assert!(
                    !frame.contains("<style") && !frame.contains("class="),
                    "layer frames need no stylesheet"
                );
            }
        }
    }

    #[test]
    fn layer_frames_calm_the_tensors_outside_the_lit_layer() {
        let m = mock();
        let l = layer_indices(&m)[0];
        let outside = m.tensors.len()
            - m.tensors
                .iter()
                .filter(|t| t.layer_index == Some(l))
                .count();
        let raw = scene_with(&m, None, Some(l), Theme::Light);
        assert_eq!(raw.matches("<g opacity=\"").count(), outside);
    }

    #[test]
    fn layer_frames_differ_only_in_the_lit_layer() {
        let m = mock();
        let (a, b) = (
            scene_with(&m, None, Some(0), Theme::Dark),
            scene_with(&m, None, Some(1), Theme::Dark),
        );
        assert_eq!(
            a.matches("class=\"iso-solid\"").count(),
            b.matches("class=\"iso-solid\"").count()
        );
        assert_eq!(a.matches("<path").count(), b.matches("<path").count());
        assert_ne!(a, b);
        assert_eq!(
            model_figure_layer_frame(&m, Theme::Dark, 1),
            model_figure_layer_frame(&m, Theme::Dark, 1),
            "deterministic"
        );
    }

    #[test]
    fn dark_splash_is_blood_red_and_light_splash_has_no_red() {
        let m = mock();
        let order = forward_order(&m);
        let dark = model_figure_frame(&m, Theme::Dark, &hl(order[3], 0.5)).to_uppercase();
        assert!(dark.contains("#C8102E"), "dark splash core");
        assert!(dark.contains("#3A0610"), "dark splash edge");
        let light = model_figure_frame(&m, Theme::Light, &hl(order[3], 0.5)).to_uppercase();
        assert!(!light.contains("C8102E") && !light.contains("E0284A"));
        let layer_light = model_figure_layer_frame(&m, Theme::Light, 0).to_uppercase();
        assert!(!layer_light.contains("C8102E") && !layer_light.contains("E0284A"));
        let layer_dark = model_figure_layer_frame(&m, Theme::Dark, 0).to_uppercase();
        assert!(layer_dark.contains("#C8102E"), "layer splash in red");
    }

    #[test]
    fn a_splash_is_drawn_once_per_frame() {
        let m = mock();
        let order = forward_order(&m);
        let frame = model_figure_frame(&m, Theme::Light, &hl(order[3], 0.5));
        assert_eq!(frame.matches("stroke=\"none\" opacity=").count(), 1);
        let layer = model_figure_layer_frame(&m, Theme::Light, 0);
        assert_eq!(layer.matches("stroke=\"none\" opacity=").count(), 1);
    }

    #[test]
    fn forward_frames_for_76_tensors_are_unchanged_in_count() {
        let m = with_tensors(&mock(), 76);
        let steps = forward_steps(&m);
        assert_eq!(steps.len(), 76 * FORWARD_STEPS);
        assert!(steps.iter().all(|h| h.lit.len() == 1));
    }

    #[test]
    fn forward_frames_stay_under_the_cap_for_300_tensors() {
        let m = with_tensors(&mock(), 300);
        let steps = forward_steps(&m);
        assert!(steps.len() <= FORWARD_MAX_FRAMES, "{} frames", steps.len());
        let mut lit: Vec<usize> = steps.iter().flat_map(|h| h.lit.clone()).collect();
        lit.sort_unstable();
        lit.dedup();
        assert_eq!(lit.len(), 300, "every tensor lights up");
        let order = forward_order(&m);
        let first_lit: Vec<usize> = steps.iter().map(|h| h.active).collect();
        assert_eq!(first_lit[0], order[0], "forward order is kept");
        assert!(first_lit.windows(2).all(|w| w[0] != w[1]));
        let svg = model_figure_frame(&m, Theme::Dark, &steps[0]);
        assert!(svg.contains("to"), "a run names its first and last tensor");
    }
}
