// SPDX-License-Identifier: AGPL-3.0-only
//! `thme figure`: a model drawn as an isometric figure with the Rust anatomy kit
//! (`engine::iso`). The model is a parts tray on a bench:
//!
//! - one tray per transformer layer, side by side along the base, a ruler tick under each;
//! - one block per tensor: footprint area proportional to its stored bytes, height proportional
//!   to its measured bits per weight (stored bytes x 8 / weights), so a quantized tensor sits
//!   visibly lower than an FP16 one;
//! - tensors outside the layers (embedding, final norm, output head) stand on the base in front
//!   of and behind the trays; the largest tensor is the one lit accent.
//!
//! Every dimension is read from the model file; the figure invents no number.

use crate::engine::iso::{
    card_svg, dots_svg, fit_projection, inset_plan, iso, line_svg, plan_outline, segment, slab_of,
    solid_svg, Card, Flat, Plan, Point3, Projection, Scene, SolidStyle, Theme, Tone, AZIMUTH,
    ELEVATION,
};
use crate::ir::model::ModelIR;
use crate::ir::tensor::{TensorIR, TensorRole};

const W: f64 = 720.0;
const H: f64 = 440.0;
/// World units of block side per sqrt(byte).
const GAP: f64 = 2.0;

/// One block to draw.
struct Block<'a> {
    tensor: &'a TensorIR,
    plan: Plan,
    height: f64,
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

/// Lays out the blocks: layers along x, tensors of a layer along y; returns blocks, tray
/// plans and the base plan.
fn layout(model: &ModelIR) -> (Vec<Block<'_>>, Vec<(usize, Plan)>, Plan, f64) {
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
    let bits_scale = 1.6; // world units of height per bit per weight
    let height = |t: &TensorIR| (measured_bits(t) * bits_scale).max(1.0);

    let mut layers: std::collections::BTreeMap<usize, Vec<&TensorIR>> = Default::default();
    let mut loose: Vec<&TensorIR> = Vec::new();
    for t in &model.tensors {
        match t.layer_index {
            Some(i) => layers.entry(i).or_default().push(t),
            None => loose.push(t),
        }
    }
    let mut blocks = Vec::new();
    let mut trays = Vec::new();
    // Loose tensors in front (smaller x) of the trays, largest first.
    loose.sort_by(|a, b| b.byte_size.cmp(&a.byte_size).then(a.name.cmp(&b.name)));
    let mut x = 0.0;
    let mut y_front = 0.0;
    let mut front_width: f64 = 0.0;
    for t in &loose {
        let s = side(t);
        blocks.push(Block {
            tensor: t,
            plan: Plan {
                x,
                y: y_front,
                w: s,
                d: s,
                r: (s * 0.06).min(1.5),
            },
            height: height(t),
        });
        y_front += s + GAP * 2.0;
        front_width = front_width.max(s);
    }
    x += front_width + GAP * 6.0;
    for (index, mut tensors) in layers {
        tensors.sort_by(|a, b| {
            role_rank(&a.role)
                .cmp(&role_rank(&b.role))
                .then(a.name.cmp(&b.name))
        });
        let width = tensors.iter().map(|t| side(t)).fold(0.0, f64::max) + GAP * 2.0;
        let mut y = GAP;
        for t in &tensors {
            let s = side(t);
            blocks.push(Block {
                tensor: t,
                plan: Plan {
                    x: x + (width - s) / 2.0,
                    y,
                    w: s,
                    d: s,
                    r: (s * 0.06).min(1.5),
                },
                height: height(t) + 1.2,
            });
            y += s + GAP;
        }
        trays.push((
            index,
            Plan {
                x,
                y: 0.0,
                w: width,
                d: y + GAP,
                r: 1.0,
            },
        ));
        x += width + GAP * 2.5;
    }
    let depth = trays.iter().map(|(_, p)| p.d).fold(y_front, f64::max);
    let base = Plan {
        x: -8.0,
        y: -8.0,
        w: x + 12.0,
        d: depth + 16.0,
        r: 4.0,
    };
    (blocks, trays, base, bits_scale)
}

/// The figure on its card as one self-contained SVG file in `theme`: every colour resolved in
/// Rust, no stylesheet.
pub fn model_figure_svg(model: &ModelIR, theme: Theme) -> String {
    let total: usize = model.tensors.iter().map(|t| t.byte_size).sum();
    let card = Card {
        fig: "Fig 1".into(),
        title: model.name.clone(),
        hint: "Model anatomy".into(),
        readout: format!(
            "{} tensors \u{b7} {:.1} MiB",
            model.tensors.len(),
            total as f64 / (1u64 << 20) as f64
        ),
        caption: vec![
            "One tray per layer, one block per tensor. Footprint area follows the tensor's stored bytes;".into(),
            "height follows its measured bits per weight (stored bytes \u{d7} 8 \u{f7} weights). The lit block".into(),
            "is the largest tensor. Every dimension is read from the model file.".into(),
        ],
    };
    card_svg(
        W,
        H,
        &format!("{} drawn as a parts tray", model.name),
        &scene(model),
        &card,
        theme,
    )
}

fn scene(model: &ModelIR) -> String {
    scene_with(model, None)
}

fn scene_with(model: &ModelIR, highlight: Option<&Highlight>) -> String {
    let (blocks, trays, base, _) = layout(model);
    let p = fit(&blocks, &base);
    let mut s = Scene::new();
    let base_h = 6.0;

    // Feet under the base, then the base with its groove, corner screws, label plate and ruler.
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
    let mut inner = line_svg(
        &plan_outline(&inset_plan(&base, 3.5), base_h, &p, 8),
        Tone::Faint,
    );
    let screws: Vec<Flat> = [
        (base.x + 2.0, base.y + 2.0),
        (base.x + base.w - 2.0, base.y + 2.0),
        (base.x + 2.0, base.y + base.d - 2.0),
        (base.x + base.w - 2.0, base.y + base.d - 2.0),
    ]
    .iter()
    .map(|&(x, y)| iso((x, y, base_h), &p))
    .collect();
    inner += &dots_svg(&screws, 0.55, Tone::Mid);
    // Ruler along the front edge (y = base.y + base.d): one major tick per layer tray.
    let ruler_y = base.y + base.d - 1.6;
    for (_, tray) in &trays {
        let cx = tray.x + tray.w / 2.0;
        inner += &line_svg(
            &segment((cx, ruler_y, base_h), (cx, ruler_y - 2.6, base_h), &p),
            Tone::Mid,
        );
    }
    let first = trays.first().map(|(_, t)| t.x).unwrap_or(base.x);
    let last = trays
        .last()
        .map(|(_, t)| t.x + t.w)
        .unwrap_or(base.x + base.w);
    inner += &line_svg(
        &segment((first, ruler_y, base_h), (last, ruler_y, base_h), &p),
        Tone::Lo,
    );
    let slab = slab_of(&base, 0.0, base_h, &p, 8, 0.8);
    s.add(
        -9e5,
        solid_svg(
            &slab,
            &SolidStyle {
                tone: Tone::Lo,
                halo: true,
                inner,
                ..Default::default()
            },
        ),
    );

    // Trays: a thin plate with a groove per layer.
    for (_, tray) in &trays {
        let t = slab_of(tray, base_h, 1.2, &p, 6, 0.4);
        let groove = line_svg(
            &plan_outline(&inset_plan(tray, 0.8), base_h + 1.2, &p, 6),
            Tone::Faint,
        );
        s.add(
            -8e5,
            solid_svg(
                &t,
                &SolidStyle {
                    tone: Tone::Mid,
                    inner: groove,
                    ..Default::default()
                },
            ),
        );
    }

    // Blocks, painted by the depth of their nearest bottom corner. Without a highlight the
    // largest tensor is the lit accent; with one, the active tensor is.
    let largest = blocks.iter().map(|b| b.tensor.byte_size).max().unwrap_or(0);
    let active = highlight.and_then(|h| model.tensors.get(h.active));
    let mut lit_done = false;
    for b in &blocks {
        let z = base_h
            + if b.tensor.layer_index.is_some() {
                1.2
            } else {
                0.0
            };
        let lit = match active {
            Some(a) => std::ptr::eq(a, b.tensor),
            None => !lit_done && b.tensor.byte_size == largest,
        };
        lit_done |= lit;
        let solid = slab_of(&b.plan, z, b.height, &p, 4, (b.plan.w * 0.04).min(0.6));
        let style = SolidStyle {
            tone: if lit { Tone::Hi } else { Tone::Mid },
            lit,
            ..Default::default()
        };
        let near = (b.plan.x + b.plan.w, b.plan.y + b.plan.d, z);
        let depth = crate::engine::iso::depth_of(near, &p);
        match highlight {
            Some(h) if lit => {
                s.add(depth - 1e-6, ink_splash(b, z, &p, h));
                let mut svg = solid_svg(&solid, &style);
                svg.push_str(&glass(b, z, &p, &solid.fill, h));
                s.add(depth, svg);
            }
            _ => s.add(depth, solid_svg(&solid, &style)),
        }
    }
    s.render()
}

/// What to light in one animation frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Highlight {
    /// Index into the model's tensors.
    pub active: usize,
    /// 0 at the start of this tensor's turn, 1 at the end.
    pub phase: f64,
    /// Ink colour (manga ink) and the sheen colour of the glass.
    pub ink: String,
    pub sheen: String,
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

fn screen_box(b: &Block, z: f64, p: &Projection) -> (f64, f64, f64, f64) {
    let (x0, y0, x1, y1) = (b.plan.x, b.plan.y, b.plan.x + b.plan.w, b.plan.y + b.plan.d);
    let mut bx = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
        for zz in [z, z + b.height] {
            let (sx, sy) = iso((x, y, zz), p);
            bx = (bx.0.min(sx), bx.1.min(sy), bx.2.max(sx), bx.3.max(sy));
        }
    }
    bx
}

/// The active block as glossy ink glass: ink over its silhouette, then a bright band sweeping
/// diagonally across it with the phase.
fn glass(b: &Block, z: f64, p: &Projection, fill: &str, h: &Highlight) -> String {
    let (x0, y0, x1, y1) = screen_box(b, z, p);
    let (w, hh) = (x1 - x0, y1 - y0);
    let c = -0.4 + h.phase * 1.8;
    let (gx, gy) = (x0 + c * w, y0 + c * hh);
    let half = 0.22 * w.max(hh);
    format!(
        "<defs><linearGradient id=\"glass\" gradientUnits=\"userSpaceOnUse\" x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"><stop offset=\"0\" stop-color=\"{s}\" stop-opacity=\"0\"/><stop offset=\"0.5\" stop-color=\"{s}\" stop-opacity=\"0.7\"/><stop offset=\"1\" stop-color=\"{s}\" stop-opacity=\"0\"/></linearGradient></defs><path d=\"{fill}\" fill=\"{ink}\" fill-opacity=\"0.9\"/><path d=\"{fill}\" fill=\"url(#glass)\"/><path d=\"{fill}\" fill=\"none\" stroke=\"{s}\" stroke-opacity=\"0.5\" stroke-width=\"0.6\"/>",
        gx - half,
        gy - half,
        gx + half,
        gy + half,
        s = h.sheen,
        ink = h.ink,
    )
}

/// A manga ink splash on the paper under the active block: an organic blot with tendrils, flung
/// droplets (some stretched along their flight) and tapered speed lines, growing with the phase
/// and fading at the end. Seeded by the tensor, so every frame is reproducible.
fn ink_splash(b: &Block, z: f64, p: &Projection, h: &Highlight) -> String {
    use crate::bench::stats::SplitMix64;
    use std::f64::consts::TAU;
    let (x0, y0, x1, y1) = screen_box(b, z, p);
    let (cx, cy) = iso((b.plan.x + b.plan.w / 2.0, b.plan.y + b.plan.d / 2.0, z), p);
    let r = ((x1 - x0).max(y1 - y0) * 1.3).clamp(42.0, 120.0);
    let t = h.phase.clamp(0.0, 1.0);
    let grow = 1.0 - (1.0 - t).powi(3);
    let fade = if t < 0.7 {
        1.0
    } else {
        (1.0 - (t - 0.7) / 0.3).max(0.0)
    };
    let squash = 0.55; // the floor is foreshortened
    let at = |a: f64, k: f64| (cx + k * a.cos(), cy + k * squash * a.sin());
    let mut rng = SplitMix64::new(0x9e37_79b9 ^ (h.active as u64).wrapping_mul(0x2545_f491));
    let mut out = format!(
        "<g fill=\"{ink}\" stroke=\"none\" opacity=\"{fade:.3}\">",
        ink = h.ink
    );

    // Blot: a closed curve through jittered points, a few pushed out into tendrils.
    let n = 26;
    let pts: Vec<(f64, f64)> = (0..n)
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
    let mid = |a: (f64, f64), b: (f64, f64)| ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
    let start = mid(pts[n - 1], pts[0]);
    let mut d = format!("M{:.1} {:.1}", start.0, start.1);
    for i in 0..n {
        let m = mid(pts[i], pts[(i + 1) % n]);
        d += &format!("Q{:.1} {:.1} {:.1} {:.1}", pts[i].0, pts[i].1, m.0, m.1);
    }
    out += &format!("<path d=\"{d}Z\"/>");

    // Droplets flung outward; the farther ones are smaller and stretched along their flight.
    for _ in 0..22 {
        let a = rng.unit() * TAU;
        let reach = 0.7 + 1.1 * rng.unit();
        let (x, y) = at(a, r * reach * grow);
        let size = (1.4 + 4.2 * rng.unit()) * (1.25 - 0.45 * reach / 1.8);
        let stretch = 1.0 + 1.6 * (reach - 0.7) * grow;
        let deg = (a.sin() * squash).atan2(a.cos()).to_degrees();
        out += &format!(
            "<ellipse cx=\"{x:.1}\" cy=\"{y:.1}\" rx=\"{:.2}\" ry=\"{:.2}\" transform=\"rotate({deg:.1} {x:.1} {y:.1})\"/>",
            size * stretch,
            size
        );
    }

    // Speed lines: tapered wedges radiating from the blot, thick at the base, sharp at the tip.
    for _ in 0..16 {
        let a = rng.unit() * TAU;
        let (r0, r1) = (
            r * (0.55 + 0.15 * rng.unit()),
            r * (1.05 + 0.75 * rng.unit()) * (0.5 + 0.5 * grow),
        );
        let half = (0.8 + 1.8 * rng.unit()) * (0.6 + 0.4 * grow);
        let (bx, by) = at(a, r0);
        let (tx, ty) = at(a, r1);
        let (nx, ny) = (-(a.sin() * squash), a.cos());
        let len = (nx * nx + ny * ny).sqrt().max(1e-9);
        let (nx, ny) = (nx / len * half, ny / len * half);
        out += &format!(
            "<path d=\"M{:.1} {:.1}L{:.1} {:.1}L{tx:.1} {ty:.1}Z\"/>",
            bx + nx,
            by + ny,
            bx - nx,
            by - ny
        );
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

/// One animation frame on its card: the active tensor glows as ink glass with an ink splash,
/// and the readout names it.
pub fn model_figure_frame(model: &ModelIR, theme: Theme, highlight: &Highlight) -> String {
    let t = &model.tensors[highlight.active.min(model.tensors.len().saturating_sub(1))];
    let card = Card {
        fig: "Fig 2".into(),
        title: model.name.clone(),
        hint: "Forward pass".into(),
        readout: format!("{} \u{b7} {}", t.name, human_bytes(t.byte_size)),
        caption: vec![
            "Each tensor lights up in forward-pass order: the embedding, each layer's projections, then".into(),
            "the output. Block sizes follow stored bytes and measured bits per weight, as in Fig 1.".into(),
        ],
    };
    card_svg(
        W,
        H,
        &format!("{} forward pass", model.name),
        &scene_with(model, Some(highlight)),
        &card,
        theme,
    )
}

/// The camera that fits the base and every block, top and bottom, into the frame.
fn fit(blocks: &[Block], base: &Plan) -> Projection {
    let mut pts: Vec<Point3> = vec![
        (base.x, base.y, -4.0),
        (base.x + base.w, base.y + base.d, -4.0),
        (base.x, base.y + base.d, 0.0),
        (base.x + base.w, base.y, 0.0),
    ];
    for b in blocks {
        pts.push((b.plan.x, b.plan.y, b.height + 8.0));
        pts.push((b.plan.x + b.plan.w, b.plan.y + b.plan.d, b.height + 8.0));
    }
    fit_projection(&pts, W, H, 26.0, AZIMUTH, ELEVATION)
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

    #[test]
    fn one_block_per_tensor_and_one_tray_per_layer() {
        let m = mock();
        let (blocks, trays, _, _) = layout(&m);
        assert_eq!(blocks.len(), m.tensors.len());
        assert_eq!(trays.len(), m.num_layers);
        let body = scene(&m);
        // Feet, base and trays plus one solid per tensor.
        assert_eq!(
            body.matches("class=\"iso-solid\"").count(),
            4 + 1 + trays.len() + blocks.len()
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
        let (blocks, _, _, scale) = layout(&m);
        let area = |b: &Block| b.plan.w * b.plan.d;
        let (a, b) = (
            &blocks[0],
            blocks
                .iter()
                .find(|x| {
                    x.tensor.byte_size > 0 && x.tensor.byte_size != blocks[0].tensor.byte_size
                })
                .unwrap(),
        );
        let ratio_area = area(a) / area(b);
        let ratio_bytes = a.tensor.byte_size as f64 / b.tensor.byte_size as f64;
        assert!(
            (ratio_area / ratio_bytes - 1.0).abs() < 1e-9,
            "area {ratio_area} vs bytes {ratio_bytes}"
        );
        let loose = blocks
            .iter()
            .find(|x| x.tensor.layer_index.is_none())
            .unwrap();
        assert!((loose.height - (measured_bits(loose.tensor) * scale).max(1.0)).abs() < 1e-9);
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
        let h = |phase| Highlight {
            active: order[3],
            phase,
            ink: "#0b0b0b".into(),
            sheen: "#ffffff".into(),
        };
        let a = model_figure_frame(&m, Theme::Light, &h(0.3));
        assert_eq!(
            a,
            model_figure_frame(&m, Theme::Light, &h(0.3)),
            "deterministic"
        );
        assert_ne!(
            a,
            model_figure_frame(&m, Theme::Light, &h(0.8)),
            "the phase moves the effect"
        );
        assert!(a.contains(&m.tensors[order[3]].name) && a.contains("url(#glass)"));
        assert!(!a.contains("class="), "frames need no stylesheet either");
    }

    #[test]
    fn deterministic_and_fits_the_frame() {
        let m = mock();
        assert_eq!(
            model_figure_svg(&m, Theme::Light),
            model_figure_svg(&m, Theme::Light)
        );
        let (blocks, _, base, _) = layout(&m);
        let p = fit(&blocks, &base);
        for b in &blocks {
            let (x, y) = iso((b.plan.x, b.plan.y, b.height), &p);
            assert!((0.0..=W).contains(&x) && (0.0..=H).contains(&y));
        }
        let svg = model_figure_svg(&m, Theme::Dark);
        assert!(svg.contains("Fig 1") && svg.contains("tensors \u{b7}"));
    }
}
