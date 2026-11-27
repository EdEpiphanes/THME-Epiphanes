// SPDX-License-Identifier: AGPL-3.0-only
//! Manga ink splash geometry and SVG rendering for the model figure and Studio's live view.

use super::Manga;
use crate::engine::iso::Flat;

/// Scale of the core of a two-tone splash inside its outer edge.
const CORE_SCALE: f64 = 0.72;
/// Droplets drawn on a two-tone splash.
const CORE_DROPS: usize = 9;

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
pub(super) fn splash_svg(centre: Flat, r: f64, seed: usize, phase: f64, m: &Manga) -> String {
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
