//! Manga x liquid-glass style kit for THME Studio, and the Calabi-Yau manifold geometry
//! behind the Live 3D "Manifold" view.
//!
//! Style: translucent glass panels over a painted backdrop (soft colour blobs, screentone,
//! speed lines), crisp ink outlines, hard offset "print" shadows, and a top sheen. Fonts:
//! Chakra Petch (SIL OFL 1.1, `assets/fonts/OFL.txt`) for the UI and display text.

use eframe::egui::{
    self, Color32, CornerRadius, FontFamily, FontId, LayerId, Mesh, Pos2, Rect, Shadow, Stroke,
    Vec2,
};
use std::sync::Arc;

use super::theme::ThemePalette;

/// Font family name for display text (headings, tabs, big numbers).
pub const DISPLAY: &str = "thme-display";

/// Corner radius of glass panels in colour themes.
pub const GLASS_RADIUS: u8 = 12;

/// Panel corner radius: near-square manga panels in black-and-white themes, rounded glass otherwise.
pub fn radius(p: &ThemePalette) -> u8 {
    if p.is_mono() {
        2
    } else {
        GLASS_RADIUS
    }
}

/// Registers Chakra Petch as the UI font and the bold cut as the display family.
/// Call once at start-up; `display()` fonts are only valid after this.
pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "chakra-medium".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "../../assets/fonts/ChakraPetch-Medium.ttf"
        ))),
    );
    fonts.font_data.insert(
        "chakra-bold".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "../../assets/fonts/ChakraPetch-Bold.ttf"
        ))),
    );
    let proportional = fonts.families.entry(FontFamily::Proportional).or_default();
    proportional.insert(0, "chakra-medium".to_owned());
    // Display text falls back to the proportional stack for glyphs Chakra Petch lacks.
    let mut display_stack = vec!["chakra-bold".to_owned()];
    display_stack.extend(proportional.iter().cloned());
    fonts
        .families
        .insert(FontFamily::Name(DISPLAY.into()), display_stack);
    ctx.set_fonts(fonts);
    ctx.style_mut(|style| {
        style
            .text_styles
            .insert(egui::TextStyle::Heading, display(17.0));
    });
}

/// A display-font `FontId` (requires `install_fonts`).
pub fn display(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(DISPLAY.into()))
}

/// Ink colour for outlines and print shadows.
pub fn ink(p: &ThemePalette) -> Color32 {
    if p.is_light() {
        Color32::from_rgb(12, 12, 14)
    } else {
        Color32::from_rgb(3, 3, 5)
    }
}

pub fn with_alpha(c: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha)
}

pub fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgba_unmultiplied(
        m(a.r(), b.r()),
        m(a.g(), b.g()),
        m(a.b(), b.b()),
        m(a.a(), b.a()),
    )
}

/// Translucent glass fill derived from a surface colour.
pub fn glass_fill(base: Color32, p: &ThemePalette) -> Color32 {
    with_alpha(base, if p.is_light() { 214 } else { 196 })
}

/// Hard offset print shadow: ink on paper themes, the spot colour on dark themes.
pub fn print_shadow(p: &ThemePalette) -> Shadow {
    Shadow {
        offset: [4, 4],
        blur: 0,
        spread: 0,
        color: if p.is_light() {
            ink(p)
        } else {
            with_alpha(p.accent, 90)
        },
    }
}

/// Glass card frame with an ink edge and a print shadow.
pub fn glass_frame(p: &ThemePalette, edge: Color32) -> egui::Frame {
    egui::Frame::NONE
        .fill(glass_fill(p.bg_card, p))
        .stroke(Stroke::new(1.5_f32, edge))
        .corner_radius(CornerRadius::same(radius(p)))
        .shadow(print_shadow(p))
}

/// Deeper glass for nested or emphasised panels.
pub fn glass_deep(p: &ThemePalette, edge: Color32) -> egui::Frame {
    egui::Frame::NONE
        .fill(glass_fill(p.bg_dark, p))
        .stroke(Stroke::new(1.5_f32, edge))
        .corner_radius(CornerRadius::same(radius(p)))
        .shadow(print_shadow(p))
}

/// Glass bar for top and bottom panels (no shadow, hairline edge).
pub fn glass_bar(p: &ThemePalette) -> egui::Frame {
    egui::Frame::NONE
        .fill(glass_fill(p.bg_card, p))
        .stroke(Stroke::new(1.0_f32, with_alpha(p.border, 120)))
}

/// Liquid-glass sheen: a bright top rim and a soft fade over the upper part of `rect`.
pub fn glass_sheen(painter: &egui::Painter, rect: Rect, radius: f32, p: &ThemePalette) {
    let white = Color32::WHITE;
    let top = rect.top() + 1.5;
    let inset = radius.min(rect.width() / 3.0);
    painter.line_segment(
        [
            Pos2::new(rect.left() + inset, top),
            Pos2::new(rect.right() - inset, top),
        ],
        Stroke::new(
            1.2_f32,
            with_alpha(white, if p.is_light() { 230 } else { 110 }),
        ),
    );
    let fade = Rect::from_min_max(
        Pos2::new(rect.left() + 2.0, rect.top() + 2.0),
        Pos2::new(rect.right() - 2.0, rect.top() + rect.height() * 0.45),
    );
    let mut mesh = Mesh::default();
    let strength = if p.is_light() { 60 } else { 22 };
    mesh.colored_vertex(fade.left_top(), with_alpha(white, strength));
    mesh.colored_vertex(fade.right_top(), with_alpha(white, strength));
    mesh.colored_vertex(fade.right_bottom(), with_alpha(white, 0));
    mesh.colored_vertex(fade.left_bottom(), with_alpha(white, 0));
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    painter.add(egui::Shape::mesh(mesh));
}

/// Full glass plate painted directly (for custom widgets): print shadow, fill, sheen, ink edge.
pub fn paint_glass_plate(painter: &egui::Painter, rect: Rect, edge: Color32, p: &ThemePalette) {
    let corner = CornerRadius::same(radius(p));
    let shadow = print_shadow(p);
    painter.rect_filled(
        rect.translate(Vec2::new(
            f32::from(shadow.offset[0]),
            f32::from(shadow.offset[1]),
        )),
        corner,
        shadow.color,
    );
    painter.rect_filled(rect, corner, glass_fill(p.bg_card, p));
    glass_sheen(painter, rect, f32::from(radius(p)), p);
    painter.rect_stroke(
        rect,
        corner,
        Stroke::new(if p.is_mono() { 2.0_f32 } else { 1.5_f32 }, edge),
        egui::StrokeKind::Inside,
    );
}

/// Soft radial blob for the liquid backdrop.
fn radial_blob(painter: &egui::Painter, center: Pos2, radius: f32, color: Color32) {
    let mut mesh = Mesh::default();
    mesh.colored_vertex(center, color);
    let ring = 48;
    for i in 0..=ring {
        let a = i as f32 / ring as f32 * std::f32::consts::TAU;
        mesh.colored_vertex(
            center + Vec2::new(a.cos(), a.sin()) * radius,
            with_alpha(color, 0),
        );
    }
    for i in 1..=ring as u32 {
        mesh.add_triangle(0, i, i + 1);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// Halftone screentone over `rect`: dots grow toward the bottom-right corner.
pub fn screentone(painter: &egui::Painter, rect: Rect, color: Color32, spacing: f32, max_r: f32) {
    let diag = rect.width() + rect.height();
    let mut y = rect.top() + spacing / 2.0;
    let mut row = 0;
    while y < rect.bottom() {
        let offset = if row % 2 == 0 { 0.0 } else { spacing / 2.0 };
        let mut x = rect.left() + offset;
        while x < rect.right() {
            let t = ((x - rect.left()) + (y - rect.top())) / diag;
            let r = max_r * ((t - 0.45) / 0.55).clamp(0.0, 1.0);
            if r > 0.25 {
                painter.circle_filled(Pos2::new(x, y), r, color);
            }
            x += spacing;
        }
        y += spacing;
        row += 1;
    }
}

/// Manga speed lines radiating from a focal point outside the top-left corner.
pub fn speed_lines(painter: &egui::Painter, rect: Rect, color: Color32, count: usize) {
    let focus = rect.left_top() + Vec2::new(-rect.width() * 0.15, -rect.height() * 0.25);
    let reach = rect.width().hypot(rect.height()) * 1.4;
    for i in 0..count {
        let a = 0.05 + 1.45 * (i as f32 + 0.5) / count as f32;
        let jitter = ((i * 7919) % 13) as f32 / 13.0;
        let start = 0.35 + 0.25 * jitter;
        let dir = Vec2::new(a.cos(), a.sin());
        painter.line_segment(
            [focus + dir * reach * start, focus + dir * reach],
            Stroke::new(0.6 + 0.8 * jitter, color),
        );
    }
}

/// Paints the app backdrop on the background layer: liquid colour blobs, speed lines and
/// screentone. Panels drawn with `glass_*` frames show it through their translucent fill.
pub fn paint_backdrop(ctx: &egui::Context, p: &ThemePalette) {
    let rect = ctx.screen_rect();
    let painter = ctx.layer_painter(LayerId::background());
    painter.rect_filled(rect, CornerRadius::ZERO, p.bg_dark);
    let light = p.is_light();
    let blob = if light { 70 } else { 46 };
    if !p.is_mono() {
        radial_blob(
            &painter,
            rect.left_top() + rect.size() * Vec2::new(0.12, 0.18),
            rect.width() * 0.42,
            with_alpha(p.accent, blob),
        );
        radial_blob(
            &painter,
            rect.left_top() + rect.size() * Vec2::new(0.88, 0.30),
            rect.width() * 0.36,
            with_alpha(p.primary, blob / 2),
        );
        radial_blob(
            &painter,
            rect.left_top() + rect.size() * Vec2::new(0.62, 0.92),
            rect.width() * 0.40,
            with_alpha(p.secondary, blob / 2),
        );
    }
    let line = with_alpha(p.text_main, if light { 20 } else { 8 });
    speed_lines(&painter, rect, line, 42);
    let dot = with_alpha(p.text_main, if light { 34 } else { 16 });
    screentone(&painter, rect, dot, 11.0, 2.2);
}

// ============================================================================
// Calabi-Yau quintic geometry
// ============================================================================

/// One coordinate patch of the Calabi-Yau cross-section, sampled on a `nu x nv` grid and
/// projected to 3D. `points[j * nu + i]`.
#[derive(Debug, Clone)]
pub struct ManifoldPatch {
    pub k1: usize,
    pub k2: usize,
    pub nu: usize,
    pub nv: usize,
    pub points: Vec<[f32; 3]>,
}

type Complex = (f32, f32);

fn c_mul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Principal-branch real power of a complex number.
fn c_pow(w: Complex, p: f32) -> Complex {
    let r = w.0.hypot(w.1);
    if r == 0.0 {
        return (0.0, 0.0);
    }
    let theta = w.1.atan2(w.0);
    let rp = r.powf(p);
    (rp * (p * theta).cos(), rp * (p * theta).sin())
}

fn c_phase(angle: f32) -> Complex {
    (angle.cos(), angle.sin())
}

/// The pair (z1, z2) on the curve z1^n + z2^n = 1 for patch (k1, k2) at parameters
/// (x, y), x in [-1, 1], y in [0, pi/2] (Hanson's parametrization).
fn quintic_point(n: usize, k1: usize, k2: usize, x: f32, y: f32) -> (Complex, Complex) {
    let nf = n as f32;
    let cosh_z = (x.cosh() * y.cos(), x.sinh() * y.sin());
    // sinh(z) / i
    let sinh_over_i = (x.cosh() * y.sin(), -(x.sinh() * y.cos()));
    let tau = std::f32::consts::TAU;
    let z1 = c_mul(c_phase(tau * k1 as f32 / nf), c_pow(cosh_z, 2.0 / nf));
    let z2 = c_mul(c_phase(tau * k2 as f32 / nf), c_pow(sinh_over_i, 2.0 / nf));
    (z1, z2)
}

/// Samples all `n * n` patches of the degree-`n` Calabi-Yau cross-section, projecting
/// (Re z1, Re z2, Im z1, Im z2) to 3D as (Re z1, Re z2, cos(a) Im z1 + sin(a) Im z2).
pub fn calabi_yau_patches(n: usize, nu: usize, nv: usize, alpha: f32) -> Vec<ManifoldPatch> {
    let n = n.max(2);
    let nu = nu.max(2);
    let nv = nv.max(2);
    let (sa, ca) = alpha.sin_cos();
    let mut patches = Vec::with_capacity(n * n);
    for k1 in 0..n {
        for k2 in 0..n {
            let mut points = Vec::with_capacity(nu * nv);
            for j in 0..nv {
                let y = std::f32::consts::FRAC_PI_2 * j as f32 / (nv - 1) as f32;
                for i in 0..nu {
                    let x = -1.0 + 2.0 * i as f32 / (nu - 1) as f32;
                    let (z1, z2) = quintic_point(n, k1, k2, x, y);
                    points.push([z1.0, z2.0, ca * z1.1 + sa * z2.1]);
                }
            }
            patches.push(ManifoldPatch {
                k1,
                k2,
                nu,
                nv,
                points,
            });
        }
    }
    patches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quintic_points_satisfy_the_curve_equation() {
        let n = 5;
        for k1 in 0..n {
            for k2 in 0..n {
                for &(x, y) in &[(-0.9f32, 0.1f32), (0.0, 0.7), (0.6, 1.4), (1.0, 0.0)] {
                    let (z1, z2) = quintic_point(n, k1, k2, x, y);
                    let mut a = (1.0, 0.0);
                    let mut b = (1.0, 0.0);
                    for _ in 0..n {
                        a = c_mul(a, z1);
                        b = c_mul(b, z2);
                    }
                    let sum = (a.0 + b.0, a.1 + b.1);
                    assert!((sum.0 - 1.0).abs() < 1e-3, "{sum:?} at {k1},{k2},{x},{y}");
                    assert!(sum.1.abs() < 1e-3, "{sum:?} at {k1},{k2},{x},{y}");
                }
            }
        }
    }

    #[test]
    fn patches_cover_the_grid_with_finite_points() {
        let patches = calabi_yau_patches(5, 9, 7, 0.8);
        assert_eq!(patches.len(), 25);
        for patch in &patches {
            assert_eq!(patch.points.len(), 9 * 7);
            assert!(patch
                .points
                .iter()
                .all(|p| p.iter().all(|v| v.is_finite() && v.abs() < 3.0)));
        }
    }

    #[test]
    fn colour_helpers_mix_and_keep_alpha() {
        let a = Color32::from_rgba_unmultiplied(0, 0, 0, 255);
        let b = Color32::from_rgba_unmultiplied(200, 100, 50, 255);
        assert_eq!(
            mix(a, b, 0.5),
            Color32::from_rgba_unmultiplied(100, 50, 25, 255)
        );
        assert_eq!(with_alpha(b, 10).a(), 10);
        let p = super::super::theme::GuiTheme::MangaNight.palette();
        assert!(glass_fill(p.bg_card, &p).a() < 255);
        assert_eq!(
            glass_frame(&p, p.border).corner_radius,
            CornerRadius::same(radius(&p))
        );
    }
}
