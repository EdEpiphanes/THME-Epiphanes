// SPDX-License-Identifier: AGPL-3.0-only
//! Isometric figures in Rust: the core of the anatomy kit (`kit/iso-kit.ts`), ported line for
//! line so the same inputs give the same SVG strings (checked against the kit's own output in
//! `tests/iso_golden.rs`).
//!
//! The camera ([`Projection`], [`iso`], [`depth_of`], [`fit_projection`]), solids extruded from
//! a floor plan with four side tones by facing plus a lighter top ([`extrude`], [`slab_of`],
//! [`cylinder`]), lines, dots and wires, a [`Scene`] that paints back to front, and the figure
//! wrapper with the kit's stylesheet ([`ISO_CSS`]). World units are about 1 px of the real
//! thing: x and y on the floor, z up. Everything is deterministic and dependency-free.

use std::fmt::Write as _;

/// A point on the page.
pub type Flat = (f64, f64);
/// A point in the world: x and y on the floor, z up.
pub type Point3 = (f64, f64, f64);
/// A point on the floor plane.
pub type Plane = (f64, f64);

pub const AZIMUTH: f64 = 45.0;
pub const ELEVATION: f64 = 30.0;
pub const SHADES: usize = 4;

/// The kit's stylesheet: tones, shades, halo, lines, dots and the card.
pub const ISO_CSS: &str = include_str!("../../assets/iso.css");

/// Camera placement on the page: origin and scale in px per world unit, azimuth and elevation
/// in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    pub origin: Flat,
    pub scale: f64,
    pub azimuth: f64,
    pub elevation: f64,
}

impl Projection {
    pub fn new(origin: Flat, scale: f64) -> Self {
        Self {
            origin,
            scale,
            azimuth: AZIMUTH,
            elevation: ELEVATION,
        }
    }
}

/// A rounded rectangle on the floor: corner `x, y`, width `w` (along x), depth `d` (along y),
/// corner radius `r`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub d: f64,
    pub r: f64,
}

/// The paths of one solid, ready for [`solid_svg`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SolidPaths {
    pub fill: String,
    pub outline: String,
    pub crease: String,
    pub top: String,
    pub shades: [String; SHADES],
    pub bevel: Option<String>,
}

/// Line and edge tones, from the kit's tone hierarchy: base and feet `Lo`, structure `Mid`,
/// the subject `Hi`, grooves `Faint`, and exactly one `Lit` accent per figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    Hi,
    Mid,
    Lo,
    Faint,
    Lit,
    Red,
    Green,
    Blue,
}

impl Tone {
    fn name(self) -> &'static str {
        match self {
            Tone::Hi => "hi",
            Tone::Mid => "mid",
            Tone::Lo => "lo",
            Tone::Faint => "faint",
            Tone::Lit => "lit",
            Tone::Red => "red",
            Tone::Green => "green",
            Tone::Blue => "blue",
        }
    }
}

const RADIANS: f64 = std::f64::consts::PI / 180.0;

struct Camera {
    sin_a: f64,
    cos_a: f64,
    sin_e: f64,
    cos_e: f64,
    k: f64,
}

fn camera_of(p: &Projection) -> Camera {
    let a = p.azimuth * RADIANS;
    let e = p.elevation * RADIANS;
    Camera {
        sin_a: a.sin(),
        cos_a: a.cos(),
        sin_e: e.sin(),
        cos_e: e.cos(),
        k: p.scale * 1.6f64.sqrt(),
    }
}

/// Projects a world point to the page.
pub fn iso(point: Point3, p: &Projection) -> Flat {
    iso_cam(point, p.origin, &camera_of(p))
}

/// [`iso`] with the camera computed once, for loops over many points.
#[inline]
fn iso_cam((x, y, z): Point3, origin: Flat, c: &Camera) -> Flat {
    (
        origin.0 + (x * c.sin_a - y * c.cos_a) * c.k,
        origin.1 + ((x * c.cos_a + y * c.sin_a) * c.sin_e - z * c.cos_e) * c.k,
    )
}

/// The floor direction pointing at the viewer.
pub fn toward_viewer(p: &Projection) -> Plane {
    let c = camera_of(p);
    (c.cos_a, c.sin_a)
}

/// Depth toward the viewer; larger is nearer, so it is painted later.
pub fn depth_of((x, y, z): Point3, p: &Projection) -> f64 {
    let c = camera_of(p);
    (x * c.cos_a + y * c.sin_a) * c.cos_e + z * c.sin_e
}

/// Appends `value` rounded to one decimal exactly as the kit prints it (`Math.round`, halves
/// toward +infinity; no trailing `.0`; never `-0`), without allocating.
fn push_tenth(out: &mut String, value: f64) {
    let n = (value * 10.0 + 0.5).floor();
    if !n.is_finite() || n.abs() >= 9.0e15 {
        let _ = write!(out, "{}", n / 10.0);
        return;
    }
    let n = n as i64;
    if n < 0 {
        out.push('-');
    }
    let a = n.unsigned_abs();
    let _ = write!(out, "{}", a / 10);
    if !a.is_multiple_of(10) {
        out.push('.');
        out.push(char::from(b'0' + (a % 10) as u8));
    }
}

/// [`push_tenth`] into a new string.
fn tenth(value: f64) -> String {
    let mut s = String::new();
    push_tenth(&mut s, value);
    s
}

/// An SVG path through `points`.
pub fn path_of(points: &[Flat], closed: bool) -> String {
    let mut s = String::with_capacity(points.len() * 12 + 1);
    for (i, (x, y)) in points.iter().enumerate() {
        s.push(if i == 0 { 'M' } else { 'L' });
        push_tenth(&mut s, *x);
        s.push(' ');
        push_tenth(&mut s, *y);
    }
    if closed && !points.is_empty() {
        s.push('Z');
    }
    s
}

/// The outline of a rounded rectangle, `steps` segments per corner.
pub fn rounded_plan(plan: &Plan, steps: usize) -> Vec<Plane> {
    let Plan { x, y, w, d, r } = *plan;
    let radius = r.min(w / 2.0).min(d / 2.0).max(0.0);
    let corners = [
        (x + w - radius, y + radius, -90.0),
        (x + w - radius, y + d - radius, 0.0),
        (x + radius, y + d - radius, 90.0),
        (x + radius, y + radius, 180.0),
    ];
    let mut out = Vec::new();
    for (cx, cy, start) in corners {
        let count = if radius > 0.0 { steps } else { 0 };
        for step in 0..=count {
            let deg = start
                + if count > 0 {
                    90.0 * step as f64 / count as f64
                } else {
                    45.0
                };
            let angle = deg * RADIANS;
            out.push((cx + radius * angle.cos(), cy + radius * angle.sin()));
        }
    }
    out
}

fn signed_area(points: &[(f64, f64)]) -> f64 {
    let n = points.len();
    let mut area = 0.0;
    for i in 0..n {
        let (ax, ay) = points[i];
        let (bx, by) = points[(i + 1) % n];
        area += ax * by - bx * ay;
    }
    area / 2.0
}

fn oriented(points: Vec<Flat>) -> Vec<Flat> {
    if signed_area(&points) < 0.0 {
        points.into_iter().rev().collect()
    } else {
        points
    }
}

fn convex_hull(points: &[Flat]) -> Vec<Flat> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    if sorted.len() < 3 {
        return sorted;
    }
    let cross = |o: Flat, a: Flat, b: Flat| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut lower: Vec<Flat> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Flat> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Runs of consecutive edges whose flag equals `want`, as vertex index lists around the ring.
fn runs_of(flags: &[bool], want: bool) -> Vec<Vec<usize>> {
    let n = flags.len();
    let start = (0..n).find(|&i| flags[i] != flags[(i + n - 1) % n]);
    let Some(start) = start else {
        return if flags.first() == Some(&want) {
            vec![(0..=n).map(|i| i % n).collect()]
        } else {
            Vec::new()
        };
    };
    let mut runs = Vec::new();
    let mut run: Option<Vec<usize>> = None;
    for step in 0..n {
        let i = (start + step) % n;
        if flags[i] == want {
            let r = run.get_or_insert_with(|| vec![i]);
            r.push((i + 1) % n);
        } else if let Some(r) = run.take() {
            runs.push(r);
        }
    }
    if let Some(r) = run {
        runs.push(r);
    }
    runs
}

/// Options for [`extrude`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtrudeOptions {
    /// The fill is the convex hull of top and bottom (true for any convex plan).
    pub convex: bool,
    /// Bevel width on the visible top edge; 0 for none.
    pub bevel: f64,
}

impl Default for ExtrudeOptions {
    fn default() -> Self {
        Self {
            convex: true,
            bevel: 0.0,
        }
    }
}

/// Extrudes a floor ring from `z` up by `height`: the silhouette fill, outline, crease, top and
/// four side-shade buckets chosen by each face's facing to the light.
pub fn extrude(
    ring: &[Plane],
    z: f64,
    height: f64,
    p: &Projection,
    options: ExtrudeOptions,
) -> SolidPaths {
    let n = ring.len();
    if n == 0 {
        return SolidPaths::default();
    }
    let counter = signed_area(ring) > 0.0;
    let (vx, vy) = toward_viewer(p);
    let c = camera_of(p);
    let normals: Vec<Plane> = (0..n)
        .map(|i| {
            let (ax, ay) = ring[i];
            let (bx, by) = ring[(i + 1) % n];
            let (dx, dy) = (bx - ax, by - ay);
            let len = dx.hypot(dy);
            let len = if len == 0.0 { 1.0 } else { len };
            if counter {
                (dy / len, -dx / len)
            } else {
                (-dy / len, dx / len)
            }
        })
        .collect();
    let facing: Vec<bool> = normals
        .iter()
        .map(|(nx, ny)| nx * vx + ny * vy > 1e-9)
        .collect();
    let top: Vec<Flat> = ring
        .iter()
        .map(|&(x, y)| iso_cam((x, y, z + height), p.origin, &c))
        .collect();
    let bottom: Vec<Flat> = ring
        .iter()
        .map(|&(x, y)| iso_cam((x, y, z), p.origin, &c))
        .collect();
    let pick = |pts: &[Flat], run: &[usize]| -> Vec<Flat> { run.iter().map(|&i| pts[i]).collect() };
    let mut outline = Vec::new();
    let mut crease = Vec::new();
    for run in runs_of(&facing, false) {
        outline.push(path_of(&pick(&top, &run), false));
    }
    for run in runs_of(&facing, true) {
        crease.push(path_of(&pick(&top, &run), false));
        if height > 0.0 {
            outline.push(path_of(&pick(&bottom, &run), false));
        }
    }
    if height > 0.0 {
        for i in 0..n {
            if facing[(i + n - 1) % n] != facing[i] {
                outline.push(path_of(&[top[i], bottom[i]], false));
            }
        }
    }
    let quad = |i: usize| {
        let j = (i + 1) % n;
        path_of(&oriented(vec![top[i], top[j], bottom[j], bottom[i]]), true)
    };
    let fill = if options.convex {
        let mut all = top.clone();
        all.extend(&bottom);
        path_of(&convex_hull(&all), true)
    } else {
        let mut parts = vec![
            path_of(&oriented(top.clone()), true),
            path_of(&oriented(bottom.clone()), true),
        ];
        parts.extend((0..n).filter(|&i| facing[i]).map(quad));
        parts.concat()
    };
    let mut buckets: [Vec<String>; SHADES] = Default::default();
    if height > 0.0 {
        let (lx, ly) = (-c.sin_a, c.cos_a);
        for i in 0..n {
            if !facing[i] {
                continue;
            }
            let (nx, ny) = normals[i];
            let lit = (nx * lx + ny * ly + 1.0) / 2.0;
            let shade = ((lit * SHADES as f64).floor().max(0.0) as usize).min(SHADES - 1);
            buckets[shade].push(quad(i));
        }
    }
    let bevel = (options.bevel > 0.0).then(|| {
        let by = options.bevel;
        let inset_top: Vec<Flat> = (0..n)
            .map(|i| {
                let (ax, ay) = normals[(i + n - 1) % n];
                let (bx, by2) = normals[i];
                let (mx, my) = (ax + bx, ay + by2);
                let len = mx.hypot(my);
                let len = if len == 0.0 { 1.0 } else { len };
                let (x, y) = ring[i];
                iso_cam(
                    (x - mx / len * by, y - my / len * by, z + height),
                    p.origin,
                    &c,
                )
            })
            .collect();
        runs_of(&facing, true)
            .iter()
            .map(|run| {
                let mut pts = vec![top[run[0]]];
                pts.extend(run[1..run.len() - 1].iter().map(|&i| inset_top[i]));
                pts.push(top[run[run.len() - 1]]);
                path_of(&pts, false)
            })
            .collect::<String>()
    });
    SolidPaths {
        fill,
        outline: outline.concat(),
        crease: crease.concat(),
        top: path_of(&oriented(top), true),
        shades: buckets.map(|b| b.concat()),
        bevel,
    }
}

/// A rounded slab.
pub fn slab_of(
    plan: &Plan,
    z: f64,
    height: f64,
    p: &Projection,
    steps: usize,
    bevel: f64,
) -> SolidPaths {
    extrude(
        &rounded_plan(plan, steps),
        z,
        height,
        p,
        ExtrudeOptions {
            convex: true,
            bevel,
        },
    )
}

/// A circle on the floor.
pub fn circle_ring(cx: f64, cy: f64, radius: f64, steps: usize) -> Vec<Plane> {
    (0..steps)
        .map(|s| {
            let a = s as f64 / steps as f64 * std::f64::consts::PI * 2.0;
            (cx + radius * a.cos(), cy + radius * a.sin())
        })
        .collect()
}

/// An upright cylinder.
#[allow(clippy::too_many_arguments)]
pub fn cylinder(
    cx: f64,
    cy: f64,
    radius: f64,
    z: f64,
    height: f64,
    p: &Projection,
    steps: usize,
    bevel: f64,
) -> SolidPaths {
    extrude(
        &circle_ring(cx, cy, radius, steps),
        z,
        height,
        p,
        ExtrudeOptions {
            convex: true,
            bevel,
        },
    )
}

/// The plan shrunk by `by` on every side.
pub fn inset_plan(plan: &Plan, by: f64) -> Plan {
    Plan {
        x: plan.x + by,
        y: plan.y + by,
        w: plan.w - by * 2.0,
        d: plan.d - by * 2.0,
        r: (plan.r - by).max(0.0),
    }
}

/// A path through floor points lifted to height `z`.
pub fn on_top(points: &[Plane], z: f64, p: &Projection, closed: bool) -> String {
    let c = camera_of(p);
    let pts: Vec<Flat> = points
        .iter()
        .map(|&(x, y)| iso_cam((x, y, z), p.origin, &c))
        .collect();
    path_of(&pts, closed)
}

/// The outline of a plan at height `z` (a groove, an inlay edge).
pub fn plan_outline(plan: &Plan, z: f64, p: &Projection, steps: usize) -> String {
    on_top(&rounded_plan(plan, steps), z, p, true)
}

/// A straight line between two world points.
pub fn segment(from: Point3, to: Point3, p: &Projection) -> String {
    path_of(&[iso(from, p), iso(to, p)], false)
}

/// Fits the camera so `points` fill `width` by `height` with `pad` around them.
pub fn fit_projection(
    points: &[Point3],
    width: f64,
    height: f64,
    pad: f64,
    azimuth: f64,
    elevation: f64,
) -> Projection {
    let unit = Projection {
        origin: (0.0, 0.0),
        scale: 1.0,
        azimuth,
        elevation,
    };
    let flat: Vec<Flat> = points.iter().map(|&pt| iso(pt, &unit)).collect();
    let left = flat.iter().map(|f| f.0).fold(f64::INFINITY, f64::min);
    let right = flat.iter().map(|f| f.0).fold(f64::NEG_INFINITY, f64::max);
    let up = flat.iter().map(|f| f.1).fold(f64::INFINITY, f64::min);
    let down = flat.iter().map(|f| f.1).fold(f64::NEG_INFINITY, f64::max);
    let scale = ((width - pad * 2.0) / (right - left)).min((height - pad * 2.0) / (down - up));
    Projection {
        origin: (
            width / 2.0 - (left + right) / 2.0 * scale,
            height / 2.0 - (up + down) / 2.0 * scale,
        ),
        scale,
        azimuth,
        elevation,
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// How a solid is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct SolidStyle {
    pub tone: Tone,
    /// Tone of the crease and bevel lines; `None` draws neither.
    pub crease: Option<Tone>,
    pub halo: bool,
    pub lit: bool,
    /// Skip shades and top (floors and inlays thinner than 1 unit).
    pub flat: bool,
    pub class: Option<String>,
    /// Extra markup drawn under the outline (seams, grooves).
    pub inner: String,
}

impl Default for SolidStyle {
    fn default() -> Self {
        Self {
            tone: Tone::Hi,
            crease: Some(Tone::Faint),
            halo: false,
            lit: false,
            flat: false,
            class: None,
            inner: String::new(),
        }
    }
}

/// Markup for one solid.
pub fn solid_svg(paths: &SolidPaths, style: &SolidStyle) -> String {
    let mut s = String::new();
    let class = match &style.class {
        Some(c) => format!("iso-solid {c}"),
        None => "iso-solid".into(),
    };
    let _ = write!(
        s,
        "<g class=\"{}\"{}>",
        escape(&class),
        if style.lit { " data-lit=\"\"" } else { "" }
    );
    if style.halo {
        let _ = write!(s, "<path class=\"iso-halo\" d=\"{}\"/>", paths.fill);
    }
    let _ = write!(s, "<path class=\"iso-fill\" d=\"{}\"/>", paths.fill);
    if !style.flat {
        for (shade, d) in paths.shades.iter().enumerate() {
            if !d.is_empty() {
                let _ = write!(
                    s,
                    "<path class=\"iso-shade\" data-shade=\"{shade}\" d=\"{d}\"/>"
                );
            }
        }
        let _ = write!(s, "<path class=\"iso-top\" d=\"{}\"/>", paths.top);
    }
    if let Some(crease) = style.crease {
        if !paths.crease.is_empty() {
            let _ = write!(
                s,
                "<path class=\"iso-line iso-crease\" data-tone=\"{}\" d=\"{}\"/>",
                crease.name(),
                paths.crease
            );
        }
        if let Some(b) = paths.bevel.as_ref().filter(|b| !b.is_empty()) {
            let _ = write!(
                s,
                "<path class=\"iso-line iso-bevel\" data-tone=\"{}\" d=\"{b}\"/>",
                crease.name()
            );
        }
    }
    s.push_str(&style.inner);
    let _ = write!(
        s,
        "<path class=\"iso-line iso-edge\" data-tone=\"{}\" d=\"{}\"/></g>",
        style.tone.name(),
        paths.outline
    );
    s
}

/// Markup for a line; an empty path gives nothing.
pub fn line_svg(d: &str, tone: Tone) -> String {
    if d.is_empty() {
        return String::new();
    }
    format!(
        "<path class=\"iso-line\" data-tone=\"{}\" d=\"{}\"/>",
        tone.name(),
        escape(d)
    )
}

/// Screws, rivets and holes as small dots.
pub fn dots_svg(points: &[Flat], size: f64, tone: Tone) -> String {
    let mut s = format!("<g class=\"iso-dots\" data-tone=\"{}\">", tone.name());
    for (x, y) in points {
        let _ = write!(
            s,
            "<circle cx=\"{}\" cy=\"{}\" r=\"{size}\"/>",
            tenth(*x),
            tenth(*y)
        );
    }
    s.push_str("</g>");
    s
}

/// A wire from one object to another, through `via`, with terminal dots on both ends.
pub fn wire_svg(from: Flat, to: Flat, via: &[Flat], tone: Tone) -> String {
    let mut pts = vec![from];
    pts.extend_from_slice(via);
    pts.push(to);
    let end = |(x, y): Flat| {
        format!(
            "<circle class=\"iso-wire-end\" cx=\"{}\" cy=\"{}\" r=\"1.2\"/>",
            tenth(x),
            tenth(y)
        )
    };
    format!(
        "<g class=\"iso-wire\" data-tone=\"{}\">{}{}{}</g>",
        tone.name(),
        line_svg(&path_of(&pts, false), tone),
        end(from),
        end(to)
    )
}

/// Parts painted back to front: each item has a depth (see [`depth_of`]); nearer items are drawn
/// later. Ties keep insertion order, so a part split into back and front halves stays in order.
#[derive(Debug, Clone, Default)]
pub struct Scene {
    items: Vec<(f64, String)>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds markup at a depth.
    pub fn add(&mut self, depth: f64, svg: String) {
        self.items.push((depth, svg));
    }

    /// Adds markup at the depth of a world point (its nearest corner, for a box).
    pub fn add_at(&mut self, at: Point3, p: &Projection, svg: String) {
        self.add(depth_of(at, p), svg);
    }

    /// The markup in painter's order.
    pub fn render(&self) -> String {
        let mut order: Vec<usize> = (0..self.items.len()).collect();
        order.sort_by(|&a, &b| self.items[a].0.total_cmp(&self.items[b].0).then(a.cmp(&b)));
        order.iter().map(|&i| self.items[i].1.as_str()).collect()
    }
}

/// Light or dark theme for a standalone figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Dark,
    Light,
}

/// Wraps `body` in an `<svg>`. With `standalone`, the stylesheet is embedded so the file opens
/// on its own in that theme.
pub fn figure_svg(
    width: f64,
    height: f64,
    label: &str,
    body: &str,
    standalone: Option<Theme>,
) -> String {
    let (class, theme, style) = match standalone {
        Some(t) => (
            "iso-svg iso",
            format!(
                " data-theme=\"{}\"",
                if t == Theme::Dark { "dark" } else { "light" }
            ),
            format!("<style>{ISO_CSS}</style>"),
        ),
        None => ("iso-svg", String::new(), String::new()),
    };
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" class=\"{class}\"{theme} viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{}\">{style}{body}</svg>",
        escape(label)
    )
}

/// The card around a figure: `Fig n` top left, a title top right, the hint bottom left and the
/// live readout bottom right; the caption goes under the card.
pub fn plate_html(
    fig: &str,
    title: &str,
    hint: &str,
    readout: &str,
    caption: &str,
    body: &str,
) -> String {
    let corner = |name: &str, text: &str, extra: &str| {
        format!(
            "<span class=\"iso-plate-corner\" data-corner=\"{name}\"{extra}>{}</span>",
            escape(text)
        )
    };
    let mut corners = corner("fig", fig, "");
    if !title.is_empty() {
        corners += &corner("title", title, "");
    }
    if !hint.is_empty() {
        corners += &corner("hint", hint, "");
    }
    corners += &corner("readout", readout, " data-readout");
    let legend = if caption.is_empty() {
        String::new()
    } else {
        format!(
            "<figcaption class=\"iso-legend\"><span class=\"iso-caption\">{}</span></figcaption>",
            escape(caption)
        )
    };
    format!(
        "<figure class=\"iso-figure\"><div class=\"iso-plate\"><div class=\"iso-plate-corners\" aria-hidden=\"true\">{corners}</div>{body}</div>{legend}</figure>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_matches_javascript() {
        assert_eq!(tenth(1.25), "1.3");
        assert_eq!(
            tenth(-1.25),
            "-1.2",
            "Math.round sends halves toward +infinity"
        );
        assert_eq!(tenth(-0.04), "0");
        assert_eq!(tenth(2.0), "2");
        assert_eq!(tenth(12.349), "12.3");
    }

    #[test]
    fn camera_and_depth() {
        let p = Projection::new((100.0, 100.0), 1.0);
        let origin = iso((0.0, 0.0, 0.0), &p);
        assert_eq!(origin, (100.0, 100.0));
        let up = iso((0.0, 0.0, 10.0), &p);
        assert!(
            up.1 < origin.1 && (up.0 - origin.0).abs() < 1e-9,
            "z goes up the page"
        );
        assert!(
            depth_of((10.0, 10.0, 0.0), &p) > depth_of((0.0, 0.0, 0.0), &p),
            "toward the viewer is nearer"
        );
    }

    #[test]
    fn a_box_shows_two_shaded_sides_and_a_top() {
        let p = Projection::new((200.0, 150.0), 2.0);
        let s = slab_of(
            &Plan {
                x: 0.0,
                y: 0.0,
                w: 40.0,
                d: 30.0,
                r: 0.0,
            },
            0.0,
            10.0,
            &p,
            8,
            0.0,
        );
        let sides = s.shades.iter().filter(|d| !d.is_empty()).count();
        assert_eq!(
            sides, 2,
            "two visible faces at 45 degrees, in different tones: {:?}",
            s.shades
        );
        assert!(s.top.ends_with('Z') && s.fill.ends_with('Z'));
        let svg = solid_svg(
            &s,
            &SolidStyle {
                halo: true,
                ..Default::default()
            },
        );
        assert!(
            svg.contains("iso-halo") && svg.contains("iso-top") && svg.contains("data-tone=\"hi\"")
        );
    }

    #[test]
    fn scene_paints_back_to_front_and_keeps_ties_in_order() {
        let mut s = Scene::new();
        s.add(5.0, "<near/>".into());
        s.add(1.0, "<far/>".into());
        s.add(5.0, "<near-front/>".into());
        assert_eq!(s.render(), "<far/><near/><near-front/>");
    }

    #[test]
    fn fit_projection_fills_the_box() {
        let pts = [
            (0.0, 0.0, 0.0),
            (100.0, 60.0, 0.0),
            (0.0, 60.0, 40.0),
            (100.0, 0.0, 40.0),
        ];
        let p = fit_projection(&pts, 600.0, 400.0, 24.0, AZIMUTH, ELEVATION);
        let flat: Vec<Flat> = pts.iter().map(|&q| iso(q, &p)).collect();
        let minx = flat.iter().map(|f| f.0).fold(f64::INFINITY, f64::min);
        let maxx = flat.iter().map(|f| f.0).fold(f64::NEG_INFINITY, f64::max);
        let miny = flat.iter().map(|f| f.1).fold(f64::INFINITY, f64::min);
        let maxy = flat.iter().map(|f| f.1).fold(f64::NEG_INFINITY, f64::max);
        assert!(minx >= 23.9 && maxx <= 576.1 && miny >= 23.9 && maxy <= 376.1);
        assert!(
            (maxx - minx - 552.0).abs() < 1e-6 || (maxy - miny - 352.0).abs() < 1e-6,
            "one axis is tight"
        );
    }
}
