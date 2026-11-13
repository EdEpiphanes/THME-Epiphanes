// SPDX-License-Identifier: AGPL-3.0-only
//! SVG visual engine (work package WP9, dossier Appendix M.6).
//!
//! Static, deterministic SVG for the dossier, the paper and the Studio: identical input gives
//! byte-identical output (fixed number formatting, no hash-map order). Five chart kinds:
//! line plots with an optional log axis (solver traces), heatmaps (layer by role), scatter
//! plots (projections, with a caption that states what the projection kept), layered dataflow
//! graphs, and Pareto fronts.
//!
//! Every chart prints its evidence label, so a chart of simulated or toy data says so. Colours
//! and the font come from a [`Theme`] read from `assets/chart-themes.json`; nothing here
//! hard-codes a palette. Dependency-free: the SVG is written by hand.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::error::{EngineError, EngineResult};

const W: f64 = 760.0;
const H: f64 = 440.0;
const LEFT: f64 = 78.0;
const RIGHT: f64 = 32.0;
const TOP: f64 = 64.0;
const BOTTOM: f64 = 70.0;

/// Colours and font for a chart, read from a theme file.
#[derive(Debug, Clone, PartialEq)]
pub struct Theme {
    pub font: String,
    pub surface: String,
    pub border: String,
    pub ink: String,
    pub ink2: String,
    pub muted: String,
    pub grid: String,
    pub axis: String,
    /// Series colours, told apart by tone.
    pub series: Vec<String>,
    pub deemph: String,
    pub mark: String,
}

impl Theme {
    /// Reads theme `name` ("light" or "dark") from a theme document such as
    /// `assets/chart-themes.json`.
    pub fn from_json(doc: &serde_json::Value, name: &str) -> EngineResult<Self> {
        let t = doc
            .get(name)
            .ok_or_else(|| EngineError::InvalidInput(format!("theme '{name}' not found")))?;
        let get = |k: &str| -> EngineResult<String> {
            t.get(k)
                .and_then(|v| v.as_str())
                .map(String::from)
                .ok_or_else(|| {
                    EngineError::InvalidInput(format!("theme '{name}' has no colour '{k}'"))
                })
        };
        let font = doc
            .get("font")
            .and_then(|v| v.as_str())
            .ok_or_else(|| EngineError::InvalidInput("theme document has no font".into()))?;
        Ok(Self {
            font: font.to_string(),
            surface: get("surface")?,
            border: get("border")?,
            ink: get("ink")?,
            ink2: get("ink2")?,
            muted: get("muted")?,
            grid: get("grid")?,
            axis: get("axis")?,
            series: vec![get("s1")?, get("s2")?],
            deemph: get("deemph")?,
            mark: get("mark")?,
        })
    }

    /// Reads a theme from a file.
    pub fn load(path: &std::path::Path, name: &str) -> EngineResult<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| EngineError::InvalidInput(format!("theme file {path:?}: {e}")))?;
        let doc: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| EngineError::InvalidInput(format!("theme file {path:?}: {e}")))?;
        Self::from_json(&doc, name)
    }
}

/// What every chart states about itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub title: String,
    /// Measured, Tested, Estimate, Simulated or Oracle (see CONTEXT.md); printed on the chart.
    pub evidence: String,
    pub x_label: String,
    pub y_label: String,
}

/// A named series of (x, y) points.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    pub name: String,
    pub points: Vec<(f64, f64)>,
}

/// Line plot; `log_y` puts the y axis on a log scale (every y must then be positive).
#[derive(Debug, Clone, PartialEq)]
pub struct LinePlot {
    pub frame: Frame,
    pub series: Vec<Series>,
    pub log_y: bool,
}

/// Heatmap: one row per label in `rows`, one column per label in `cols`; `None` is a missing
/// cell, drawn empty.
#[derive(Debug, Clone, PartialEq)]
pub struct Heatmap {
    pub frame: Frame,
    pub rows: Vec<String>,
    pub cols: Vec<String>,
    pub values: Vec<Vec<Option<f64>>>,
}

/// Scatter plot. `caption` is required: for a projection it must say what was kept, for
/// example "2 of 11 dimensions; 40% of variance".
#[derive(Debug, Clone, PartialEq)]
pub struct Scatter {
    pub frame: Frame,
    pub points: Vec<(f64, f64, String)>,
    pub caption: String,
}

/// Dataflow graph: nodes are laid out in columns by topological depth; edge labels name the
/// format at each boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataflow {
    pub frame: Frame,
    pub nodes: Vec<(String, String)>,
    pub edges: Vec<(String, String, String)>,
}

/// Kind of point on a Pareto chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointKind {
    Candidate,
    Baseline,
    Control,
}

/// One row of an interval plot: a label, a point estimate and its interval.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalRow {
    pub label: String,
    pub value: f64,
    pub low: f64,
    pub high: f64,
}

/// Dot plot with interval whiskers, one row per entry; `reference` draws a vertical line (for
/// example zero for differences, or a gate margin).
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalPlot {
    pub frame: Frame,
    pub rows: Vec<IntervalRow>,
    pub reference: Option<f64>,
}

/// Pareto chart of cost (x, lower is better) against loss (y, lower is better). The front of
/// non-dominated candidates is drawn as a step line.
#[derive(Debug, Clone, PartialEq)]
pub struct Pareto {
    pub frame: Frame,
    pub points: Vec<(f64, f64, String, PointKind)>,
}

/// Fixed-format number for labels and coordinates.
fn num(x: f64) -> String {
    let s = format!("{x:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".into()
    } else {
        s.to_string()
    }
}

/// Short label for an axis tick.
fn tick_label(x: f64) -> String {
    let a = x.abs();
    if a != 0.0 && !(1e-3..1e5).contains(&a) {
        format!("{x:.0e}")
    } else if a >= 100.0 {
        format!("{x:.0}")
    } else {
        let s = format!("{x:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn check_frame(f: &Frame) -> EngineResult<()> {
    if f.evidence.trim().is_empty() {
        return Err(EngineError::InvalidInput(
            "every chart needs an evidence label (Measured, Tested, Estimate, Simulated or Oracle)"
                .into(),
        ));
    }
    Ok(())
}

fn finite(what: &str, values: impl Iterator<Item = f64>) -> EngineResult<()> {
    for (index, v) in values.enumerate() {
        if !v.is_finite() {
            return Err(EngineError::NonFinite {
                what: what.to_string(),
                index,
            });
        }
    }
    Ok(())
}

/// About five round tick values covering [lo, hi].
fn nice_ticks(lo: f64, hi: f64) -> Vec<f64> {
    let span = (hi - lo).max(1e-12);
    let raw = span / 5.0;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|s| span / s <= 6.0)
        .unwrap_or(10.0 * mag);
    let first = (lo / step).ceil() * step;
    (0..)
        .map(|i| first + i as f64 * step)
        .take_while(|t| *t <= hi + step * 1e-9)
        .take(12)
        .collect()
}

/// A linear or log axis mapping data to pixels.
struct Axis {
    lo: f64,
    hi: f64,
    log: bool,
    p0: f64,
    p1: f64,
}

impl Axis {
    fn new(values: &[f64], log: bool, p0: f64, p1: f64) -> Self {
        let tr = |v: f64| if log { v.log10() } else { v };
        let mut lo = values.iter().copied().map(tr).fold(f64::INFINITY, f64::min);
        let mut hi = values
            .iter()
            .copied()
            .map(tr)
            .fold(f64::NEG_INFINITY, f64::max);
        if !lo.is_finite() {
            (lo, hi) = (0.0, 1.0);
        }
        if log {
            lo = lo.floor();
            hi = hi.ceil().max(lo + 1.0);
        } else if hi - lo < 1e-12 {
            (lo, hi) = (lo - 0.5, hi + 0.5);
        } else {
            let pad = (hi - lo) * 0.05;
            (lo, hi) = (lo - pad, hi + pad);
        }
        Self {
            lo,
            hi,
            log,
            p0,
            p1,
        }
    }

    fn px(&self, v: f64) -> f64 {
        let t = if self.log { v.log10() } else { v };
        self.p0 + (t - self.lo) / (self.hi - self.lo) * (self.p1 - self.p0)
    }

    fn ticks(&self) -> Vec<f64> {
        if self.log {
            (self.lo as i32..=self.hi as i32)
                .map(|e| 10f64.powi(e))
                .collect()
        } else {
            nice_ticks(self.lo, self.hi)
        }
    }
}

/// Opens an SVG with background, title, axis labels and the evidence label.
fn open(frame: &Frame, theme: &Theme) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {W} {H}\" width=\"{W}\" height=\"{H}\" font-family=\"{}\">",
        esc(&theme.font)
    );
    let _ = write!(
        s,
        "<rect x=\"0\" y=\"0\" width=\"{W}\" height=\"{H}\" fill=\"{}\"/>",
        theme.surface
    );
    let _ = write!(
        s,
        "<text x=\"{LEFT}\" y=\"34\" font-size=\"18\" font-weight=\"700\" fill=\"{}\">{}</text>",
        theme.ink,
        esc(&frame.title)
    );
    let _ = write!(
        s,
        "<text class=\"evidence\" x=\"{}\" y=\"34\" font-size=\"12\" text-anchor=\"end\" fill=\"{}\">Evidence: {}</text>",
        W - RIGHT,
        theme.muted,
        esc(&frame.evidence)
    );
    let _ = write!(
        s,
        "<text x=\"{}\" y=\"{}\" font-size=\"12\" text-anchor=\"middle\" fill=\"{}\">{}</text>",
        num(LEFT + (W - LEFT - RIGHT) / 2.0),
        H - 18.0,
        theme.ink2,
        esc(&frame.x_label)
    );
    let _ = write!(
        s,
        "<text x=\"18\" y=\"{}\" font-size=\"12\" text-anchor=\"middle\" fill=\"{}\" transform=\"rotate(-90 18 {})\">{}</text>",
        num(TOP + (H - TOP - BOTTOM) / 2.0),
        theme.ink2,
        num(TOP + (H - TOP - BOTTOM) / 2.0),
        esc(&frame.y_label)
    );
    s
}

fn axes(s: &mut String, x: &Axis, y: &Axis, theme: &Theme) {
    for t in y.ticks() {
        let py = num(y.px(t));
        let _ = write!(
            s,
            "<line x1=\"{LEFT}\" y1=\"{py}\" x2=\"{}\" y2=\"{py}\" stroke=\"{}\" stroke-width=\"1\"/>",
            W - RIGHT,
            theme.grid
        );
        let _ = write!(
            s,
            "<text x=\"{}\" y=\"{py}\" dy=\"4\" font-size=\"11\" text-anchor=\"end\" fill=\"{}\">{}</text>",
            LEFT - 8.0,
            theme.muted,
            tick_label(t)
        );
    }
    for t in x.ticks() {
        let px = num(x.px(t));
        let _ = write!(
            s,
            "<text x=\"{px}\" y=\"{}\" font-size=\"11\" text-anchor=\"middle\" fill=\"{}\">{}</text>",
            H - BOTTOM + 18.0,
            theme.muted,
            tick_label(t)
        );
    }
    let _ = write!(
        s,
        "<line x1=\"{LEFT}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
        H - BOTTOM,
        W - RIGHT,
        H - BOTTOM,
        theme.axis
    );
}

/// Line plot; a non-positive or non-finite y on a log axis is an error, not a clipped point.
pub fn line_plot(plot: &LinePlot, theme: &Theme) -> EngineResult<String> {
    check_frame(&plot.frame)?;
    if plot.series.is_empty() || plot.series.iter().any(|s| s.points.is_empty()) {
        return Err(EngineError::InvalidInput(
            "a line plot needs series with points".into(),
        ));
    }
    let xs: Vec<f64> = plot
        .series
        .iter()
        .flat_map(|s| s.points.iter().map(|p| p.0))
        .collect();
    let ys: Vec<f64> = plot
        .series
        .iter()
        .flat_map(|s| s.points.iter().map(|p| p.1))
        .collect();
    finite("x", xs.iter().copied())?;
    finite("y", ys.iter().copied())?;
    if plot.log_y {
        if let Some(i) = ys.iter().position(|y| *y <= 0.0) {
            return Err(EngineError::InvalidInput(format!(
                "log axis: y value {} at index {i} is not positive",
                ys[i]
            )));
        }
    }
    let x = Axis::new(&xs, false, LEFT, W - RIGHT);
    let y = Axis::new(&ys, plot.log_y, H - BOTTOM, TOP);
    let mut s = open(&plot.frame, theme);
    axes(&mut s, &x, &y, theme);
    for (k, series) in plot.series.iter().enumerate() {
        let colour = &theme.series[k % theme.series.len()];
        let dash = if k % 2 == 1 {
            " stroke-dasharray=\"6 4\""
        } else {
            ""
        };
        let path: Vec<String> = series
            .points
            .iter()
            .map(|(a, b)| format!("{},{}", num(x.px(*a)), num(y.px(*b))))
            .collect();
        let _ = write!(
            s,
            "<polyline class=\"series\" fill=\"none\" stroke=\"{colour}\" stroke-width=\"2\"{dash} points=\"{}\"/>",
            path.join(" ")
        );
        for (a, b) in &series.points {
            let _ = write!(
                s,
                "<circle class=\"point\" cx=\"{}\" cy=\"{}\" r=\"3\" fill=\"{colour}\"/>",
                num(x.px(*a)),
                num(y.px(*b))
            );
        }
        if let Some((a, b)) = series.points.last() {
            let _ = write!(
                s,
                "<text x=\"{}\" y=\"{}\" dx=\"6\" dy=\"4\" font-size=\"12\" fill=\"{}\">{}</text>",
                num(x.px(*a)),
                num(y.px(*b)),
                theme.ink,
                esc(&series.name)
            );
        }
    }
    s.push_str("</svg>\n");
    Ok(s)
}

/// Mixes two `#rrggbb` colours; `t` = 0 gives `a`, 1 gives `b`.
fn mix(a: &str, b: &str, t: f64) -> EngineResult<String> {
    let rgb = |c: &str| -> EngineResult<[u8; 3]> {
        let h = c
            .strip_prefix('#')
            .filter(|h| h.len() == 6)
            .ok_or_else(|| EngineError::InvalidInput(format!("colour {c} is not #rrggbb")))?;
        let mut out = [0u8; 3];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u8::from_str_radix(&h[2 * i..2 * i + 2], 16)
                .map_err(|_| EngineError::InvalidInput(format!("colour {c} is not #rrggbb")))?;
        }
        Ok(out)
    };
    let (ca, cb) = (rgb(a)?, rgb(b)?);
    let t = t.clamp(0.0, 1.0);
    let ch =
        |i: usize| (f64::from(ca[i]) + (f64::from(cb[i]) - f64::from(ca[i])) * t).round() as u8;
    Ok(format!("#{:02x}{:02x}{:02x}", ch(0), ch(1), ch(2)))
}

/// Heatmap from surface (lowest value) to ink (highest), each cell labelled with its value.
pub fn heatmap(map: &Heatmap, theme: &Theme) -> EngineResult<String> {
    check_frame(&map.frame)?;
    if map.rows.is_empty() || map.cols.is_empty() || map.values.len() != map.rows.len() {
        return Err(EngineError::InvalidInput(
            "heatmap needs one value row per row label".into(),
        ));
    }
    if let Some(r) = map
        .values
        .iter()
        .position(|row| row.len() != map.cols.len())
    {
        return Err(EngineError::InvalidInput(format!(
            "heatmap row {r} has {} values for {} columns",
            map.values[r].len(),
            map.cols.len()
        )));
    }
    let present: Vec<f64> = map.values.iter().flatten().flatten().copied().collect();
    finite("heatmap value", present.iter().copied())?;
    let lo = present.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = present.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let left = LEFT + 70.0;
    let cw = (W - left - RIGHT) / map.cols.len() as f64;
    let ch = (H - TOP - BOTTOM) / map.rows.len() as f64;
    let mut s = open(&map.frame, theme);
    for (c, label) in map.cols.iter().enumerate() {
        let _ = write!(
            s,
            "<text x=\"{}\" y=\"{}\" font-size=\"11\" text-anchor=\"middle\" fill=\"{}\">{}</text>",
            num(left + (c as f64 + 0.5) * cw),
            H - BOTTOM + 18.0,
            theme.muted,
            esc(label)
        );
    }
    for (r, label) in map.rows.iter().enumerate() {
        let y0 = TOP + r as f64 * ch;
        let _ = write!(
            s,
            "<text x=\"{}\" y=\"{}\" font-size=\"11\" text-anchor=\"end\" fill=\"{}\">{}</text>",
            left - 8.0,
            num(y0 + ch / 2.0 + 4.0),
            theme.muted,
            esc(label)
        );
        for (c, v) in map.values[r].iter().enumerate() {
            let x0 = left + c as f64 * cw;
            match v {
                Some(v) => {
                    let t = if hi > lo { (v - lo) / (hi - lo) } else { 1.0 };
                    let fill = mix(&theme.surface, &theme.ink, t)?;
                    let text = if t > 0.55 { &theme.surface } else { &theme.ink };
                    let _ = write!(
                        s,
                        "<rect class=\"cell\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{fill}\" stroke=\"{}\"/>",
                        num(x0),
                        num(y0),
                        num(cw),
                        num(ch),
                        theme.grid
                    );
                    let _ = write!(
                        s,
                        "<text x=\"{}\" y=\"{}\" font-size=\"11\" text-anchor=\"middle\" fill=\"{text}\">{}</text>",
                        num(x0 + cw / 2.0),
                        num(y0 + ch / 2.0 + 4.0),
                        tick_label(*v)
                    );
                }
                None => {
                    let _ = write!(
                        s,
                        "<rect class=\"missing\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-dasharray=\"3 3\"/>",
                        num(x0),
                        num(y0),
                        num(cw),
                        num(ch),
                        theme.deemph
                    );
                }
            }
        }
    }
    s.push_str("</svg>\n");
    Ok(s)
}

/// Scatter plot with its required caption under the title.
pub fn scatter(plot: &Scatter, theme: &Theme) -> EngineResult<String> {
    check_frame(&plot.frame)?;
    if plot.caption.trim().is_empty() {
        return Err(EngineError::InvalidInput(
            "a scatter plot needs a caption saying what the view keeps (for a projection, the share of variance)".into(),
        ));
    }
    if plot.points.is_empty() {
        return Err(EngineError::InvalidInput(
            "a scatter plot needs points".into(),
        ));
    }
    let xs: Vec<f64> = plot.points.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = plot.points.iter().map(|p| p.1).collect();
    finite("x", xs.iter().copied())?;
    finite("y", ys.iter().copied())?;
    let x = Axis::new(&xs, false, LEFT, W - RIGHT);
    let y = Axis::new(&ys, false, H - BOTTOM, TOP);
    let mut s = open(&plot.frame, theme);
    let _ = write!(
        s,
        "<text class=\"caption\" x=\"{LEFT}\" y=\"52\" font-size=\"12\" fill=\"{}\">{}</text>",
        theme.ink2,
        esc(&plot.caption)
    );
    axes(&mut s, &x, &y, theme);
    for (a, b, label) in &plot.points {
        let _ = write!(
            s,
            "<circle class=\"point\" cx=\"{}\" cy=\"{}\" r=\"4\" fill=\"{}\" stroke=\"{}\"/>",
            num(x.px(*a)),
            num(y.px(*b)),
            theme.series[0],
            theme.mark
        );
        if !label.is_empty() {
            let _ = write!(
                s,
                "<text x=\"{}\" y=\"{}\" dx=\"6\" dy=\"-6\" font-size=\"10\" fill=\"{}\">{}</text>",
                num(x.px(*a)),
                num(y.px(*b)),
                theme.ink2,
                esc(label)
            );
        }
    }
    s.push_str("</svg>\n");
    Ok(s)
}

/// Layered dataflow graph; a cycle, a duplicate node or an edge to an unknown node is an error.
pub fn dataflow(graph: &Dataflow, theme: &Theme) -> EngineResult<String> {
    check_frame(&graph.frame)?;
    let mut index = BTreeMap::new();
    for (i, (id, _)) in graph.nodes.iter().enumerate() {
        if index.insert(id.as_str(), i).is_some() {
            return Err(EngineError::InvalidInput(format!(
                "node '{id}' appears twice"
            )));
        }
    }
    let n = graph.nodes.len();
    if n == 0 {
        return Err(EngineError::InvalidInput(
            "a dataflow graph needs nodes".into(),
        ));
    }
    let mut edges = Vec::new();
    for (a, b, label) in &graph.edges {
        let (Some(&i), Some(&j)) = (index.get(a.as_str()), index.get(b.as_str())) else {
            return Err(EngineError::InvalidInput(format!(
                "edge {a} -> {b} names an unknown node"
            )));
        };
        edges.push((i, j, label));
    }
    // Longest-path depth by Kahn's algorithm; leftover nodes mean a cycle.
    let mut indeg = vec![0usize; n];
    for &(_, j, _) in &edges {
        indeg[j] += 1;
    }
    let mut depth = vec![0usize; n];
    let mut ready: BTreeSet<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut done = 0;
    while let Some(i) = ready.pop_first() {
        done += 1;
        for &(a, b, _) in &edges {
            if a == i {
                depth[b] = depth[b].max(depth[a] + 1);
                indeg[b] -= 1;
                if indeg[b] == 0 {
                    ready.insert(b);
                }
            }
        }
    }
    if done < n {
        return Err(EngineError::InvalidInput(
            "dataflow graph has a cycle".into(),
        ));
    }
    let layers = depth.iter().max().copied().unwrap_or(0) + 1;
    let mut slot = vec![0usize; n];
    let mut per_layer = vec![0usize; layers];
    for i in 0..n {
        slot[i] = per_layer[depth[i]];
        per_layer[depth[i]] += 1;
    }
    let (bw, bh) = (120.0, 34.0);
    let col = (W - LEFT - RIGHT - bw) / (layers.max(2) - 1) as f64;
    let pos = |i: usize| {
        let rows = per_layer[depth[i]] as f64;
        let gap = (H - TOP - BOTTOM) / rows;
        (
            LEFT + depth[i] as f64 * col,
            TOP + gap * (slot[i] as f64 + 0.5) - bh / 2.0,
        )
    };
    let mut s = open(&graph.frame, theme);
    for &(a, b, label) in &edges {
        let ((ax, ay), (bx, by)) = (pos(a), pos(b));
        let (x1, y1, x2, y2) = (ax + bw, ay + bh / 2.0, bx, by + bh / 2.0);
        let _ = write!(
            s,
            "<line class=\"edge\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
            num(x1),
            num(y1),
            num(x2),
            num(y2),
            theme.ink2
        );
        let _ = write!(
            s,
            "<polygon points=\"{},{} {},{} {},{}\" fill=\"{}\"/>",
            num(x2),
            num(y2),
            num(x2 - 7.0),
            num(y2 - 4.0),
            num(x2 - 7.0),
            num(y2 + 4.0),
            theme.ink2
        );
        if !label.is_empty() {
            let _ = write!(
                s,
                "<text x=\"{}\" y=\"{}\" font-size=\"10\" text-anchor=\"middle\" fill=\"{}\" stroke=\"{}\" stroke-width=\"4\" paint-order=\"stroke\">{}</text>",
                num((x1 + x2) / 2.0),
                num((y1 + y2) / 2.0 - 5.0),
                theme.ink2,
                theme.surface,
                esc(label)
            );
        }
    }
    for (i, (_, label)) in graph.nodes.iter().enumerate() {
        let (x, y) = pos(i);
        let _ = write!(
            s,
            "<rect class=\"node\" x=\"{}\" y=\"{}\" width=\"{bw}\" height=\"{bh}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"2\"/>",
            num(x),
            num(y),
            theme.surface,
            theme.border
        );
        let _ = write!(
            s,
            "<text x=\"{}\" y=\"{}\" font-size=\"12\" text-anchor=\"middle\" fill=\"{}\">{}</text>",
            num(x + bw / 2.0),
            num(y + bh / 2.0 + 4.0),
            theme.ink,
            esc(label)
        );
    }
    s.push_str("</svg>\n");
    Ok(s)
}

/// Indices of the candidates no other candidate dominates (no higher cost, no higher loss,
/// strictly better in one), sorted by cost.
pub fn pareto_front(points: &[(f64, f64, String, PointKind)]) -> Vec<usize> {
    let cands: Vec<usize> = (0..points.len())
        .filter(|&i| points[i].3 == PointKind::Candidate)
        .collect();
    let mut front: Vec<usize> = cands
        .iter()
        .copied()
        .filter(|&i| {
            !cands.iter().any(|&j| {
                let (a, b) = (&points[j], &points[i]);
                j != i && a.0 <= b.0 && a.1 <= b.1 && (a.0 < b.0 || a.1 < b.1)
            })
        })
        .collect();
    front.sort_by(|&a, &b| points[a].0.total_cmp(&points[b].0).then(a.cmp(&b)));
    front
}

/// Pareto chart: candidates as dots, the front as a step line, the baseline as a square and
/// random controls as hollow dots.
pub fn pareto(chart: &Pareto, theme: &Theme) -> EngineResult<String> {
    check_frame(&chart.frame)?;
    if chart.points.is_empty() {
        return Err(EngineError::InvalidInput(
            "a Pareto chart needs points".into(),
        ));
    }
    let xs: Vec<f64> = chart.points.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = chart.points.iter().map(|p| p.1).collect();
    finite("cost", xs.iter().copied())?;
    finite("loss", ys.iter().copied())?;
    let x = Axis::new(&xs, false, LEFT, W - RIGHT);
    let y = Axis::new(&ys, false, H - BOTTOM, TOP);
    let mut s = open(&chart.frame, theme);
    axes(&mut s, &x, &y, theme);
    let front = pareto_front(&chart.points);
    if front.len() > 1 {
        let mut path = Vec::new();
        for (k, &i) in front.iter().enumerate() {
            let (px, py) = (x.px(chart.points[i].0), y.px(chart.points[i].1));
            if k > 0 {
                let prev_y = y.px(chart.points[front[k - 1]].1);
                path.push(format!("{},{}", num(px), num(prev_y)));
            }
            path.push(format!("{},{}", num(px), num(py)));
        }
        let _ = write!(
            s,
            "<polyline class=\"front\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\" points=\"{}\"/>",
            theme.ink,
            path.join(" ")
        );
    }
    for (cost, loss, label, kind) in &chart.points {
        let (px, py) = (num(x.px(*cost)), num(y.px(*loss)));
        match kind {
            PointKind::Candidate => {
                let _ = write!(
                    s,
                    "<circle class=\"candidate\" cx=\"{px}\" cy=\"{py}\" r=\"4\" fill=\"{}\"/>",
                    theme.series[0]
                );
            }
            PointKind::Control => {
                let _ = write!(
                    s,
                    "<circle class=\"control\" cx=\"{px}\" cy=\"{py}\" r=\"4\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                    theme.series[1]
                );
            }
            PointKind::Baseline => {
                let _ = write!(
                    s,
                    "<rect class=\"baseline\" x=\"{}\" y=\"{}\" width=\"9\" height=\"9\" fill=\"{}\"/>",
                    num(x.px(*cost) - 4.5),
                    num(y.px(*loss) - 4.5),
                    theme.ink
                );
            }
        }
        if !label.is_empty() {
            let _ = write!(
                s,
                "<text x=\"{px}\" y=\"{py}\" dx=\"7\" dy=\"-6\" font-size=\"10\" fill=\"{}\">{}</text>",
                theme.ink2,
                esc(label)
            );
        }
    }
    s.push_str("</svg>\n");
    Ok(s)
}

/// Interval plot; each row needs low <= value <= high, all finite.
pub fn interval_plot(plot: &IntervalPlot, theme: &Theme) -> EngineResult<String> {
    check_frame(&plot.frame)?;
    if plot.rows.is_empty() {
        return Err(EngineError::InvalidInput(
            "an interval plot needs rows".into(),
        ));
    }
    for (i, r) in plot.rows.iter().enumerate() {
        finite("interval value", [r.low, r.value, r.high].into_iter())?;
        if !(r.low <= r.value && r.value <= r.high) {
            return Err(EngineError::InvalidInput(format!(
                "row {i} ('{}'): needs low <= value <= high, got {} <= {} <= {}",
                r.label, r.low, r.value, r.high
            )));
        }
    }
    let mut xs: Vec<f64> = plot.rows.iter().flat_map(|r| [r.low, r.high]).collect();
    xs.extend(plot.reference);
    let left = LEFT + 110.0;
    let x = Axis::new(&xs, false, left, W - RIGHT);
    let row_h = (H - TOP - BOTTOM) / plot.rows.len() as f64;
    let mut s = open(&plot.frame, theme);
    for t in x.ticks() {
        let px = num(x.px(t));
        let _ = write!(
            s,
            "<line x1=\"{px}\" y1=\"{TOP}\" x2=\"{px}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
            H - BOTTOM,
            theme.grid
        );
        let _ = write!(
            s,
            "<text x=\"{px}\" y=\"{}\" font-size=\"11\" text-anchor=\"middle\" fill=\"{}\">{}</text>",
            H - BOTTOM + 18.0,
            theme.muted,
            tick_label(t)
        );
    }
    if let Some(r) = plot.reference {
        let px = num(x.px(r));
        let _ = write!(
            s,
            "<line class=\"reference\" x1=\"{px}\" y1=\"{TOP}\" x2=\"{px}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\" stroke-dasharray=\"5 4\"/>",
            H - BOTTOM,
            theme.ink2
        );
    }
    for (i, r) in plot.rows.iter().enumerate() {
        let cy = num(TOP + row_h * (i as f64 + 0.5));
        let _ = write!(
            s,
            "<text x=\"{}\" y=\"{cy}\" dy=\"4\" font-size=\"12\" text-anchor=\"end\" fill=\"{}\">{}</text>",
            left - 10.0,
            theme.ink,
            esc(&r.label)
        );
        let _ = write!(
            s,
            "<line class=\"whisker\" x1=\"{}\" y1=\"{cy}\" x2=\"{}\" y2=\"{cy}\" stroke=\"{}\" stroke-width=\"2\"/>",
            num(x.px(r.low)),
            num(x.px(r.high)),
            theme.ink2
        );
        let _ = write!(
            s,
            "<circle class=\"estimate\" cx=\"{}\" cy=\"{cy}\" r=\"5\" fill=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
            num(x.px(r.value)),
            theme.series[0],
            theme.mark
        );
        let _ = write!(
            s,
            "<text x=\"{}\" y=\"{cy}\" dx=\"8\" dy=\"-8\" font-size=\"10\" fill=\"{}\" stroke=\"{}\" stroke-width=\"4\" paint-order=\"stroke\">{} [{}, {}]</text>",
            num(x.px(r.value)),
            theme.ink2,
            theme.surface,
            tick_label(r.value),
            tick_label(r.low),
            tick_label(r.high)
        );
    }
    s.push_str("</svg>\n");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> Theme {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/chart-themes.json");
        Theme::load(&path, "light").unwrap()
    }

    fn frame() -> Frame {
        Frame {
            title: "Toy chart".into(),
            evidence: "Simulated".into(),
            x_label: "x".into(),
            y_label: "y".into(),
        }
    }

    fn count(svg: &str, needle: &str) -> usize {
        svg.matches(needle).count()
    }

    #[test]
    fn line_plot_counts_and_log_axis_rules() {
        let plot = LinePlot {
            frame: frame(),
            series: vec![
                Series {
                    name: "RK4".into(),
                    points: (1..=6).map(|i| (i as f64, 10f64.powi(-i))).collect(),
                },
                Series {
                    name: "Euler".into(),
                    points: (1..=4).map(|i| (i as f64, 10f64.powi(-i / 2))).collect(),
                },
            ],
            log_y: true,
        };
        let svg = line_plot(&plot, &theme()).unwrap();
        assert_eq!(count(&svg, "class=\"series\""), 2);
        assert_eq!(count(&svg, "class=\"point\""), 10);
        assert!(svg.contains("Evidence: Simulated"));
        assert!(svg.contains(">1e-6<"), "log ticks are powers of ten");
        let mut bad = plot.clone();
        bad.series[0].points[2].1 = 0.0;
        assert!(
            matches!(line_plot(&bad, &theme()), Err(EngineError::InvalidInput(m)) if m.contains("not positive"))
        );
        bad.series[0].points[2].1 = f64::NAN;
        assert!(matches!(
            line_plot(&bad, &theme()),
            Err(EngineError::NonFinite { .. })
        ));
    }

    #[test]
    fn identical_input_gives_identical_bytes() {
        let plot = LinePlot {
            frame: frame(),
            series: vec![Series {
                name: "a".into(),
                points: vec![(0.0, 1.0), (1.0, 3.0)],
            }],
            log_y: false,
        };
        assert_eq!(
            line_plot(&plot, &theme()).unwrap(),
            line_plot(&plot, &theme()).unwrap()
        );
        let map = Heatmap {
            frame: frame(),
            rows: vec!["L0".into(), "L1".into()],
            cols: vec!["q".into(), "v".into()],
            values: vec![vec![Some(0.1), Some(0.9)], vec![None, Some(0.5)]],
        };
        assert_eq!(
            heatmap(&map, &theme()).unwrap(),
            heatmap(&map, &theme()).unwrap()
        );
    }

    #[test]
    fn heatmap_counts_cells_and_rejects_ragged_rows() {
        let map = Heatmap {
            frame: frame(),
            rows: (0..3).map(|i| format!("layer {i}")).collect(),
            cols: vec!["Q".into(), "K".into(), "V".into(), "MLP".into()],
            values: vec![
                vec![Some(1.0), Some(2.0), Some(8.0), Some(0.5)],
                vec![Some(1.5), None, Some(6.0), Some(0.4)],
                vec![Some(0.9), Some(1.1), Some(9.5), None],
            ],
        };
        let svg = heatmap(&map, &theme()).unwrap();
        assert_eq!(count(&svg, "class=\"cell\""), 10);
        assert_eq!(count(&svg, "class=\"missing\""), 2);
        // The largest value takes the ink colour, the smallest the surface colour.
        assert!(svg.contains(&format!("fill=\"{}\"", theme().ink)));
        let mut bad = map.clone();
        bad.values[1].pop();
        assert!(heatmap(&bad, &theme()).is_err());
    }

    #[test]
    fn scatter_needs_its_caption() {
        let plot = Scatter {
            frame: frame(),
            points: vec![
                (0.0, 0.0, "a".into()),
                (1.0, 2.0, "b".into()),
                (2.0, 1.0, String::new()),
            ],
            caption: "2 of 11 dimensions; 40% of variance kept".into(),
        };
        let svg = scatter(&plot, &theme()).unwrap();
        assert_eq!(count(&svg, "class=\"point\""), 3);
        assert!(svg.contains("40% of variance kept"));
        let mut bad = plot;
        bad.caption = " ".into();
        assert!(scatter(&bad, &theme()).is_err());
    }

    #[test]
    fn dataflow_layers_by_depth_and_refuses_cycles() {
        let node = |id: &str| (id.to_string(), id.to_uppercase());
        let edge = |a: &str, b: &str, f: &str| (a.to_string(), b.to_string(), f.to_string());
        let graph = Dataflow {
            frame: frame(),
            nodes: vec![node("emb"), node("attn"), node("mlp"), node("head")],
            edges: vec![
                edge("emb", "attn", "F16"),
                edge("attn", "mlp", "Q8_0"),
                edge("emb", "mlp", "F16"),
                edge("mlp", "head", "TQ1_0"),
            ],
        };
        let svg = dataflow(&graph, &theme()).unwrap();
        assert_eq!(count(&svg, "class=\"node\""), 4);
        assert_eq!(count(&svg, "class=\"edge\""), 4);
        assert!(svg.contains(">Q8_0<"));
        let mut cyclic = graph.clone();
        cyclic.edges.push(edge("head", "emb", "F16"));
        assert!(
            matches!(dataflow(&cyclic, &theme()), Err(EngineError::InvalidInput(m)) if m.contains("cycle"))
        );
        let mut unknown = graph;
        unknown.edges.push(edge("emb", "nowhere", ""));
        assert!(dataflow(&unknown, &theme()).is_err());
    }

    #[test]
    fn pareto_front_matches_brute_force() {
        let p = |c: f64, l: f64, k: PointKind| (c, l, String::new(), k);
        let points = vec![
            p(1.0, 9.0, PointKind::Candidate),
            p(2.0, 5.0, PointKind::Candidate),
            p(3.0, 6.0, PointKind::Candidate), // dominated by (2, 5)
            p(4.0, 2.0, PointKind::Candidate),
            p(4.0, 3.0, PointKind::Candidate), // dominated by (4, 2)
            p(0.5, 1.0, PointKind::Control),   // controls never join the front
            p(5.0, 1.0, PointKind::Baseline),
        ];
        assert_eq!(pareto_front(&points), [0, 1, 3]);
        let svg = pareto(
            &Pareto {
                frame: frame(),
                points,
            },
            &theme(),
        )
        .unwrap();
        assert_eq!(count(&svg, "class=\"candidate\""), 5);
        assert_eq!(count(&svg, "class=\"control\""), 1);
        assert_eq!(count(&svg, "class=\"baseline\""), 1);
        assert_eq!(count(&svg, "class=\"front\""), 1);
    }

    #[test]
    fn interval_plot_rows_and_ordering() {
        let row = |l: &str, v: f64, lo: f64, hi: f64| IntervalRow {
            label: l.into(),
            value: v,
            low: lo,
            high: hi,
        };
        let plot = IntervalPlot {
            frame: frame(),
            rows: vec![
                row("base", 0.70, 0.67, 0.73),
                row("Q8_0", 0.69, 0.66, 0.72),
                row("TQ1", 0.41, 0.38, 0.44),
            ],
            reference: Some(0.68),
        };
        let svg = interval_plot(&plot, &theme()).unwrap();
        assert_eq!(count(&svg, "class=\"estimate\""), 3);
        assert_eq!(count(&svg, "class=\"whisker\""), 3);
        assert_eq!(count(&svg, "class=\"reference\""), 1);
        assert!(svg.contains("0.69 [0.66, 0.72]"));
        let mut bad = plot;
        bad.rows[1].low = 0.70;
        assert!(interval_plot(&bad, &theme()).is_err());
    }

    #[test]
    fn evidence_label_and_theme_are_required() {
        let mut f = frame();
        f.evidence = String::new();
        let plot = LinePlot {
            frame: f,
            series: vec![Series {
                name: "a".into(),
                points: vec![(0.0, 1.0)],
            }],
            log_y: false,
        };
        assert!(line_plot(&plot, &theme()).is_err());
        let doc = serde_json::json!({"font": "x", "light": {"surface": "#fff"}});
        assert!(Theme::from_json(&doc, "light").is_err());
        assert!(Theme::from_json(&doc, "sepia").is_err());
        let dark = Theme::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/chart-themes.json"),
            "dark",
        )
        .unwrap();
        assert_ne!(dark.surface, theme().surface);
        assert!(mix("#000000", "#ffffff", 0.5).unwrap() == "#808080");
        assert!(mix("black", "#ffffff", 0.5).is_err());
    }
}
