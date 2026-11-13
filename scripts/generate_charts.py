#!/usr/bin/env python3
"""
THME benchmark chart generator.

Renders the measured reports in docs/benchmarks/data/ as static SVG charts in
docs/benchmarks/charts/, one light and one dark file per chart. Standard library
only; output is deterministic, so regenerating without new data produces no diff.

Every plotted value is read from a report written by a real forward pass
(examples/production_verification.rs, precision_sweep.rs, depth_probe.rs).
This script contains no benchmark numbers of its own.
"""

import argparse
import json
import math
import os
import sys

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

THEMES = {
    "light": {
        "surface": "#ffffff", "border": "#0b0b0b", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "s1": "#2a78d6", "s2": "#eb6834", "deemph": "#898781",
        "good": "#0ca30c", "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19", "border": "#2c2c2a", "ink": "#f0efec", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "s1": "#3987e5", "s2": "#d95926", "deemph": "#898781",
        "good": "#0ca30c", "critical": "#d03b3b",
    },
}

ROLE_NAMES = {
    "attention_q": "Attention Q", "attention_k": "Attention K", "attention_v": "Attention V",
    "attention_out": "Attention out", "attention_qkv": "Attention QKV (fused)", "mlp_gate": "MLP gate", "mlp_up": "MLP up", "mlp_down": "MLP down",
}


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width, height, theme, title, subtitle, description):
        self.t = theme
        self.w = width
        self.h = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img" font-family=\'{FONT}\'>',
            f"<title>{esc(title)}</title>",
            f"<desc>{esc(description)}</desc>",
            f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="2" stroke-width="2" '
            f'fill="{theme["surface"]}" stroke="{theme["border"]}"/>',
        ]
        self.text(24, 34, title, size=16, weight=600)
        self.text(24, 54, subtitle, size=12, color=theme["ink2"])

    def text(self, x, y, content, size=12, color=None, weight=400, anchor="start"):
        color = color or self.t["ink"]
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}">{esc(content)}</text>'
        )

    def line(self, x1, y1, x2, y2, color, width=1):
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
            f'stroke-width="{width}"/>'
        )

    def polyline(self, points, color):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def dot(self, x, y, color, r=4):
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{self.t["surface"]}" '
            'stroke-width="2"/>'
        )

    def hbar(self, x, y, width, height, color):
        """Horizontal bar: square at the baseline, 4px rounded data end."""
        r = min(4, width / 2, height / 2)
        self.parts.append(
            f'<path d="M{x:.1f},{y:.1f} h{width - r:.1f} a{r},{r} 0 0 1 {r},{r} v{height - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{width - r:.1f} z" fill="{color}"/>'
        )

    def swatch(self, x, y, color, label):
        self.parts.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{color}"/>')
        self.text(x + 16, y, label, size=12, color=self.t["ink2"])

    def line_key(self, x, y, color, label):
        self.line(x, y - 4, x + 18, y - 4, color, 2)
        self.dot(x + 9, y - 4, color, r=3)
        self.text(x + 26, y, label, size=12, color=self.t["ink2"])

    def status(self, x, y, ok, label):
        color = self.t["good"] if ok else self.t["critical"]
        self.parts.append(f'<circle cx="{x + 7}" cy="{y - 4}" r="7" fill="{color}"/>')
        if ok:
            path = f"M{x + 3.6},{y - 4} l2.4,2.6 l4.4,-5"
        else:
            path = f"M{x + 4.2},{y - 6.8} l5.6,5.6 M{x + 9.8},{y - 6.8} l-5.6,5.6"
        self.parts.append(
            f'<path d="{path}" fill="none" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" '
            'stroke-linejoin="round"/>'
        )
        self.text(x + 20, y, label, size=12, color=self.t["ink2"])

    def render(self):
        return "\n".join(self.parts + ["</svg>"]) + "\n"


def chart_gate(report, theme):
    ev = report["evaluation"]
    variants = {v["id"]: v for v in ev["variants"]}
    base = variants["baseline"]
    svg = Svg(
        760, 196, theme,
        f"Perplexity on the full WikiText-2 test split, {report['model_name']}",
        f"{base['scored_tokens']:,} scored tokens · lower is better · quality gate: Δ ≤ {ev['max_ppl_delta']}",
        "Three headline perplexity figures: source weights, exported GGUF, and the surgery recipe.",
    )
    tiles = [
        ("Source weights", base, None),
        ("Exported GGUF, uniform Q8_0", variants.get("exported_gguf"), "Inside the gate"),
        ("Surgery recipe, no retraining", variants.get("recipe"), "Outside the gate"),
    ]
    width = (760 - 48 - 2 * 16) / 3
    for i, (label, variant, _) in enumerate(tiles):
        if variant is None:
            continue
        x = 24 + i * (width + 16)
        svg.parts.append(
            f'<rect x="{x:.1f}" y="74" width="{width:.1f}" height="100" rx="6" fill="none" '
            f'stroke="{theme["border"]}"/>'
        )
        svg.text(x + 16, 98, label, size=12, color=theme["ink2"])
        svg.text(x + 16, 134, f"{variant['perplexity']:.2f}", size=30, weight=600)
        if variant["delta_ppl"] is None:
            svg.text(x + 16, 158, "Baseline", size=12, color=theme["ink2"])
        else:
            ok = bool(variant["accepted"])
            svg.status(x + 16, 158, ok, f"Δ {variant['delta_ppl']:+.2f} · {'inside' if ok else 'outside'} the gate")
    return svg


def log_scale(value, lo, hi, y0, y1):
    return y0 + (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (y1 - y0)


def chart_layers(sweep, theme):
    rows = [r for r in sweep["variants"] if r["group"] == "layer"]
    base = next(r for r in sweep["variants"] if r["group"] == "baseline")
    svg = Svg(
        760, 360, theme,
        "Perplexity cost of ternarizing one layer at a time",
        f"Δ perplexity vs baseline {base['perplexity']:.2f} · all {rows[0]['targeted_tensors']} linear tensors of the layer · "
        "log scale · no retraining",
        "Per-layer increase in perplexity when that layer alone is ternarized. Layer 0 is far more sensitive.",
    )
    left, right, top, bottom = 64, 736, 84, 300
    lo, hi = 0.2, 3000.0
    for tick in (1, 10, 100, 1000):
        y = log_scale(tick, lo, hi, bottom, top)
        svg.line(left, y, right, y, theme["grid"])
        svg.text(left - 8, y + 4, f"+{tick:,}", size=11, color=theme["muted"], anchor="end")
    svg.line(left, bottom, right, bottom, theme["axis"])
    step = (right - left - 24) / (len(rows) - 1)
    points = []
    for i, r in enumerate(rows):
        x = left + 12 + i * step
        y = log_scale(max(r["delta_ppl"], lo), lo, hi, bottom, top)
        points.append((x, y, r["delta_ppl"]))
        if len(rows) <= 10 or i % 5 == 0 or i == len(rows) - 1:
            svg.text(x, bottom + 18, str(i), size=11, color=theme["muted"], anchor="middle")
    svg.text((left + right) / 2, bottom + 40, "Layer", size=12, color=theme["ink2"], anchor="middle")
    svg.polyline([(x, y) for x, y, _ in points], theme["s1"])
    for x, y, _ in points:
        svg.dot(x, y, theme["s1"])
    deltas = [d for _, _, d in points]
    inner = deltas[1:-1]
    labelled = {0, len(points) - 1, 1 + inner.index(max(inner)), 1 + inner.index(min(inner))}
    for i in sorted(labelled):
        x, y, d = points[i]
        text = f"Layer {i}: +{d:,.0f}" if d >= 100 else f"Layer {i}: +{d:.1f}"
        anchor = "start" if i == 0 else ("end" if i == len(points) - 1 else "middle")
        dx = -4 if i == 0 else (0 if anchor == "middle" else 4)
        dy = -12 if d != min(inner) else 20
        svg.text(x + dx, y + dy, text, size=11, color=theme["ink2"], anchor=anchor)
    return svg


def chart_roles(sweep, theme):
    rows = sorted((r for r in sweep["variants"] if r["group"] == "role"), key=lambda r: -r["delta_ppl"])
    base = next(r for r in sweep["variants"] if r["group"] == "baseline")
    svg = Svg(
        760, 96 + 34 * len(rows), theme,
        "Perplexity cost of ternarizing one tensor role across all layers",
        f"Δ perplexity vs baseline {base['perplexity']:.2f} · share of model weights in parentheses · no retraining",
        "Ranked bars of perplexity increase per tensor role when that role is ternarized in every layer.",
    )
    left, right = 232, 660
    top = 80
    peak = max(r["delta_ppl"] for r in rows)
    svg.line(left, top - 6, left, top + 34 * len(rows) - 12, theme["axis"])
    for i, r in enumerate(rows):
        y = top + i * 34
        name = ROLE_NAMES.get(r["id"].replace("role_ternary_", ""), r["id"])
        svg.text(left - 12, y + 13, name, size=12, anchor="end")
        svg.text(left - 110, y + 13, f"({100 * r['targeted_param_fraction']:.1f}%)", size=11,
                 color=theme["muted"], anchor="end")
        width = (right - left) * r["delta_ppl"] / peak
        svg.hbar(left, y, width, 18, theme["s1"])
        svg.text(left + width + 8, y + 13, f"+{r['delta_ppl']:.1f}", size=12, color=theme["ink2"])
    return svg


def chart_placement(sweep, theme):
    recipe = next(r for r in sweep["variants"] if r["group"] == "recipe")
    randoms = sorted((r for r in sweep["variants"] if r["group"] == "random"), key=lambda r: r["delta_ppl"])
    base = next(r for r in sweep["variants"] if r["group"] == "baseline")
    rows = [("Sensitivity-guided recipe", recipe, theme["s1"])]
    rows += [(f"Random placement, seed {r['id'][-1]}", r, theme["deemph"]) for r in randoms]
    svg = Svg(
        760, 130 + 34 * len(rows), theme,
        "Placement matters: the recipe against random placement",
        f"Δ perplexity vs baseline {base['perplexity']:.2f} · same tensor counts and shapes in every row · "
        "all rows are outside the 0.45 gate",
        "Perplexity increase for the surgery recipe compared with three random placements of the same formats.",
    )
    left, right = 232, 660
    top = 80
    peak = max(r["delta_ppl"] for _, r, _ in rows)
    svg.line(left, top - 6, left, top + 34 * len(rows) - 12, theme["axis"])
    for i, (name, r, color) in enumerate(rows):
        y = top + i * 34
        svg.text(left - 12, y + 13, name, size=12, anchor="end")
        width = (right - left) * r["delta_ppl"] / peak
        svg.hbar(left, y, width, 18, color)
        svg.text(left + width + 8, y + 13, f"+{r['delta_ppl']:.1f}", size=12, color=theme["ink2"])
    legend_y = top + 34 * len(rows) + 20
    svg.swatch(left, legend_y, theme["s1"], "Recipe")
    svg.swatch(left + 90, legend_y, theme["deemph"], "Random placement (exploratory, n = 3)")
    return svg


def chart_depth(probe, theme):
    rows = probe["exit_readout"]
    depth = probe["num_layers"]
    svg = Svg(
        760, 392, theme,
        "How early do predictions settle?",
        f"Share of {probe['scored_tokens']:,} tokens · read out through the model's own output head after each "
        "layer · no trained exit heads",
        "Share of tokens whose top prediction matches the full-depth prediction, by number of layers applied.",
    )
    left, right, top, bottom = 64, 716, 112, 328
    svg.line_key(left, 86, theme["s1"], "Top prediction equals the full-depth prediction")
    svg.line_key(left + 340, 86, theme["s2"], "Settled: stays equal from this depth on")

    def px(layer):
        return left + (right - left) * layer / depth

    def py(fraction):
        return bottom - (bottom - top) * fraction

    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        svg.line(left, py(tick), right, py(tick), theme["grid"] if tick else theme["axis"])
        svg.text(left - 8, py(tick) + 4, f"{int(tick * 100)}%", size=11, color=theme["muted"], anchor="end")
    for layer in range(0, depth + 1, 5):
        svg.text(px(layer), bottom + 18, str(layer), size=11, color=theme["muted"], anchor="middle")
    svg.text((left + right) / 2, bottom + 40, "Layers applied", size=12, color=theme["ink2"], anchor="middle")

    agree = [(px(r["layers_applied"]), py(r["top1_agrees_with_final"])) for r in rows]
    settled = [(px(r["layers_applied"]), py(r["settled_fraction"])) for r in rows]
    svg.polyline(agree, theme["s1"])
    svg.polyline(settled, theme["s2"])
    for share in (0.3, 0.8):
        layer = round(depth * share)
        r = rows[layer]
        x, y = px(layer), py(r["settled_fraction"])
        # Label sits in clear space above both lines, tied to its point by a leader.
        label_y = py(r["top1_agrees_with_final"] + 0.22)
        svg.line(x, label_y + 6, x, y - 6, theme["muted"])
        svg.dot(x, y, theme["s2"])
        svg.text(x, label_y, f"{int(share * 100)}% depth: {100 * r['settled_fraction']:.1f}% settled",
                 size=11, color=theme["ink2"], anchor="middle")
    svg.dot(*agree[-1], theme["s1"])
    return svg


def chart_pareto(search, theme):
    """Measured size vs quality: searched recipes against random placement of the same formats."""
    s = search["search"]
    tol = search["evaluation"]["max_ppl_delta"]
    points = sorted(s["pareto"], key=lambda p: p["achieved_fraction"])
    svg = Svg(
        760, 400, theme,
        f"Measured size vs quality, {search['model_name']}",
        f"Validation slice, {s['validation_tokens']:,} tokens · formats {', '.join(s['formats'])} · "
        f"Δ perplexity vs source · lower and further left is better",
        "Perplexity increase against nominal linear-tensor size for searched recipes and random placements.",
    )
    left, right, top, bottom = 72, 730, 110, 330
    values = [0.0, tol] + [p["validation_delta"] for p in points]
    values += [d for p in points for d in p["control_deltas"]]
    y_max = max(values) * 1.08 or 1.0
    y_min = min(0.0, min(values))

    def px(fraction):
        return left + (right - left) * fraction

    def py(delta):
        return bottom - (bottom - top) * (delta - y_min) / (y_max - y_min)

    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        svg.line(left, py(value), right, py(value), theme["grid"] if i else theme["axis"])
        svg.text(left - 8, py(value) + 4, f"{value:+.2f}" if abs(y_max) < 10 else f"{value:+,.0f}",
                 size=11, color=theme["muted"], anchor="end")
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        svg.text(px(f), bottom + 18, f"{int(f * 100)}%", size=11, color=theme["muted"], anchor="middle")
    svg.text((left + right) / 2, bottom + 40, "Nominal size of the linear tensors, share of stored size",
             size=12, color=theme["ink2"], anchor="middle")
    svg.line(left, py(tol), right, py(tol), theme["critical"], 1.5)
    svg.text(right, py(tol) - 6, f"Gate Δ ≤ {tol}", size=11, color=theme["ink2"], anchor="end")
    for p in points:
        for d in p["control_deltas"]:
            svg.dot(px(p["achieved_fraction"]), py(d), theme["deemph"], r=3.5)
    line = [(px(1.0), py(0.0))] + [(px(p["achieved_fraction"]), py(p["validation_delta"])) for p in reversed(points)]
    svg.polyline(sorted(line), theme["s1"])
    for x, y in line:
        svg.dot(x, y, theme["s1"])
    svg.line_key(left, 86, theme["s1"], "Searched recipe (and source at 100%)")
    svg.swatch(left + 280, 86, theme["deemph"], "Random placement, same formats")
    return svg

def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="THME benchmark chart generator")
    parser.add_argument("--data-dir", default="docs/benchmarks/smollm-135m/data", help="Directory holding the measured reports")
    parser.add_argument("--out-dir", default="docs/benchmarks/smollm-135m/charts", help="Directory for generated SVG charts")
    args = parser.parse_args()

    report = load(os.path.join(args.data_dir, "verification_report.json"))
    sweep = load(os.path.join(args.data_dir, "precision_sweep.json"))
    probe = load(os.path.join(args.data_dir, "depth_probe.json"))
    search = load(os.path.join(args.data_dir, "search_report.json"))
    charts = []
    if report and report.get("evaluation", {}).get("measured"):
        charts.append(("quality-gate", lambda t: chart_gate(report, t)))
    if sweep:
        groups = {r["group"] for r in sweep["variants"]}
        if "layer" in groups:
            charts.append(("layer-sensitivity", lambda t: chart_layers(sweep, t)))
        if "role" in groups:
            charts.append(("role-sensitivity", lambda t: chart_roles(sweep, t)))
        if {"recipe", "random"} <= groups:
            charts.append(("placement", lambda t: chart_placement(sweep, t)))
    if probe:
        charts.append(("depth-settling", lambda t: chart_depth(probe, t)))
    if search and search.get("search", {}).get("pareto"):
        charts.append(("pareto", lambda t: chart_pareto(search, t)))
    if not charts:
        print(f"error: no measured reports found in {args.data_dir}", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    for name, build in charts:
        for mode, theme in THEMES.items():
            path = os.path.join(args.out_dir, f"{name}-{mode}.svg")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(build(theme).render())
        print(f"Generated {name} (light, dark)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
