#!/usr/bin/env python3
"""
Render the README "Status" board from the repository's own records.

Sources (nothing is typed in by hand):
  - ROADMAP.md: v0.1.x sub-milestone checkboxes and v0.2.x sub-milestone tables
  - BACKLOG.md: active tickets (between the backlog:start/end markers) by area and status
  - src/ and tests/: #[test] functions per module
  - docs/benchmarks/*/data/*.json: committed measured reports

Writes docs/status/status-light.svg and docs/status/status-dark.svg in the manga panel style
(ink outlines, hard print shadows, screentone). Output is deterministic, so CI regenerates it
and fails when the committed board is stale.

  python scripts/generate_status.py
"""

import glob
import html
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "status")
FONT = "'Chakra Petch', 'Bahnschrift', 'Arial Narrow', 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, monospace"

THEMES = {
    "light": {"paper": "#ffffff", "ground": "#f6f6f4", "ink": "#0b0b0b", "muted": "#555555", "tone": "#0b0b0b"},
    "dark": {"paper": "#141414", "ground": "#0b0b0b", "ink": "#f4f2ec", "muted": "#a8a6a0", "tone": "#f4f2ec"},
}


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        return handle.read()


def roadmap_milestones():
    """[(series, version, name, [state x9])] with state in done / progress / open."""
    text = read("ROADMAP.md")
    milestones = []
    # v0.1.x: "#### [x] v0.1.N.0 — Name (Completed)" followed by "- [x] **v0.1.N.k**" items.
    for match in re.finditer(r"^#### \[[ x]\] v0\.1\.(\d+)\.0 — (.+?)(?: Baseline)?(?: \(.*\))?$", text, re.M):
        number, name = match.group(1), match.group(2).strip()
        block = text[match.end():].split("\n#### ", 1)[0]
        states = ["done" if box == "x" else "open"
                  for box in re.findall(r"^\s*- \[([ x])\] \*\*v0\.1\.%s\.\d\*\*" % number, block, re.M)]
        milestones.append(("v0.1", f"v0.1.{number}", name, states))
    # v0.2.x: "## v0.2.N — Name" followed by "| v0.2.N.k | ... |" table rows.
    for match in re.finditer(r"^## v0\.2\.(\d+) — (.+?)(?: \(.*\))?$", text, re.M):
        number, name = match.group(1), match.group(2).strip()
        block = text[match.end():].split("\n## ", 1)[0]
        states = []
        for row in re.findall(r"^\| v0\.2\.%s\.\d \|.*$" % number, block, re.M):
            if "**Done**" in row:
                states.append("done")
            elif "Implemented" in row or "in progress" in row.lower():
                states.append("progress")
            else:
                states.append("open")
        milestones.append(("v0.2", f"v0.2.{number}", name, states))
    return milestones


def backlog_by_area():
    text = read("BACKLOG.md")
    start, end = text.find("<!-- backlog:start -->"), text.find("<!-- backlog:end -->")
    if start < 0 or end < start:
        raise SystemExit("error: backlog markers not found in BACKLOG.md")
    areas = {}
    for line in text[start:end].splitlines():
        if not line.startswith("- **Milestone**"):
            continue
        fields = {k.strip().lower(): v.strip() for k, v in re.findall(r"\*\*(\w[\w ]*)\*\*:\s*([^·]+)", line)}
        area = fields.get("area", "other")
        status = fields.get("status", "triage")
        if status == "done":
            continue  # finished tickets leave the active board
        areas.setdefault(area, {"ready": 0, "blocked": 0, "triage": 0})
        areas[area][status if status in ("ready", "blocked") else "triage"] += 1
    return sorted(areas.items(), key=lambda item: (-sum(item[1].values()), item[0]))


def tests_by_module():
    counts = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.rs"), recursive=True)
                       + glob.glob(os.path.join(ROOT, "tests", "*.rs"))):
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        parts = rel.split("/")
        if parts[0] == "tests":
            module = "integration"
        elif len(parts) > 2:
            module = parts[1]
        else:
            module = parts[1].removesuffix(".rs")
        with open(path, encoding="utf-8") as handle:
            n = handle.read().count("#[test]")
        if n:
            counts[module] = counts.get(module, 0) + n
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def measured_reports():
    return len(glob.glob(os.path.join(ROOT, "docs", "benchmarks", "*", "data", "*.json")))


def esc(text):
    return html.escape(str(text), quote=True)


class Svg:
    def __init__(self, width, theme):
        self.width, self.t, self.parts, self.y = width, theme, [], 0

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, weight=600, fill=None, anchor="start", font=FONT, spacing=0):
        tracking = ' letter-spacing="%s"' % spacing if spacing else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{fill or self.t["ink"]}" text-anchor="{anchor}"'
                 f'{tracking}>{esc(s)}</text>')

    def panel(self, x, y, w, h, shadow=4):
        t = self.t
        self.add(f'<rect x="{x + shadow}" y="{y + shadow}" width="{w}" height="{h}" fill="{t["ink"]}"/>')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{t["paper"]}" '
                 f'stroke="{t["ink"]}" stroke-width="2.5"/>')

    def cell(self, x, y, w, h, state):
        t = self.t
        fill = {"done": t["ink"], "ready": t["ink"], "progress": "url(#tone)", "open": t["paper"],
                "triage": t["paper"], "blocked": "url(#hatch)"}[state]
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
                 f'stroke="{t["ink"]}" stroke-width="1.2"/>')

    def render(self, height):
        t = self.t
        defs = (f'<defs><pattern id="tone" width="5" height="5" patternUnits="userSpaceOnUse">'
                f'<rect width="5" height="5" fill="{t["paper"]}"/>'
                f'<circle cx="2.5" cy="2.5" r="1.25" fill="{t["tone"]}"/></pattern>'
                f'<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                f'<rect width="6" height="6" fill="{t["paper"]}"/>'
                f'<line x1="0" y1="0" x2="0" y2="6" stroke="{t["tone"]}" stroke-width="2"/></pattern>'
                f'<pattern id="ground" width="9" height="9" patternUnits="userSpaceOnUse">'
                f'<rect width="9" height="9" fill="{t["ground"]}"/>'
                f'<circle cx="4.5" cy="4.5" r="0.7" fill="{t["tone"]}" opacity="0.18"/></pattern></defs>')
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{height}" '
                f'viewBox="0 0 {self.width} {height}" role="img" aria-label="THME project status">'
                f'<title>THME project status, generated from ROADMAP.md, BACKLOG.md and the test suite</title>')
        bg = f'<rect width="{self.width}" height="{height}" fill="url(#ground)"/>'
        return head + defs + bg + "".join(self.parts) + "</svg>\n"


def badge(svg, x, y, label, value, filled):
    t = svg.t
    lw = 9 + 7.2 * len(label)
    vw = max(34, 12 + 8.0 * len(value))
    svg.add(f'<rect x="{x + 3}" y="{y + 3}" width="{lw + vw}" height="24" fill="{t["ink"]}"/>')
    svg.add(f'<rect x="{x}" y="{y}" width="{lw}" height="24" fill="{t["paper"]}" stroke="{t["ink"]}" stroke-width="2"/>')
    svg.add(f'<rect x="{x + lw}" y="{y}" width="{vw}" height="24" fill="{t["ink"] if filled else "url(#tone)"}" '
            f'stroke="{t["ink"]}" stroke-width="2"/>')
    svg.text(x + lw / 2, y + 16.5, label, size=11, anchor="middle", spacing=0.5)
    if filled:
        svg.text(x + lw + vw / 2, y + 16.5, value, size=12.5, weight=700, fill=t["paper"], anchor="middle")
    else:
        svg.add(f'<rect x="{x + lw + vw / 2 - 4.5 * len(value) - 3}" y="{y + 5}" width="{9 * len(value) + 6}" '
                f'height="14" fill="{t["paper"]}"/>')
        svg.text(x + lw + vw / 2, y + 16.5, value, size=12.5, weight=700, anchor="middle")
    return x + lw + vw + 14


def milestone_grid(svg, x, y, items, columns, pw, ph):
    for index, (_, version, name, states) in enumerate(items):
        cx = x + (index % columns) * (pw + 14)
        cy = y + (index // columns) * (ph + 16)
        svg.panel(cx, cy, pw, ph)
        done = sum(s == "done" for s in states)
        svg.text(cx + 10, cy + 19, version, size=13, weight=700)
        svg.text(cx + pw - 10, cy + 19, f"{done}/{len(states)}", size=12, weight=700, anchor="end", font=MONO)
        label = name if len(name) <= 24 else name[:23] + "…"
        svg.text(cx + 10, cy + 34, label, size=10.5, weight=500, fill=svg.t["muted"])
        gx, gy, gw = cx + 10, cy + 42, pw - 20
        size = (gw - 2 * 4) / 3
        cell_h = (ph - 52 - 2 * 4) / 3
        for k, state in enumerate(states[:9]):
            svg.cell(gx + (k % 3) * (size + 4), gy + (k // 3) * (cell_h + 4), size, cell_h, state)
    rows = (len(items) + columns - 1) // columns
    return y + rows * (ph + 16)


def bars(svg, x, y, w, rows, segments, legend):
    t = svg.t
    total_max = max((sum(v[s] for s in segments) if isinstance(v, dict) else v) for _, v in rows) or 1
    label_w = 120
    for i, (name, value) in enumerate(rows):
        ry = y + i * 22
        svg.text(x, ry + 13, name, size=11.5, weight=500)
        bx = x + label_w
        svg.add(f'<rect x="{bx}" y="{ry + 2}" width="{w - label_w - 40}" height="14" fill="{t["paper"]}" '
                f'stroke="{t["ink"]}" stroke-width="1.2"/>')
        counts = [value[s] for s in segments] if isinstance(value, dict) else [value]
        scale = (w - label_w - 40) / total_max
        cursor = bx
        for state, n in zip(segments, counts):
            if n:
                svg.cell(cursor, ry + 2, n * scale, 14, state)
                cursor += n * scale
        svg.text(x + w - 30, ry + 13, sum(counts), size=11.5, weight=700, anchor="start", font=MONO)
    ly = y + len(rows) * 22 + 6
    lx = x + label_w
    for state, label in legend:
        svg.cell(lx, ly, 12, 12, state)
        svg.text(lx + 17, ly + 10.5, label, size=10.5, weight=500, fill=t["muted"])
        lx += 26 + 6.6 * len(label)
    return ly + 22


def build(theme_name, data):
    milestones, backlog, tests, reports = data
    t = THEMES[theme_name]
    width = 960
    svg = Svg(width, t)
    v01 = [m for m in milestones if m[0] == "v0.1"]
    v02 = [m for m in milestones if m[0] == "v0.2"]
    subs = [s for m in milestones for s in m[3]]
    done_subs = sum(s == "done" for s in subs)
    done_ms = sum(1 for m in milestones if m[3] and all(s == "done" for s in m[3]))
    test_total = sum(n for _, n in tests)

    svg.text(24, 40, "STATUS", size=26, weight=700, spacing=2)
    svg.add(f'<line x1="24" y1="50" x2="{width - 24}" y2="50" stroke="{t["ink"]}" stroke-width="3"/>')
    x = 24
    x = badge(svg, x, 64, "MILESTONES", f"{done_ms}/{len(milestones)}", True)
    x = badge(svg, x, 64, "SUB-MILESTONES", f"{done_subs}/{len(subs)}", True)
    x = badge(svg, x, 64, "TESTS", str(test_total), True)
    x = badge(svg, x, 64, "MEASURED REPORTS", str(reports), False)

    y = 116
    lx = 24
    for state, label in (("done", "Done"), ("progress", "In progress"), ("open", "Open")):
        svg.cell(lx, y, 12, 12, state)
        svg.text(lx + 17, y + 10.5, label, size=10.5, weight=500, fill=t["muted"])
        lx += 30 + 6.6 * len(label)

    done01 = sum(s == "done" for m in v01 for s in m[3])
    total01 = sum(len(m[3]) for m in v01)
    svg.text(24, y + 40, f"v0.1 series: {done01}/{total01} sub-milestones (implemented and tested)", size=14, weight=700)
    pw = (width - 48 - 4 * 14) / 5
    y = milestone_grid(svg, 24, y + 52, v01, 5, pw, 108)

    done02 = sum(s == "done" for m in v02 for s in m[3])
    total02 = sum(len(m[3]) for m in v02)
    svg.text(24, y + 22, f"v0.2 series: {done02}/{total02} sub-milestones (measured core and beyond)", size=14, weight=700)
    y = milestone_grid(svg, 24, y + 34, v02, 5, pw, 108)

    col_w = (width - 48 - 24) / 2
    top = y + 14
    svg.text(24, top + 14, f"Active backlog by area ({sum(sum(v.values()) for _, v in backlog)} tickets)", size=14, weight=700)
    end_left = bars(svg, 24, top + 26, col_w, backlog, ("ready", "blocked", "triage"),
                    (("ready", "Ready"), ("blocked", "Blocked"), ("open", "Triage")))
    svg.text(24 + col_w + 24, top + 14, f"Tests by module ({test_total})", size=14, weight=700)
    end_right = bars(svg, 24 + col_w + 24, top + 26, col_w, tests[:len(backlog) + 2] if len(tests) > len(backlog) + 2 else tests,
                     ("done",), ())
    height = int(max(end_left, end_right) + 30)
    svg.text(width - 24, height - 12, "Generated by scripts/generate_status.py from ROADMAP.md, BACKLOG.md and the test suite",
             size=9.5, weight=500, fill=t["muted"], anchor="end")
    return svg.render(height)


def main():
    data = (roadmap_milestones(), backlog_by_area(), tests_by_module(), measured_reports())
    if not data[0]:
        print("error: no milestones parsed from ROADMAP.md", file=sys.stderr)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    for theme in THEMES:
        path = os.path.join(OUT_DIR, f"status-{theme}.svg")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(build(theme, data))
        print("wrote", os.path.relpath(path, ROOT).replace(os.sep, "/"))
    milestones = data[0]
    print(f"{len(milestones)} milestones, {sum(len(m[3]) for m in milestones)} sub-milestones, "
          f"{sum(n for _, n in data[2])} tests, {data[3]} measured reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
