# Brand Guidelines: Epiphanes // THME

> Last updated: 2026-11-11
> Version: 1.0.0
> Organization: Epiphanes
> Product: THME (Topological Heterogeneous Manifold Engine)
> Design Motif: OLED Obsidian Black Developer Interface

---

## Quick Reference

| Element | Value |
|---------|-------|
| Company | Epiphanes |
| Product | THME (Topological Heterogeneous Manifold Engine) |
| CLI Alias | llm-tweaker |
| Primary Color | #00E5FF |
| Secondary Color | #10B981 |
| Accent Color | #F59E0B |
| Canvas Background | #09090B |
| Primary Font | JetBrains Mono, Inter |
| Tone | Rigorous, Systems-Focused, Empirical, Zero-Slop |

---

## 1. Color Palette

The color system is engineered for high-contrast OLED developer dashboards, hardware-calibrated telemetry, and terminal matrix inspections.

### Primary Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Obsidian Cyan | #00E5FF | rgb(0,229,255) | Active cockpit tabs, primary controls, live streaming token highlights |
| Primary Dark | #00B4D8 | rgb(0,180,216) | Hover states, active borders, secondary charts |
| Primary Light | #80F2FF | rgb(128,242,255) | Glow overlays, selected text highlights |

### Secondary Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Emerald Green | #10B981 | rgb(16,185,129) | Quality gate accepted status, throughput speedup confirmations, BitNet pass |
| Emerald Dark | #059669 | rgb(5,150,105) | Success hover states, confirmed ledger items |
| Emerald Light | #6EE7B7 | rgb(110,231,183) | Verified checkmarks, positive delta badges |

### Accent Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Amber Warning | #F59E0B | rgb(245,158,11) | Memory ceiling pressure, compute-bound kneepoints, fallback paths |
| Crimson Error | #EF4444 | rgb(239,68,68) | Perplexity regressions, gate rejections, critical panics |
| Delta Indigo | #6366F1 | rgb(99,102,241) | Speculative decoding trees, Hadamard transforms, SVD energy decay |

### Neutral Palette (OLED Obsidian Black)

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Background | #09090B | rgb(9,9,11) | Root canvas, OLED background, terminal black |
| Surface | #18181B | rgb(24,24,27) | Component cards, telemetry panels, inspector sidebars |
| Surface Raised | #27272A | rgb(39,39,42) | Hovered card state, dropdown containers, table headers |
| Border | #27272A | rgb(39,39,42) | 1px structural dividing lines, subtle containment borders |
| Text Primary | #FFFFFF | rgb(255,255,255) | High-contrast KPI headers, active parameter readouts (19.06:1 AAA) |
| Text Secondary | #A1A1AA | rgb(161,161,170) | Parameter descriptions, diagnostic subtitles, inactive controls |
| Text Muted | #71717A | rgb(113,113,122) | Inactive tabs, disabled options, timestamp metadata |

### Accessibility & Contrast Compliance

- Primary White on OLED Black (#FFFFFF on #09090B): 19.06:1 contrast ratio (exceeds WCAG 2.1 AAA 7:1 requirement).
- Obsidian Cyan on Surface (#00E5FF on #18181B): 11.4:1 contrast ratio (exceeds WCAG 2.1 AAA).
- Emerald on Surface (#10B981 on #18181B): 6.8:1 contrast ratio (WCAG 2.1 AA for large and medium text).
- Amber on Surface (#F59E0B on #18181B): 7.3:1 contrast ratio (WCAG 2.1 AAA).

---

## 2. Typography

Engineering interfaces prioritize readability, zero ambiguity between numerical characters (`0`, `O`, `1`, `l`, `I`), and tabular numerical alignment.

### Typefaces

- **Primary Code & Monospace**: `JetBrains Mono`, `Fira Code`, `Consolas`, `monospace`
  - Usage: Weight matrices, memory metrics, CLI commands, telemetry feeds, JSON manifests.
- **Primary Proportional / UI**: `Inter`, `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `sans-serif`
  - Usage: Navigation tabs, panel titles, documentation prose, modal dialogs.

### Type Scale

| Role | Size | Weight | Line Height | Usage |
|------|------|--------|-------------|-------|
| Display | 24px | 700 (Bold) | 32px | Studio header, main brand logotype |
| Heading 1 | 18px | 600 (SemiBold) | 24px | Tab titles, modal titles, KPI primary values |
| Heading 2 | 14px | 600 (SemiBold) | 20px | Section headers, card titles |
| Body | 13px | 400 (Regular) | 18px | Standard UI labels, body explanations |
| Monospace / Code | 12px | 500 (Medium) | 16px | Data readouts, tensor dimensions, CLI syntax |
| Micro / Caption | 10px | 500 (Medium) | 14px | Status bar indicators, hardware knee markers |

---

## 3. Brand Voice & Editorial Tone

The voice of Epiphanes and THME is that of a seasoned systems researcher and performance engineer:

- **Mathematical Precision**: Always reference specific mathematical operations (e.g. *singular value decomposition*, *Frobenius norm*, *Hadamard Walsh transform*, *Fisher information matrix*, *2-bit ternary bitstream*) rather than generic descriptions.
- **Empirical Honesty**: Non-negotiable standard from `AGENTS.md`. Never present simulated projections as physical hardware benchmarks. Clearly demarcate simulated telemetry from measured runtime profiles.
- **Conciseness & High Information Density**: Communicate through structured tables, ASCII waterfalls, and direct diffs.
- **Strictly No Emojis**: Emojis are forbidden in all official documentation, user interfaces, code comments, tests, and commit messages.
- **Zero AI-Generated Slop**: Avoid filler jargon, overused superlatives, and artificial hype words.

---

## 4. Logo & Insignia Standards

THME brand identity combines neoclassical architectural engraving with topological manifold geometry.

### 4.1 Master Emblem: Neoclassical Insignia (`assets/brand/thme_insignia_emblem.svg`)
The primary emblem for README heroes, splash screens, documentation covers, and official releases:
- **Central Bust**: Epiphaneia (Divine Insight / Sophia), personifying the high-rank unquantized Teacher LLM prior to model surgery.
- **Topological Compass**: Outer coordinate dial with 360-degree ticks, orbital coordinate nodes, and directional vector arrows (`↑`).
- **Signal Modulation**: Undulating waveform passing through the coordinate origin, symbolizing Fast Walsh-Hadamard Transform (FWHT) and spectral kurtosis outlier suppression.
- **Heterogeneous Lattice**: 4-tier memory hierarchy and tensor wireframe mesh in the upper quadrant.
- **Acronym Cipher**: Nordic/Celtic knotwork breastplate emblem interlocking `T`, `H`, `M`, and `E`.

### 4.2 Application Sigil & Favicon (`assets/brand/thme_sigil_icon.svg`)
The scalable vector glyph for desktop app icons, browser favicons, dock icons, and CLI output:
- **Scalability**: Pure geometric Bézier curves designed for crisp legibility from 16×16px to 512×512px.
- **Contrast**: Meets WCAG 2.1 AAA (19.06:1 on OLED Obsidian `#09090B`).

### 4.3 Monogram & ASCII Lockup
For terminal environments without vector rendering:

```
+---+---+
| T | H |
+---+---+  THME // TOPOLOGICAL HETEROGENEOUS MANIFOLD ENGINE
| M | E |  EPIPHANES SYSTEMS
+---+---+
```

- **Clearspace**: Minimum 16px padding on all sides of the logomark.
- **Monochrome Inversion**: On light backgrounds (print or export documentation), invert canvas to pure white (`#FFFFFF`) with Obsidian Slate (`#09090B`) mark strokes.
- **Minimum Digital Size**: 16×16 pixels for favicon; 24×24 pixels for app icons.
