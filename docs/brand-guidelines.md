# Brand Guidelines: Epiphanes // THME

> Last updated: 2026-11-13 · Version 2.0.0 · Organization: Epiphanes
> Product: THME (Topological Heterogeneous Manifold Engine) · CLI alias: `llm-tweaker`
> Design motif: classic black-and-white manga panel, with liquid-glass surfaces

---

## Quick reference

| Element | Value |
| :--- | :--- |
| Ink | `#0B0B0B` (outlines, type, solid tone, print shadows) |
| Paper | `#FFFFFF` panels on a `#F6F6F4` ground |
| Tone | Greys only, carried by screentone dots and hatching rather than hue |
| Spot colour | Red `#E0284A` / `#C8102E`, only in the optional Manga Night and Manga Ink themes |
| Display and UI type | Chakra Petch Bold / Medium (SIL OFL 1.1, `assets/fonts/`) |
| Data type | Monospace (Studio default mono; JetBrains Mono or Consolas in SVGs) |
| Motif | Calabi-Yau quintic cross-section, drawn from its equation, never traced |
| Voice | Rigorous, empirical, plain: every figure is measured or labelled |

## 1. Identity

THME's look is a manga title page: paper, ink, screentone and panel frames. The default Studio
theme (**Manga Panel**) uses no colour at all. Format families, states and charts are told apart
by grey tone and dot density. Red survives as an optional spot colour in the **Manga Night**
(dark) and **Manga Ink** (paper) themes.

Glass is the second layer: panels are translucent over a painted backdrop of screentone and
speed lines, with a bright top rim (sheen), a crisp ink edge and a hard offset print shadow.

## 2. Assets

| File | What it is | Source |
| :--- | :--- | :--- |
| `assets/brand/thme-banner.gif` | Animated title spread: the mascot breathing; the manifold turning in 3D and through its 4D projection inside focus lines | `scripts/generate_banner.py` |
| `assets/brand/thme-banner.png` | First frame of the banner (static fallback) | `scripts/generate_banner.py` |
| `assets/brand/thme-mark-256.png` | Logo mark: the ink-shaded manifold in a panel frame | `scripts/generate_banner.py` |
| `assets/brand/thme-mascot.png` | Mascot artwork, black-and-white manga | Supplied artwork; to be refined by a human artist |
| `assets/brand/thme-icon-256.png`, `-64.png` | App and window icon (mascot face crop) | `scripts/generate_brand_art.ps1` |
| `assets/gifs/*.gif` | Recordings of the real Studio | `scripts/record_gif.ps1` |
| `docs/status/status-*.svg` | Status board in the panel style | `scripts/generate_status.py` |

The manifold in the banner, the logo mark and the Studio's Live 3D view all come from the same
parametrization (`src/gui/manga.rs`, `scripts/generate_banner.py`): each point solves
z1^5 + z2^5 = 1, and a test checks it.

## 3. Panels and surfaces

- **Panel frame**: 2 to 2.5 px ink edge, near-square corners (2 px) in the Manga Panel theme,
  rounded glass corners (12 px) in the colour themes.
- **Print shadow**: hard, no blur, offset 4 px down and right; ink on paper, the spot colour at
  low opacity on dark grounds.
- **Sheen**: a 1 px white rim along the top edge and a soft fade over the upper 45 %.
- **Screentone**: dots on a 9 to 11 px grid, growing toward one corner; never behind body text
  at full strength.
- **Speed and focus lines**: thin ink wedges radiating from a point outside the frame (backdrop)
  or toward a subject (banner).

## 4. Type

- Headings, tabs, KPI values and the wordmark: **Chakra Petch Bold**, uppercase for tabs.
- Body and controls: **Chakra Petch Medium**.
- Numbers in tables and code: monospace.
- The wordmark is set as outlined paper letters over a solid ink print shadow.

## 5. Mascot

The mascot holds the manifold: the person behind the measurements, holding the shape of the
model. In motion she breathes slowly (about 1 % scale, two breaths per loop). She is not
redrawn by scripts; changes to her design go through a human artist.

## 6. Do and don't

- Do keep data honest: brand art may be illustrative, but any figure that shows numbers must
  come from a committed report or say that it is illustrative or simulated.
- Do keep the default UI free of hue; use tone and pattern.
- Don't use the red spot colour for decoration in the Manga Panel theme.
- Don't trace or copy other people's manifold renderings; generate from the equation.
- Don't use third-party logos or trademarks as part of THME marks.
