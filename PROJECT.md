# Project: THME Desktop Studio Elevation, Showcase Visuals & README Redesign

## Architecture
The THME (Topological Heterogeneous Manifold Engine) user-facing experience is elevated across three integrated layers:
1. **Desktop Studio GUI Subsystem (`src/gui/`)**:
   - Modern dark-mode developer tool aesthetic (Linear / Vercel / Raycast style).
   - Pure Rust theming engine in `src/gui/theme.rs` with Tailwind Zinc (950/900/800/700) and Slate primitives.
   - Elevated components: KPI stat cards (no decorative stripes, 18pt typography), waterfall sensitivity charts (gridlines, hover tooltips), minimalist sliders with progressive fill, and clean tab navigation.
   - Zero emojis in UI or codebase; mathematical/technical Unicode glyphs preserved.
2. **Showcase Visuals Engine (`examples/generate_showcase_gifs.rs` & `assets/`)**:
   - Pure Rust, zero-external-dependency GIF89a generator with an embedded LZW encoder.
   - 920x520 widescreen, 24-color elevated dark mode palette, 30-32 frames @ 110ms delay.
   - 5 capability flows: Cockpit/AutoDecider, Roofline Profiler, Speculative Decoding, 8 Model Paradigms, PyTorch/CUDA Bridge.
   - Authentic telemetry directly connected to the `thme` crate. File sizes <= 1.2 MB each.
3. **Repository Showcase Documentation (`README.md`)**:
   - Top-level `## Interactive Capabilities Showcase` positioned between Section 1 and Section 2.
   - 5 high-conversion bento decks featuring 920px animated previews, technical blockquotes, 2-column engineering spec tables, reproducible CLI commands, and links to `tutorials/*.md`.
   - Star driver badges and zero emojis.
4. **CI/CD Quality & Governance**:
   - `cargo fmt --check` passes.
   - `cargo clippy --all-targets -- -D warnings` passes with exit code 0.
   - `cargo test` passes all tests with exit code 0.
   - Zero developer machine path leaks (personal username or absolute filesystem paths).
   - Conventional commits and push to `origin/main`.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Dedicated Theme Module `src/gui/theme.rs` | Modular theme engine: ZincDark, DeepSlate, ObsidianMono, NordicFrost, SolarizedDark | M1 | Survey |
| 2 | egui Visuals & Spacing Calibration | Complete `configure_visuals` with `slider_trailing_fill`, `extreme_bg_color`, 1px borders | M1 | Survey |
| 3 | Polished KPI Stat Cards | 54px cards, 6px radius, prominent 18pt typography, no decorative stripes | M1 | Survey |
| 4 | Refined Waterfall Chart | 4 horizontal gridlines (25/50/75/100%), 3px bar tops, interactive hover tooltips | M1 | Survey |
| 5 | Modern Interactive Sliders | Minimalist track, progressive colored fill, clean parameter labels | M1 | Survey |
| 6 | Button Contrast & Bug Fix | Replace hardcoded `Color32::BLACK` on dark buttons with `Color32::WHITE`/`p.text_main` | M1 | Survey |
| 7 | Zero Emojis in GUI Code | Verify 0 emojis in `src/gui/`, preserve technical Unicode glyphs (`α`, `β`, `•`, etc.) | M1 | Survey |
| 8 | Pure Rust GIF Generator | Self-contained LZW GIF89a encoder in `examples/generate_showcase_gifs.rs` | M2 | Survey |
| 9 | Flow 1 GIF: Cockpit & AutoDecider | Model ingestion, sensitivity waterfall, SVD energy decay, Pareto knapsack | M2 | Survey |
| 10 | Flow 2 GIF: Roofline Profiler | H100 SXM5 ceilings, batch scaling (1->64), knee (295.34 FLOP/B), 4-tier cache | M2 | Survey |
| 11 | Flow 3 GIF: Speculative Decoding | Medusa tree branching, 2D causal mask, rejection sampling, Leviathan speedup >1.5x | M2 | Survey |
| 12 | Flow 4 GIF: 8 Model Paradigms | Dense MLP to MoE upcycling (+77.9%), SLM BitNet b1.58 ternary compaction (-81.9%) | M2 | Survey |
| 13 | Flow 5 GIF: PyTorch & CUDA Bridge | LoRA zero-copy fold, instant standalone `train_cuda.py` script generation | M2 | Survey |
| 14 | README Showcase Section Placement | Insert `## Interactive Capabilities Showcase` between Sections 1 and 2 | M3 | Survey |
| 15 | README Top Navigation Bar Link | Add `<a href="#interactive-capabilities-showcase">Showcase</a> •` | M3 | Survey |
| 16 | 5 Bento Feature Decks in README | 920px GIF embeds, blockquotes, 2-column tables, CLI one-liners, tutorial links | M3 | Survey |
| 17 | README Star Conversion & Badges | High-visibility GitHub badges, community CTA, zero emojis | M3 | Survey |
| 18 | Quality Gate: Clippy, Fmt, Tests | Zero warnings, clean format, all unit & integration tests pass | M4 | Survey |
| 19 | Path Hygiene Guard | Zero developer username or machine path leaks | M4 | Survey |
| 20 | Git Commit & Push | Conventional commits and push to `origin/main` | M4 | Survey |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Desktop Studio GUI Elevation | `src/gui/theme.rs`, `src/gui/mod.rs`, `src/gui/app.rs` | none | DONE |
| M2 | Showcase Animated GIFs Generation | `examples/generate_showcase_gifs.rs`, `assets/*.gif` | none | DONE |
| M3 | README Showcase Redesign | `README.md` showcase section & top nav bar | M2 | DONE |
| M4 | Final Milestone: Quality Gate & Push | Full test suite, clippy, fmt, path check, push to `origin/main` | M1, M2, M3 | DONE |

---

## Interface Contracts

### Theme Engine ↔ Desktop Studio App
- `pub mod theme;` in `src/gui/mod.rs`
- `pub enum GuiTheme { ZincDark, DeepSlate, ObsidianMono, NordicFrost, SolarizedDark }`
- `pub struct ThemePalette { bg_dark, bg_card, bg_input, border, text_main, text_muted, primary, secondary, accent, success, warning, danger }`
- `pub fn configure_visuals(ctx: &egui::Context, p: &ThemePalette)`
- `pub fn draw_kpi_card(ui: &mut egui::Ui, title: &str, value: &str, badge: Option<(&str, Color32)>, p: &ThemePalette)`
- `pub fn draw_waterfall_chart(ui: &mut egui::Ui, layers: &[LayerNode], p: &ThemePalette)`
- `pub fn draw_tab_button(ui: &mut egui::Ui, title: &str, tab_idx: usize, active_tab: &mut usize, p: &ThemePalette)`

### GIF Generator ↔ Asset File Output
- `cargo run --release --example generate_showcase_gifs`
- Outputs to `assets/`:
  - `assets/showcase_cockpit_autodecider.gif` (and alias `assets/cockpit_layer_surgery.gif`)
  - `assets/showcase_roofline_profiler.gif` (and alias `assets/roofline_profiler.gif`)
  - `assets/showcase_speculative_decoding.gif` (and alias `assets/speculative_decoding.gif`)
  - `assets/showcase_paradigm_converter.gif` (and alias `assets/paradigm_converter.gif`)
  - `assets/showcase_pytorch_cuda_bridge.gif` (and alias `assets/pytorch_cuda_bridge.gif`)

### README ↔ Asset Resolution
- Embedded `<img src="assets/showcase_cockpit_autodecider.gif" width="920" />`
- Embedded `<img src="assets/showcase_roofline_profiler.gif" width="920" />`
- Embedded `<img src="assets/showcase_speculative_decoding.gif" width="920" />`
- Embedded `<img src="assets/showcase_paradigm_converter.gif" width="920" />`
- Embedded `<img src="assets/showcase_pytorch_cuda_bridge.gif" width="920" />`
- All tutorial links point to verified files in `tutorials/`.

---

## Code Layout
- `src/gui/theme.rs`: Color tokens, `GuiTheme`, `ThemePalette`, visual styles, reusable UI components.
- `src/gui/app.rs`: State machine, tab views, event loops.
- `src/gui/mod.rs`: GUI exports and eframe startup.
- `examples/generate_showcase_gifs.rs`: Headless pure-Rust GIF generator binary.
- `assets/`: Committed animated showcase GIF files.
- `README.md`: Root documentation and showcase presentation.
- Exclusive ownership: Concurrent workers MUST NOT write to the same file.
