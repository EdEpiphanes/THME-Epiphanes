//! Modern Dark-Mode Developer Tool Theme Engine for THME Desktop Studio
//!
//! Provides OLED, Zinc, and Deep Slate color palettes calibrated for
//! Linear/Vercel/Raycast developer tool aesthetics with zero emojis,
//! WCAG AAA contrast compliance, and elevated reusable components.

use eframe::egui;
use egui::{Color32, CornerRadius, Pos2, Rect, Stroke, Vec2};
use serde::{Deserialize, Serialize};

use super::app::LayerNode;

// ============================================================================
// Master Color Tokens (Tailwind Zinc, Slate, and Developer Accents)
// ============================================================================

/// Pitch Black OLED / Zinc 950 Base (`#09090B`)
pub const ZINC_950: Color32 = Color32::from_rgb(9, 9, 11);
/// Elevated Surface / Zinc 900 Card (`#18181B`)
pub const ZINC_900: Color32 = Color32::from_rgb(24, 24, 27);
/// Mid-Elevated Surface / Zinc 850 (`#1F1F23`)
pub const ZINC_850: Color32 = Color32::from_rgb(31, 31, 35);
/// Inputs, Sliders, and Subtle Track / Zinc 800 (`#27272A`)
pub const ZINC_800: Color32 = Color32::from_rgb(39, 39, 42);
/// Subtle Card Borders / Zinc 700 (`#3F3F46`)
pub const ZINC_700: Color32 = Color32::from_rgb(63, 63, 70);
/// Inactive & Muted Borders / Zinc 600 (`#52525B`)
pub const ZINC_600: Color32 = Color32::from_rgb(82, 82, 91);
/// Secondary Muted Labels / Zinc 500 (`#71717A`)
pub const ZINC_500: Color32 = Color32::from_rgb(113, 113, 122);
/// Primary Muted Text / Zinc 400 (`#A1A1AA`)
pub const ZINC_400: Color32 = Color32::from_rgb(161, 161, 170);
/// High-Legibility Body Text / Zinc 200 (`#E4E4E7`)
pub const ZINC_200: Color32 = Color32::from_rgb(228, 228, 231);
/// Crisp Pure Main Text / Zinc 50 (`#FAFAFA`)
pub const ZINC_50: Color32 = Color32::from_rgb(250, 250, 250);

/// Deep Midnight Base / Slate 950 (`#020617`)
pub const SLATE_950: Color32 = Color32::from_rgb(2, 6, 23);
/// Midnight Card Surface / Slate 900 (`#0F172A`)
pub const SLATE_900: Color32 = Color32::from_rgb(15, 23, 42);
/// Midnight Input & Trough / Slate 800 (`#1E293B`)
pub const SLATE_800: Color32 = Color32::from_rgb(30, 41, 59);
/// Midnight Border / Slate 700 (`#334155`)
pub const SLATE_700: Color32 = Color32::from_rgb(51, 65, 85);
/// Slate Muted Text (`#94A3B8`)
pub const SLATE_400: Color32 = Color32::from_rgb(148, 163, 184);
/// Slate Crisp Main Text (`#F8FAFC`)
pub const SLATE_50: Color32 = Color32::from_rgb(248, 250, 252);

/// Cobalt Blue 500 (`#3B82F6`) - Linear Action Accent
pub const ACCENT_BLUE: Color32 = Color32::from_rgb(59, 130, 246);
/// Sky Blue 400 (`#38BDF8`) - Vercel Focus Accent
pub const ACCENT_SKY: Color32 = Color32::from_rgb(56, 189, 248);
/// Indigo 500 (`#6366F1`)
pub const ACCENT_INDIGO: Color32 = Color32::from_rgb(99, 102, 241);
/// Iris Violet 500 (`#8B5CF6`)
pub const ACCENT_VIOLET: Color32 = Color32::from_rgb(139, 92, 246);
/// Calibrated Emerald 500 (`#10B981`) - Success
pub const ACCENT_EMERALD: Color32 = Color32::from_rgb(16, 185, 129);
/// Calibrated Amber 500 (`#F59E0B`) - Warning
pub const ACCENT_AMBER: Color32 = Color32::from_rgb(245, 158, 11);
/// Refined Rose 500 (`#F43F5E`) - Danger
pub const ACCENT_ROSE: Color32 = Color32::from_rgb(244, 63, 94);
/// Surgical Cyan 500 (`#06B6D4`) - Manifold Teal
pub const ACCENT_CYAN: Color32 = Color32::from_rgb(6, 182, 212);

// ============================================================================
// GuiTheme Enum
// ============================================================================

/// UI/UX Pro Max Multi-Theme Palette Engine for THME Studio
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GuiTheme {
    /// Default modern developer tool theme based on Tailwind Zinc (950/900/800/700)
    #[default]
    #[serde(alias = "ObsidianCyan")]
    ZincDark,
    /// Deep Slate theme with midnight navy undertones
    #[serde(alias = "CyberpunkNeon")]
    DeepSlate,
    /// Pure pitch-black OLED monochrome theme with surgical cyan accents
    #[serde(alias = "MatrixEmerald")]
    ObsidianMono,
    /// Arctic Slate theme with sky-polar and glacier accents
    NordicFrost,
    /// Refined warm charcoal theme with amber gold accents
    #[serde(alias = "SolarizedAmber")]
    SolarizedDark,
}

impl GuiTheme {
    /// Canonical list of all themes for selectors and iteration
    pub const ALL: &'static [GuiTheme] = &[
        GuiTheme::ZincDark,
        GuiTheme::DeepSlate,
        GuiTheme::ObsidianMono,
        GuiTheme::NordicFrost,
        GuiTheme::SolarizedDark,
    ];

    /// Human-readable display label
    pub fn name(&self) -> &'static str {
        match self {
            GuiTheme::ZincDark => "Zinc Dark",
            GuiTheme::DeepSlate => "Deep Slate",
            GuiTheme::ObsidianMono => "Obsidian Mono",
            GuiTheme::NordicFrost => "Nordic Frost",
            GuiTheme::SolarizedDark => "Solarized Dark",
        }
    }

    /// Produce the calibrated 12-slot `ThemePalette`
    pub fn palette(&self) -> ThemePalette {
        match self {
            GuiTheme::ZincDark => ThemePalette {
                bg_dark: ZINC_950,
                bg_card: ZINC_900,
                bg_input: ZINC_800,
                border: ZINC_700,
                text_main: ZINC_50,
                text_muted: ZINC_400,
                primary: ACCENT_BLUE,
                secondary: ACCENT_VIOLET,
                accent: ACCENT_CYAN,
                success: ACCENT_EMERALD,
                warning: ACCENT_AMBER,
                danger: ACCENT_ROSE,
            },
            GuiTheme::DeepSlate => ThemePalette {
                bg_dark: SLATE_950,
                bg_card: SLATE_900,
                bg_input: SLATE_800,
                border: SLATE_700,
                text_main: SLATE_50,
                text_muted: SLATE_400,
                primary: ACCENT_SKY,
                secondary: ACCENT_INDIGO,
                accent: Color32::from_rgb(45, 212, 191),
                success: Color32::from_rgb(52, 211, 153),
                warning: Color32::from_rgb(251, 146, 60),
                danger: Color32::from_rgb(248, 113, 113),
            },
            GuiTheme::ObsidianMono => ThemePalette {
                bg_dark: Color32::from_rgb(0, 0, 0),
                bg_card: Color32::from_rgb(18, 18, 18),
                bg_input: Color32::from_rgb(28, 28, 28),
                border: Color32::from_rgb(48, 48, 48),
                text_main: Color32::from_rgb(255, 255, 255),
                text_muted: Color32::from_rgb(160, 160, 160),
                primary: Color32::from_rgb(255, 255, 255),
                secondary: Color32::from_rgb(180, 180, 180),
                accent: ACCENT_CYAN,
                success: ACCENT_EMERALD,
                warning: ACCENT_AMBER,
                danger: ACCENT_ROSE,
            },
            GuiTheme::NordicFrost => ThemePalette {
                bg_dark: Color32::from_rgb(15, 20, 32),
                bg_card: Color32::from_rgb(24, 32, 48),
                bg_input: Color32::from_rgb(36, 48, 70),
                border: Color32::from_rgb(54, 70, 96),
                text_main: Color32::from_rgb(240, 245, 252),
                text_muted: Color32::from_rgb(148, 163, 184),
                primary: ACCENT_SKY,
                secondary: ACCENT_INDIGO,
                accent: Color32::from_rgb(45, 212, 191),
                success: Color32::from_rgb(52, 211, 153),
                warning: Color32::from_rgb(251, 146, 60),
                danger: Color32::from_rgb(248, 113, 113),
            },
            GuiTheme::SolarizedDark => ThemePalette {
                bg_dark: Color32::from_rgb(18, 17, 16),
                bg_card: Color32::from_rgb(28, 27, 25),
                bg_input: Color32::from_rgb(42, 40, 37),
                border: Color32::from_rgb(64, 61, 57),
                text_main: Color32::from_rgb(248, 246, 240),
                text_muted: Color32::from_rgb(160, 156, 148),
                primary: Color32::from_rgb(217, 119, 6),
                secondary: Color32::from_rgb(234, 88, 12),
                accent: Color32::from_rgb(180, 83, 9),
                success: Color32::from_rgb(132, 204, 22),
                warning: Color32::from_rgb(245, 158, 11),
                danger: Color32::from_rgb(239, 68, 68),
            },
        }
    }
}

// ============================================================================
// ThemePalette Struct
// ============================================================================

/// 12-Slot Calibrated Developer Color Palette
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThemePalette {
    /// Root application background (OLED pitch black or deep navy base)
    pub bg_dark: Color32,
    /// Elevated cards, panels, and sidebars
    pub bg_card: Color32,
    /// Input controls, slider troughs, and unselected button states
    pub bg_input: Color32,
    /// Subtle structural 1px borders
    pub border: Color32,
    /// High-contrast primary text
    pub text_main: Color32,
    /// Secondary muted labels and captions
    pub text_muted: Color32,
    /// Primary brand/action accent (e.g. Cobalt Blue or Sky)
    pub primary: Color32,
    /// Secondary accent for highlights and badges
    pub secondary: Color32,
    /// Tertiary accent for special indicators
    pub accent: Color32,
    /// Semantic success (Emerald)
    pub success: Color32,
    /// Semantic warning (Amber)
    pub warning: Color32,
    /// Semantic danger (Rose/Red)
    pub danger: Color32,
}

impl Default for ThemePalette {
    fn default() -> Self {
        GuiTheme::ZincDark.palette()
    }
}

// ============================================================================
// Visuals & Spacing Calibration
// ============================================================================

/// Apply full egui Visuals, widget states, and spacing calibration
pub fn configure_visuals(ctx: &egui::Context, p: &ThemePalette) {
    let mut visuals = egui::Visuals::dark();
    visuals.dark_mode = true;
    visuals.panel_fill = p.bg_dark;
    visuals.window_fill = p.bg_card;
    visuals.window_stroke = Stroke::new(1.0_f32, p.border);
    visuals.window_corner_radius = CornerRadius::same(6);
    visuals.menu_corner_radius = CornerRadius::same(6);
    visuals.override_text_color = Some(p.text_main);
    visuals.extreme_bg_color = p.bg_input; // Used for text input troughs and slider tracks
    visuals.faint_bg_color = p.bg_card;
    visuals.slider_trailing_fill = true;

    // Selection styling with primary accent
    visuals.selection.stroke = Stroke::new(1.0_f32, p.primary);
    visuals.selection.bg_fill = p.primary.linear_multiply(0.25);

    // Non-interactive widgets
    visuals.widgets.noninteractive.bg_fill = p.bg_card;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, p.border);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, p.text_muted);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(4);

    // Inactive buttons and controls
    visuals.widgets.inactive.bg_fill = p.bg_input;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, p.border);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, p.text_muted);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(4);

    // Hovered widgets
    visuals.widgets.hovered.bg_fill = p.border;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, p.text_muted);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.5_f32, p.text_main);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(4);

    // Active / Pressed widgets
    visuals.widgets.active.bg_fill = p.primary.linear_multiply(0.20);
    visuals.widgets.active.bg_stroke = Stroke::new(1.5_f32, p.primary);
    visuals.widgets.active.fg_stroke = Stroke::new(1.5_f32, p.primary);
    visuals.widgets.active.corner_radius = CornerRadius::same(4);

    // Open menus / comboboxes
    visuals.widgets.open.bg_fill = p.bg_input;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0_f32, p.primary);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0_f32, p.text_main);
    visuals.widgets.open.corner_radius = CornerRadius::same(4);

    ctx.set_visuals(visuals);

    // Spacing and layout calibration
    ctx.style_mut(|style| {
        style.spacing.item_spacing = Vec2::new(8.0, 6.0);
        style.spacing.button_padding = Vec2::new(10.0, 5.0);
        style.spacing.slider_width = 160.0;
    });
}

// ============================================================================
// Elevated Reusable UI Components
// ============================================================================

/// Render an elevated KPI stat card (54px height, 6px radius, prominent 16pt value, no decorative stripes)
pub fn draw_kpi_card(
    ui: &mut egui::Ui,
    title: &str,
    value: &str,
    badge: Option<(&str, Color32)>,
    p: &ThemePalette,
) {
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 54.0), egui::Sense::hover());

    // Subtle border lighting and background shift on hover
    let stroke_col = if resp.hovered() {
        p.text_muted
    } else {
        p.border
    };
    let bg_col = if resp.hovered() {
        p.bg_card.linear_multiply(1.08)
    } else {
        p.bg_card
    };

    ui.painter().rect_filled(rect, 6.0_f32, bg_col);
    ui.painter().rect_stroke(
        rect,
        6.0_f32,
        Stroke::new(1.0_f32, stroke_col),
        egui::StrokeKind::Outside,
    );

    // Category title (muted uppercase 8.5pt)
    ui.painter().text(
        Pos2::new(rect.min.x + 12.0, rect.min.y + 14.0),
        egui::Align2::LEFT_CENTER,
        title.to_uppercase(),
        egui::FontId::proportional(8.5),
        p.text_muted,
    );

    // Primary Metric Value (prominent, high contrast 16.0pt)
    ui.painter().text(
        Pos2::new(rect.min.x + 12.0, rect.min.y + 36.0),
        egui::Align2::LEFT_CENTER,
        value,
        egui::FontId::proportional(16.0),
        p.text_main,
    );

    // Optional subtle badge or status indicator in top-right
    if let Some((badge_text, badge_col)) = badge {
        ui.painter().text(
            Pos2::new(rect.max.x - 12.0, rect.min.y + 14.0),
            egui::Align2::RIGHT_CENTER,
            badge_text,
            egui::FontId::monospace(8.0),
            badge_col,
        );
    }
}

/// Render a refined waterfall sensitivity chart (180px height, 4 gridlines, 3px rounded tops, hover tooltips)
pub fn draw_waterfall_chart(ui: &mut egui::Ui, layers: &[LayerNode], p: &ThemePalette) {
    let (rect, resp) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 180.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 6.0_f32, p.bg_card);
    painter.rect_stroke(
        rect,
        6.0_f32,
        Stroke::new(1.0_f32, p.border),
        egui::StrokeKind::Outside,
    );

    let graph_top = rect.min.y + 24.0;
    let graph_bottom = rect.max.y - 28.0;
    let graph_height = graph_bottom - graph_top;

    // Draw subtle horizontal gridlines at 25%, 50%, 75%, 100%
    for fraction in [0.25f32, 0.50, 0.75, 1.0] {
        let gy = graph_bottom - (graph_height * fraction);
        painter.line_segment(
            [
                Pos2::new(rect.min.x + 36.0, gy),
                Pos2::new(rect.max.x - 16.0, gy),
            ],
            Stroke::new(0.5_f32, p.border.linear_multiply(0.5)),
        );
        painter.text(
            Pos2::new(rect.min.x + 30.0, gy),
            egui::Align2::RIGHT_CENTER,
            format!("{:.0}%", fraction * 100.0),
            egui::FontId::monospace(7.0),
            p.text_muted.linear_multiply(0.7),
        );
    }

    let n = layers.len().max(1);
    let start_x = rect.min.x + 40.0;
    let available_w = rect.width() - 56.0;
    let bar_slot = available_w / n as f32;
    let bar_w = (bar_slot - 4.0).clamp(4.0, 32.0);

    let mouse_pos = resp.hover_pos();

    for (i, layer) in layers.iter().enumerate() {
        let bx = start_x + (i as f32 * bar_slot) + (bar_slot - bar_w) / 2.0;
        let reach = layer.reach_probability.clamp(0.0, 1.0);
        let bh = (graph_height * reach).max(2.0);
        let by = graph_bottom - bh;

        let bar_rect = Rect::from_min_size(Pos2::new(bx, by), Vec2::new(bar_w, bh));
        let is_hovered = mouse_pos.is_some_and(|pos| bar_rect.contains(pos));

        let color = if layer.beta_ternary >= 0.5 {
            p.success
        } else if layer.alpha_binary >= 0.35 {
            p.primary
        } else {
            p.warning
        };

        // Render bar with 3px rounded tops
        painter.rect_filled(
            bar_rect,
            3.0_f32,
            if is_hovered {
                color.linear_multiply(1.2)
            } else {
                color
            },
        );

        // Layer axis label (L0..Ln)
        painter.text(
            Pos2::new(bx + bar_w / 2.0, rect.max.y - 12.0),
            egui::Align2::CENTER_CENTER,
            format!("L{}", i),
            egui::FontId::monospace(7.5),
            if is_hovered {
                p.text_main
            } else {
                p.text_muted
            },
        );

        // Interactive hover tooltip displaying layer metrics
        if is_hovered {
            egui::show_tooltip_at_pointer(
                ui.ctx(),
                ui.layer_id(),
                resp.id.with(i),
                |ui: &mut egui::Ui| {
                    ui.label(
                        egui::RichText::new(format!("Layer {}: {}", i, layer.layer_type)).strong(),
                    );
                    ui.label(format!("Reach Probability: {:.1}%", reach * 100.0));
                    ui.label(format!(
                        "Bit-width: {:.0}% α / {:.0}% β / {:.0}% γ",
                        layer.alpha_binary * 100.0,
                        layer.beta_ternary * 100.0,
                        layer.gamma_phase * 100.0
                    ));
                    ui.label(format!(
                        "Est. Latency: {:.2} ms",
                        layer.benchmarked_latency_ms
                    ));
                },
            );
        }
    }
}

/// Render a sleek segmented pill tab button
pub fn draw_tab_button(
    ui: &mut egui::Ui,
    title: &str,
    tab_idx: usize,
    active_tab: &mut usize,
    p: &ThemePalette,
) {
    let is_active = *active_tab == tab_idx;
    let text_col = if is_active { p.text_main } else { p.text_muted };
    let bg_col = if is_active {
        p.bg_input
    } else {
        Color32::TRANSPARENT
    };
    let stroke = if is_active {
        Stroke::new(1.0_f32, p.primary.linear_multiply(0.8))
    } else {
        Stroke::NONE
    };

    let btn = ui.add(
        egui::Button::new(
            egui::RichText::new(title)
                .size(11.0)
                .strong()
                .color(text_col),
        )
        .fill(bg_col)
        .stroke(stroke)
        .corner_radius(4.0_f32),
    );

    if btn.clicked() {
        *active_tab = tab_idx;
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_theme_is_zinc_dark() {
        assert_eq!(GuiTheme::default(), GuiTheme::ZincDark);
        assert_eq!(GuiTheme::default().name(), "Zinc Dark");
    }

    #[test]
    fn test_all_themes_have_valid_palettes() {
        for theme in GuiTheme::ALL {
            assert!(!theme.name().is_empty());
            let p = theme.palette();
            // All alpha values must be 255 (fully opaque)
            assert_eq!(p.bg_dark.a(), 255);
            assert_eq!(p.bg_card.a(), 255);
            assert_eq!(p.bg_input.a(), 255);
            assert_eq!(p.border.a(), 255);
            assert_eq!(p.text_main.a(), 255);
            assert_eq!(p.text_muted.a(), 255);
            assert_eq!(p.primary.a(), 255);
            assert_eq!(p.secondary.a(), 255);
            assert_eq!(p.accent.a(), 255);
            assert_eq!(p.success.a(), 255);
            assert_eq!(p.warning.a(), 255);
            assert_eq!(p.danger.a(), 255);
        }
    }

    #[test]
    fn test_palette_contrast_ratios() {
        let p = GuiTheme::ZincDark.palette();
        // ZincDark text_main is #FAFAFA (very bright), bg_dark is #09090B (very dark)
        assert!(p.text_main.r() >= 240);
        assert!(p.text_main.g() >= 240);
        assert!(p.text_main.b() >= 240);
        assert!(p.bg_dark.r() <= 15);
        assert!(p.bg_dark.g() <= 15);
        assert!(p.bg_dark.b() <= 15);

        // Surface elevation hierarchy
        // bg_dark < bg_card < bg_input
        assert!(p.bg_dark.r() < p.bg_card.r());
        assert!(p.bg_card.r() < p.bg_input.r());
    }

    #[test]
    fn test_theme_serde_roundtrip() {
        for theme in GuiTheme::ALL {
            let json = serde_json::to_string(theme).expect("serialize");
            let deserialized: GuiTheme = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*theme, deserialized);
        }

        // Test backward-compatibility aliases
        let legacy_cyan: GuiTheme = serde_json::from_str("\"ObsidianCyan\"").expect("alias");
        assert_eq!(legacy_cyan, GuiTheme::ZincDark);

        let legacy_neon: GuiTheme = serde_json::from_str("\"CyberpunkNeon\"").expect("alias");
        assert_eq!(legacy_neon, GuiTheme::DeepSlate);

        let legacy_matrix: GuiTheme = serde_json::from_str("\"MatrixEmerald\"").expect("alias");
        assert_eq!(legacy_matrix, GuiTheme::ObsidianMono);

        let legacy_amber: GuiTheme = serde_json::from_str("\"SolarizedAmber\"").expect("alias");
        assert_eq!(legacy_amber, GuiTheme::SolarizedDark);
    }

    #[test]
    fn test_configure_visuals_does_not_panic() {
        let ctx = egui::Context::default();
        let p = GuiTheme::ZincDark.palette();
        configure_visuals(&ctx, &p);

        // Verify visuals applied
        let visuals = ctx.style().visuals.clone();
        assert!(visuals.dark_mode);
        assert!(visuals.slider_trailing_fill);
        assert_eq!(visuals.panel_fill, p.bg_dark);
        assert_eq!(visuals.window_fill, p.bg_card);
        assert_eq!(visuals.extreme_bg_color, p.bg_input);
        assert_eq!(visuals.override_text_color, Some(p.text_main));
    }
}
