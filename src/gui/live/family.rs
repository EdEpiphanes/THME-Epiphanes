// SPDX-License-Identifier: AGPL-3.0-only
//! Tensor format families and theme-aware colours for the live view.

use eframe::egui::Color32;

use crate::gui::manga;
use crate::gui::theme::ThemePalette;
use crate::ir::TensorFormat;

/// Representation families a tensor can belong to. Binary and phase families are part of
/// the THME research design but have no implementation yet; they appear in the legend
/// marked as such so the view never implies they are running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Dense,
    Int8,
    Int4,
    Ternary,
    Binary,
    Phase,
}

impl Family {
    pub fn of(format: &TensorFormat) -> Self {
        match format {
            TensorFormat::FP32
            | TensorFormat::FP16
            | TensorFormat::BF16
            | TensorFormat::Custom(_) => Self::Dense,
            TensorFormat::INT8 | TensorFormat::Q8_0 | TensorFormat::Q5_K | TensorFormat::Q6_K => {
                Self::Int8
            }
            TensorFormat::Q4_0 | TensorFormat::Q4_K | TensorFormat::Q2_K => Self::Int4,
            TensorFormat::Ternary => Self::Ternary,
            TensorFormat::Bit1_5 => Self::Binary,
        }
    }

    pub(super) fn from_dtype(dtype: &str) -> Self {
        match dtype {
            "Q8_0" | "Q8_1" | "Q8_K" | "I8" => Self::Int8,
            d if d.starts_with("Q4")
                || d.starts_with("Q5")
                || d.starts_with("Q2")
                || d.starts_with("Q3")
                || d.starts_with("IQ") =>
            {
                Self::Int4
            }
            "TQ1_0" | "TQ2_0" => Self::Ternary,
            _ => Self::Dense,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense (FP32/FP16/BF16)",
            Self::Int8 => "5 to 8-bit blocks",
            Self::Int4 => "4-bit and below",
            Self::Ternary => "Ternary {-1, 0, +1}",
            Self::Binary => "Binary (not implemented)",
            Self::Phase => "Phase (not implemented)",
        }
    }

    /// Categorical hues validated for dark surfaces; identity is also carried by the legend.
    fn color(self) -> Color32 {
        match self {
            Self::Dense => Color32::from_rgb(0x39, 0x87, 0xe5),
            Self::Int8 => Color32::from_rgb(0x19, 0x9e, 0x70),
            Self::Int4 => Color32::from_rgb(0xc9, 0x85, 0x00),
            Self::Ternary => Color32::from_rgb(0xd9, 0x59, 0x26),
            Self::Binary => Color32::from_rgb(0x90, 0x85, 0xe9),
            Self::Phase => Color32::from_rgb(0xd5, 0x51, 0x81),
        }
    }
}

const ANOMALY: Color32 = Color32::from_rgb(0xd0, 0x3b, 0x3b);

/// Anomaly marker: red in colour themes, solid ink in the black-and-white panel theme.
pub(super) fn anomaly(p: &ThemePalette) -> Color32 {
    if p.is_mono() {
        manga::ink(p)
    } else {
        ANOMALY
    }
}

impl Family {
    /// Family colour for the active theme. Black-and-white themes use distinct grey tones
    /// (and, in the manifold, screentone densities) instead of hue.
    pub(super) fn color_for(self, p: &ThemePalette) -> Color32 {
        if !p.is_mono() {
            return self.color();
        }
        Color32::from_gray(match self {
            Self::Dense => 46,
            Self::Int8 => 92,
            Self::Int4 => 132,
            Self::Ternary => 168,
            Self::Binary => 196,
            Self::Phase => 220,
        })
    }

    /// Screentone dot radius used for this family on the manifold in mono themes.
    pub(super) fn tone_radius(self) -> f32 {
        match self {
            Self::Dense => 0.0,
            Self::Int8 => 0.7,
            Self::Int4 => 1.0,
            Self::Ternary => 1.3,
            Self::Binary => 1.6,
            Self::Phase => 1.9,
        }
    }
}

/// The family most tensors of a layer use (Dense when unknown).
pub(super) fn dominant_family(entries: Option<&Vec<(String, Family)>>) -> Family {
    let Some(entries) = entries else {
        return Family::Dense;
    };
    let mut best = (Family::Dense, 0usize);
    for family in ALL_FAMILIES {
        let count = entries.iter().filter(|(_, f)| *f == family).count();
        if count > best.1 {
            best = (family, count);
        }
    }
    best.0
}

pub(super) const ALL_FAMILIES: [Family; 6] = [
    Family::Dense,
    Family::Int8,
    Family::Int4,
    Family::Ternary,
    Family::Binary,
    Family::Phase,
];
