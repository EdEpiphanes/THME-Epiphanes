// SPDX-License-Identifier: AGPL-3.0-only
//! Documentation capture mode for THME Studio.
//!
//! When `THME_GUI_CAPTURE_DIR` is set, the Studio opens, shows each tab in turn, saves
//! a screenshot of it to that directory as an uncompressed BMP, and closes. The images
//! are the real application rendering its built-in demo model, so README visuals can be
//! regenerated whenever the GUI changes (`scripts/capture_studio.ps1`).
//!
//! With `THME_GUI_RECORD=<tab>:<frames>` as well, it instead records one tab as a frame
//! sequence (`rec-000.bmp`, ...), about one frame every `RECORD_STRIDE` render frames, for
//! README GIFs (`scripts/record_gif.ps1`).

use eframe::egui;
use std::io::Write;
use std::path::{Path, PathBuf};

/// File-name stems for tabs 0..=10, in tab order.
pub const TAB_NAMES: [&str; 11] = [
    "dashboard",
    "models",
    "layer-tweaker",
    "matrix-engine",
    "benchmarks",
    "history",
    "audit-log",
    "surgery-export",
    "dev-lab",
    "co-surgery",
    "live-3d",
];

/// Frames to let a tab lay out and settle before it is captured.
const SETTLE_FRAMES: u32 = 45;

/// Frames to wait before recording starts (the Live 3D demo needs its first window).
const RECORD_SETTLE_FRAMES: u32 = 150;

/// Render frames between recorded frames (about 15 recorded frames per second at 60 FPS).
const RECORD_STRIDE: u32 = 4;

struct Record {
    tab: usize,
    total: usize,
    saved: usize,
}

pub struct CaptureState {
    dir: PathBuf,
    tab: usize,
    frames: u32,
    requested: bool,
    record: Option<Record>,
}

impl CaptureState {
    /// Enables capture mode when `THME_GUI_CAPTURE_DIR` is set.
    pub fn from_env() -> Option<Self> {
        std::env::var_os("THME_GUI_CAPTURE_DIR").map(|dir| Self {
            dir: PathBuf::from(dir),
            tab: 0,
            frames: 0,
            requested: false,
            record: std::env::var("THME_GUI_RECORD")
                .ok()
                .and_then(|spec| parse_record(&spec)),
        })
    }

    /// Advances the capture sequence and returns the tab to display this frame.
    pub fn step(&mut self, ctx: &egui::Context) -> usize {
        if self.record.is_some() {
            return self.record_step(ctx);
        }
        if self.tab >= TAB_NAMES.len() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return TAB_NAMES.len() - 1;
        }
        let shot = ctx.input(|input| {
            input.events.iter().find_map(|event| match event {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = shot {
            let path = self
                .dir
                .join(format!("{:02}-{}.bmp", self.tab, TAB_NAMES[self.tab]));
            if let Err(error) = write_bmp(&path, &image) {
                eprintln!("THME Studio capture: failed to write {path:?}: {error}");
            }
            self.tab += 1;
            self.frames = 0;
            self.requested = false;
        } else {
            self.frames += 1;
            if self.frames >= SETTLE_FRAMES && !self.requested {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
                self.requested = true;
            }
        }
        ctx.request_repaint();
        self.tab.min(TAB_NAMES.len() - 1)
    }
}

impl CaptureState {
    fn record_step(&mut self, ctx: &egui::Context) -> usize {
        let Some(record) = self.record.as_mut() else {
            return 0;
        };
        let tab = record.tab;
        if record.saved >= record.total {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return tab;
        }
        let shot = ctx.input(|input| {
            input.events.iter().find_map(|event| match event {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        self.frames += 1;
        if let Some(image) = shot {
            let path = self.dir.join(format!("rec-{:03}.bmp", record.saved));
            if let Err(error) = write_bmp(&path, &image) {
                eprintln!("THME Studio capture: failed to write {path:?}: {error}");
            }
            record.saved += 1;
            self.requested = false;
        } else if !self.requested
            && self.frames >= RECORD_SETTLE_FRAMES
            && self.frames.is_multiple_of(RECORD_STRIDE)
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            self.requested = true;
        }
        ctx.request_repaint();
        tab
    }
}

/// Parses `<tab>:<frames>`, e.g. `10:90`; the tab is clamped to the last tab.
fn parse_record(spec: &str) -> Option<Record> {
    let (tab, frames) = spec.split_once(':')?;
    let tab = tab.trim().parse::<usize>().ok()?.min(TAB_NAMES.len() - 1);
    let total = frames.trim().parse::<usize>().ok()?.clamp(1, 600);
    Some(Record {
        tab,
        total,
        saved: 0,
    })
}

/// Writes a 24-bit bottom-up BMP; no image codec dependency is needed.
fn write_bmp(path: &Path, image: &egui::ColorImage) -> std::io::Result<()> {
    let [width, height] = image.size;
    let row_bytes = (width * 3).div_ceil(4) * 4;
    let data_len = row_bytes * height;
    let to_u32 = |value: usize| {
        u32::try_from(value)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "image too large"))
    };
    let mut out = Vec::with_capacity(54 + data_len);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&to_u32(54 + data_len)?.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&to_u32(width)?.to_le_bytes());
    out.extend_from_slice(&to_u32(height)?.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&to_u32(data_len)?.to_le_bytes());
    out.extend_from_slice(&[0; 16]);
    for row in image.pixels.chunks_exact(width).rev() {
        for pixel in row {
            out.extend_from_slice(&[pixel.b(), pixel.g(), pixel.r()]);
        }
        out.resize(out.len() + row_bytes - width * 3, 0);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::File::create(path)?.write_all(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_spec_parses_and_clamps() {
        let r = parse_record("10:90").unwrap();
        assert_eq!((r.tab, r.total, r.saved), (10, 90, 0));
        assert_eq!(parse_record("99:5").unwrap().tab, TAB_NAMES.len() - 1);
        assert_eq!(parse_record("0:100000").unwrap().total, 600);
        assert!(parse_record("live").is_none());
        assert!(parse_record("x:3").is_none());
    }

    #[test]
    fn bmp_has_padded_rows_and_correct_header() {
        let image = egui::ColorImage::new([3, 2], egui::Color32::from_rgb(10, 20, 30));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.bmp");
        write_bmp(&path, &image).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        // 3 px * 3 bytes = 9, padded to 12 per row, 2 rows.
        assert_eq!(bytes.len(), 54 + 24);
        assert_eq!(&bytes[..2], b"BM");
        assert_eq!(&bytes[54..57], &[30, 20, 10]);
    }
}
