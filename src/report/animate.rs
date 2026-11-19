// SPDX-License-Identifier: AGPL-3.0-only
//! Animated figures in Rust: SVG frames rasterized with resvg and encoded as a looping GIF with
//! the gif crate. No browser, Node or Python step.

use anyhow::{anyhow, Context, Result};
use std::path::Path;

use resvg::{tiny_skia, usvg};

use crate::engine::iso::Theme;
use crate::ir::model::ModelIR;
use crate::report::figure::{
    forward_steps, layer_indices, model_figure_frame, model_figure_layer_frame,
};

/// Each forward-pass frame's delay in 1/100 s.
const DELAY: u16 = 7;
/// One layer per 0.4 s in the layers animation.
const LAYER_DELAY: u16 = 40;
/// The Chakra Petch faces of the manga card, so the frames do not depend on installed fonts.
const CHAKRA_BOLD: &[u8] = include_bytes!("../../assets/fonts/ChakraPetch-Bold.ttf");
const CHAKRA_MEDIUM: &[u8] = include_bytes!("../../assets/fonts/ChakraPetch-Medium.ttf");

/// Rasterizes SVG frames and writes them as a looping GIF. Every frame must have the same size.
pub struct GifWriter {
    options: usvg::Options<'static>,
    encoder: Option<gif::Encoder<std::fs::File>>,
    path: std::path::PathBuf,
    size: Option<(u16, u16)>,
    /// Write each frame as a PNG here too (for checking frames by eye).
    png_dir: Option<std::path::PathBuf>,
    frames: usize,
}

impl GifWriter {
    pub fn create(path: &Path, png_dir: Option<&Path>) -> Result<Self> {
        let mut options = usvg::Options::default();
        options.fontdb_mut().load_system_fonts();
        options.fontdb_mut().load_font_data(CHAKRA_BOLD.to_vec());
        options.fontdb_mut().load_font_data(CHAKRA_MEDIUM.to_vec());
        if let Some(d) = png_dir {
            std::fs::create_dir_all(d)?;
        }
        Ok(Self {
            options,
            encoder: None,
            path: path.to_path_buf(),
            size: None,
            png_dir: png_dir.map(Path::to_path_buf),
            frames: 0,
        })
    }

    /// Adds one SVG frame shown for `delay` hundredths of a second.
    pub fn frame(&mut self, svg: &str, delay: u16) -> Result<()> {
        let tree = usvg::Tree::from_str(svg, &self.options).context("frame is not valid SVG")?;
        let size = tree.size().to_int_size();
        let (w, h) = (u16::try_from(size.width())?, u16::try_from(size.height())?);
        if let Some(first) = self.size {
            if first != (w, h) {
                return Err(anyhow!(
                    "frame {} is {w}x{h}, the first was {}x{}",
                    self.frames,
                    first.0,
                    first.1
                ));
            }
        }
        let mut pixmap = tiny_skia::Pixmap::new(size.width(), size.height())
            .ok_or_else(|| anyhow!("empty frame"))?;
        resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());
        if let Some(dir) = &self.png_dir {
            pixmap.save_png(dir.join(format!("frame-{:03}.png", self.frames)))?;
        }
        if self.encoder.is_none() {
            let file = std::fs::File::create(&self.path)
                .with_context(|| format!("creating {:?}", self.path))?;
            let mut e = gif::Encoder::new(file, w, h, &[])?;
            e.set_repeat(gif::Repeat::Infinite)?;
            self.encoder = Some(e);
            self.size = Some((w, h));
        }
        let mut rgba = pixmap.take();
        let mut frame = gif::Frame::from_rgba_speed(w, h, &mut rgba, 10);
        frame.delay = delay;
        if let Some(e) = self.encoder.as_mut() {
            e.write_frame(&frame)?;
        }
        self.frames += 1;
        Ok(())
    }

    /// Finishes the file; returns the number of frames.
    pub fn finish(self) -> Result<usize> {
        if self.frames == 0 {
            return Err(anyhow!("no frames"));
        }
        drop(self.encoder);
        Ok(self.frames)
    }
}

/// The forward-pass animation on the `theme` card: every tensor lights up in forward order as
/// ink glass with an ink splash (black on paper in the light theme, blood-red in the dark).
/// Returns the number of frames.
pub fn model_forward_gif(
    model: &ModelIR,
    theme: Theme,
    path: &Path,
    png_dir: Option<&Path>,
) -> Result<usize> {
    let mut gif = GifWriter::create(path, png_dir)?;
    for h in forward_steps(model) {
        gif.frame(&model_figure_frame(model, theme, &h), DELAY)?;
    }
    gif.finish()
}

/// The layers animation on the `theme` card: one frame per layer, every tensor of that layer
/// lit in turn, with one splash under it. Returns the number of frames.
pub fn model_layers_gif(
    model: &ModelIR,
    theme: Theme,
    path: &Path,
    png_dir: Option<&Path>,
) -> Result<usize> {
    let layers = layer_indices(model);
    if layers.is_empty() {
        return Err(anyhow!("model has no layer structure"));
    }
    let mut gif = GifWriter::create(path, png_dir)?;
    for l in layers {
        gif.frame(&model_figure_layer_frame(model, theme, l), LAYER_DELAY)?;
    }
    gif.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_looping_gif_and_refuses_mismatched_frames() {
        let dir = tempfile::tempdir().unwrap();
        let svg = |w: u32| {
            format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"20\"><rect width=\"{w}\" height=\"20\" fill=\"#000\"/></svg>")
        };
        let path = dir.path().join("t.gif");
        let mut g = GifWriter::create(&path, None).unwrap();
        g.frame(&svg(30), 5).unwrap();
        g.frame(&svg(30), 5).unwrap();
        assert!(g.frame(&svg(31), 5).is_err());
        assert_eq!(g.finish().unwrap(), 2);
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"GIF89a"));
        let mut decoder = gif::DecodeOptions::new()
            .read_info(std::io::Cursor::new(bytes))
            .unwrap();
        let mut n = 0;
        while decoder.read_next_frame().unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, 2);
        assert!(GifWriter::create(&dir.path().join("e.gif"), None)
            .unwrap()
            .finish()
            .is_err());
    }

    fn mock() -> ModelIR {
        let dir = tempfile::tempdir().unwrap();
        let (model_dir, _) = crate::cli::evaluate::tests::synthetic_checkpoint(dir.path());
        crate::parser::ModelImporter::import(&model_dir).unwrap()
    }

    fn frame_count(path: &Path) -> usize {
        let bytes = std::fs::read(path).unwrap();
        let mut decoder = gif::DecodeOptions::new()
            .read_info(std::io::Cursor::new(bytes))
            .unwrap();
        let mut frames = 0;
        while decoder.read_next_frame().unwrap().is_some() {
            frames += 1;
        }
        frames
    }

    #[test]
    fn layers_gif_has_one_frame_per_layer_and_is_deterministic() {
        let m = mock();
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a.gif"), dir.path().join("b.gif"));
        let n = model_layers_gif(&m, Theme::Light, &a, None).unwrap();
        assert_eq!(n, layer_indices(&m).len());
        model_layers_gif(&m, Theme::Light, &b, None).unwrap();
        assert_eq!(
            std::fs::read(&a).unwrap(),
            std::fs::read(&b).unwrap(),
            "deterministic"
        );
        assert_eq!(frame_count(&a), n);
    }

    #[test]
    fn dark_gifs_have_the_same_frame_counts_as_the_light_ones() {
        let m = mock();
        let dir = tempfile::tempdir().unwrap();
        let (f, d) = (dir.path().join("f.gif"), dir.path().join("fd.gif"));
        let n = model_forward_gif(&m, Theme::Light, &f, None).unwrap();
        assert_eq!(n, forward_steps(&m).len());
        assert_eq!(model_forward_gif(&m, Theme::Dark, &d, None).unwrap(), n);
        assert_eq!(frame_count(&d), n);
    }

    #[test]
    fn layers_gif_refuses_a_model_without_layer_structure() {
        let mut m = mock();
        for t in &mut m.tensors {
            t.layer_index = None;
        }
        let dir = tempfile::tempdir().unwrap();
        let err = model_layers_gif(&m, Theme::Light, &dir.path().join("x.gif"), None).unwrap_err();
        assert!(err.to_string().contains("no layer structure"), "{err}");
    }
}
