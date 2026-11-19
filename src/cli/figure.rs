// SPDX-License-Identifier: AGPL-3.0-only
//! `thme figure`: draw a model as an isometric figure (one SVG per theme, no stylesheet) and,
//! with `--gif`, four animations: forward pass and layers, each on the light and dark card.

use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::engine::iso::Theme;
use crate::parser::ModelImporter;
use crate::report::animate::{model_forward_gif, model_layers_gif};
use crate::report::figure::model_figure_svg;

#[derive(Args, Debug)]
pub struct FigureArgs {
    /// Model file or folder (GGUF, SafeTensors folder, .epiph); scanned before it is read
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Output folder; writes <name>-dark.svg and <name>-light.svg
    #[arg(long)]
    pub out: PathBuf,

    /// Also write <name>-forward.gif, <name>-forward-dark.gif, <name>-layers.gif and
    /// <name>-layers-dark.gif
    #[arg(long)]
    pub gif: bool,

    /// With --gif, also write every frame as a PNG into this folder: the light forward frames go
    /// to DIR, the others to DIR/forward-dark, DIR/layers and DIR/layers-dark
    #[arg(long, requires = "gif")]
    pub frames: Option<PathBuf>,
}

pub fn execute(args: FigureArgs) -> Result<()> {
    let model = ModelImporter::import(&args.model)?;
    std::fs::create_dir_all(&args.out)?;
    let stem: String = model
        .name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    for (name, theme) in [("dark", Theme::Dark), ("light", Theme::Light)] {
        let path = args.out.join(format!("{stem}-{name}.svg"));
        std::fs::write(&path, model_figure_svg(&model, theme))?;
        println!("wrote {}", path.display());
    }
    if args.gif {
        let sub = |name: &str| args.frames.as_ref().map(|d| d.join(name));
        let jobs: [(&str, Theme, Option<PathBuf>); 4] = [
            ("forward", Theme::Light, args.frames.clone()),
            ("forward-dark", Theme::Dark, sub("forward-dark")),
            ("layers", Theme::Light, sub("layers")),
            ("layers-dark", Theme::Dark, sub("layers-dark")),
        ];
        for (name, theme, png_dir) in jobs {
            let path = args.out.join(format!("{stem}-{name}.gif"));
            let frames = if name.starts_with("forward") {
                model_forward_gif(&model, theme, &path, png_dir.as_deref())?
            } else {
                model_layers_gif(&model, theme, &path, png_dir.as_deref())?
            };
            println!("wrote {} ({frames} frames)", path.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::figure::{forward_steps, layer_indices};

    #[test]
    fn gif_flag_writes_the_four_animations_with_their_frame_counts() {
        let dir = tempfile::tempdir().unwrap();
        let (model_dir, _) = crate::cli::evaluate::tests::synthetic_checkpoint(dir.path());
        let out = dir.path().join("out");
        execute(FigureArgs {
            model: model_dir.clone(),
            out: out.clone(),
            gif: true,
            frames: None,
        })
        .unwrap();
        let model = ModelImporter::import(&model_dir).unwrap();
        let frames_of = |name: &str| {
            let file = std::fs::read_dir(&out)
                .unwrap()
                .map(|e| e.unwrap().path())
                .find(|p| p.to_string_lossy().ends_with(&format!("-{name}.gif")))
                .unwrap();
            let bytes = std::fs::read(file).unwrap();
            let mut dec = gif::DecodeOptions::new()
                .read_info(std::io::Cursor::new(bytes))
                .unwrap();
            let mut n = 0;
            while dec.read_next_frame().unwrap().is_some() {
                n += 1;
            }
            n
        };
        let forward = forward_steps(&model).len();
        let layers = layer_indices(&model).len();
        assert_eq!(frames_of("forward"), forward);
        assert_eq!(frames_of("forward-dark"), forward);
        assert_eq!(frames_of("layers"), layers);
        assert_eq!(frames_of("layers-dark"), layers);
        assert_eq!(std::fs::read_dir(&out).unwrap().count(), 6, "2 SVG + 4 GIF");
    }
}
