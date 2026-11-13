// SPDX-License-Identifier: AGPL-3.0-only
//! `thme figure`: draw a model as an isometric figure (one SVG per theme, no stylesheet).

use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::engine::iso::Theme;
use crate::parser::ModelImporter;
use crate::report::figure::model_figure_svg;

#[derive(Args, Debug)]
pub struct FigureArgs {
    /// Model file or folder (GGUF, SafeTensors folder, .epiph); scanned before it is read
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Output folder; writes <name>-dark.svg and <name>-light.svg
    #[arg(long)]
    pub out: PathBuf,
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
    Ok(())
}
