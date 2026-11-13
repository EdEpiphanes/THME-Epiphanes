//! `thme epiph`: pack any model THME can read into one `.epiph` file, and verify one.

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::epiph::{self, PackOptions};
use crate::exporter::compute_file_sha256;
use crate::inference::CalibrationFile;
use crate::parser::ModelImporter;

#[derive(Args, Debug)]
pub struct EpiphArgs {
    #[command(subcommand)]
    pub command: EpiphCommand,
}

#[derive(Subcommand, Debug)]
pub enum EpiphCommand {
    /// Pack a model (SafeTensors directory, GGUF or .epiph) with its metadata into one .epiph file
    Pack(PackArgs),
    /// Check an .epiph file: header, layout, every SHA-256 digest, and every tensor's shape and type
    Verify {
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
}

#[derive(Args, Debug)]
pub struct PackArgs {
    #[arg(value_name = "MODEL_PATH")]
    pub model: PathBuf,

    /// Output file (extension .epiph)
    #[arg(long)]
    pub out: PathBuf,

    /// Recipe JSON that produced this model
    #[arg(long)]
    pub recipe: Option<PathBuf>,

    /// Calibration statistics from `thme calibrate`, attached to the file
    #[arg(long)]
    pub calibration: Option<PathBuf>,

    /// Report JSON (e.g. from `thme evaluate`) embedded as a quality entry labelled `imported`
    #[arg(long)]
    pub quality: Vec<PathBuf>,

    /// .epiph file this one was derived from (recorded by its SHA-256)
    #[arg(long)]
    pub parent: Vec<PathBuf>,

    /// Free annotation, `key=value`
    #[arg(long)]
    pub annotate: Vec<String>,
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("Failed to read {path:?}"))?;
    serde_json::from_str(&text).with_context(|| format!("Failed to parse {path:?}"))
}

pub fn pack(args: &PackArgs) -> Result<epiph::PackSummary> {
    if !epiph::is_epiph(&args.out) {
        bail!("--out must end in .{}", epiph::EXTENSION);
    }
    let model = ModelImporter::import(&args.model)?;
    let mut quality = Vec::new();
    for path in &args.quality {
        quality.push(json!({
            "evidence": "imported",
            "file_name": path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "sha256": compute_file_sha256(path)?,
            "report": read_json(path)?,
        }));
    }
    let mut annotations = BTreeMap::new();
    for item in &args.annotate {
        let (k, v) = item
            .split_once('=')
            .with_context(|| format!("--annotate '{item}' is not key=value"))?;
        annotations.insert(k.trim().to_string(), v.trim().to_string());
    }
    let options = PackOptions {
        recipe: args.recipe.as_deref().map(read_json).transpose()?,
        quality,
        calibration: args
            .calibration
            .as_deref()
            .map(CalibrationFile::load)
            .transpose()?,
        parents: args
            .parent
            .iter()
            .map(compute_file_sha256)
            .collect::<Result<_>>()?,
        annotations,
        created_unix: None,
    };
    epiph::write(&model, &options, &args.out)
}

pub fn execute(args: EpiphArgs) -> Result<()> {
    match args.command {
        EpiphCommand::Pack(pack_args) => {
            let s = pack(&pack_args)?;
            println!(
                "Packed {} tensors and {} attachments into {:?} ({} bytes, sha256 {})",
                s.tensors, s.attachments, pack_args.out, s.file_bytes, s.sha256
            );
        }
        EpiphCommand::Verify { file } => {
            let r = epiph::verify(&file)?;
            println!(
                "OK: {} tensors and {} attachments verified ({} payload bytes)",
                r.tensors, r.attachments, r.payload_bytes
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::evaluate::tests::synthetic_checkpoint;

    #[test]
    fn pack_from_the_command_line_and_verify() {
        let dir = tempfile::tempdir().unwrap();
        let (model, _) = synthetic_checkpoint(dir.path());
        let report = dir.path().join("report.json");
        std::fs::write(&report, r#"{"evaluation": {"verdict": "pass"}}"#).unwrap();
        let first = dir.path().join("a.epiph");
        let args = PackArgs {
            model: model.clone(),
            out: first.clone(),
            recipe: None,
            calibration: None,
            quality: vec![report],
            parent: vec![],
            annotate: vec!["stage=base".into()],
        };
        pack(&args).unwrap();
        // Re-pack from the .epiph itself, recording lineage.
        let second = dir.path().join("b.epiph");
        let child = PackArgs {
            model: first.clone(),
            out: second.clone(),
            parent: vec![first.clone()],
            quality: vec![],
            annotate: vec![],
            ..args
        };
        pack(&child).unwrap();
        let (_, manifest) = epiph::read(&second).unwrap();
        assert_eq!(
            manifest.lineage.parents,
            vec![compute_file_sha256(&first).unwrap()]
        );
        let (_, first_manifest) = epiph::read(&first).unwrap();
        assert_eq!(first_manifest.quality[0]["evidence"], "imported");
        assert_eq!(first_manifest.annotations["stage"], "base");
        assert!(epiph::verify(&second).is_ok());

        let bad = PackArgs {
            model,
            out: dir.path().join("x.gguf"),
            recipe: None,
            calibration: None,
            quality: vec![],
            parent: vec![],
            annotate: vec!["no-equals".into()],
        };
        assert!(pack(&bad).is_err());
    }
}
