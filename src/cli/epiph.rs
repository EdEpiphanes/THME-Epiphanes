// SPDX-License-Identifier: AGPL-3.0-only
//! `thme epiph`: pack any model THME can read into one `.epiph` file, and verify one.

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::epiph::sign::{self, SignatureStatus};
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
    /// Check an .epiph file: header, layout, every SHA-256 digest, every tensor's shape and type,
    /// and its signature (against the Epiphanes key unless --public-key is given)
    Verify {
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Minisign public key (.pub) to check the signature with
        #[arg(long)]
        public_key: Option<PathBuf>,
        /// Fail unless the file carries a valid signature
        #[arg(long)]
        require_signature: bool,
    },
    /// Write the Hugging Face files attached to an .epiph file (tokenizer, config) into a folder
    Extract {
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Folder to write the files into
        #[arg(long)]
        out: PathBuf,
    },
    /// Sign an .epiph file in place with a Minisign secret key (replaces an earlier signature)
    Sign {
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Minisign secret key file; it is read, never copied or printed
        #[arg(long)]
        secret_key: PathBuf,
        /// Name of an environment variable holding the key's password (omit for unencrypted keys)
        #[arg(long)]
        password_env: Option<String>,
        /// Matching public key, to check the new signature before writing it
        #[arg(long)]
        public_key: Option<PathBuf>,
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

    /// Apply --recipe and store each tensor in its selected format (Q8_0, Q4_K, ternary, ...)
    #[arg(long, requires = "recipe")]
    pub quantize: bool,

    /// With --quantize, skip recipe operations that cannot be applied instead of failing
    #[arg(long, requires = "quantize")]
    pub allow_partial: bool,
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("Failed to read {path:?}"))?;
    serde_json::from_str(&text).with_context(|| format!("Failed to parse {path:?}"))
}

fn load_public_key(path: &Path) -> Result<minisign::PublicKey> {
    let text = std::fs::read_to_string(path).with_context(|| format!("Failed to read {path:?}"))?;
    sign::public_key_from_text(&text)
}

pub fn pack(args: &PackArgs) -> Result<epiph::PackSummary> {
    if !epiph::is_epiph(&args.out) {
        bail!("--out must end in .{}", epiph::EXTENSION);
    }
    let mut model = ModelImporter::import(&args.model)?;
    if args.quantize {
        let recipe = args
            .recipe
            .as_deref()
            .context("--quantize needs --recipe")?;
        model = crate::cli::export::apply_recipe(&model, recipe, args.allow_partial)?;
    }
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
        store_selected: args.quantize,
        hf_files: if args.model.is_dir() {
            epiph::HF_FILES
                .iter()
                .map(|name| args.model.join(name))
                .filter(|p| p.is_file())
                .collect()
        } else {
            Vec::new()
        },
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
        EpiphCommand::Verify {
            file,
            public_key,
            require_signature,
        } => {
            let r = epiph::verify(&file)?;
            println!(
                "OK: {} tensors and {} attachments verified ({} payload bytes)",
                r.tensors, r.attachments, r.payload_bytes
            );
            let key = public_key.as_deref().map(load_public_key).transpose()?;
            match sign::check(&file, key.as_ref())? {
                SignatureStatus::Valid {
                    key_id,
                    by_epiphanes,
                    ..
                } => println!(
                    "Signature: valid, key {key_id}{}",
                    if by_epiphanes { " (Epiphanes)" } else { "" }
                ),
                SignatureStatus::UnknownKey { key_id } => {
                    println!(
                        "Signature: by key {key_id}, not Epiphanes; pass --public-key to check it"
                    );
                    if require_signature {
                        bail!("signature not verified");
                    }
                }
                SignatureStatus::Unsigned => {
                    println!("Signature: none");
                    if require_signature {
                        bail!("file is not signed");
                    }
                }
            }
        }
        EpiphCommand::Extract { file, out } => {
            let names = epiph::extract_hf_files(&file, &out)?;
            println!(
                "Wrote {} files to {out:?}: {}",
                names.len(),
                names.join(", ")
            );
        }
        EpiphCommand::Sign {
            file,
            secret_key,
            password_env,
            public_key,
        } => {
            let password = match password_env {
                Some(var) => Some(
                    std::env::var(&var)
                        .with_context(|| format!("environment variable {var} is not set"))?,
                ),
                None => None,
            };
            let text = std::fs::read_to_string(&secret_key)
                .with_context(|| format!("Failed to read the secret key file {secret_key:?}"))?;
            let key_box = minisign::SecretKeyBox::from_string(&text)
                .map_err(|e| anyhow::anyhow!("not a Minisign secret key file: {e}"))?;
            // Never prompt: an encrypted key needs --password-env.
            let sk = match password {
                Some(pw) => minisign::SecretKey::from_box(key_box, Some(pw)),
                None => minisign::SecretKey::from_unencrypted_box(key_box),
            }
            .map_err(|e| {
                anyhow::anyhow!(
                    "cannot open the secret key ({e}); for a password-protected key pass --password-env VAR"
                )
            })?;
            let pk = public_key.as_deref().map(load_public_key).transpose()?;
            sign::sign(&file, &sk, pk.as_ref())?;
            println!("Signed {file:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::evaluate::tests::synthetic_checkpoint;

    #[test]
    fn sign_and_verify_through_the_commands_with_key_files() {
        let dir = tempfile::tempdir().unwrap();
        let (model, _) = synthetic_checkpoint(dir.path());
        let file = dir.path().join("s.epiph");
        pack(&PackArgs {
            model,
            out: file.clone(),
            recipe: None,
            calibration: None,
            quality: vec![],
            parent: vec![],
            annotate: vec![],
            quantize: false,
            allow_partial: false,
        })
        .unwrap();
        let pair = minisign::KeyPair::generate_unencrypted_keypair().unwrap();
        let sk_path = dir.path().join("test.key");
        let pk_path = dir.path().join("test.pub");
        std::fs::write(&sk_path, pair.sk.to_box(None).unwrap().into_string()).unwrap();
        std::fs::write(&pk_path, pair.pk.to_box().unwrap().into_string()).unwrap();
        let verify = |key: Option<PathBuf>, require: bool| {
            execute(EpiphArgs {
                command: EpiphCommand::Verify {
                    file: file.clone(),
                    public_key: key,
                    require_signature: require,
                },
            })
        };
        assert!(
            verify(None, true).is_err(),
            "unsigned file fails --require-signature"
        );
        execute(EpiphArgs {
            command: EpiphCommand::Sign {
                file: file.clone(),
                secret_key: sk_path,
                password_env: None,
                public_key: Some(pk_path.clone()),
            },
        })
        .unwrap();
        assert!(verify(Some(pk_path), true).is_ok());
        assert!(verify(None, true).is_err(), "not an Epiphanes signature");
        assert!(verify(None, false).is_ok());
    }

    #[test]
    fn quantize_applies_the_recipe_and_needs_one() {
        let dir = tempfile::tempdir().unwrap();
        let (model, _) = synthetic_checkpoint(dir.path());
        let recipe = crate::cli::evaluate::tests::write_recipe(dir.path(), false);
        let base = PackArgs {
            model,
            out: dir.path().join("q.epiph"),
            recipe: Some(recipe.clone()),
            calibration: None,
            quality: vec![],
            parent: vec![],
            annotate: vec![],
            quantize: true,
            allow_partial: false,
        };
        pack(&base).unwrap();
        let (m, manifest) = epiph::read(&base.out).unwrap();
        assert!(m.tensors.iter().any(|t| t.original_dtype == "Q8_0"));
        assert!(manifest.recipe.is_some(), "the recipe is embedded too");
        let missing = PackArgs {
            recipe: None,
            out: dir.path().join("x.epiph"),
            ..base
        };
        assert!(format!("{:#}", pack(&missing).unwrap_err()).contains("--recipe"));
    }

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
            quantize: false,
            allow_partial: false,
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
            quantize: false,
            allow_partial: false,
        };
        assert!(pack(&bad).is_err());
    }
}
