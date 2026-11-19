// SPDX-License-Identifier: AGPL-3.0-only
//! `thme discover`: list the models already on this machine, from the tools users came from.
//! Read-only: nothing is downloaded, loaded or written.

use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::discover::{self, EnvSnapshot};

#[derive(Args, Debug)]
pub struct DiscoverArgs {
    /// Extra folder to search for models (repeatable)
    #[arg(long = "path", value_name = "DIR")]
    pub paths: Vec<PathBuf>,

    /// Print JSON instead of a table
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: DiscoverArgs) -> Result<()> {
    let found = discover::discover(&args.paths, &EnvSnapshot::from_process());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&discover::render_json(&found))?
        );
    } else {
        println!("{}", discover::render_table(&found.models));
        for warning in &found.warnings {
            eprintln!("warning: {warning}");
        }
    }
    Ok(())
}
