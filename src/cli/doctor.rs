// SPDX-License-Identifier: AGPL-3.0-only
//! `thme doctor`: say what works on this machine and what to try next.
//! Read-only: nothing is downloaded, loaded or written.

use anyhow::Result;
use clap::Args;

use crate::doctor;

#[derive(Args, Debug)]
pub struct DoctorArgs {
    /// Print JSON instead of text
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: DoctorArgs) -> Result<()> {
    let facts = doctor::collect_host_facts();
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&doctor::render_json(&facts))?
        );
    } else {
        println!("{}", doctor::render_text(&facts));
    }
    Ok(())
}
