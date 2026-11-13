// SPDX-License-Identifier: AGPL-3.0-only
//! `thme sandbox`: run a program in the developer sandbox (docs/sandbox.md).

use anyhow::{bail, Result};
use clap::{Args, Subcommand};
use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use crate::sandbox::{check, run, Ended, Policy};

#[derive(Args, Debug)]
pub struct SandboxArgs {
    #[command(subcommand)]
    pub command: SandboxCommand,
}

#[derive(Subcommand, Debug)]
pub enum SandboxCommand {
    /// Show which isolation layers this host can enforce
    Check,
    /// Run PROGRAM isolated: no network, writes only in --work, time and memory caps
    Run {
        /// Work folder: the only writable place, and the working directory
        #[arg(long)]
        work: PathBuf,
        /// Extra readable path (file or folder); repeatable
        #[arg(long = "read-only")]
        read_only: Vec<PathBuf>,
        /// Environment variable NAME=VALUE passed in; nothing else is inherited
        #[arg(long = "env")]
        env: Vec<String>,
        /// Wall-clock limit in seconds
        #[arg(long, default_value_t = 60)]
        time: u64,
        /// CPU-time limit in seconds
        #[arg(long, default_value_t = 30)]
        cpu: u64,
        /// Memory limit in MiB
        #[arg(long, default_value_t = 1024)]
        memory: u64,
        /// Process limit
        #[arg(long, default_value_t = 16)]
        processes: u32,
        #[arg(value_name = "PROGRAM")]
        program: PathBuf,
        #[arg(
            value_name = "ARGS",
            trailing_var_arg = true,
            allow_hyphen_values = true
        )]
        args: Vec<OsString>,
    },
    /// Escape-test probe (used by the test suite)
    #[command(hide = true)]
    Probe {
        action: String,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

pub fn execute(args: SandboxArgs) -> Result<()> {
    match args.command {
        SandboxCommand::Check => {
            let support = check();
            println!("Sandbox on {}:", support.os);
            if support.layers.is_empty() {
                println!(
                    "  no backend for this OS yet: `thme sandbox run` refuses to start anything"
                );
            }
            for l in &support.layers {
                let mark = if l.supported { "ok " } else { "NO " };
                println!("  {mark} {:<16} {} ({})", l.name, l.mechanism, l.note);
            }
            if !support.all_supported() {
                bail!("this host cannot enforce every sandbox layer");
            }
            Ok(())
        }
        SandboxCommand::Run {
            work,
            read_only,
            env,
            time,
            cpu,
            memory,
            processes,
            program,
            args,
        } => {
            let mut policy = Policy::new(work);
            policy.read_only = read_only;
            policy.wall_time = Duration::from_secs(time);
            policy.cpu_time = Duration::from_secs(cpu);
            policy.memory_bytes = memory << 20;
            policy.max_processes = processes;
            for kv in env {
                let Some((k, v)) = kv.split_once('=') else {
                    bail!("--env needs NAME=VALUE, got '{kv}'");
                };
                policy.env.push((k.to_string(), v.to_string()));
            }
            let outcome = run(&policy, &program, &args, &[])?;
            std::io::stdout().write_all(&outcome.stdout)?;
            std::io::stderr().write_all(&outcome.stderr)?;
            eprintln!(
                "[sandbox] {:?} after {:.2} s",
                outcome.ended,
                outcome.wall.as_secs_f64()
            );
            match outcome.ended {
                Ended::Exited(0) => Ok(()),
                other => bail!("sandboxed program ended with {other:?}"),
            }
        }
        SandboxCommand::Probe { action, args } => {
            std::process::exit(crate::sandbox::probe::run(&action, &args));
        }
    }
}
