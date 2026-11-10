pub mod inspect;
pub mod plan;
pub mod profile;
pub mod report;
pub mod tweak;
pub mod validate;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "thme",
    author = "Epiphanes <https://github.com/EdEpiphanes>",
    version,
    about = "Topological Heterogeneous Manifold Engine (THME) by Epiphanes - Open-source LLM optimization & profiling framework",
    long_about = "THME provides zero-copy model inspection, topological layer sensitivity analysis, KV-cache projection, Pareto precision planning, and quality gate verification."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Launch the interactive THME Native Desktop Studio GUI
    #[command(alias = "ui")]
    Gui,

    /// Inspect model weights, tensor shapes, and architecture zero-copy
    Inspect(inspect::InspectArgs),

    /// Profile layer sensitivity and KV-cache memory scaling
    Profile(profile::ProfileArgs),

    /// Generate heterogeneous precision Pareto allocation plan
    Plan(plan::PlanArgs),

    /// Run quality gate check between baseline and candidate
    #[command(alias = "check")]
    Validate(validate::ValidateArgs),

    /// Apply Teacher LLM surgery recipe or restructuring plan
    #[command(alias = "surgery")]
    Tweak(tweak::TweakArgs),

    /// View or export optimization plan reports
    Report(report::ReportArgs),
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Gui => crate::gui::launch_gui(),
        Commands::Inspect(args) => inspect::execute(args),
        Commands::Profile(args) => profile::execute(args),
        Commands::Plan(args) => plan::execute(args),
        Commands::Validate(args) => validate::execute(args),
        Commands::Tweak(args) => tweak::execute(args),
        Commands::Report(args) => report::execute(args),
    }
}
