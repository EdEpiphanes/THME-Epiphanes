pub mod calibrate;
pub mod co_surgery;
pub mod convert;
pub mod epiph;
pub mod evaluate;
pub mod export;
pub mod inspect;
pub mod interop;
pub mod mcp;
pub mod plan;
pub mod profile;
pub mod report;
pub mod roofline;
pub mod search;
pub mod speculate;
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

    /// Measure a surgery recipe's perplexity cost with the reference forward pass
    Evaluate(evaluate::EvaluateArgs),

    /// Search for a mixed-precision recipe from measured costs under a size budget
    Search(search::SearchArgs),

    /// Record per-input-channel activation statistics on calibration tokens
    Calibrate(calibrate::CalibrateArgs),

    /// Run quality gate check between baseline and candidate
    #[command(alias = "check")]
    Validate(validate::ValidateArgs),

    /// Apply Teacher LLM surgery recipe or restructuring plan
    #[command(alias = "surgery")]
    Tweak(tweak::TweakArgs),

    /// View or export optimization plan reports
    Report(report::ReportArgs),

    /// Cross-convert model topology across 8 specialized AI paradigms (LLM, LCM, LAM, MoE, VLM, SLM, MLM, SAM)
    Convert(convert::ConvertArgs),

    /// Synthesize draft models and evaluate multi-token speculative decoding speedups
    Speculate(speculate::SpeculateArgs),

    /// Profile model layers against empirical hardware roofline ceilings and cache hierarchy
    Roofline(roofline::RooflineArgs),

    /// Export a source-backed model as GGUF or SafeTensors
    Export(export::ExportArgs),

    /// Pack or verify Epiphanes model files (.epiph): weights plus all their metadata
    Epiph(epiph::EpiphArgs),

    /// Generate PyTorch / CUDA fine-tuning scripts or Hugging Face configurations
    Interop(interop::InteropArgs),

    /// Run Model Context Protocol (MCP) server for VS Code, Copilot & AI agents
    #[command(alias = "daemon")]
    Mcp(mcp::McpArgs),

    /// Run autonomous Closed-Loop Teacher-Student Co-Surgery optimization
    #[command(name = "co-surgery", alias = "cosurgery")]
    CoSurgery(co_surgery::CoSurgeryArgs),
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Gui => crate::gui::launch_gui(),
        Commands::Inspect(args) => inspect::execute(args),
        Commands::Profile(args) => profile::execute(args),
        Commands::Plan(args) => plan::execute(args),
        Commands::Evaluate(args) => evaluate::execute(args),
        Commands::Search(args) => search::execute(args),
        Commands::Calibrate(args) => calibrate::execute(args),
        Commands::Validate(args) => validate::execute(args),
        Commands::Tweak(args) => tweak::execute(args),
        Commands::Report(args) => report::execute(args),
        Commands::Convert(args) => convert::execute(args),
        Commands::Speculate(args) => speculate::execute(args),
        Commands::Roofline(args) => roofline::execute(args),
        Commands::Export(args) => export::execute(args),
        Commands::Epiph(args) => epiph::execute(args),
        Commands::Interop(args) => interop::execute(args),
        Commands::Mcp(args) => mcp::execute(args),
        Commands::CoSurgery(args) => co_surgery::execute(args),
    }
}
