// SPDX-License-Identifier: AGPL-3.0-only
pub mod bench;
pub mod calibrate;
pub mod chat;
pub mod chat_template;
pub mod cluster;
pub mod co_surgery;
pub mod convert;
pub mod discover;
pub mod doctor;
pub mod download;
pub mod epiph;
pub mod evaluate;
pub mod export;
pub mod figure;
pub mod forward;
pub mod inspect;
pub mod interop;
pub mod mcp;
pub mod pathfinder;
pub mod plan;
pub mod profile;
pub mod report;
pub mod roofline;
pub mod scan;
pub mod search;
pub mod speculate;
pub mod telemetry;
pub mod tokenize;
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
    /// Load models even when the safety scan finds high or critical issues (prints them)
    #[arg(long, global = true)]
    pub allow_unsafe_model: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Launch the THME Studio (the separate `thme-gui` program next to this one)
    #[command(alias = "ui")]
    Gui,

    /// Inspect model weights, tensor shapes, and architecture zero-copy
    Inspect(inspect::InspectArgs),

    /// Profile layer sensitivity and KV-cache memory scaling
    Profile(profile::ProfileArgs),

    /// Generate heterogeneous precision Pareto allocation plan
    Plan(plan::PlanArgs),

    /// Plan run settings (context, KV precision, predicted memory) from a model header
    Pathfinder(pathfinder::PathfinderArgs),

    /// Measure a surgery recipe's perplexity cost with the reference forward pass
    Evaluate(evaluate::EvaluateArgs),

    /// Run authentic reference forward pass on token sequence and output logits
    #[command(alias = "logits")]
    Forward(forward::ForwardArgs),

    /// Chat with a model using incremental KV-cached greedy decoding
    Chat(chat::ChatArgs),

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

    /// Classify multi-PC interconnects (Thunderbolt / Ethernet / Wi-Fi) and partition layers
    Cluster(cluster::ClusterArgs),

    /// Synthesize draft models and evaluate multi-token speculative decoding speedups
    Speculate(speculate::SpeculateArgs),

    /// Profile model layers against empirical hardware roofline ceilings and cache hierarchy
    Roofline(roofline::RooflineArgs),

    /// Export a source-backed model as GGUF or SafeTensors
    Export(export::ExportArgs),

    /// Pack or verify Epiphanes model files (.epiph): weights plus all their metadata
    Epiph(epiph::EpiphArgs),

    /// Security scan of a model file or folder for code execution, hidden payloads and malformed headers
    Scan(scan::ScanArgs),

    /// List models already on this machine (Hugging Face, Ollama, LM Studio, llama.cpp); read-only
    Discover(discover::DiscoverArgs),

    /// Say what works on this machine (OS, memory, disk, GPU adapter, models) and what to try next
    Doctor(doctor::DoctorArgs),

    /// Benchmark registry: list and check task files and result records
    Bench(bench::BenchArgs),

    /// Draw a model as an isometric figure: one tray per layer, one block per tensor
    Figure(figure::FigureArgs),

    /// Generate PyTorch / CUDA fine-tuning scripts or Hugging Face configurations
    Interop(interop::InteropArgs),

    /// Run Model Context Protocol (MCP) server for VS Code, Copilot & AI agents
    #[command(alias = "daemon")]
    Mcp(mcp::McpArgs),

    /// Run autonomous Closed-Loop Teacher-Student Co-Surgery optimization
    #[command(name = "co-surgery", alias = "cosurgery")]
    CoSurgery(co_surgery::CoSurgeryArgs),

    /// Query real-time CPU, RAM, and discrete GPU hardware telemetry
    Telemetry(telemetry::TelemetryArgs),

    /// Encode text with a model's tokenizer.json and decode it back
    Tokenize(tokenize::TokenizeArgs),

    /// Render a model's own chat template for a JSON list of messages
    ChatTemplate(chat_template::ChatTemplateArgs),

    /// Download a model from the Hugging Face hub, then run the safety scan on it
    Download(download::DownloadArgs),
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    crate::parser::allow_unsafe_models(cli.allow_unsafe_model);
    match cli.command {
        Commands::Gui => launch_studio(),
        Commands::Inspect(args) => inspect::execute(args),
        Commands::Profile(args) => profile::execute(args),
        Commands::Plan(args) => plan::execute(args),
        Commands::Pathfinder(args) => pathfinder::execute(args),
        Commands::Evaluate(args) => evaluate::execute(args),
        Commands::Forward(args) => forward::execute(args),
        Commands::Chat(args) => chat::execute(args),
        Commands::Search(args) => search::execute(args),
        Commands::Calibrate(args) => calibrate::execute(args),
        Commands::Validate(args) => validate::execute(args),
        Commands::Tweak(args) => tweak::execute(args),
        Commands::Report(args) => report::execute(args),
        Commands::Convert(args) => convert::execute(args),
        Commands::Cluster(args) => cluster::execute(args),
        Commands::Speculate(args) => speculate::execute(args),
        Commands::Roofline(args) => roofline::execute(args),
        Commands::Export(args) => export::execute(args),
        Commands::Epiph(args) => epiph::execute(args),
        Commands::Scan(args) => scan::execute(args),
        Commands::Discover(args) => discover::execute(args),
        Commands::Doctor(args) => doctor::execute(args),
        Commands::Bench(args) => bench::execute(args),
        Commands::Figure(args) => figure::execute(args),
        Commands::Interop(args) => interop::execute(args),
        Commands::Mcp(args) => mcp::execute(args),
        Commands::CoSurgery(args) => co_surgery::execute(args),
        Commands::Telemetry(args) => telemetry::execute(args),
        Commands::Tokenize(args) => tokenize::execute(args),
        Commands::ChatTemplate(args) => chat_template::execute(args),
        Commands::Download(args) => download::execute(args),
    }
}

/// Starts the Studio, which is its own program (`thme-gui`), so this headless CLI carries none of
/// the GUI stack.
fn launch_studio() -> Result<()> {
    let exe = std::env::current_exe()?;
    let studio = exe.with_file_name(format!("thme-gui{}", std::env::consts::EXE_SUFFIX));
    if !studio.is_file() {
        anyhow::bail!(
            "the Studio is a separate program and {} was not found next to {}; install the full build or run thme-gui directly",
            studio.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            exe.display()
        );
    }
    let status = std::process::Command::new(&studio).status()?;
    if !status.success() {
        anyhow::bail!("thme-gui exited with {status}");
    }
    Ok(())
}
