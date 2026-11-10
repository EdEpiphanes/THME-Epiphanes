use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::allocator::ParetoAllocator;
use crate::parser::ModelImporter;
use crate::profiling::SensitivityProfiler;
use crate::report::ReportPrinter;

#[derive(Args, Debug)]
pub struct PlanArgs {
    /// Path to model directory
    #[arg(value_name = "MODEL_PATH")]
    pub path: PathBuf,

    /// Target VRAM budget in gigabytes (e.g. 8.0, 16.0)
    #[arg(short, long)]
    pub budget_gb: Option<f64>,

    /// Maximum tolerated degradation percentage (e.g. 1.5)
    #[arg(short, long, default_value_t = 2.0)]
    pub max_degradation: f64,

    /// Output file for allocation plan
    #[arg(short, long, default_value = "optimization_plan.json")]
    pub output: PathBuf,
}

pub fn execute(args: PlanArgs) -> Result<()> {
    println!(
        "Computing heterogeneous precision Pareto plan for: {:?}",
        args.path
    );
    let mut model_ir = ModelImporter::import(&args.path)?;

    // Profile sensitivity first
    SensitivityProfiler::profile_model(&mut model_ir);

    // Solve Pareto allocation
    let plan = ParetoAllocator::plan_allocation(&model_ir, args.budget_gb, args.max_degradation);

    ReportPrinter::print_allocation_plan(&plan);
    plan.save_to_file(&args.output)?;
    println!(
        "\nOptimization plan successfully written to: {:?}",
        args.output
    );

    Ok(())
}
