use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

use crate::allocator::AllocationPlan;
use crate::report::ReportPrinter;

#[derive(Args, Debug)]
pub struct ReportArgs {
    /// Path to optimization_plan.json
    #[arg(value_name = "PLAN_PATH")]
    pub plan_path: PathBuf,

    /// Export to Markdown format file
    #[arg(short, long)]
    pub markdown: Option<PathBuf>,
}

pub fn execute(args: ReportArgs) -> Result<()> {
    let plan = AllocationPlan::load_from_file(&args.plan_path)?;
    ReportPrinter::print_allocation_plan(&plan);

    if let Some(md_path) = args.markdown {
        let mut md = String::new();
        md.push_str(&format!(
            "# THME Optimization Report: {}\n\n",
            plan.model_name
        ));
        md.push_str("## Executive Summary\n\n");
        md.push_str(&format!(
            "- **Original Weight Footprint:** {:.2} GB\n",
            plan.original_size_gb
        ));
        md.push_str(&format!(
            "- **Target Planned Footprint:** {:.2} GB\n",
            plan.planned_size_gb
        ));
        md.push_str(&format!(
            "- **Achieved Compression Ratio:** {:.2}x\n",
            plan.compression_ratio
        ));
        md.push_str(&format!(
            "- **Estimated Quality Retention:** {:.1}%\n\n",
            plan.estimated_quality_retention_pct
        ));
        md.push_str("## Layer Allocations\n\n");
        md.push_str("| Layer | Assigned Format | Original (MB) | Optimized (MB) |\n");
        md.push_str("|---|---|---|---|\n");
        for l in &plan.layer_allocations {
            md.push_str(&format!(
                "| Layer {:02} | {} | {:.1} | {:.1} |\n",
                l.layer_index,
                l.assigned_format.display_name(),
                l.original_mb,
                l.optimized_mb
            ));
        }

        std::fs::write(&md_path, md)?;
        println!("\nMarkdown report exported to: {:?}", md_path);
    }

    Ok(())
}
