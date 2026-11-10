use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;
use std::str::FromStr;

use crate::ir::ParadigmConverter;
use crate::parser::ModelImporter;
use crate::profiling::cache_sim::CacheSimulator4Tier;
use crate::profiling::hardware::{HardwareDevice, HardwareSpec, HardwareTarget};
use crate::profiling::roofline::RooflineModel;
use crate::profiling::thermal::{DynamicThermalModel, ThermalConfig};

#[derive(Args, Debug, Clone)]
pub struct RooflineArgs {
    /// Path to target model file (.safetensors or .gguf) (omit to use synthetic baseline)
    #[arg(short, long, value_name = "MODEL_PATH")]
    pub model: Option<PathBuf>,

    /// Target hardware architecture (h100, rtx4090, m3_max, mi300x, emerald_rapids)
    #[arg(short = 'w', long, default_value = "h100")]
    pub hardware: String,

    /// Inference batch size B (default: 1)
    #[arg(short = 'b', long, default_value_t = 1)]
    pub batch_size: usize,

    /// Sequence context length S (default: 2048)
    #[arg(short = 's', long, default_value_t = 2048)]
    pub seq_len: usize,

    /// Model FlashAttention-2 tiling arithmetic intensity instead of standard attention
    #[arg(long, default_value_t = true)]
    pub flash_attn: bool,

    /// Simulate dynamic TDP power consumption and thermal throttling
    #[arg(long)]
    pub thermal: bool,

    /// Operating ambient or die temperature in degrees Celsius (e.g. 25.0, 75.0, 88.0)
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Output report in machine-readable JSON format
    #[arg(long)]
    pub json: bool,
}

pub fn execute(args: RooflineArgs) -> Result<()> {
    let model = if let Some(ref path) = args.model {
        println!("Loading model from: {:?}", path);
        ModelImporter::import(path)
            .with_context(|| format!("Failed to import model from {:?}", path))?
    } else {
        println!("Using default 12-layer synthetic LLM baseline for roofline profiling.");
        ParadigmConverter::create_synthetic_baseline()
    };

    let hw = HardwareTarget::from_str(&args.hardware)
        .ok()
        .map(HardwareSpec::from_target)
        .or_else(|| HardwareDevice::find_device(&args.hardware))
        .unwrap_or_else(HardwareDevice::predefined_h100);

    let thermal_factor = if args.thermal || args.temperature.is_some() {
        let temp = args.temperature.unwrap_or_else(|| {
            let tm = DynamicThermalModel::new(hw.clone(), ThermalConfig::default());
            tm.steady_state_temp(1.0, 1.0) as f32
        });
        DynamicThermalModel::calculate_throttling_factor(temp, hw.tdp_watts)
    } else {
        1.0
    };

    let report = RooflineModel::profile_model(
        &model,
        &hw,
        args.batch_size,
        args.seq_len,
        args.flash_attn,
        thermal_factor,
    )?;

    if args.json {
        let json_str = serde_json::to_string_pretty(&report)?;
        println!("{}", json_str);
    } else {
        let effective_peak = report.peak_compute_tflops;
        let knee = report.knee_intensity_flop_per_byte;
        println!("{}", report.format_ascii_report(&hw, effective_peak, knee));

        let mut mem_bound = 0;
        let mut comp_bound = 0;
        for pt in &report.points {
            match pt.classification {
                crate::profiling::BoundClassification::MemoryBound { .. } => mem_bound += 1,
                crate::profiling::BoundClassification::ComputeBound { .. } => comp_bound += 1,
            }
        }
        let total = mem_bound + comp_bound;
        let mem_pct = if total > 0 {
            (mem_bound as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        let comp_pct = if total > 0 {
            (comp_bound as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("EXECUTION PROFILE SUMMARY:");
        println!(
            " - Target Device       : {} (Architecture: {})",
            hw.name, hw.architecture
        );
        println!(
            " - Peak Compute Ceiling: {:.1} TFLOPS ({})",
            effective_peak,
            report.precision.display_name()
        );
        println!(
            " - Memory Bandwidth    : {:.1} GB/s ({:.2} TB/s)",
            hw.dram_bandwidth_gbps,
            hw.memory_bandwidth_tbps()
        );
        println!(" - Machine Balance Knee: {:.2} FLOPs/Byte", knee);
        println!(
            " - Thermal Throttling  : {:.1}% (Throttle Factor: {:.2})",
            (1.0 - thermal_factor) * 100.0,
            thermal_factor
        );
        println!(
            " - Bound Breakdown     : {} Memory-Bound ({:.1}%), {} Compute-Bound ({:.1}%)",
            mem_bound, mem_pct, comp_bound, comp_pct
        );

        let cache_sim = CacheSimulator4Tier::new(hw.clone());
        let working_set_bytes = (model.hidden_size * model.hidden_size * 2 * 4).max(16_000_000);
        let cache_report = cache_sim.simulate_layer_working_set(working_set_bytes);
        println!(
            " - Cache Simulation    : Effective Latency: {:.2} ns | Effective BW: {:.1} GB/s | DRAM Spill: {:.2} MB",
            cache_report.effective_latency_ns,
            cache_report.effective_bandwidth_gbps,
            cache_report.dram_spill_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "================================================================================"
        );
    }

    Ok(())
}
