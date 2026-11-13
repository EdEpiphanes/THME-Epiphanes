// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::{Context, Result};
use clap::Args;
use std::fs;
use std::path::PathBuf;

use crate::interop::{
    HuggingFaceConfigGenerator, PyTorchCUDATrainingStrategy, PyTorchInteropConfig,
    PyTorchScriptGenerator,
};
use crate::parser::ModelImporter;

#[derive(Args, Debug)]
pub struct InteropArgs {
    /// Fine-tuning strategy: lora, qlora, full, unsloth, deepspeed-zero2, deepspeed-zero3
    #[arg(short, long, default_value = "lora")]
    pub strategy: String,

    /// Model HuggingFace ID or local path
    #[arg(short, long, default_value = "meta-llama/Meta-Llama-3-8B")]
    pub model: String,

    /// Dataset path (JSONL, CSV, or Text)
    #[arg(short, long, default_value = "dataset.jsonl")]
    pub dataset: String,

    /// Output destination path for the generated PyTorch CUDA training script
    #[arg(long, default_value = "train_cuda.py")]
    pub output_script: PathBuf,

    /// Optional path to export Hugging Face compatible config.json
    #[arg(long)]
    pub output_config: Option<PathBuf>,

    /// Optional local model file (.safetensors or .gguf) to inspect for config generation
    #[arg(long)]
    pub model_file: Option<PathBuf>,

    /// Target CUDA hardware acceleration
    #[arg(long, default_value_t = true)]
    pub cuda: bool,

    /// Training batch size per GPU
    #[arg(long, default_value_t = 4)]
    pub batch_size: usize,

    /// Learning rate
    #[arg(long, default_value_t = 2e-4)]
    pub lr: f64,

    /// Training epochs
    #[arg(long, default_value_t = 3)]
    pub epochs: usize,

    /// LoRA rank dimension r
    #[arg(long, default_value_t = 16)]
    pub rank: usize,

    /// LoRA scaling alpha
    #[arg(long, default_value_t = 32)]
    pub alpha: usize,
}

pub fn execute(args: InteropArgs) -> Result<()> {
    let strategy = match args.strategy.to_lowercase().as_str() {
        "qlora" => PyTorchCUDATrainingStrategy::QLoRA,
        "full" => PyTorchCUDATrainingStrategy::FullFineTune,
        "unsloth" => PyTorchCUDATrainingStrategy::UnslothFast,
        "deepspeed-zero2" | "zero2" => PyTorchCUDATrainingStrategy::DeepSpeedZeRO2,
        "deepspeed-zero3" | "zero3" => PyTorchCUDATrainingStrategy::DeepSpeedZeRO3,
        _ => PyTorchCUDATrainingStrategy::LoRA,
    };

    let config = PyTorchInteropConfig {
        model_id_or_path: args.model,
        dataset_path: args.dataset,
        strategy,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        num_epochs: args.epochs,
        lora_rank: args.rank,
        lora_alpha: args.alpha,
        ..Default::default()
    };

    println!("=== THME PyTorch & CUDA Interoperability Bridge ===");
    println!("Strategy:       {}", config.strategy.name());
    println!("Target Model:   {}", config.model_id_or_path);
    println!("Dataset:        {}", config.dataset_path);
    println!(
        "Hardware Target: {}",
        if args.cuda {
            "NVIDIA CUDA / Tensor Cores"
        } else {
            "CPU Fallback"
        }
    );

    // Generate PyTorch CUDA script
    let script = PyTorchScriptGenerator::generate_training_script(&config, None);
    fs::write(&args.output_script, script).with_context(|| {
        format!(
            "Failed to write PyTorch training script to {:?}",
            args.output_script
        )
    })?;
    println!(
        "Generated Script: {:?} (Ready for 'python {:?}')",
        args.output_script, args.output_script
    );

    // If config export requested
    if let Some(cfg_path) = args.output_config {
        let model_ir = if let Some(m_path) = args.model_file {
            ModelImporter::import(&m_path).with_context(|| {
                format!(
                    "Failed to load model file for config export from {:?}",
                    m_path
                )
            })?
        } else {
            crate::ir::ModelIR::default()
        };

        let hf_config = HuggingFaceConfigGenerator::generate_config_json(&model_ir);
        let config_str = serde_json::to_string_pretty(&hf_config)?;
        fs::write(&cfg_path, config_str)
            .with_context(|| format!("Failed to write HuggingFace config to {:?}", cfg_path))?;
        println!("Exported Config:  {:?} (Hugging Face format)", cfg_path);
    }

    println!("Status: Ready for external PyTorch / CUDA fine-tuning pipelines.");
    Ok(())
}
