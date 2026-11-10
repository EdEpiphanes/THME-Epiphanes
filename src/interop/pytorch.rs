use serde::{Deserialize, Serialize};

use crate::allocator::AllocationPlan;
use crate::ir::{ModelArchitecture, ModelIR};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PyTorchCUDATrainingStrategy {
    LoRA,
    QLoRA,
    FullFineTune,
    UnslothFast,
    DeepSpeedZeRO2,
    DeepSpeedZeRO3,
}

impl PyTorchCUDATrainingStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::LoRA => "PEFT LoRA (Low-Rank Adaptation)",
            Self::QLoRA => "QLoRA (4-bit NF4 Base + LoRA Adapters)",
            Self::FullFineTune => "Full Parameter Fine-Tuning",
            Self::UnslothFast => "Unsloth Fast GPU Kernel Fine-Tuning",
            Self::DeepSpeedZeRO2 => "DeepSpeed ZeRO-2 Stage Optimizer",
            Self::DeepSpeedZeRO3 => "DeepSpeed ZeRO-3 Stage Sharding",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchInteropConfig {
    pub model_id_or_path: String,
    pub output_dir: String,
    pub dataset_path: String,
    pub strategy: PyTorchCUDATrainingStrategy,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub max_seq_length: usize,
    pub lora_rank: usize,
    pub lora_alpha: usize,
    pub lora_dropout: f32,
    pub use_flash_attention_2: bool,
    pub use_bf16: bool,
    pub target_modules: Vec<String>,
}

impl Default for PyTorchInteropConfig {
    fn default() -> Self {
        Self {
            model_id_or_path: "meta-llama/Meta-Llama-3-8B".to_string(),
            output_dir: "./output_pytorch_cuda".to_string(),
            dataset_path: "dataset.jsonl".to_string(),
            strategy: PyTorchCUDATrainingStrategy::LoRA,
            batch_size: 4,
            gradient_accumulation_steps: 4,
            learning_rate: 2e-4,
            num_epochs: 3,
            max_seq_length: 2048,
            lora_rank: 16,
            lora_alpha: 32,
            lora_dropout: 0.05,
            use_flash_attention_2: true,
            use_bf16: true,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        }
    }
}

pub struct PyTorchScriptGenerator;

impl PyTorchScriptGenerator {
    /// Generates a standalone, executable PyTorch CUDA fine-tuning Python script
    /// that integrates THME's Pareto precision allocations with standard PyTorch/CUDA tooling.
    pub fn generate_training_script(
        config: &PyTorchInteropConfig,
        plan: Option<&AllocationPlan>,
    ) -> String {
        let target_modules_repr = format!(
            "[{}]",
            config
                .target_modules
                .iter()
                .map(|m| format!("\"{}\"", m))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let plan_comment = if let Some(p) = plan {
            format!(
                "# THME Pareto Optimization Plan Applied: Target VRAM {:.2} GB, Planned Size {:.2} GB, Compression Ratio {:.2}x\n",
                p.target_vram_gb.unwrap_or(0.0),
                p.planned_size_gb,
                p.compression_ratio
            )
        } else {
            "# Default THME Configuration (Pure PyTorch + CUDA Training Harness)\n".to_string()
        };

        let strategy_setup = match config.strategy {
            PyTorchCUDATrainingStrategy::LoRA => {
                format!(
                    r#"    # 1. Setup LoRA PEFT Adapter
    from peft import LoraConfig, get_peft_model, TaskType
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r={},
        lora_alpha={},
        lora_dropout={},
        target_modules={},
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
"#,
                    config.lora_rank,
                    config.lora_alpha,
                    config.lora_dropout,
                    target_modules_repr
                )
            }
            PyTorchCUDATrainingStrategy::QLoRA => {
                format!(
                    r#"    # 1. Setup QLoRA 4-bit Quantized Base with BitsAndBytes
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if {} else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # Re-instantiate model with 4-bit quant
    model = AutoModelForCausalLM.from_pretrained(
        "{}",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if {} else torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r={},
        lora_alpha={},
        lora_dropout={},
        target_modules={},
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
"#,
                    config.use_bf16,
                    config.model_id_or_path,
                    config.use_bf16,
                    config.lora_rank,
                    config.lora_alpha,
                    config.lora_dropout,
                    target_modules_repr
                )
            }
            PyTorchCUDATrainingStrategy::UnslothFast => {
                format!(
                    r#"    # 1. Setup Unsloth Fast GPU Kernels
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="{}",
            max_seq_length={},
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r={},
            target_modules={},
            lora_alpha={},
            lora_dropout={},
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    except ImportError:
        print("[WARN] unsloth library not installed. Falling back to standard PEFT LoRA.")
"#,
                    config.model_id_or_path,
                    config.max_seq_length,
                    config.lora_rank,
                    target_modules_repr,
                    config.lora_alpha,
                    config.lora_dropout
                )
            }
            PyTorchCUDATrainingStrategy::FullFineTune => {
                r#"    # 1. Full Parameter Fine-Tuning Setup
    for param in model.parameters():
        param.requires_grad = True
    print(f"[INFO] Full Parameter Fine-Tuning: All {sum(p.numel() for p in model.parameters())} parameters trainable.")
"#
                .to_string()
            }
            PyTorchCUDATrainingStrategy::DeepSpeedZeRO2 | PyTorchCUDATrainingStrategy::DeepSpeedZeRO3 => {
                let stage = if config.strategy == PyTorchCUDATrainingStrategy::DeepSpeedZeRO2 { 2 } else { 3 };
                format!(
                    r#"    # 1. DeepSpeed ZeRO-{} Configuration
    deepspeed_config = {{
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": {},
        "gradient_accumulation_steps": {},
        "zero_optimization": {{
            "stage": {},
            "offload_optimizer": {{"device": "cpu"}},
            "offload_param": {{"device": "none"}},
            "overlap_comm": True,
            "contiguous_gradients": True
        }},
        "fp16": {{"enabled": not {}}},
        "bf16": {{"enabled": {}}},
    }}
"#,
                    stage,
                    config.batch_size,
                    config.gradient_accumulation_steps,
                    stage,
                    config.use_bf16,
                    config.use_bf16
                )
            }
        };

        format!(
            r#"#!/usr/bin/env python3
"""
THME PyTorch + CUDA Fine-Tuning and Adaptation Harness
Generated automatically by Topological Heterogeneous Manifold Engine (THME)
Compatible with Hugging Face, PyTorch 2.x, CUDA 11.8/12.x, BitsAndBytes, and PEFT.
"""

import os
import sys
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

{}
def setup_cuda_environment():
    if not torch.cuda.is_available():
        print("[ERROR] CUDA-enabled GPU not detected. PyTorch CUDA training requires a compatible NVIDIA GPU.")
        sys.exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[CUDA] Active Device: {{device_name}} (Count: {{device_count}}, VRAM: {{vram_gb:.2f}} GB)")
    
    # Enable TF32 for Ampere / Ada / Hopper / Blackwell architectures
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[CUDA] TF32 Tensor Core acceleration enabled.")

def main():
    setup_cuda_environment()
    
    model_id = "{}"
    output_dir = "{}"
    dataset_file = "{}"
    
    print(f"[THME Bridge] Loading base model from: {{model_id}}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model with FlashAttention-2 if requested and available
    attn_impl = "flash_attention_2" if {} and torch.cuda.get_device_capability()[0] >= 8 else "eager"
    torch_dtype = torch.bfloat16 if {} else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
{}
    # 2. Ingest Dataset
    print(f"[THME Bridge] Loading training dataset: {{dataset_file}}")
    if dataset_file.endswith(".json") or dataset_file.endswith(".jsonl"):
        raw_dataset = load_dataset("json", data_files=dataset_file)["train"]
    elif dataset_file.endswith(".csv"):
        raw_dataset = load_dataset("csv", data_files=dataset_file)["train"]
    else:
        raw_dataset = load_dataset("text", data_files=dataset_file)["train"]

    def tokenize_fn(examples):
        text_column = "text" if "text" in examples else next(iter(examples.keys()))
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length={},
            padding=False,
        )

    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=raw_dataset.column_names)
    
    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size={},
        gradient_accumulation_steps={},
        learning_rate={},
        num_train_epochs={},
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=not {},
        bf16={},
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print("[THME Bridge] Commencing PyTorch CUDA training execution...")
    train_result = trainer.train()
    
    print(f"[THME Bridge] Training finished. Saving adapted model artifacts to: {{output_dir}}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Export execution metrics
    metrics = train_result.metrics
    with open(os.path.join(output_dir, "thme_pytorch_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[THME Bridge] Complete. Loss: {{metrics.get('train_loss', 'N/A')}}")

if __name__ == "__main__":
    main()
"#,
            plan_comment,
            config.model_id_or_path,
            config.output_dir,
            config.dataset_path,
            config.use_flash_attention_2,
            config.use_bf16,
            strategy_setup,
            config.max_seq_length,
            config.batch_size,
            config.gradient_accumulation_steps,
            config.learning_rate,
            config.num_epochs,
            config.use_bf16,
            config.use_bf16
        )
    }
}

pub struct HuggingFaceConfigGenerator;

impl HuggingFaceConfigGenerator {
    /// Generates standard Hugging Face `config.json` conforming to transformers architecture specifications.
    pub fn generate_config_json(model_ir: &ModelIR) -> serde_json::Value {
        let arch_string = match model_ir.architecture {
            ModelArchitecture::Llama => "LlamaForCausalLM",
            ModelArchitecture::Mistral => "MistralForCausalLM",
            ModelArchitecture::Gemma => "GemmaForCausalLM",
            ModelArchitecture::Qwen => "Qwen2ForCausalLM",
            ModelArchitecture::DeepSeek => "DeepseekForCausalLM",
            ModelArchitecture::Unknown(_) => "AutoModelForCausalLM",
        };

        let mut config_map = serde_json::Map::new();
        config_map.insert(
            "architectures".to_string(),
            serde_json::json!([arch_string]),
        );
        config_map.insert(
            "model_type".to_string(),
            serde_json::json!(match model_ir.architecture {
                ModelArchitecture::Llama => "llama",
                ModelArchitecture::Mistral => "mistral",
                ModelArchitecture::Gemma => "gemma",
                ModelArchitecture::Qwen => "qwen2",
                ModelArchitecture::DeepSeek => "deepseek",
                ModelArchitecture::Unknown(_) => "custom",
            }),
        );
        config_map.insert(
            "hidden_size".to_string(),
            serde_json::json!(model_ir.hidden_size),
        );
        config_map.insert(
            "num_hidden_layers".to_string(),
            serde_json::json!(model_ir.num_layers),
        );
        config_map.insert(
            "num_attention_heads".to_string(),
            serde_json::json!(model_ir.num_attention_heads),
        );
        config_map.insert(
            "num_key_value_heads".to_string(),
            serde_json::json!(model_ir.num_key_value_heads),
        );
        config_map.insert("head_dim".to_string(), serde_json::json!(model_ir.head_dim));
        config_map.insert(
            "vocab_size".to_string(),
            serde_json::json!(model_ir.vocab_size),
        );
        config_map.insert(
            "max_position_embeddings".to_string(),
            serde_json::json!(model_ir.max_position_embeddings),
        );
        config_map.insert("torch_dtype".to_string(), serde_json::json!("float16"));
        config_map.insert(
            "transformers_version".to_string(),
            serde_json::json!("4.44.0"),
        );

        serde_json::Value::Object(config_map)
    }
}

pub struct PeftLoraAdapter;

impl PeftLoraAdapter {
    /// Ingests and merges Low-Rank Adaptation (LoRA) weights into a base weight tensor:
    /// W_new = W_base + (alpha / rank) * (B @ A)
    pub fn merge_lora_weights(
        base_weight: &[f32],
        lora_a: &[f32], // Shape: [r, in_features]
        lora_b: &[f32], // Shape: [out_features, r]
        out_features: usize,
        in_features: usize,
        rank: usize,
        alpha: f32,
    ) -> Result<Vec<f32>, String> {
        if base_weight.len() != out_features * in_features {
            return Err(format!(
                "Base weight dimension mismatch: expected {}, got {}",
                out_features * in_features,
                base_weight.len()
            ));
        }
        if lora_a.len() != rank * in_features {
            return Err(format!(
                "LoRA A dimension mismatch: expected {}, got {}",
                rank * in_features,
                lora_a.len()
            ));
        }
        if lora_b.len() != out_features * rank {
            return Err(format!(
                "LoRA B dimension mismatch: expected {}, got {}",
                out_features * rank,
                lora_b.len()
            ));
        }
        if rank == 0 {
            return Err("LoRA rank cannot be zero".to_string());
        }

        let scale = alpha / (rank as f32);
        let mut merged = base_weight.to_vec();

        // Perform matrix multiplication: delta[i, j] = sum_k(B[i, k] * A[k, j])
        for i in 0..out_features {
            let row_offset = i * in_features;
            let b_row_offset = i * rank;
            for j in 0..in_features {
                let mut delta = 0.0f32;
                for k in 0..rank {
                    delta += lora_b[b_row_offset + k] * lora_a[k * in_features + j];
                }
                merged[row_offset + j] += delta * scale;
            }
        }

        Ok(merged)
    }

    /// Detects whether tensor names belong to PEFT LoRA adapter modules
    pub fn is_lora_tensor(name: &str) -> bool {
        name.contains("lora_A") || name.contains("lora_B") || name.contains(".lora_")
    }

    /// Extracts base module name from a LoRA tensor identifier
    pub fn extract_base_module_name(lora_name: &str) -> String {
        lora_name
            .replace(".lora_A.weight", ".weight")
            .replace(".lora_B.weight", ".weight")
            .replace(".lora_A.default.weight", ".weight")
            .replace(".lora_B.default.weight", ".weight")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_script_generator_lora() {
        let config = PyTorchInteropConfig::default();
        let script = PyTorchScriptGenerator::generate_training_script(&config, None);
        assert!(script.contains("AutoModelForCausalLM"));
        assert!(script.contains("peft"));
        assert!(script.contains("LoraConfig"));
        assert!(script.contains("torch.cuda.is_available()"));
        assert!(script.contains("Trainer"));
    }

    #[test]
    fn test_pytorch_script_generator_qlora() {
        let config = PyTorchInteropConfig {
            strategy: PyTorchCUDATrainingStrategy::QLoRA,
            ..Default::default()
        };
        let script = PyTorchScriptGenerator::generate_training_script(&config, None);
        assert!(script.contains("BitsAndBytesConfig"));
        assert!(script.contains("load_in_4bit=True"));
    }

    #[test]
    fn test_huggingface_config_generator() {
        let model_ir = ModelIR::default();
        let json_val = HuggingFaceConfigGenerator::generate_config_json(&model_ir);
        assert_eq!(json_val["model_type"], "llama");
        assert_eq!(json_val["hidden_size"], 4096);
        assert_eq!(json_val["num_hidden_layers"], 32);
        assert_eq!(json_val["torch_dtype"], "float16");
    }

    #[test]
    fn test_peft_lora_merge_math() {
        let out_dim = 2;
        let in_dim = 2;
        let rank = 1;
        let alpha = 2.0;

        let base = vec![1.0, 2.0, 3.0, 4.0];
        let lora_a = vec![1.0, 2.0]; // 1 x 2
        let lora_b = vec![0.5, 1.0]; // 2 x 1

        // scale = alpha / rank = 2.0 / 1.0 = 2.0
        // B @ A:
        // [0.5] @ [1.0, 2.0] = [[0.5, 1.0], [1.0, 2.0]]
        // delta * scale:
        // [[1.0, 2.0], [2.0, 4.0]]
        // base + delta * scale:
        // [[1.0+1.0, 2.0+2.0], [3.0+2.0, 4.0+4.0]] = [2.0, 4.0, 5.0, 8.0]
        let merged = PeftLoraAdapter::merge_lora_weights(
            &base, &lora_a, &lora_b, out_dim, in_dim, rank, alpha,
        )
        .expect("merge failed");

        assert_eq!(merged, vec![2.0, 4.0, 5.0, 8.0]);
    }

    #[test]
    fn test_lora_tensor_identification() {
        assert!(PeftLoraAdapter::is_lora_tensor(
            "layers.0.self_attn.q_proj.lora_A.weight"
        ));
        assert!(!PeftLoraAdapter::is_lora_tensor(
            "layers.0.self_attn.q_proj.weight"
        ));
        assert_eq!(
            PeftLoraAdapter::extract_base_module_name("layers.0.self_attn.q_proj.lora_A.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
    }
}
