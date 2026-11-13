// SPDX-License-Identifier: AGPL-3.0-only
use thme::interop::{
    HuggingFaceConfigGenerator, PeftLoraAdapter, PyTorchCUDATrainingStrategy, PyTorchInteropConfig,
    PyTorchScriptGenerator,
};
use thme::ir::{ModelArchitecture, ModelIR};

#[test]
fn test_pytorch_script_generator_strategies() {
    // 1. LoRA
    let config_lora = PyTorchInteropConfig {
        strategy: PyTorchCUDATrainingStrategy::LoRA,
        ..Default::default()
    };
    let lora_script = PyTorchScriptGenerator::generate_training_script(&config_lora, None);
    assert!(lora_script.contains("from peft import LoraConfig, get_peft_model"));
    assert!(lora_script.contains("r=16"));
    assert!(lora_script.contains("lora_alpha=32"));
    assert!(lora_script.contains("torch.cuda.is_available()"));

    // 2. QLoRA
    let config_qlora = PyTorchInteropConfig {
        strategy: PyTorchCUDATrainingStrategy::QLoRA,
        ..Default::default()
    };
    let qlora_script = PyTorchScriptGenerator::generate_training_script(&config_qlora, None);
    assert!(qlora_script.contains("BitsAndBytesConfig"));
    assert!(qlora_script.contains("load_in_4bit=True"));
    assert!(qlora_script.contains("bnb_4bit_quant_type=\"nf4\""));

    // 3. Unsloth Fast
    let config_unsloth = PyTorchInteropConfig {
        strategy: PyTorchCUDATrainingStrategy::UnslothFast,
        ..Default::default()
    };
    let unsloth_script = PyTorchScriptGenerator::generate_training_script(&config_unsloth, None);
    assert!(unsloth_script.contains("FastLanguageModel"));
    assert!(unsloth_script.contains("unsloth"));

    // 4. DeepSpeed ZeRO-2
    let config_ds2 = PyTorchInteropConfig {
        strategy: PyTorchCUDATrainingStrategy::DeepSpeedZeRO2,
        ..Default::default()
    };
    let ds2_script = PyTorchScriptGenerator::generate_training_script(&config_ds2, None);
    assert!(ds2_script.contains("\"stage\": 2"));

    // 5. DeepSpeed ZeRO-3
    let config_ds3 = PyTorchInteropConfig {
        strategy: PyTorchCUDATrainingStrategy::DeepSpeedZeRO3,
        ..Default::default()
    };
    let ds3_script = PyTorchScriptGenerator::generate_training_script(&config_ds3, None);
    assert!(ds3_script.contains("\"stage\": 3"));
}

#[test]
fn test_huggingface_config_generation_all_architectures() {
    let architectures = vec![
        (ModelArchitecture::Llama, "LlamaForCausalLM", "llama"),
        (ModelArchitecture::Mistral, "MistralForCausalLM", "mistral"),
        (ModelArchitecture::Gemma, "GemmaForCausalLM", "gemma"),
        (ModelArchitecture::Qwen, "Qwen2ForCausalLM", "qwen2"),
        (
            ModelArchitecture::DeepSeek,
            "DeepseekForCausalLM",
            "deepseek",
        ),
    ];

    for (arch, expected_class, expected_type) in architectures {
        let model_ir = ModelIR {
            architecture: arch,
            ..Default::default()
        };
        let cfg = HuggingFaceConfigGenerator::generate_config_json(&model_ir);

        assert_eq!(cfg["architectures"][0], expected_class);
        assert_eq!(cfg["model_type"], expected_type);
        assert_eq!(cfg["hidden_size"], 4096);
        assert_eq!(cfg["num_hidden_layers"], 32);
        assert_eq!(cfg["torch_dtype"], "float16");
    }
}

#[test]
fn test_peft_lora_merge_correctness_and_errors() {
    let out_dim = 2;
    let in_dim = 3;
    let rank = 1;
    let alpha = 1.0;

    let base = vec![1.0; 6]; // 2 x 3
    let lora_a = vec![2.0, 3.0, 4.0]; // 1 x 3
    let lora_b = vec![0.5, 1.0]; // 2 x 1

    // delta = B @ A = [[0.5*2, 0.5*3, 0.5*4], [1.0*2, 1.0*3, 1.0*4]]
    //       = [[1.0, 1.5, 2.0], [2.0, 3.0, 4.0]]
    // base + delta = [[2.0, 2.5, 3.0], [3.0, 4.0, 5.0]]
    let merged =
        PeftLoraAdapter::merge_lora_weights(&base, &lora_a, &lora_b, out_dim, in_dim, rank, alpha)
            .expect("merge failed");

    assert_eq!(merged, vec![2.0, 2.5, 3.0, 3.0, 4.0, 5.0]);

    // Dimension mismatch checks (infallible error handling)
    assert!(PeftLoraAdapter::merge_lora_weights(
        &base,
        &[1.0],
        &lora_b,
        out_dim,
        in_dim,
        rank,
        alpha
    )
    .is_err());
    assert!(PeftLoraAdapter::merge_lora_weights(
        &base,
        &lora_a,
        &[1.0],
        out_dim,
        in_dim,
        rank,
        alpha
    )
    .is_err());
    assert!(PeftLoraAdapter::merge_lora_weights(
        &[1.0],
        &lora_a,
        &lora_b,
        out_dim,
        in_dim,
        rank,
        alpha
    )
    .is_err());
    assert!(PeftLoraAdapter::merge_lora_weights(
        &base, &lora_a, &lora_b, out_dim, in_dim, 0, alpha
    )
    .is_err());
}
