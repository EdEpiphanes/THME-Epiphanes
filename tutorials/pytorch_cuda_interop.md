# PyTorch and CUDA Interoperability Guide: Generating Training Harnesses, Hugging Face Configs, and LoRA Weight Merging

## Executive Summary

While THME is engineered in pure Rust to provide zero-copy memory profiling, structural layer decomposition, and sub-millisecond AST model surgery, fine-tuning and weight adaptation workflows predominantly execute within the Python, PyTorch, and NVIDIA CUDA ecosystems.

THME bridges this gap with its PyTorch and CUDA Interoperability Bridge (`thme export` / `thme interop`). It programmatically synthesizes standalone, production-ready Python training harnesses (`train_cuda.py`) supporting PEFT LoRA, BitsAndBytes 4-bit QLoRA, Unsloth, and DeepSpeed ZeRO-2/3. Furthermore, THME serializes official Hugging Face `config.json` specifications directly from `ModelIR`, and provides an in-engine pure Rust LoRA weight merger ($W_{\text{eff}} = W_0 + \frac{\alpha}{r} B \times A$) for zero-copy weight integration.

---

## Architectural Overview

The interoperability workflow functions as a bi-directional bridge between pure Rust and PyTorch / CUDA:

```text
[THME Rust Engine]
       │
       ├─► 1. thme plan / convert ──────► Mutated ModelIR
       │
       ├─► 2. thme export ──────────────► train_cuda.py + config.json
       │                                         │
       │                                         ▼
[CUDA GPU Cluster] ◄────────────────────── [PyTorch / Hugging Face]
  (train_cuda.py executes fine-tuning)       (PEFT LoRA / QLoRA / ZeRO)
       │
       ▼
  adapter_model.safetensors
       │
       ▼
[THME Rust Engine] ◄────────────────────── In-Engine LoRA Weight Merger
  (W_eff = W_base + (alpha / r) * B x A)     (Zero-copy tensor integration)
```

1. **Model Export**: THME analyzes model topology and exports optimized training harnesses and Hugging Face configuration files.
2. **External Fine-Tuning**: The generated `train_cuda.py` script trains low-rank adapters ($A$ and $B$) using native PyTorch 2.x and CUDA Tensor Cores.
3. **Weight Ingestion**: The resulting adapter weights are ingested back into THME's pure Rust engine and merged directly into base model weights without requiring Python runtimes in production.

---

## Prerequisites and Environment Setup

### 1. Host System Requirements
- CUDA-compatible GPU: NVIDIA Ampere (RTX 30xx, A100), Ada Lovelace (RTX 40xx), Hopper (H100), or Blackwell.
- CUDA Toolkit: Version 11.8 or 12.x installed with active drivers.

### 2. Python Environment Setup
Create a dedicated virtual environment and install the required machine learning dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft bitsandbytes accelerate
```

Verify CUDA availability in Python:

```bash
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## Step 1: Synthesizing a Standalone PEFT LoRA Training Script

Generate a complete, self-contained PyTorch training harness configured for parameter-efficient fine-tuning (PEFT LoRA):

```bash
./target/release/thme export \
  --strategy lora \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset dataset.jsonl \
  --output-script train_cuda.py \
  --rank 16 \
  --alpha 32 \
  --lr 2e-4 \
  --epochs 3 \
  --cuda
```

Expected output:

```text
=== THME PyTorch & CUDA Interoperability Bridge ===
Strategy:       PEFT LoRA (Low-Rank Adaptation)
Target Model:   meta-llama/Meta-Llama-3-8B
Dataset:        dataset.jsonl
Hardware Target: NVIDIA CUDA / Tensor Cores
Generated Script: "train_cuda.py" (Ready for 'python "train_cuda.py"')
Status: Ready for external PyTorch / CUDA fine-tuning pipelines.
```

### Generated Script Structure
The synthesized `train_cuda.py` script includes:
- Automated CUDA environment detection and TF32 Tensor Core activation:
  ```python
  if torch.cuda.get_device_capability()[0] >= 8:
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
  ```
- Target linear module configuration targeting attention projection heads (`q_proj`, `k_proj`, `v_proj`, `o_proj`):
  ```python
  from peft import LoraConfig, get_peft_model, TaskType

  lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      lora_dropout=0.05,
      bias="none",
      task_type=TaskType.CAUSAL_LM,
  )
  ```
- Hugging Face `Trainer` setup with FP16/BF16 mixed-precision and gradient accumulation.

---

## Step 2: Generating QLoRA and DeepSpeed ZeRO-3 Harnesses

For large models exceeding single-GPU memory limits, generate harnesses configured for 4-bit quantized base models (QLoRA) or multi-GPU distributed data parallelism (DeepSpeed).

### 4-Bit QLoRA (BitsAndBytes NormalFloat4)

```bash
./target/release/thme export \
  --strategy qlora \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset dataset.jsonl \
  --output-script train_qlora.py
```

Expected output:

```text
=== THME PyTorch & CUDA Interoperability Bridge ===
Strategy:       QLoRA (4-bit NF4 Base + LoRA Adapters)
Target Model:   meta-llama/Meta-Llama-3-8B
Dataset:        dataset.jsonl
Hardware Target: NVIDIA CUDA / Tensor Cores
Generated Script: "train_qlora.py" (Ready for 'python "train_qlora.py"')
Status: Ready for external PyTorch / CUDA fine-tuning pipelines.
```

The generated `train_qlora.py` injects `BitsAndBytesConfig` with double quantization and NF4 representation:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### DeepSpeed ZeRO-3 Distributed Training

For multi-node cluster fine-tuning:

```bash
./target/release/thme export \
  --strategy deepspeed-zero3 \
  --model meta-llama/Meta-Llama-3-70B \
  --dataset dataset.jsonl \
  --output-script train_deepspeed.py
```

---

## Step 3: Exporting Hugging Face Compatible config.json

When modifying layer counts, hidden dimensions, or attention heads in THME, downstream inference engines (such as transformers, text-generation-inference, or vLLM) require a matching `config.json` adhering to the official Hugging Face architecture schema.

Export a verified `config.json` from a model directory or SafeTensors archive:

```bash
./target/release/thme export \
  --model-file examples/mock_llama \
  --output-config config.json
```

Expected output:

```text
=== THME PyTorch & CUDA Interoperability Bridge ===
Strategy:       PEFT LoRA (Low-Rank Adaptation)
Target Model:   meta-llama/Meta-Llama-3-8B
Dataset:        dataset.jsonl
Hardware Target: NVIDIA CUDA / Tensor Cores
Generated Script: "train_cuda.py" (Ready for 'python "train_cuda.py"')
Exported Config:  "config.json" (Hugging Face format)
Status: Ready for external PyTorch / CUDA fine-tuning pipelines.
```

Inspect the exported `config.json`:

```json
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "head_dim": 32,
  "hidden_size": 256,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 8,
  "num_hidden_layers": 4,
  "num_key_value_heads": 4,
  "torch_dtype": "float16",
  "transformers_version": "4.44.0",
  "vocab_size": 32000
}
```

THME automatically identifies the model family (`LlamaForCausalLM`, `MistralForCausalLM`, `Qwen2ForCausalLM`, `DeepseekForCausalLM`, etc.) and formats the parameters accordingly.

---

## Step 4: Executing External CUDA Fine-Tuning

With the training harness and configuration exported, launch fine-tuning:

```bash
# 1. Prepare sample instruction dataset (JSONL format)
echo '{"text": "<s>[INST] What is topological manifold engine? [/INST] THME is a high-performance framework.</s>"}' > dataset.jsonl

# 2. Execute training
python3 train_cuda.py
```

During execution, PyTorch trains the adapter matrices and serializes the fine-tuned adapter weights to `./lora_output/adapter_model.safetensors`.

---

## Step 5: Pure Rust In-Engine LoRA Weight Merging

Once training completes, deploying separate base model weights and LoRA adapters in inference introduces pointer indirection and kernel launch overhead. THME provides an in-engine LoRA weight merger (`src/interop/pytorch.rs:PeftLoraAdapter`) that computes the effective weight matrix directly in Rust:

$$W_{\text{eff}} = W_{\text{base}} + \Delta W = W_{\text{base}} + \frac{\alpha}{r} (B \times A)$$

Where:
- $W_{\text{base}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$: Base model linear projection matrix.
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$: Down-projection adapter matrix (`lora_A`).
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$: Up-projection adapter matrix (`lora_B`).
- $\alpha$: LoRA scaling parameter.
- $r$: LoRA rank dimension.
- Scaling factor: $\gamma = \frac{\alpha}{r}$.

### Mathematical Implementation in Pure Rust

The matrix multiplication and scaling execute with zero heap allocations across weight buffers:

```rust
pub fn merge_lora_weights(
    base_weight: &mut [f32],
    lora_a: &[f32],
    lora_b: &[f32],
    d_out: usize,
    d_in: usize,
    rank: usize,
    alpha: f32,
) -> Result<(), &'static str> {
    let scaling = alpha / rank as f32;

    // Compute W_eff[i, j] += scaling * sum_k(B[i, k] * A[k, j])
    for i in 0..d_out {
        for j in 0..d_in {
            let mut delta = 0.0f32;
            for k in 0..rank {
                let b_val = lora_b[i * rank + k];
                let a_val = lora_a[k * d_in + j];
                delta += b_val * a_val;
            }
            base_weight[i * d_in + j] += scaling * delta;
        }
    }
    Ok(())
}
```

This in-engine merge guarantees zero inference overhead while eliminating Python dependencies from the production serving path.

---

## CLI Parameter Reference

| Flag | Short | Type | Default | Description |
|:---|:---|:---|:---|:---|
| `--strategy` | `-s` | String | "lora" | Fine-tuning strategy (`lora`, `qlora`, `full`, `unsloth`, `deepspeed-zero2`, `deepspeed-zero3`) |
| `--model` | `-m` | String | "meta-llama/Meta-Llama-3-8B" | Hugging Face model ID or path for the generated script |
| `--dataset` | `-d` | Path | "dataset.jsonl" | Path to instruction fine-tuning dataset |
| `--output-script`| | Path | "train_cuda.py" | Destination filename for synthesized Python training script |
| `--output-config`| | Path | None | Destination path to export Hugging Face `config.json` |
| `--model-file` | | Path | None | Local model directory or `.safetensors` file to inspect for config export |
| `--cuda` | | Flag | true | Target NVIDIA CUDA hardware acceleration |
| `--batch-size` | | Integer | 4 | Per-GPU training batch size |
| `--lr` | | Float | 0.0002 | Optimizer learning rate |
| `--epochs` | | Integer | 3 | Training epochs |
| `--rank` | | Integer | 16 | LoRA rank dimension $r$ |
| `--alpha` | | Integer | 32 | LoRA scaling factor $\alpha$ |

---

## Troubleshooting and Common Issues

### Issue 1: Model File Path for Config Export Fails
- **Symptom**: `Error: Path is not a valid directory or safetensors file: "examples/mock_llama/config.json"`.
- **Cause**: `--model-file` expects the directory path or the binary `.safetensors` file path, not the path to an individual JSON configuration file.
- **Remedy**: Pass `--model-file examples/mock_llama` or `--model-file examples/mock_llama/model.safetensors`.

### Issue 2: CUDA Out-of-Memory During Fine-Tuning
- **Symptom**: `torch.cuda.OutOfMemoryError: CUDA out of memory`.
- **Cause**: The base model in FP16 exceeds GPU VRAM capacity during backward gradient passes.
- **Remedy**: Switch the training strategy to QLoRA:
  ```bash
  ./target/release/thme export --strategy qlora --output-script train_cuda.py
  ```

---

## Next Steps and Related Guides

- [Roofline Profiling Tutorial](roofline_profiling.md): Evaluate GPU compute ceilings and memory bandwidth for training kernels.
- [Model Context Protocol (MCP) Setup](mcp_copilot_setup.md): Automate model training and surgery from GitHub Copilot.
- [Quickstart Guide](quickstart.md): Zero-copy inspection, layer sensitivity profiling, and quality gates.
- [README Overview](../README.md): Central architectural documentation and production roadmap.
