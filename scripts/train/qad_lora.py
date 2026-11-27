#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Quantization-aware distillation with LoRA recovery (ROADMAP v0.2.3, external PyTorch trainer).

  teacher = the unquantized model
  student = the same model with every weight THME's Q4_0 recipe touches replaced by its Q4_0
            round trip (quant.py), frozen, plus LoRA adapters on q, k, v, o, gate, up, down
  loss    = KL(teacher next-token distribution || student next-token distribution), mean over
            every position of every window in the batch; only the adapters train (AdamW)

Writes <out-dir>/adapter, train_log.json (loss at every step, taken before that step's update, so
step 0 is the zero-adapter loss), merged/ (the student base with the adapter merged in, fp16
safetensors, for `thme evaluate`) and result.json. Evaluation uses eval_ppl.py (THME's windowing).

  python qad_lora.py --model-dir M --train-tokens T.u32 --eval-tokens E.u32 --out-dir O \
      --steps 200 --device cpu
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import eval_ppl
import quant

ROLES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def read_u32(path):
    return eval_ppl.read_token_file(path)


def windows_from_tokens(tokens, window):
    """Consecutive non-overlapping full windows, shape [n, window]."""
    n = len(tokens) // window
    if n == 0:
        raise ValueError(f"{len(tokens)} training tokens hold no window of {window}")
    return torch.tensor(tokens[: n * window]).reshape(n, window)


def batch_indices(seed, n_windows, batch_size, steps):
    """Window indices of every step, drawn with replacement from a generator of its own."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, n_windows, (batch_size,), generator=g) for _ in range(steps)]


def kl_loss(teacher_logits, student_logits):
    """Mean over positions of KL(p_teacher || p_student), in float32."""
    lt = F.log_softmax(teacher_logits.float(), dim=-1)
    ls = F.log_softmax(student_logits.float(), dim=-1)
    return (lt.exp() * (lt - ls)).sum(-1).mean()


def _load(model_dir, dtype):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)


def build_teacher(model_dir, device, dtype):
    model = _load(model_dir, dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_student(model_dir, device):
    """float32 model with the THME Q4_0 tensors rounded. Returns (model, rounded names)."""
    model = _load(model_dir, torch.float32).to(device).eval()
    return model, quant.quantize_model_(model)


def add_lora(model, rank, alpha, seed):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)  # the LoRA A matrices are drawn from the global generator
    cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=ROLES, lora_dropout=0.0, bias="none")
    return get_peft_model(model, cfg)


def merge(peft_model):
    """The base model with the adapter folded into its weights (modifies `peft_model`)."""
    return peft_model.merge_and_unload()


def load_merged(path):
    return _load(path, torch.float16)


def run(a):
    started = time.perf_counter()
    os.makedirs(a.out_dir, exist_ok=True)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    torch.manual_seed(a.seed)
    train = windows_from_tokens(read_u32(a.train_tokens), a.window).to(device)
    eval_tokens = read_u32(a.eval_tokens)
    ev = lambda m: eval_ppl.evaluate_ppl(m, eval_tokens, a.window, device=device)["perplexity"]

    teacher = build_teacher(a.model_dir, device, DTYPES[a.teacher_dtype])
    student, names = build_student(a.model_dir, device)
    ppl_teacher, ppl_q4 = ev(teacher), ev(student)
    model = add_lora(student, a.rank, a.alpha, a.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    log = []
    model.train()
    train_started = time.perf_counter()
    micro = getattr(a, "micro_batch", None) or a.batch_size
    for step, idx in enumerate(batch_indices(a.seed, len(train), a.batch_size, a.steps)):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for piece in idx.to(device).split(micro):
            ids = train[piece]
            with torch.no_grad():
                t_logits = teacher(input_ids=ids).logits
            # Every window has the same number of positions, so weighting each piece's mean by its share
            # of the windows gives exactly the mean of the whole batch.
            loss = kl_loss(t_logits, model(input_ids=ids).logits) * (len(piece) / len(idx))
            loss.backward()
            step_loss += loss.item()
        opt.step()
        log.append({"step": step, "loss": step_loss})
    train_seconds = time.perf_counter() - train_started

    model.eval()
    ppl_final = ev(model)
    model.save_pretrained(os.path.join(a.out_dir, "adapter"))
    merged = merge(model).to(torch.float16)
    merged.save_pretrained(os.path.join(a.out_dir, "merged"), safe_serialization=True)

    adapter_file = os.path.join(a.out_dir, "adapter", "adapter_model.safetensors")
    result = {
        "model_dir_name": os.path.basename(os.path.normpath(a.model_dir)),
        "device": device,
        "teacher_dtype": a.teacher_dtype,
        "seed": a.seed,
        "steps": a.steps,
        "batch_size": a.batch_size,
        "micro_batch": micro,
        "window": a.window,
        "rank": a.rank,
        "alpha": a.alpha,
        "lr": a.lr,
        "q4_0_tensors": len(names),
        "adapter_bytes": os.path.getsize(adapter_file),
        "trainable_params": sum(p.numel() for p in params),
        "trainable_tensors": len(params),
        "total_params": total_params,
        "train_seconds": train_seconds,
        "wall_seconds": time.perf_counter() - started,
        "first_loss": log[0]["loss"] if log else None,
        "last_loss": log[-1]["loss"] if log else None,
        "eval_ppl": ppl_final,
        "eval_ppl_teacher": ppl_teacher,
        "eval_ppl_student_q4_0": ppl_q4,
        "eval_tokens_file": os.path.basename(a.eval_tokens),
    }
    with open(os.path.join(a.out_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    with open(os.path.join(a.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--train-tokens", required=True)
    ap.add_argument("--eval-tokens", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="windows per forward pass (gradient accumulation to fit memory); default: the whole batch")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--teacher-dtype", choices=list(DTYPES), default="float32")
    return ap.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=1))
