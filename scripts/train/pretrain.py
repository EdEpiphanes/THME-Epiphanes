#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Small-model pretraining for the .epiph testbed (THME-3110, docs/research/epiph-testbed-plan.md).

Five arms are supported:
  scratch  random initialisation, Pythia-14M GPT-NeoX shape, Pythia tokenizer ids
  pythia   initialised from the published Pythia-14M weights, Pythia tokenizer ids
  byte     random initialisation, Pythia-14M GPT-NeoX shape, vocabulary 256 (raw UTF-8 bytes)
  llama    random initialisation, 6-layer Llama shape (width 128, 4 heads, SwiGLU 352), Pythia tokenizer ids
  loop     Llama shape with weight-shared middle layer(s) (prelude + shared*k + coda), exported untied

Data is a flat binary file of ids (uint16 for the token arms, uint8 for bytes). Windows are drawn at random
offsets from a seeded generator. A checkpoint (Hugging Face safetensors folder) is written every --ckpt-every
steps and a report JSON records every setting, the data hash and the losses, so a run can be repeated.

  python scripts/train/pretrain.py --arm scratch --model-dir models/pythia-14m --data train.u16 --val val.u16 \
      --out target/testbed/arm-a --steps 20000 --ckpt-every 2000
"""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.utils.parametrize
from transformers import AutoModelForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM

ARMS = ("scratch", "pythia", "byte", "llama", "loop")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def loop_schedule(shared_layers=1, loop_count=4):
    if shared_layers < 1 or loop_count < 1:
        raise ValueError("loop_shared_layers and loop_count must be >= 1")
    return [0] + list(range(1, 1 + shared_layers)) * loop_count + [1 + shared_layers]


def llama_config(model_dir, num_layers=None):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("model_type") == "llama":
        cfg = LlamaConfig.from_pretrained(model_dir)
        if num_layers is not None and cfg.num_hidden_layers != num_layers:
            cfg.num_hidden_layers = num_layers
            if getattr(cfg, "layer_types", None) is not None:
                cfg.layer_types = ["full_attention"] * num_layers
        cfg.tie_word_embeddings = False
        cfg.attention_bias = False
        cfg.mlp_bias = False
        return cfg
    return LlamaConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=128,
        num_hidden_layers=num_layers if num_layers is not None else 6,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=352,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )


def unique_params(model):
    seen, params = set(), []
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return params


def build_model(arm, model_dir, seed, loop_shared_layers=1, loop_count=4):
    torch.manual_seed(seed)
    if arm == "pythia":
        return GPTNeoXForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    if arm == "llama":
        return LlamaForCausalLM(llama_config(model_dir))
    if arm == "loop":
        sched = loop_schedule(loop_shared_layers, loop_count)
        cfg = llama_config(model_dir, num_layers=len(sched))
        cfg.use_cache = False
        model = LlamaForCausalLM(cfg)
        layers = model.model.layers
        unique = [layers[0]] + [layers[1 + j] for j in range(loop_shared_layers)] + [layers[-1]]
        for pos, u in enumerate(sched):
            layers[pos] = unique[u]
        return model
    cfg = GPTNeoXConfig.from_pretrained(model_dir)
    if arm == "byte":
        cfg.vocab_size = 256
        cfg.bos_token_id = cfg.eos_token_id = None
    return GPTNeoXForCausalLM(cfg)


class FakeQuant(torch.nn.Module):
    """Parametrization: forward weight = w + blend * (quantise(w) - w), the second term detached (straight-through)."""

    def __init__(self, regime):
        super().__init__()
        self.regime, self.blend = regime, 1.0

    def forward(self, w):
        if self.blend == 0.0:
            return w
        with torch.no_grad():
            q = _regimes()[self.regime](w.detach().float())[0].to(w.dtype)
        return w + self.blend * (q - w).detach()


def _regimes():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import testbed_s0
    return testbed_s0.REGIMES


FQ_WEIGHTS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def apply_fake_quant(model, regime, positions):
    """Put FakeQuant on the linear weights of the given layer positions; shared layer objects are wrapped once."""
    if regime not in _regimes():
        raise ValueError(f"unknown regime {regime}")
    done, fqs = set(), []
    for pos in positions:
        layer = model.model.layers[pos]
        if id(layer) in done:
            continue
        done.add(id(layer))
        for name in FQ_WEIGHTS:
            mod = layer
            for part in name.split("."):
                mod = getattr(mod, part)
            fq = FakeQuant(regime)
            torch.nn.utils.parametrize.register_parametrization(mod, "weight", fq)
            fqs.append(fq)
    return fqs


def export_quantised(model):
    """A plain Llama with the fake-quantised weights written in place of the latent ones (the model is left untouched)."""
    fqs = [fq for m in model.modules() if torch.nn.utils.parametrize.is_parametrized(m, "weight")
           for fq in m.parametrizations.weight]
    old = [fq.blend for fq in fqs]
    for fq in fqs:
        fq.blend = 1.0
    try:
        sd = {}
        for k, v in model.state_dict().items():
            if ".parametrizations.weight." in k:
                prefix = k.split(".parametrizations.weight.")[0]
                sd[prefix + ".weight"] = model.get_submodule(prefix).weight.detach().clone()
            else:
                sd[k] = v.detach().clone()
    finally:
        for fq, b in zip(fqs, old):
            fq.blend = b
    out = LlamaForCausalLM(copy.deepcopy(model.config))
    out.load_state_dict(sd)
    return out


def load_ids(path, arm):
    return np.memmap(path, dtype=np.uint8 if arm == "byte" else np.uint16, mode="r")


def batch(ids, rng, size, seq, device):
    starts = rng.integers(0, len(ids) - seq - 1, size=size)
    x = np.stack([ids[s:s + seq + 1] for s in starts]).astype(np.int64)
    t = torch.from_numpy(x).to(device)
    return t[:, :-1], t[:, 1:]


def lr_at(step, steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, ids, seq, windows, device):
    model.eval()
    total, count = 0.0, 0
    for i in range(windows):
        s = i * seq
        if s + seq + 1 > len(ids):
            break
        x = torch.from_numpy(ids[s:s + seq + 1].astype(np.int64)).to(device)[None]
        logits = model(x[:, :-1]).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0], x[0, 1:], reduction="sum")
        total += loss.item()
        count += seq
    model.train()
    return total / max(1, count)


def save(model, out, step, model_dir, arm):
    d = os.path.join(out, f"step-{step:06d}")
    if any(torch.nn.utils.parametrize.is_parametrized(m, "weight") for m in model.modules()):
        model = export_quantised(model)
    if arm == "loop":
        # Materialize an untied Llama checkpoint with cloned tensors so safetensors never sees shared storage.
        rng_state = torch.get_rng_state()
        try:
            export_cfg = copy.deepcopy(model.config)
            export_cfg.use_cache = True
            export_model = LlamaForCausalLM(export_cfg)
        finally:
            torch.set_rng_state(rng_state)
        export_model.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        export_model.save_pretrained(d, safe_serialization=True)
    else:
        model.save_pretrained(d, safe_serialization=True)
    if arm != "byte":  # token arms keep the Pythia tokenizer next to the weights
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = os.path.join(model_dir, f)
            if os.path.exists(src):
                shutil.copy(src, d)
    return d


def distill_loss(ce_sum, student_logits, teacher_logits, weight, temp):
    s_log_p = torch.nn.functional.log_softmax(student_logits / temp, dim=-1)
    t_log_p = torch.nn.functional.log_softmax(teacher_logits / temp, dim=-1)
    kl_sum = torch.nn.functional.kl_div(s_log_p, t_log_p, reduction="sum", log_target=True) * (temp * temp)
    return (1.0 - weight) * ce_sum + weight * kl_sum


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.makedirs(args.out, exist_ok=True)
    loop_shared_layers = getattr(args, "loop_shared_layers", 1)
    loop_count = getattr(args, "loop_count", 4)
    distill_from = getattr(args, "distill_from", None)
    distill_weight = getattr(args, "distill_weight", 0.5)
    distill_temp = getattr(args, "distill_temp", 1.0)
    if distill_temp <= 0:
        raise ValueError("distill_temp must be positive")
    if not (0.0 <= distill_weight <= 1.0):
        raise ValueError("distill_weight must be in [0, 1]")

    model = build_model(args.arm, args.model_dir, args.seed, loop_shared_layers, loop_count)
    init_from = getattr(args, "init_from", None)
    if init_from:
        model.load_state_dict(LlamaForCausalLM.from_pretrained(init_from, torch_dtype=torch.float32).state_dict())
    model = model.to(device)
    fq_regime = getattr(args, "fq_regime", None)
    fqs = []
    if fq_regime:
        fq_layers = [int(v) for v in str(getattr(args, "fq_layers", "1")).split(",")]
        fqs = apply_fake_quant(model, fq_regime, fq_layers)
    fq_blend_steps = getattr(args, "fq_blend_steps", 500)
    model.train()
    params = unique_params(model)

    teacher = None
    if distill_from:
        teacher = AutoModelForCausalLM.from_pretrained(distill_from, torch_dtype=torch.float32).to(device)
        teacher.eval()
        teacher.config.use_cache = False
        for p in teacher.parameters():
            p.requires_grad_(False)
        if teacher.config.vocab_size != model.config.vocab_size:
            raise ValueError(
                f"teacher vocab_size ({teacher.config.vocab_size}) != student ({model.config.vocab_size})"
            )

    ids = load_ids(args.data, args.arm)
    val = load_ids(args.val, args.arm)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    n_params = sum(p.numel() for p in params)
    report = {
        "report": "pretrain", "arm": args.arm, "seed": args.seed, "steps": args.steps, "batch": args.batch,
        "seq": args.seq, "lr": args.lr, "warmup": args.warmup, "weight_decay": args.weight_decay,
        "parameters": n_params, "device": device,
        "data_sha256": sha256(args.data), "val_sha256": sha256(args.val), "data_ids": int(len(ids)),
        "checkpoints": [], "train_loss_every_100": [],
    }
    if args.arm == "loop":
        report["unique_parameters"] = n_params
        report["layer_schedule"] = loop_schedule(loop_shared_layers, loop_count)
    if teacher is not None:
        report["distill_from"] = os.path.basename(os.path.normpath(distill_from))
        report["distill_weight"] = distill_weight
        report["distill_temp"] = distill_temp

    t0, run_loss = time.time(), 0.0
    for step in range(args.steps + 1):
        if step % args.ckpt_every == 0 or step == args.steps:
            path = save(model, args.out, step, args.model_dir, args.arm)
            vl = evaluate(model, val, args.seq, args.val_windows, device)
            report["checkpoints"].append({"step": step, "dir": os.path.basename(path), "val_loss": vl,
                                          "val_nats_per_id": vl, "seconds": round(time.time() - t0, 1)})
            print(f"step {step} val_loss {vl:.4f}", flush=True)
        if step == args.steps:
            break
        for fq in fqs:
            fq.blend = min(1.0, (step + 1) / max(1, fq_blend_steps))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        x, y = batch(ids, rng, args.batch, args.seq, device)
        micro = args.micro_batch or args.batch
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        cache = torch.nn.utils.parametrize.cached() if fqs else contextlib.nullcontext()
        cache.__enter__()  # quantise each weight once per step, not once per loop pass and micro-batch
        for i in range(0, args.batch, micro):  # gradient accumulation: same batch, less memory
            x_mb, y_mb = x[i:i + micro], y[i:i + micro]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x_mb).logits
            ce_part = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                        y_mb.reshape(-1), reduction="sum")
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x_mb).logits.float()
                part = distill_loss(ce_part, logits.float(), t_logits, distill_weight, distill_temp)
            else:
                part = ce_part
            (part / (args.batch * args.seq)).backward()
            loss_sum += part.item()
        cache.__exit__(None, None, None)
        loss = torch.tensor(loss_sum / (args.batch * args.seq))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        loss_val = loss.item()
        if teacher is not None:
            if step == 0:
                report["first_train_loss"] = loss_val
            report["last_train_loss"] = loss_val
        run_loss += loss_val
        if (step + 1) % 100 == 0:
            report["train_loss_every_100"].append(round(run_loss / 100, 5))
            run_loss = 0.0
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--model-dir", required=True, help="Pythia-14M folder (config, tokenizer, weights for arm pythia)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=0, help="windows per forward pass; 0 means the whole batch")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--val-windows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--loop-shared-layers", type=int, default=1)
    ap.add_argument("--loop-count", type=int, default=4)
    ap.add_argument("--distill-from", default=None, help="HF teacher checkpoint folder for KL distillation")
    ap.add_argument("--distill-weight", type=float, default=0.5)
    ap.add_argument("--distill-temp", type=float, default=1.0)
    ap.add_argument("--init-from", default=None, help="HF Llama folder to start from (arms llama and loop)")
    ap.add_argument("--fq-regime", default=None, help="fake-quantise (straight-through) with a testbed_s0 regime")
    ap.add_argument("--fq-layers", default="1", help="comma list of layer positions to fake-quantise")
    ap.add_argument("--fq-blend-steps", type=int, default=500)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
