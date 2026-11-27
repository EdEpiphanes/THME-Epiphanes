#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Small-model pretraining for the .epiph testbed (THME-3110, docs/research/epiph-testbed-plan.md).

Three arms share Pythia-14M's shape (6 layers, width 128, 4 heads, GPT-NeoX blocks):
  scratch  random initialisation, Pythia tokenizer ids
  pythia   initialised from the published Pythia-14M weights, Pythia tokenizer ids
  byte     random initialisation, vocabulary 256 (raw UTF-8 bytes, no tokenizer)

Data is a flat binary file of ids (uint16 for the token arms, uint8 for bytes). Windows are drawn at random
offsets from a seeded generator. A checkpoint (Hugging Face safetensors folder) is written every --ckpt-every
steps and a report JSON records every setting, the data hash and the losses, so a run can be repeated.

  python scripts/train/pretrain.py --arm scratch --model-dir models/pythia-14m --data train.u16 --val val.u16 \
      --out target/testbed/arm-a --steps 20000 --ckpt-every 2000
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

ARMS = ("scratch", "pythia", "byte")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(arm, model_dir, seed):
    torch.manual_seed(seed)
    if arm == "pythia":
        return GPTNeoXForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    cfg = GPTNeoXConfig.from_pretrained(model_dir)
    if arm == "byte":
        cfg.vocab_size = 256
        cfg.bos_token_id = cfg.eos_token_id = None
    return GPTNeoXForCausalLM(cfg)


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
    model.save_pretrained(d, safe_serialization=True)
    if arm != "byte":  # token arms keep the Pythia tokenizer next to the weights
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = os.path.join(model_dir, f)
            if os.path.exists(src):
                shutil.copy(src, d)
    return d


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.makedirs(args.out, exist_ok=True)
    model = build_model(args.arm, args.model_dir, args.seed).to(device)
    model.train()
    ids = load_ids(args.data, args.arm)
    val = load_ids(args.val, args.arm)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    report = {
        "report": "pretrain", "arm": args.arm, "seed": args.seed, "steps": args.steps, "batch": args.batch,
        "seq": args.seq, "lr": args.lr, "warmup": args.warmup, "weight_decay": args.weight_decay,
        "parameters": sum(p.numel() for p in model.parameters()), "device": device,
        "data_sha256": sha256(args.data), "val_sha256": sha256(args.val), "data_ids": int(len(ids)),
        "checkpoints": [], "train_loss_every_100": [],
    }
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
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        x, y = batch(ids, rng, args.batch, args.seq, device)
        micro = args.micro_batch or args.batch
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for i in range(0, args.batch, micro):  # gradient accumulation: same batch, less memory
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x[i:i + micro]).logits
            part = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                     y[i:i + micro].reshape(-1), reduction="sum")
            (part / (args.batch * args.seq)).backward()
            loss_sum += part.item()
        loss = torch.tensor(loss_sum / (args.batch * args.seq))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.item()
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
    run(ap.parse_args())


if __name__ == "__main__":
    main()
