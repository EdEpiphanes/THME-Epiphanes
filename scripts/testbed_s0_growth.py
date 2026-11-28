#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""How damage grows as k consecutive middle layers (1..k) are quantised. Linear growth means additive; faster means compounding.
python scripts/testbed_s0_growth.py --model <folder> --tokens <u32> --json out.json"""
import argparse, json, math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import testbed_s0 as t
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--tokens"); ap.add_argument("--json")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
w = t.load_windows(a.tokens, 1024)
base, _, _ = t.perplexity(m, w, dev)
rows = []
for regime in ("phase_P4_a4_r2", "scalar3_control", "lloyd3_control", "ternary"):
    for k in (1, 2, 3, 4):
        saved, _, _ = t.quantise_layers(m, list(range(1, k + 1)), regime)
        p, _, _ = t.perplexity(m, w, dev)
        t.restore(m, saved)
        rows.append({"regime": regime, "k": k, "perplexity": p, "delta_nats": math.log(p / base)})
        print(regime, k, round(p, 2), round(math.log(p / base), 4), flush=True)
json.dump({"baseline": base, "rows": rows}, open(a.json, "w"), indent=2)
