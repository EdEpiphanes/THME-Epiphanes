#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Test perplexity of one or more HF checkpoint folders on the committed WikiText-2 test tokens (float32, eager).
python scripts/testbed_eval_ppl.py --tokens <u32> --out out.json folder [folder ...]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
import testbed_s0 as t
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--tokens"); ap.add_argument("--out"); ap.add_argument("folders", nargs="+")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
w = t.load_windows(a.tokens, 1024)
res = {}
for f in a.folders:
    m = AutoModelForCausalLM.from_pretrained(f, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    res[f.replace("\\", "/")] = t.perplexity(m, w, dev)[0]
    print(f, round(res[f.replace("\\", "/")], 3), flush=True)
json.dump(res, open(a.out, "w"), indent=2)
