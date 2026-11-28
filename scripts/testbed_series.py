#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Packs a training run's checkpoints as .epiph files (each recording the previous one as its parent) and measures
every checkpoint with `thme evaluate` on the committed WikiText-2 test tokens (THME-3110). Standard library only.

The recipe is one harmless setting; the number kept from each report is the unmodified baseline variant.

  py scripts/testbed_series.py --run-dir target/testbed/arm-scratch --thme target-nodbg/release/thme.exe \
      --tokens target/testbed/eval/wikitext2_test.pythia-14m.u32 --json docs/benchmarks/epiph-testbed/data/arm_a_checkpoints.json
"""

import argparse
import hashlib
import json
import os
import re
import subprocess

RECIPE = {"name": "baseline-probe", "author": "Epiphanes", "target_model": "epiph-testbed", "version": "1.0",
          "description": "one-layer FP32 setting so that evaluate reports the unmodified baseline variant",
          "operations": [{"type": "SetPrecision",
                          "params": {"start_layer": 0, "end_layer": 0, "role": None, "format": "FP32"}}]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"error: {' '.join(cmd[:3])} failed: {p.stderr.strip()[-300:]}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--thme", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--note", default="")
    ap.add_argument("--context", type=int, default=1024, help="evaluation window in tokens (2048 for the byte arm)")
    a = ap.parse_args()

    steps = sorted(d for d in os.listdir(a.run_dir) if re.fullmatch(r"step-\d{6}", d))
    recipe = os.path.join(a.run_dir, "baseline-probe.json")
    with open(recipe, "w", encoding="utf-8", newline="\n") as f:
        json.dump(RECIPE, f)
    train = json.load(open(os.path.join(a.run_dir, "report.json"), encoding="utf-8"))
    val = {c["dir"]: c["val_loss"] for c in train["checkpoints"]}
    rows, parent = [], None
    for d in steps:
        epiph = os.path.join(a.run_dir, d + ".epiph")
        cmd = [a.thme, "epiph", "pack", os.path.join(a.run_dir, d), "--out", epiph, "--annotate", f"checkpoint={d}"]
        if parent:
            cmd += ["--parent", parent]
        run(cmd)
        report = os.path.join(a.run_dir, f"eval-{d}.json")
        run([a.thme, "evaluate", epiph, "--recipe", recipe, "--tokens", a.tokens, "--device", a.device,
             "--eval-context", str(a.context), "--report", report])
        r = json.load(open(report, encoding="utf-8"))
        base = next(v for v in r["evaluation"]["variants"] if v["id"] == "baseline")
        rows.append({"checkpoint": d, "step": int(d.split("-")[1]), "epiph_sha256": sha256(epiph),
                     "epiph_bytes": os.path.getsize(epiph), "wikitext2_test_perplexity": base["perplexity"],
                     "bits_per_token": base["bits_per_token"],
                     "wikitext103_validation_loss_nats": val.get(d), "device_used": r["host"].get("used"),
                     "backend": r.get("host", {}).get("backend")})
        print(d, base["perplexity"], flush=True)
        parent = epiph
    out = {"report": "testbed_checkpoint_series", "arm": train["arm"], "training_report": "report.json of the run",
           "training": {k: train[k] for k in ("seed", "steps", "batch", "seq", "lr", "warmup", "weight_decay",
                                              "parameters", "data_sha256", "val_sha256", "device")},
           "evaluation": {"tokens_sha256": sha256(a.tokens), "context_len": a.context,
                          "note": "perplexity of the unmodified checkpoint from thme evaluate; GPU path uses f16 weights"},
           "conditions": a.note, "checkpoints": rows, "provenance": {"thme": os.path.basename(a.thme)}}
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
