#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Runs the six THME-3080 A1a recipes through `thme evaluate` and writes the results next to the
pre-registered rule in docs/benchmarks/pythia-14m/data/embedding_vs_linear.json.

  py scripts/embedding_vs_linear.py --thme target/release/thme --model models/pythia-14m \
      --tokens target/eval/wikitext2_test.pythia-14m.u32

The rule is read from the file's `pre_registration` block and applied as written.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

RECIPES = "docs/benchmarks/pythia-14m/recipes"
DATA = "docs/benchmarks/pythia-14m/data/embedding_vs_linear.json"
COMMITTED_Q8_DELTA = 0.8335124268154317  # verification_report.json, uniform Q8_0
GATE_MARGIN = 0.45


def run(args, name, work):
    report = os.path.join(work, f"{name}.json")
    cmd = [args.thme, "evaluate", args.model, "--recipe", f"{RECIPES}/{name}.json",
           "--tokens", args.tokens, "--eval-context", "1024", "--device", args.device,
           "--report", report, "--baseline-cache-dir", os.path.join(work, "baseline-cache")]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(report, encoding="utf-8") as f:
        r = json.load(f)
    base, rec = r["evaluation"]["variants"]
    return {
        "recipe": f"{RECIPES}/{name}.json",
        "format_counts": r["recipe"]["format_counts"],
        "baseline_perplexity": base["perplexity"],
        "perplexity": rec["perplexity"],
        "delta_ppl": rec["delta_ppl"],
        "gate_accepted": r["evaluation"]["gate_passed"],
        "gate_verdict": r["evaluation"]["gate_verdict"],
        "artifact_bytes": r["export"].get("artifact_bytes"),
        "device": r["host"]["device"],
        "backend": r["host"]["backend"],
        "corpus": r["evaluation"]["corpus"],
        "thme_version": r["thme_version"],
    }


def verdict(runs, fmt):
    e, l, both = (runs[f"a1a_{g}_{fmt}"]["delta_ppl"] for g in ("emb", "lin", "both"))
    big = max(e, l)
    if abs(e - l) < 0.2 * big:
        cause = "both contribute"
    else:
        cause = "E (embeddings and output head)" if e > l else "L (linear layers)"
    return {
        "delta_E": e, "delta_L": l, "delta_both": both,
        "E_fails_alone": e > GATE_MARGIN, "L_fails_alone": l > GATE_MARGIN,
        "cause_by_rule": cause,
        "additivity": {"sum_of_singles": e + l, "both": both, "gap": both - (e + l)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thme", default="target/release/thme")
    p.add_argument("--model", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--work-dir", default="target/embedding_vs_linear")
    args = p.parse_args()
    with open(DATA, encoding="utf-8") as f:
        doc = json.load(f)
    os.makedirs(args.work_dir, exist_ok=True)
    names = sorted(os.path.basename(x)[:-5] for x in glob.glob(f"{RECIPES}/a1a_*.json"))
    runs = {n: run(args, n, args.work_dir) for n in names}
    doc["results"] = {
        "runs": runs,
        "verdict_q8_0": verdict(runs, "q8_0"),
        "verdict_q4_0": verdict(runs, "q4_0"),
        "reproduction": {
            "committed_delta": COMMITTED_Q8_DELTA,
            "measured_delta": runs["a1a_both_q8_0"]["delta_ppl"],
            "within_0_02": abs(runs["a1a_both_q8_0"]["delta_ppl"] - COMMITTED_Q8_DELTA) <= 0.02,
        },
    }
    doc["status"] = "measured"
    with open(DATA, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(json.dumps(doc["results"]["verdict_q8_0"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
