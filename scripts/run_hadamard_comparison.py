#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Runs the pre-registered Hadamard rotation comparison (THME-2206) and fills the results block of
docs/benchmarks/<model>/data/hadamard_comparison.json. Standard library only.

Four deterministic arms, each applied to every weight matrix of the model: unrotated and rotated
4-bit codebook, unrotated and rotated absmean ternary. The ceiling and Q4_0 round-to-nearest are
read from awq_calibration.json. The decision rule in the JSON is applied exactly as written.

  py scripts/run_hadamard_comparison.py --model-dir models/pythia-14m \
      --tokens-dir target/gptq-eval/pythia-14m --work-dir target/hadamard-run/pythia-14m \
      --json docs/benchmarks/pythia-14m/data/hadamard_comparison.json --thme target/release/thme
"""

import argparse
import hashlib
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("awq", os.path.join(HERE, "run_awq_calibration.py"))
awq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awq)

LINEAR_ROLES = ["MlpGate", "MlpUp", "MlpDown", "AttentionQ", "AttentionK", "AttentionV",
                "AttentionOut", "AttentionQkv"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unrotated_int4():
    return {
        "id": "thme.codebook_int4_unrotated", "name": "Unrotated 4-bit codebook", "author": "Epiphanes",
        "version": "1.0.0", "category": "Quantization",
        "description": "The 15 symmetric levels and absolute-maximum scale of thme.hadamard_int4, no rotation (THME-2206 control)",
        "effective_bits_per_param": 4.0, "compression_ratio": 4.0, "hardware_speedup_factor": 1.0,
        "target_roles": LINEAR_ROLES,
        "rule": {"type": "QuantizeCodebook",
                 "params": {"levels": [k / 7.0 for k in range(-7, 8)], "scale_mode": "AbsMax"}},
    }


def verdict(unrotated, rotated):
    """Pre-registered rule: 1 percent of the unrotated perplexity."""
    gap = unrotated - rotated
    limit = 0.01 * unrotated
    return {
        "unrotated": unrotated, "rotated": rotated, "unrotated_minus_rotated": gap,
        "relative_change": -gap / unrotated,
        "rotation": "better" if gap >= limit else "worse" if -gap >= limit else "no_difference_shown",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--thme", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        proto = json.load(f)
    with open(os.path.join(os.path.dirname(args.json), proto["comparators"]["file"]), encoding="utf-8") as f:
        prior = json.load(f)
    tag = proto["model_repo"].split("/")[-1]
    test = os.path.join(args.tokens_dir, f"wikitext2_test.{tag}.u32")
    if sha256(test) != proto["evaluation"]["token_file_sha256"]:
        raise SystemExit(f"error: {test} does not match the pre-registered hash")
    os.makedirs(args.work_dir, exist_ok=True)
    layers = prior["model_config"]["num_hidden_layers"]

    algo_file = os.path.join(args.work_dir, "algorithm-int4-unrotated.json")
    with open(algo_file, "w", encoding="utf-8") as f:
        json.dump(unrotated_int4(), f, indent=2)

    def evaluate(label, algorithm_id, algorithms=None):
        report = os.path.join(args.work_dir, f"report-{label}.json")
        if not os.path.exists(report):
            rec = awq.recipe(label, proto["model_name"], layers,
                             ("ApplyCustomAlgorithm", {"algorithm_id": algorithm_id}))
            rec_path = os.path.join(args.work_dir, f"recipe-{label}.json")
            with open(rec_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            print(f"running {proto['model_name']} {label} ...", flush=True)
            cmd = [args.thme, "evaluate", args.model_dir, "--recipe", rec_path, "--tokens", test,
                   "--report", report, "--device", args.device]
            if algorithms:
                cmd += ["--algorithms", algorithms]
            awq.run(cmd, os.path.join(args.work_dir, f"evaluate-{label}.log"))
        if not os.path.exists(report):
            raise SystemExit(f"error: evaluate {label} wrote no report")
        with open(report, encoding="utf-8") as f:
            return awq.extract_run(json.load(f))

    runs = {}
    for pair in proto["pairs"]:
        un, ro = pair["unrotated"]["algorithm_id"], pair["rotated"]["algorithm_id"]
        runs[f"{pair['id']}_unrotated"] = evaluate(f"{pair['id']}-unrotated", un,
                                                   algo_file if pair["id"] == "int4" else None)
        runs[f"{pair['id']}_rotated"] = evaluate(f"{pair['id']}-rotated", ro)

    q4 = prior["results"]["Q4_0"]
    proto["results"] = {
        "ceiling": q4["ceiling"]["perplexity"], "q4_0_rtn": q4["rtn"]["perplexity"], "runs": runs,
        "verdicts": {p["id"]: verdict(runs[f"{p['id']}_unrotated"]["perplexity"],
                                      runs[f"{p['id']}_rotated"]["perplexity"]) for p in proto["pairs"]},
    }
    proto["status"] = "measured"
    proto["pending"] = []
    proto["provenance"] = {"thme": os.path.basename(args.thme), "device": args.device}  # no local paths in committed reports
    with open(args.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(proto, f, indent=2)
        f.write("\n")
    print(json.dumps(proto["results"]["verdicts"], indent=2))


if __name__ == "__main__":
    main()
