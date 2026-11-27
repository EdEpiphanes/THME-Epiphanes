#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Runs the pre-registered GPTQ comparison (THME-2205) and fills the results block of
docs/benchmarks/<model>/data/gptq_comparison.json. Standard library only.

One arm: `thme.gptq_q4_0` on every weight matrix with calibration statistics from the seed-1234
token file. RTN, AWQ and the ceiling are read from awq_calibration.json (not re-run). The decision
rule in the JSON is applied exactly as written; this script never changes the protocol fields.

  py scripts/run_gptq_comparison.py --model-dir models/pythia-14m \
      --tokens-dir target/gptq-eval/pythia-14m --work-dir target/gptq-run/pythia-14m \
      --json docs/benchmarks/pythia-14m/data/gptq_comparison.json --thme target/release/thme
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("awq", os.path.join(HERE, "run_awq_calibration.py"))
awq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awq)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verdict(rtn, awq_mean, gptq, sigma):
    """The pre-registered decision rule, applied to numbers only."""
    return {
        "sigma": sigma,
        "rtn_minus_gptq": rtn - gptq,
        "beats_rtn": (rtn - gptq) > 2 * sigma,
        "awq_mean_minus_gptq": awq_mean - gptq,
        "vs_awq": (
            "gptq_better" if (awq_mean - gptq) > 2 * sigma
            else "gptq_worse" if (gptq - awq_mean) > 2 * sigma
            else "no_difference_shown"
        ),
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
    awq_path = os.path.join(os.path.dirname(args.json), proto["comparators"]["file"])
    with open(awq_path, encoding="utf-8") as f:
        prior = json.load(f)
    q = prior["results"][proto["format"]]
    awq_runs = [r["perplexity"] for r in q["awq"]["runs"]]
    sigma = statistics.stdev(awq_runs)
    rtn, ceiling = q["rtn"]["perplexity"], q["ceiling"]["perplexity"]
    awq_mean = statistics.mean(awq_runs)

    tag = proto["model_repo"].split("/")[-1]
    test = os.path.join(args.tokens_dir, f"wikitext2_test.{tag}.u32")
    cal_tok = os.path.join(args.tokens_dir, f"wikitext2_train.{tag}.cal16x2048.s1234.u32")
    for path, want in ((test, proto["evaluation"]["token_file_sha256"]),
                       (cal_tok, proto["calibration"]["token_file_sha256"])):
        if sha256(path) != want:
            raise SystemExit(f"error: {path} does not match the pre-registered hash")
    os.makedirs(args.work_dir, exist_ok=True)

    cal_out = os.path.join(args.work_dir, "calibration-1234.json")
    if not os.path.exists(cal_out):
        code = awq.run([args.thme, "calibrate", args.model_dir, "--tokens", cal_tok, "--context", "2048",
                        "--max-tokens", "32768", "--out", cal_out, "--device", args.device],
                       os.path.join(args.work_dir, "calibrate.log"))
        if code != 0 or not os.path.exists(cal_out):
            raise SystemExit("error: calibrate failed (see calibrate.log)")

    layers = prior["model_config"]["num_hidden_layers"]
    rec = awq.recipe("gptq-q4_0", proto["model_name"], layers,
                     ("ApplyCustomAlgorithm", {"algorithm_id": proto["algorithm_id"]}))
    rec_path = os.path.join(args.work_dir, "recipe-gptq-q4_0.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    results = {}
    for gate_name, gate_args in (("v1", []), ("v2", ["--gate", "configs/gates/thme-perplexity-v2.json"])):
        report = os.path.join(args.work_dir, f"report-gptq-{gate_name}.json")
        if not os.path.exists(report):
            print(f"running {proto['model_name']} gptq ({gate_name}) ...", flush=True)
            awq.run([args.thme, "evaluate", args.model_dir, "--recipe", rec_path, "--tokens", test,
                     "--report", report, "--device", args.device, "--calibration", cal_out] + gate_args,
                    os.path.join(args.work_dir, f"evaluate-gptq-{gate_name}.log"))
        if not os.path.exists(report):
            raise SystemExit(f"error: no report for gate {gate_name}")
        with open(report, encoding="utf-8") as f:
            results[gate_name] = awq.extract_run(json.load(f))

    gptq = results["v1"]["perplexity"]
    proto["results"] = {
        "ceiling": ceiling, "rtn": rtn, "awq_runs": awq_runs, "awq_mean": awq_mean,
        "gptq": results["v1"], "gptq_gate_v2": results["v2"],
        "verdict": verdict(rtn, awq_mean, gptq, sigma),
        "gptq_minus_ceiling": gptq - ceiling,
    }
    proto["status"] = "measured"
    proto["pending"] = []
    proto["provenance"] = {"thme": args.thme, "device": args.device}
    with open(args.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(proto, f, indent=2)
        f.write("\n")
    print(json.dumps(proto["results"]["verdict"], indent=2))


if __name__ == "__main__":
    main()
