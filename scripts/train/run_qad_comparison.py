#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Runs the pre-registered QAD-LoRA comparison (docs/benchmarks/smollm-135m/data/qad_lora.json) and
fills its results block. Standard library only; it drives scripts/train/qad_lora.py.

Phases, in the order the protocol fixes:
  0  replication gate: a run with zero steps measures the unquantized teacher and the plain Q4_0
     student on the TEST file; both must match THME's committed numbers within 1 percent
  1  pilot: seed 0, each learning rate of the grid, scored on the VALIDATION file only
  2  scored runs: seeds 1, 2, 3 at the frozen learning rate, scored on the TEST file
  3  verdicts: the 2 x sigma rule against round-to-nearest, AWQ and GPTQ, the drift protocol, the gate

  python run_qad_comparison.py --python venv/python.exe --model-dir models/SmolLM-135M \
      --tokens-dir DIR --work-dir DIR --json docs/benchmarks/smollm-135m/data/qad_lora.json
"""

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("drift", os.path.join(HERE, "..", "drift.py"))
drift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drift)

GATE_MARGIN = 0.45


def sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_trainer(args, out_dir, seed, steps, lr, eval_file):
    result = os.path.join(out_dir, "result.json")
    if os.path.exists(result):  # resumable: a finished run is not repeated
        with open(result, encoding="utf-8") as f:
            return json.load(f)
    cmd = [args.python, os.path.join(HERE, "qad_lora.py"), "--model-dir", args.model_dir,
           "--train-tokens", os.path.join(args.tokens_dir, "qad_train_main.u32"), "--eval-tokens", eval_file,
           "--out-dir", out_dir, "--seed", str(seed), "--steps", str(steps), "--batch-size", "8",
           "--micro-batch", "2", "--window", "1024", "--rank", "16", "--alpha", "32", "--lr", str(lr), "--device", "cuda",
           "--teacher-dtype", "float32"]
    print("running", out_dir, flush=True)
    log = os.path.join(args.work_dir, os.path.basename(out_dir) + ".log")
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE).returncode
    if code != 0 or not os.path.exists(result):
        raise SystemExit(f"error: {out_dir} failed (exit {code}); see {log}")
    with open(result, encoding="utf-8") as f:
        return json.load(f)


def verdicts(mean, sigma, comparators):
    out = {}
    for name, value in comparators.items():
        gap = value - mean  # positive: QAD-LoRA is lower (better)
        out[name] = {"comparator_perplexity": value, "comparator_minus_qad": gap,
                     "verdict": "qad_better" if gap > 2 * sigma else "qad_worse" if -gap > 2 * sigma else "no_difference_shown"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    with open(args.json, encoding="utf-8") as f:
        proto = json.load(f)
    pre = proto["preregistration"]
    cmp_ = proto["comparators"]

    files = {"train": "qad_train_main.u32", "val": "qad_validation.u32"}
    want = {"train": proto["data"]["train"]["sha256"], "val": proto["data"]["validation"]["sha256"]}
    for key, name in files.items():
        if sha256(os.path.join(args.tokens_dir, name)) != want[key]:
            raise SystemExit(f"error: {name} does not match the pre-registered hash")
    test = os.path.join(args.tokens_dir, f"wikitext2_test.{proto['model_repo'].split('/')[-1]}.u32")
    if sha256(test) != proto["data"]["test"]["token_file_sha256"]:
        raise SystemExit("error: the test token file does not match the pre-registered hash")
    val = os.path.join(args.tokens_dir, files["val"])

    # Phase 0: replication gate
    rep = run_trainer(args, os.path.join(args.work_dir, "replication"), 0, 0, 0.0001, test)
    gate = {"teacher": rep["eval_ppl_teacher"], "student_q4_0": rep["eval_ppl_student_q4_0"],
            "teacher_error": drift.relative_bias(rep["eval_ppl_teacher"], cmp_["ceiling"]),
            "student_error": drift.relative_bias(rep["eval_ppl_student_q4_0"], cmp_["rtn_q4_0"])}
    gate["passed"] = abs(gate["teacher_error"]) <= 0.01 and abs(gate["student_error"]) <= 0.01
    proto["results"] = {"replication_gate": gate}
    if not gate["passed"]:
        proto["status"] = "replication_failed"
        proto["pending"] = ["replication gate failed: no verdict is given"]
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(proto, f, indent=2)
            f.write("\n")
        raise SystemExit(f"replication gate FAILED: {gate}")

    # Phase 1: pilot
    pilot = {}
    for lr in pre["pilot"]["learning_rates"]:
        r = run_trainer(args, os.path.join(args.work_dir, f"pilot-lr{lr}"), pre["pilot"]["seed"], 256, lr, val)
        pilot[str(lr)] = {"validation_perplexity": r["eval_ppl"], "validation_perplexity_q4_0": r["eval_ppl_student_q4_0"],
                          "first_loss": r["first_loss"], "last_loss": r["last_loss"]}
    chosen = min(pre["pilot"]["learning_rates"], key=lambda lr: (pilot[str(lr)]["validation_perplexity"], lr))
    proto["results"]["pilot"] = {"runs": pilot, "chosen_learning_rate": chosen}

    # Phase 2: scored runs
    scored = []
    for seed in pre["scored_runs"]["seeds"]:
        scored.append(run_trainer(args, os.path.join(args.work_dir, f"scored-seed{seed}"), seed, 256, chosen, test))
    ppls = [r["eval_ppl"] for r in scored]
    mean = statistics.mean(ppls)
    sd = statistics.stdev(ppls)
    sigma = max(sd, cmp_["awq_q4_0_sd"])

    # Phase 3: verdicts
    nominal = cmp_["ceiling"]
    b_pre = drift.relative_bias(scored[0]["eval_ppl_teacher"], nominal)
    b_post = drift.relative_bias(scored[-1]["eval_ppl_teacher"], nominal)
    stability = drift.accept(b_pre, b_post, ppls, 0.05)
    delta = mean - nominal
    proto["results"].update({
        "scored_runs": [{"seed": r["seed"], "perplexity": r["eval_ppl"], "teacher_perplexity": r["eval_ppl_teacher"],
                         "q4_0_perplexity": r["eval_ppl_student_q4_0"], "first_loss": r["first_loss"],
                         "last_loss": r["last_loss"], "adapter_bytes": r["adapter_bytes"],
                         "trainable_params": r["trainable_params"], "train_seconds": r["train_seconds"],
                         "tokens_seen": r["steps"] * r["batch_size"] * r["window"]} for r in scored],
        "mean_perplexity": mean, "sd_perplexity": sd, "sigma": sigma,
        "verdicts": verdicts(mean, sigma, {"rtn_q4_0": cmp_["rtn_q4_0"], "awq_q4_0_mean": cmp_["awq_q4_0_mean"],
                                           "gptq_q4_0": cmp_["gptq_q4_0"]}),
        "gate": {"delta_vs_ceiling": delta, "v1": "passed" if delta <= GATE_MARGIN else "failed",
                 "v2_note": "the baseline perplexity is below the switch point 20.0, so v2 uses the same absolute margin"},
        "drift": {"reference_nominal": nominal, "bias_pre": b_pre, "bias_post": b_post, "stability": stability,
                  "guard_band_verdict_v1": drift.guarded_verdict(delta, GATE_MARGIN, abs(stability["drift"]) * nominal)},
        "cost": {"adapter_bytes": scored[0]["adapter_bytes"], "trainable_params": scored[0]["trainable_params"],
                 "total_params": scored[0]["total_params"]},
    })
    proto["status"] = "measured" if stability["accepted"] else "unstable"
    proto["pending"] = [] if stability["accepted"] else ["batch flagged unstable by the drift protocol: no verdict is used"]
    proto["provenance"] = {"python": os.path.basename(args.python), "trainer": "scripts/train/qad_lora.py", "device": "cuda"}  # no local paths in committed reports
    with open(args.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(proto, f, indent=2)
        f.write("\n")
    print(json.dumps({"replication": gate, "chosen_lr": chosen, "mean": mean, "sd": sd, "sigma": sigma,
                      "verdicts": proto["results"]["verdicts"], "stability": stability["accepted"]}, indent=2))


if __name__ == "__main__":
    main()
