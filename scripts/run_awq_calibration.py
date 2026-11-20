#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Runs the pre-registered AWQ calibration comparison (THME-2204) and fills the results block of
docs/benchmarks/<model>/data/awq_calibration.json. Standard library only.

For every format in the file it drives the `thme` CLI through the arms of the protocol:
ceiling and RTN (one run each), AWQ, control A (seed-1234 scales permuted with seeds 1..5),
control B1 (shuffled calibration tokens) and control B2 (uniform random ids). Every perplexity
and gate verdict is read from a `thme evaluate` report; nothing is computed or assumed here.
The protocol fields of the JSON (arms, seeds, alpha grid, decision rule, hashes) are never
modified; only status, results, pending and provenance are written.

Token files come from scripts/prepare_eval_tokens.py (see the sidecars and the hashes in the
JSON); every file hash is checked before the first run and a mismatch stops the script.

  py scripts/run_awq_calibration.py --model-dir models/pythia-14m \
      --tokens-dir target/awq-eval/pythia-14m --work-dir target/awq-run/pythia-14m \
      --json docs/benchmarks/pythia-14m/data/awq_calibration.json --thme target/release/thme
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

CONTROL_A_SEEDS = [1, 2, 3, 4, 5]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_files(protocol, tokens_dir):
    """Name -> (path, expected sha256) for every token file the protocol uses."""
    tag = protocol["model_repo"].split("/")[-1]
    cal = protocol["calibration"]
    stem = f"wikitext2_train.{tag}.cal{cal['windows']}x{cal['window_len']}"
    files = {"test": (f"wikitext2_test.{tag}.u32", protocol["evaluation"]["token_file_sha256"])}
    for entry in cal["files"]:
        files[f"cal-{entry['seed']}"] = (f"{stem}.s{entry['seed']}.u32", entry["token_file_sha256"])
    for entry in cal["shuffled_files"]:
        files[f"shuf-{entry['seed']}"] = (f"{stem}.s{entry['seed']}.shuffled.u32", entry["token_file_sha256"])
    rnd = cal["random_ids"]
    vocab = rnd["random_ids"]["vocab_size"]
    files["rand"] = (f"{stem}.s{rnd['seed']}.randomids.v{vocab}.u32", rnd["token_file_sha256"])
    return {k: (os.path.join(tokens_dir, name), sha) for k, (name, sha) in files.items()}


def verify_hashes(files):
    """Raises SystemExit naming every file that is missing or whose hash differs."""
    bad = []
    for key, (path, expected) in files.items():
        if not os.path.exists(path):
            bad.append(f"{key}: missing {path}")
        elif sha256_file(path) != expected:
            bad.append(f"{key}: hash differs for {path}")
    if bad:
        raise SystemExit("error: token files do not match the pre-registered hashes:\n  " + "\n  ".join(bad))


def awq_rule(fmt, alpha_grid, permute_seed=None):
    params = {"format": fmt, "alpha_grid": alpha_grid}
    if permute_seed is not None:
        params["scales"] = {"PermutedScales": {"seed": permute_seed}}
    return {"type": "ActivationAware", "params": params}


def algorithm_definition(algorithm_id, fmt, alpha_grid, permute_seed=None):
    roles = ["AttentionQ", "AttentionK", "AttentionV", "AttentionQkv", "AttentionOut",
             "MlpGate", "MlpUp", "MlpDown"]
    return {
        "id": algorithm_id, "name": algorithm_id, "author": "Epiphanes", "version": "1.0.0",
        "category": "Quantization",
        "description": f"AWQ-style {fmt} for the THME-2204 comparison"
                       + ("" if permute_seed is None else f" (control A, permutation seed {permute_seed})"),
        "effective_bits_per_param": 4.5, "compression_ratio": 16.0 / 4.5, "hardware_speedup_factor": 1.0,
        "target_roles": roles, "rule": awq_rule(fmt, alpha_grid, permute_seed),
    }


def recipe(name, model, layers, operation):
    return {
        "name": name, "author": "Epiphanes", "target_model": model, "version": "1.0",
        "description": "THME-2204 AWQ calibration comparison arm",
        "operations": [{"type": operation[0], "params": {"start_layer": 0, "end_layer": layers - 1,
                                                          "role": None, **operation[1]}}],
    }


def extract_run(report):
    """The numbers one `thme evaluate` report contributes (all read, none computed)."""
    variants = {v["id"]: v for v in report["evaluation"]["variants"]}
    return {
        "baseline_perplexity": variants["baseline"]["perplexity"],
        "perplexity": variants["recipe"]["perplexity"],
        "delta_ppl": variants["recipe"]["delta_ppl"],
        "gate_accepted": report["evaluation"]["gate_passed"],
        "thme_version": report.get("thme_version"),
        "logical_cpus": report.get("host", {}).get("logical_cpus"),
    }


def build_results(protocol, runs):
    """Results block in the documented result_format. `runs` maps (format, arm, seed) -> extract_run."""
    bench = protocol["evaluation"]["benchmark"]
    meta = {"benchmark_id": bench["id"], "benchmark_version": bench["version"], "split": bench["split"]}
    results = {}
    for fmt in protocol["formats"]:
        ceilings = {r["baseline_perplexity"] for (f, _, _), r in runs.items() if f == fmt}
        if len(ceilings) != 1:
            raise ValueError(f"{fmt}: ceiling is not one value across runs: {sorted(ceilings)}")
        block = {"ceiling": {"status": "measured", "perplexity": ceilings.pop(), **meta}}
        rtn = runs[(fmt, "rtn", None)]
        block["rtn"] = {"status": "measured", "perplexity": rtn["perplexity"], **meta}
        for arm in protocol["arms"]:
            if "seeds" not in arm:
                continue
            seeded = []
            for seed in arm["seeds"]:
                r = runs[(fmt, arm["id"], seed)]
                # alpha is chosen per tensor inside the CLI and is not reported: null, not guessed.
                seeded.append({"seed": seed, "perplexity": r["perplexity"], "alpha": None,
                               "delta_ppl": r["delta_ppl"], "gate_accepted": r["gate_accepted"]})
            block[arm["id"]] = {"status": "measured", **meta, "seeds": arm["seeds"], "runs": seeded}
        results[fmt] = block
    return results


def merge(protocol, results, provenance):
    """Copy of `protocol` with only the result-bearing fields replaced."""
    out = json.loads(json.dumps(protocol))
    out["results"] = results
    out["status"] = "measured"
    out["status_note"] = ("Every arm was measured on the full WikiText-2 test split by the reference CPU "
                          "forward pass. The decision rule was applied as pre-registered.")
    out["pending"] = [p for p in protocol["pending"]
                      if not p.startswith(("Measured perplexity", "A collector"))]
    out["provenance"] = provenance
    return out


def run(cmd, log):
    with open(log, "w", encoding="utf-8") as f:
        # A failed quality gate exits non-zero but still writes the report; the caller checks for it.
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def drive(args):
    with open(args.json, encoding="utf-8") as f:
        protocol = json.load(f)
    files = token_files(protocol, args.tokens_dir)
    verify_hashes(files)
    os.makedirs(args.work_dir, exist_ok=True)
    grid = protocol["preregistration"]["alpha_grid"]
    layers = protocol["model_config"]["num_hidden_layers"]
    model = protocol["model_name"]
    cal = protocol["calibration"]
    context, max_tokens = str(cal["window_len"]), str(cal["tokens"])

    def path(name):
        return os.path.join(args.work_dir, name)

    def write_json(name, value):
        with open(path(name), "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
        return path(name)

    def calibrate(key):
        out = path(f"calibration-{key}.json")
        if not os.path.exists(out):
            code = run([args.thme, "calibrate", args.model_dir, "--tokens", files[key][0], "--context", context,
                        "--max-tokens", max_tokens, "--out", out], path(f"calibrate-{key}.log"))
            if code != 0 or not os.path.exists(out):
                raise SystemExit(f"error: calibrate {key} failed (see {path(f'calibrate-{key}.log')})")
        return out

    def evaluate(label, recipe_file, calibration=None, algorithms=None):
        report = path(f"report-{label}.json")
        if not os.path.exists(report):
            cmd = [args.thme, "evaluate", args.model_dir, "--recipe", recipe_file, "--tokens", files["test"][0],
                   "--report", report]
            if calibration:
                cmd += ["--calibration", calibration]
            if algorithms:
                cmd += ["--algorithms", algorithms]
            run(cmd, path(f"evaluate-{label}.log"))
            if not os.path.exists(report):
                raise SystemExit(f"error: evaluate {label} wrote no report (see {path(f'evaluate-{label}.log')})")
        with open(report, encoding="utf-8") as f:
            return extract_run(json.load(f))

    runs = {}
    for fmt in protocol["formats"]:
        low = fmt.lower()
        rtn_recipe = write_json(f"recipe-rtn-{low}.json", recipe(
            f"rtn-{low}", model, layers, ("SetPrecision", {"format": fmt})))
        runs[(fmt, "rtn", None)] = evaluate(f"{low}-rtn", rtn_recipe)
        print(f"{model} {fmt} rtn: {runs[(fmt, 'rtn', None)]['perplexity']}", flush=True)
        awq_id = f"thme.awq_{low}"
        awq_recipe = write_json(f"recipe-awq-{low}.json", recipe(
            f"awq-{low}", model, layers, ("ApplyCustomAlgorithm", {"algorithm_id": awq_id})))
        for arm in protocol["arms"]:
            if "seeds" not in arm:
                continue
            for seed in arm["seeds"]:
                label = f"{low}-{arm['id']}-{seed}"
                if arm["id"] == "awq":
                    r = evaluate(label, awq_recipe, calibrate(f"cal-{seed}"))
                elif arm["id"] == "control_b1":
                    r = evaluate(label, awq_recipe, calibrate(f"shuf-{seed}"))
                elif arm["id"] == "control_b2":
                    r = evaluate(label, awq_recipe, calibrate("rand"))
                elif arm["id"] == "control_a":
                    cid = f"{awq_id}_control_a_s{seed}"
                    algo = write_json(f"algorithm-{low}-control-a-{seed}.json",
                                      algorithm_definition(cid, fmt, grid, seed))
                    ca_recipe = write_json(f"recipe-control-a-{low}-{seed}.json", recipe(
                        f"control-a-{low}-{seed}", model, layers,
                        ("ApplyCustomAlgorithm", {"algorithm_id": cid})))
                    r = evaluate(label, ca_recipe, calibrate("cal-1234"), algo)
                else:
                    raise SystemExit(f"error: unknown seeded arm {arm['id']}")
                runs[(fmt, arm["id"], seed)] = r
                print(f"{model} {fmt} {arm['id']} seed {seed}: {r['perplexity']}", flush=True)

    first = next(iter(runs.values()))
    provenance = {
        "thme_version": first["thme_version"],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None,
        "threads": first["logical_cpus"],
        "hardware": f"{platform.system()} {platform.machine()}, {platform.processor()}; "
                    "THME reference CPU forward pass (f32)",
        "note": "Recorded by scripts/run_awq_calibration.py. Alpha is chosen per tensor inside the CLI and is not reported.",
    }
    merged = merge(protocol, build_results(protocol, runs), provenance)
    with open(args.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    print(f"wrote {args.json}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Run the pre-registered AWQ calibration comparison")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tokens-dir", required=True, help="Output directory of scripts/prepare_eval_tokens.py")
    p.add_argument("--work-dir", required=True, help="Scratch directory for recipes, statistics and reports (resumable)")
    p.add_argument("--json", required=True, help="awq_calibration.json to fill")
    p.add_argument("--thme", default="target/release/thme")
    sys.exit(drive(p.parse_args()))


if __name__ == "__main__":
    main()
