#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Paired block-bootstrap statistics for an S0 report (scripts/testbed_s0.py, THME-3110). Standard library plus numpy.

Every variant is scored on the same windows, so differences are paired. Windows are consecutive pieces of one corpus, so
the bootstrap resamples blocks of consecutive windows (default 10), not single windows.

  python scripts/testbed_s0_stats.py --json docs/benchmarks/epiph-testbed/data/s0_arm_a.json --out stats.json
  python scripts/testbed_s0_stats.py --selftest
"""

import argparse
import json

import numpy as np

PAIRS = [  # (regime under test, control at the same rate)
    ("phase_P4_a4_r2", "scalar3_control"),
    ("phase_P4_a4_r2", "lloyd3_control"),
    ("phase_P5_a3_r3", "lloyd3_control"),
    ("phase_P6_a5_r1", "lloyd3_control"),
    ("phase_P5_a3_r3", "phase_P4_a4_r2"),
    ("phase_P6_a5_r1", "phase_P4_a4_r2"),
    ("phase_P2_a3_r1", "scalar2_control"),
    ("ternary", "scalar_3level_control"),
]
MIDDLE = ("layer1", "layer2", "layer3", "layer4")


def resample_indices(n, block, reps, rng):
    """Moving-block bootstrap: `reps` index arrays of length n made of blocks of `block` consecutive windows."""
    starts_max = max(1, n - block + 1)
    nblocks = -(-n // block)
    starts = rng.integers(0, starts_max, size=(reps, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    return idx


def ci(values, level=0.95):
    lo, hi = np.percentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def analyse(rep, block=10, reps=2000, seed=1234):
    rng = np.random.default_rng(seed)
    tok = np.asarray(rep["window_tokens"], dtype=float)
    base = np.asarray(rep["baseline_window_nll"], dtype=float)
    table = {(r["scope"], r["regime"]): np.asarray(r["window_nll"], dtype=float) for r in rep["rows"]}
    idx = resample_indices(len(tok), block, reps, rng)

    def delta(sum_w, rows):  # extra cross-entropy per token (nats) over the resampled windows
        return (sum_w[rows].sum(1) - base[rows].sum(1)) / tok[rows].sum(1)

    full = np.arange(len(tok))[None, :]
    out = {"pairs": [], "gamma": [], "block": block, "reps": reps}
    scopes = sorted({s for s, _ in table})
    for test, ctl in PAIRS:
        for scope in scopes:
            if (scope, test) not in table or (scope, ctl) not in table:
                continue
            d = lambda rows: delta(table[(scope, test)], rows) - delta(table[(scope, ctl)], rows)
            point, boot = float(d(full)[0]), d(idx)
            lo, hi = ci(boot)
            verdict = "better" if hi < 0 else "worse" if lo > 0 else "unresolved"
            out["pairs"].append({"scope": scope, "test": test, "control": ctl, "delta_nats_per_token": point,
                                 "ci95": [lo, hi], "verdict": verdict})
    regimes = sorted({r for _, r in table})
    for regime in regimes:
        if ("middle_1_to_4", regime) not in table or any((s, regime) not in table for s in MIDDLE):
            continue
        def gamma(rows):
            joint = delta(table[("middle_1_to_4", regime)], rows)
            single = sum(delta(table[(s, regime)], rows) for s in MIDDLE)
            return joint / single
        point, boot = float(gamma(full)[0]), gamma(idx)
        lo, hi = ci(boot)
        out["gamma"].append({"regime": regime, "gamma": point, "ci95": [lo, hi]})
    return out


def selftest():
    rng = np.random.default_rng(0)
    n = 120
    tok = [1023] * n
    base = list(rng.normal(3000, 40, n))
    same = [b + 100 for b in base]
    better = [b + 50 for b in base]
    rows = [{"scope": "layer1", "regime": "a", "window_nll": same}, {"scope": "layer1", "regime": "b", "window_nll": same},
            {"scope": "layer1", "regime": "c", "window_nll": better}]
    rep = {"window_tokens": tok, "baseline_window_nll": base, "rows": rows}
    global PAIRS
    PAIRS = [("a", "b"), ("c", "a")]
    res = analyse(rep, reps=200)
    p = {(x["test"], x["control"]): x for x in res["pairs"]}
    assert p[("a", "b")]["delta_nats_per_token"] == 0 and p[("a", "b")]["ci95"] == [0.0, 0.0]
    assert p[("c", "a")]["verdict"] == "better" and p[("c", "a")]["delta_nats_per_token"] < 0
    idx = resample_indices(n, 10, 5, rng)
    assert idx.shape == (5, n) and idx.min() >= 0 and idx.max() < n
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--out")
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rep = json.load(open(a.json, encoding="utf-8"))
    res = analyse(rep, block=a.block)
    res["source"] = a.json.replace("\\", "/").split("/")[-1]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    for p in res["pairs"]:
        print(f'{p["scope"]:14} {p["test"]:16} vs {p["control"]:22} {p["delta_nats_per_token"]:+.4f} '
              f'[{p["ci95"][0]:+.4f}, {p["ci95"][1]:+.4f}] {p["verdict"]}')
    for g in res["gamma"]:
        print(f'gamma {g["regime"]:22} {g["gamma"]:.2f} [{g["ci95"][0]:.2f}, {g["ci95"][1]:.2f}]')


if __name__ == "__main__":
    main()
