#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Offline tests for the AWQ calibration collector (THME-2204): canned `thme evaluate` reports,
no model and no CLI. Run with: python -m unittest scripts/test_run_awq_calibration.py
"""

import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_awq_calibration", HERE / "run_awq_calibration.py")
rac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rac)

PROTOCOL = json.loads(
    (HERE.parent / "docs" / "benchmarks" / "pythia-14m" / "data" / "awq_calibration.json").read_text(encoding="utf-8")
)


def canned(ppl, base=10.0, accepted=True):
    return {
        "thme_version": "0.0.0-test",
        "host": {"logical_cpus": 4},
        "evaluation": {
            "gate_passed": accepted,
            "variants": [
                {"id": "baseline", "perplexity": base, "delta_ppl": None},
                {"id": "recipe", "perplexity": ppl, "delta_ppl": ppl - base},
            ],
        },
    }


def all_runs(base=10.0):
    runs = {}
    for fmt in PROTOCOL["formats"]:
        runs[(fmt, "rtn", None)] = rac.extract_run(canned(50.0, base, False))
        for arm in PROTOCOL["arms"]:
            for seed in arm.get("seeds", []):
                runs[(fmt, arm["id"], seed)] = rac.extract_run(canned(20.0 + seed / 100, base))
    return runs


class CollectorTest(unittest.TestCase):
    def test_extract_reads_report_fields(self):
        r = rac.extract_run(canned(12.5, 10.0, False))
        self.assertEqual((r["baseline_perplexity"], r["perplexity"], r["gate_accepted"]), (10.0, 12.5, False))
        self.assertEqual(r["delta_ppl"], 2.5)

    def test_results_follow_result_format_and_leave_protocol_alone(self):
        results = rac.build_results(PROTOCOL, all_runs())
        merged = rac.merge(PROTOCOL, results, {"thme_version": "t"})
        for fmt in PROTOCOL["formats"]:
            block = merged["results"][fmt]
            self.assertEqual(block["ceiling"]["perplexity"], 10.0)
            self.assertEqual(block["rtn"]["status"], "measured")
            self.assertEqual([r["seed"] for r in block["awq"]["runs"]], [1234, 1, 2, 3, 4])
            self.assertEqual([r["seed"] for r in block["control_a"]["runs"]], [1, 2, 3, 4, 5])
            self.assertEqual(len(block["control_b2"]["runs"]), 1)
            self.assertEqual(block["awq"]["split"], "test")
        for key in ("preregistration", "arms", "calibration", "evaluation", "size_check", "formats"):
            self.assertEqual(merged[key], PROTOCOL[key])
        self.assertEqual(merged["status"], "measured")
        self.assertFalse(any(p.startswith("Measured perplexity") for p in merged["pending"]))

    def test_ceiling_must_be_one_value(self):
        runs = all_runs()
        runs[("Q4_0", "awq", 1)] = rac.extract_run(canned(20.0, base=11.0))
        with self.assertRaises(ValueError):
            rac.build_results(PROTOCOL, runs)

    def test_control_a_rule_and_hash_mismatch(self):
        rule = rac.awq_rule("Q4_0", [0.0, 1.0], permute_seed=3)
        self.assertEqual(rule["params"]["scales"], {"PermutedScales": {"seed": 3}})
        self.assertNotIn("scales", rac.awq_rule("Q4_0", [0.0])["params"])
        with self.assertRaises(SystemExit):
            rac.verify_hashes({"x": ("does-not-exist.u32", "0" * 64)})


if __name__ == "__main__":
    unittest.main()
