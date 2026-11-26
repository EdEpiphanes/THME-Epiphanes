#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Tests for the AWQ calibration section of the dossier (THME-2204).

The pre-registered rule is checked on synthetic numbers only; no measured result is
invented. The pending report committed under docs/benchmarks is the schema anchor.
Run with: python -m unittest scripts/test_awq_section.py
"""

import copy
import importlib.util
import json
import statistics
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEEDS = [1234, 1, 2, 3, 4]
CONTROL_A_SEEDS = [1, 2, 3, 4, 5]

spec = importlib.util.spec_from_file_location("generate_dossier", HERE / "generate_dossier.py")
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)


def pending_report():
    path = REPO / "docs" / "benchmarks" / "pythia-14m" / "data" / "awq_calibration.json"
    return json.loads(path.read_text(encoding="utf-8"))


def seeded(values, seeds):
    return {
        "status": "measured",
        "benchmark_id": "wikitext2-raw",
        "benchmark_version": "1",
        "split": "test",
        "runs": [{"seed": s, "perplexity": v, "alpha": 0.5} for s, v in zip(seeds, values)],
    }


def single(value):
    return {
        "status": "measured",
        "benchmark_id": "wikitext2-raw",
        "benchmark_version": "1",
        "split": "test",
        "perplexity": value,
    }


def measured_report(awq, a, b1, rtn=30.0, ceiling=28.0):
    report = copy.deepcopy(pending_report())
    report["status"] = "measured"
    block = {
        "ceiling": single(ceiling),
        "rtn": single(rtn),
        "awq": seeded(awq, SEEDS),
        "control_a": seeded(a, CONTROL_A_SEEDS),
        "control_b1": seeded(b1, SEEDS),
        "control_b2": seeded([rtn + 1.0], [1234]),
    }
    report["results"] = {"Q4_0": copy.deepcopy(block), "Q4_K": copy.deepcopy(block)}
    return report


class SampleStatsTest(unittest.TestCase):
    def test_sample_sd_uses_n_minus_one(self):
        stats = gd.awq_stats([1.0, 2.0, 3.0])
        self.assertAlmostEqual(stats["sd"], 1.0)
        self.assertEqual((stats["n"], stats["min"], stats["max"]), (3, 1.0, 3.0))


class PreRegisteredRuleTest(unittest.TestCase):
    AWQ = [20.0, 21.0, 19.0, 20.5, 20.2]
    CONTROL_A = [28.0, 29.0, 27.0, 28.5, 28.1]
    CONTROL_B1 = [27.0, 28.0, 26.0, 27.5, 27.2]

    def evaluate(self, awq=AWQ, a=CONTROL_A, b1=CONTROL_B1, rtn=30.0, ceiling=28.0):
        report = measured_report(awq, a, b1, rtn=rtn, ceiling=ceiling)
        return gd.awq_evaluate(report["results"]["Q4_0"], ceiling=ceiling, rtn=rtn)

    def test_pooled_sd_is_the_root_mean_variance_of_the_three_seeded_arms(self):
        ev = self.evaluate()
        expected = (
            statistics.variance(self.AWQ)
            + statistics.variance(self.CONTROL_A)
            + statistics.variance(self.CONTROL_B1)
        ) / 3
        self.assertAlmostEqual(ev["pooled_sd"], expected**0.5)

    def test_awq_wins_when_gap_exceeds_twice_pooled_sd_and_every_seed_agrees(self):
        ev = self.evaluate()
        self.assertGreater(ev["gap"], 2 * ev["pooled_sd"])
        self.assertTrue(ev["all_seeds_below_rtn"])
        self.assertTrue(ev["awq_wins"])

    def test_one_seed_against_the_sign_is_not_a_win(self):
        ev = self.evaluate(awq=[20.0, 21.0, 19.0, 31.0, 20.2])
        self.assertGreater(ev["gap"], 2 * ev["pooled_sd"], "the magnitude alone would pass")
        self.assertFalse(ev["all_seeds_below_rtn"])
        self.assertFalse(ev["awq_wins"])

    def test_a_gap_inside_twice_pooled_sd_is_not_a_win(self):
        awq = [24.0, 29.9, 25.0, 29.5, 28.0]
        ev = self.evaluate(awq=awq)
        self.assertTrue(ev["all_seeds_below_rtn"])
        self.assertLess(ev["gap"], 2 * ev["pooled_sd"])
        self.assertFalse(ev["awq_wins"])

    def test_gate_compares_the_mean_with_the_ceiling(self):
        self.assertTrue(self.evaluate(awq=[28.0, 28.2, 27.9, 28.1, 28.0])["gate_pass"])
        self.assertFalse(self.evaluate(awq=[29.0, 29.1, 28.9, 29.0, 29.0])["gate_pass"])

    def test_control_is_beaten_only_by_the_paired_rule(self):
        ev = self.evaluate()
        self.assertTrue(ev["controls"]["control_a"]["awq_beats_control"])
        self.assertTrue(ev["controls"]["control_b1"]["awq_beats_control"])
        # Control A at seed position 3 is below AWQ there: the pairing fails.
        crossed = self.evaluate(a=[28.0, 29.0, 18.5, 28.5, 28.1])
        self.assertFalse(crossed["controls"]["control_a"]["awq_beats_control"])

    def test_missing_seeds_are_an_error(self):
        report = measured_report(self.AWQ[:4], self.CONTROL_A, self.CONTROL_B1)
        with self.assertRaises(ValueError):
            gd.awq_evaluate(report["results"]["Q4_0"], ceiling=28.0, rtn=30.0)


class SectionRenderingTest(unittest.TestCase):
    def test_pending_report_renders_as_pending_with_the_rule_and_no_verdict(self):
        text = "\n".join(gd.awq_section(pending_report(), "H"))
        self.assertIn("PENDING", text)
        self.assertIn("2 x the pooled SD", text)
        self.assertIn("C4", text)
        self.assertNotIn("AWQ beats round-to-nearest:", text)

    def test_measured_report_renders_the_verdict_and_control_rows(self):
        report = measured_report(
            [20.0, 21.0, 19.0, 20.5, 20.2],
            [28.0, 29.0, 27.0, 28.5, 28.1],
            [27.0, 28.0, 26.0, 27.5, 27.2],
        )
        text = "\n".join(gd.awq_section(report, "H"))
        self.assertIn("AWQ beats round-to-nearest: yes", text)
        self.assertIn("| Control A", text)
        self.assertIn("| Control B1", text)
        self.assertIn("| Control B2 (stress", text)
        self.assertIn("pooled SD", text)

    def test_partly_measured_report_is_refused(self):
        report = measured_report(
            [20.0, 21.0, 19.0, 20.5, 20.2],
            [28.0, 29.0, 27.0, 28.5, 28.1],
            [27.0, 28.0, 26.0, 27.5, 27.2],
        )
        report["results"]["Q4_K"]["rtn"] = {"status": "pending"}
        with self.assertRaises(ValueError):
            gd.awq_section(report, "H")


if __name__ == "__main__":
    unittest.main()
