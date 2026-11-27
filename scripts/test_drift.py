#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for scripts/drift.py. Expected values are worked by hand (the 100 ppm sensor example)."""

import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("drift", os.path.join(HERE, "drift.py"))
drift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drift)


class SensorExample(unittest.TestCase):
    def test_bias_is_relative_to_the_nominal_value(self):
        self.assertAlmostEqual(drift.relative_bias(101.0, 100.0), 0.01)
        self.assertAlmostEqual(drift.relative_bias(97.0, 100.0), -0.03)

    def test_first_run_is_a_warm_up_and_is_not_averaged(self):
        runs = [[200.0, 200.0], [100.0, 102.0], [98.0, 100.0]]
        self.assertEqual(drift.run_means(runs), [101.0, 99.0])
        self.assertEqual(drift.run_means(runs, skip_warmup=False), [200.0, 101.0, 99.0])
        self.assertEqual(drift.run_means([[5.0, 7.0]]), [6.0])  # n = 1 keeps the only run

    def test_spread_is_range_over_mean(self):
        self.assertAlmostEqual(drift.spread([98.0, 102.0, 100.0]), 0.04)
        self.assertAlmostEqual(drift.spread([94.0, 100.0, 106.0]), 0.12)

    def test_drift_during_the_batch_is_post_minus_pre(self):
        self.assertAlmostEqual(drift.batch_drift(0.01, 0.03), 0.02)

    def test_five_percent_example(self):
        ok = drift.accept(0.01, 0.03, [98.0, 102.0, 100.0], tolerance=0.05)
        self.assertTrue(ok["accepted"])
        self.assertEqual(ok["reasons"], [])
        bad = drift.accept(0.01, 0.03, [94.0, 100.0, 106.0], tolerance=0.05)
        self.assertFalse(bad["accepted"])
        self.assertEqual(bad["reasons"], ["spread 0.1200 exceeds 2 x tolerance 0.1000"])

    def test_a_large_reference_bias_rejects_the_batch(self):
        r = drift.accept(0.01, 0.07, [100.0, 100.0], tolerance=0.05)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reasons"], ["post-run bias 0.0700 exceeds tolerance 0.0500"])


class Correction(unittest.TestCase):
    def test_bias_is_interpolated_between_the_checks(self):
        self.assertAlmostEqual(drift.interpolated_bias(0.02, 0.04, 0.5), 0.03)
        self.assertAlmostEqual(drift.interpolated_bias(0.02, 0.04, 0.0), 0.02)
        self.assertAlmostEqual(drift.interpolated_bias(0.02, 0.04, 1.0), 0.04)

    def test_corrected_value_removes_the_bias(self):
        self.assertAlmostEqual(drift.corrected(103.0, 0.02, 0.04, 0.5), 100.0)


class GuardBand(unittest.TestCase):
    """Lower is better: the candidate's change must stay under the margin, drift counted against it."""

    def test_pass_only_if_it_passes_at_the_unfavourable_edge(self):
        self.assertEqual(drift.guarded_verdict(0.30, 0.45, 0.10), "pass")  # 0.40 <= 0.45

    def test_inconclusive_when_the_edges_disagree(self):
        self.assertEqual(drift.guarded_verdict(0.40, 0.45, 0.10), "inconclusive")  # 0.50 > 0.45, 0.30 <= 0.45

    def test_fail_when_it_fails_even_at_the_favourable_edge(self):
        self.assertEqual(drift.guarded_verdict(0.60, 0.45, 0.10), "fail")  # 0.50 > 0.45

    def test_no_drift_gives_the_plain_verdict(self):
        self.assertEqual(drift.guarded_verdict(0.44, 0.45, 0.0), "pass")
        self.assertEqual(drift.guarded_verdict(0.46, 0.45, 0.0), "fail")


class Attribution(unittest.TestCase):
    def test_pearson(self):
        self.assertAlmostEqual(drift.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(drift.pearson([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertEqual(drift.pearson([1, 1, 1], [1, 2, 3]), 0.0)  # no variance: no correlation claimed

    def test_factors_are_ranked_by_absolute_correlation(self):
        readings = [100.0, 102.0, 104.0, 106.0]
        factors = {"run_index": [1, 2, 3, 4], "gpu_temp": [60, 61, 59, 60], "other_processes": [3, 2, 1, 0]}
        ranked = drift.attribute(readings, factors)
        self.assertEqual([name for name, _ in ranked][:2], ["run_index", "other_processes"])
        self.assertAlmostEqual(ranked[0][1], 1.0)
        self.assertAlmostEqual(ranked[1][1], -1.0)


if __name__ == "__main__":
    unittest.main()
