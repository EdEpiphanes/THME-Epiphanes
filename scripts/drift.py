#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Measurement drift helpers (docs/research/measurement-drift-protocol.md). Standard library only.

A reading is a measured value; a run is one repeat; the reference check is a measurement whose true
value is known. These functions compute bias, drift and spread, decide whether a batch is inside its
tolerance, apply a guard band to a gate verdict, and rank factors that may explain a deviation.
"""

import math


def relative_bias(reading, nominal):
    """(reading - nominal) / nominal."""
    return (reading - nominal) / nominal


def run_means(runs, skip_warmup=True):
    """Mean of every run; the first run is a warm-up and is left out when there is more than one."""
    chosen = runs[1:] if skip_warmup and len(runs) > 1 else runs
    return [sum(r) / len(r) for r in chosen]


def spread(means):
    """(max - min) / mean of the run means."""
    return (max(means) - min(means)) / (sum(means) / len(means))


def batch_drift(bias_pre, bias_post):
    """Change of the reference bias over the batch."""
    return bias_post - bias_pre


def accept(bias_pre, bias_post, means, tolerance):
    """Rule 3 of the protocol: both biases within the tolerance, spread within twice it."""
    reasons = []
    if abs(bias_pre) > tolerance:
        reasons.append(f"pre-run bias {abs(bias_pre):.4f} exceeds tolerance {tolerance:.4f}")
    if abs(bias_post) > tolerance:
        reasons.append(f"post-run bias {abs(bias_post):.4f} exceeds tolerance {tolerance:.4f}")
    s = spread(means)
    if s > 2 * tolerance + 1e-12:
        reasons.append(f"spread {s:.4f} exceeds 2 x tolerance {2 * tolerance:.4f}")
    return {"accepted": not reasons, "reasons": reasons, "spread": s,
            "drift": batch_drift(bias_pre, bias_post)}


def interpolated_bias(bias_pre, bias_post, fraction):
    """Bias at a point of the batch, `fraction` of the way from the first check to the second."""
    return bias_pre + (bias_post - bias_pre) * fraction


def corrected(value, bias_pre, bias_post, fraction):
    """The value with the interpolated reference bias removed (reported next to the raw value)."""
    return value / (1.0 + interpolated_bias(bias_pre, bias_post, fraction))


def guarded_verdict(delta, margin, drift_abs):
    """Lower is better. Pass only if the change stays under the margin at the unfavourable edge of the
    drift band; fail only if it is over the margin even at the favourable edge; otherwise inconclusive."""
    if delta + drift_abs <= margin:
        return "pass"
    if delta - drift_abs > margin:
        return "fail"
    return "inconclusive"


def pearson(xs, ys):
    """Pearson correlation; 0.0 when either side has no variance (no correlation is claimed)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def attribute(readings, factors):
    """Factors ranked by the absolute correlation of their values with the readings. A pointer for the
    next experiment, not a cause."""
    ranked = [(name, pearson(values, readings)) for name, values in factors.items()]
    return sorted(ranked, key=lambda item: -round(abs(item[1]), 12))
