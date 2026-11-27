#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Re-judges the committed perplexities under the numbers printed in US EPA Performance Specification 16
(40 CFR Part 60 Appendix B, the version dated 3/29/2023), transposed to THME's terms:

  reading   = perplexity of the reference (the unquantized model on THME's reference path)
  PEMS      = the candidate (a quantized, recovered or exported model)
  RATA      = section 13.1: relative accuracy at most 10 percent above 100, at most 20 percent from 10 to 100,
              and below 10 an absolute mean difference of at most 2
  RAA       = section 13.5: 10 percent above 100, 20 percent from 20 to 100, absolute difference at most 2
              at 20 or less (the audit's text puts the absolute threshold at 20 where the RATA's says 10)

Only the summary means are used here. PS-16's statistic adds a confidence coefficient to the mean difference;
that needs paired per-window differences, which the committed reports do not hold, so it is NOT computed.
The relative limit is applied to the mean difference alone, which is therefore slightly more lenient than PS-16.
Run: python scripts/ps16_check.py
"""

import json

D = "docs/benchmarks/"


def load(path):
    return json.load(open(D + path, encoding="utf-8"))


def limit(ref, high, mid_floor, abs_limit):
    """Allowed absolute difference for a reference reading `ref` (None, band name, limit)."""
    if ref > high:
        return "high", 0.10 * ref
    if ref >= mid_floor:
        return "mid", 0.20 * ref
    return "low", abs_limit


RATA = dict(high=100.0, mid_floor=10.0, abs_limit=2.0)
RAA = dict(high=100.0, mid_floor=20.0, abs_limit=2.0)


def rows():
    out = []
    for model, tag in (("pythia-14m", "Pythia-14M"), ("smollm-135m", "SmolLM-135M")):
        awq = load(f"{model}/data/awq_calibration.json")["results"]["Q4_0"]
        gptq = load(f"{model}/data/gptq_comparison.json")["results"]
        ref = awq["ceiling"]["perplexity"]
        awq_mean = sum(r["perplexity"] for r in awq["awq"]["runs"]) / len(awq["awq"]["runs"])
        out += [(tag, "Q4_0 round-to-nearest", ref, awq["rtn"]["perplexity"]),
                (tag, "Q4_0 AWQ (mean of 5 seeds)", ref, awq_mean),
                (tag, "Q4_0 GPTQ", ref, gptq["gptq"]["perplexity"])]
    q = load("smollm-135m/data/qad_lora.json")["results"]
    ref = load("smollm-135m/data/awq_calibration.json")["results"]["Q4_0"]["ceiling"]["perplexity"]
    out += [("SmolLM-135M", "Q4_0 + LoRA recovery (mean of 3 seeds)", ref, q["mean_perplexity"]),
            ("SmolLM-135M", "Q8_0 uniform export", 19.80, 19.83),
            ("Pythia-14M", "Q8_0 uniform export", 114.29, 115.12)]
    return out


def main():
    print("| model | candidate | reference | candidate | difference | relative | gate v2 | PS-16 RATA band | RATA | PS-16 RAA band | RAA |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- | :--- | :--- |")
    for model, name, ref, cand in rows():
        diff = cand - ref
        v2 = "pass" if diff <= (0.45 if ref < 20.0 else 0.0225 * ref) else "fail"
        b1, l1 = limit(ref, **RATA)
        b2, l2 = limit(ref, **RAA)
        r1 = "pass" if abs(diff) <= l1 else "fail"
        r2 = "pass" if abs(diff) <= l2 else "fail"
        print(f"| {model} | {name} | {ref:.2f} | {cand:.2f} | {diff:+.2f} | {diff / ref * 100:+.1f}% | {v2} | {b1} (limit {l1:.2f}) | {r1} | {b2} (limit {l2:.2f}) | {r2} |")


if __name__ == "__main__":
    main()
