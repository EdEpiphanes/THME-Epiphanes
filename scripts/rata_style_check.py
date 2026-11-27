#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Re-judges the committed perplexities under a band rule shaped like a relative accuracy test (RATA):
the reference reading decides which limit applies.

  reference >= HIGH            : the difference may be at most 10 percent of the reference
  LOW <= reference < HIGH      : the difference may be at most 20 percent of the reference (assumed middle band)
  reference < LOW              : the difference may be at most an absolute number (swept: 0.45, 1.0, 2.0)

HIGH = 100 and LOW = 20 follow the owner's description; the middle-band 20 percent and the absolute limit
are NOT given and are assumptions here. The script reads only committed reports and prints a table.
Run: python scripts/rata_style_check.py
"""

import json

D = "docs/benchmarks/"
HIGH, LOW, REL_HIGH, REL_MID = 100.0, 20.0, 0.10, 0.20
ABS_SWEEP = (0.45, 1.0, 2.0)


def load(path):
    return json.load(open(D + path, encoding="utf-8"))


def band_limit(ref, abs_limit):
    if ref >= HIGH:
        return "high", REL_HIGH * ref
    if ref >= LOW:
        return "mid", REL_MID * ref
    return "low", abs_limit


def gate_v1(ref, cand):
    return cand - ref <= 0.45


def gate_v2(ref, cand):
    return cand - ref <= (0.45 if ref < 20.0 else 0.0225 * ref)


def rows():
    out = []
    for model, tag in (("pythia-14m", "Pythia-14M"), ("smollm-135m", "SmolLM-135M")):
        awq = load(f"{model}/data/awq_calibration.json")["results"]["Q4_0"]
        gptq = load(f"{model}/data/gptq_comparison.json")["results"]
        ref = awq["ceiling"]["perplexity"]
        awq_mean = sum(r["perplexity"] for r in awq["awq"]["runs"]) / len(awq["awq"]["runs"])
        out.append((tag, "Q4_0 round-to-nearest", ref, awq["rtn"]["perplexity"]))
        out.append((tag, "Q4_0 AWQ (mean of 5 seeds)", ref, awq_mean))
        out.append((tag, "Q4_0 GPTQ", ref, gptq["gptq"]["perplexity"]))
    q = load("smollm-135m/data/qad_lora.json")["results"]
    out.append(("SmolLM-135M", "Q4_0 + LoRA recovery (mean of 3 seeds)", load("smollm-135m/data/awq_calibration.json")["results"]["Q4_0"]["ceiling"]["perplexity"], q["mean_perplexity"]))
    # Q8_0 uniform exports, quoted in docs/quality-gate.md (gate v2 section)
    out.append(("SmolLM-135M", "Q8_0 uniform export", 19.80, 19.83))
    out.append(("Pythia-14M", "Q8_0 uniform export", 114.29, 115.12))
    return out


def main():
    print("| model | candidate | reference | candidate | difference | relative | gate v1 | gate v2 | band | rule-A=0.45 | A=1.0 | A=2.0 |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- | :--- | :--- | :--- |")
    changes = 0
    for model, name, ref, cand in rows():
        diff = cand - ref
        verdicts = []
        for a in ABS_SWEEP:
            band, limit = band_limit(ref, a)
            verdicts.append("pass" if abs(diff) <= limit else "fail")
        v1 = "pass" if gate_v1(ref, cand) else "fail"
        v2 = "pass" if gate_v2(ref, cand) else "fail"
        if v2 != verdicts[0]:
            changes += 1
        print(f"| {model} | {name} | {ref:.2f} | {cand:.2f} | {diff:+.2f} | {diff / ref * 100:+.1f}% | {v1} | {v2} | {band_limit(ref, 0.45)[0]} | "
              + " | ".join(verdicts) + " |")
    print()
    print(f"verdicts that differ between gate v2 and the band rule with A=0.45: {changes}")


if __name__ == "__main__":
    main()
