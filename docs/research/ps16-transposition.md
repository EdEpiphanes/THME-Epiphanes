# US EPA Performance Specification 16 transposed to THME's gate

27 November 2026. The owner pointed to US EPA Performance Specification 16 (PS-16, predictive emissions
monitoring systems, 40 CFR Part 60 Appendix B) and its relative accuracy audits as the model for how
THME should set tolerances, "transposed in our own terms", which is what the Epiphanes math does. This
note records what PS-16 says, how each piece maps to THME, what the committed results look like under
its numbers, and what is not computed yet. It replaces the assumptions in
`docs/research/rata-style-tolerance-check.md`, which guessed the bands before the text was read.

## What PS-16 says (section numbers of the version dated 3/29/2023)

A PEMS predicts an emission from process inputs; a reference method (RM) measures it. The PEMS is
accepted if it agrees with the RM.

| Section | Rule |
| :--- | :--- |
| 3.10 | Relative accuracy (RA) is the average difference between PEMS and RM over a number of comparison runs, plus a 2.5 percent confidence coefficient, divided by the average of the RM runs |
| 12.2, Eq. 16-1 to 16-4 | mean of the differences `d`, their standard deviation `Sd`, confidence coefficient `cc = t(0.025) * Sd / sqrt(n)`, and `RA = (|d| + cc) / RM mean` (Eq. 16-4 itself was an image in the extracted text; the formula is the one section 3.10 states) |
| 13.1 RATA | RA at most **10 percent** if measurements are above 100 ppm; at most **20 percent** between 100 and 10 ppm; below 10 ppm the absolute mean difference at most **2 ppm**. Diluent-only PEMS may use plus or minus 1 percent absolute |
| 13.5 RAA (quarterly audit, three runs) | the average of three runs must not differ from the PEMS by more than 10 percent of the RM above 100 ppm, 20 percent between 100 and 10 ppm, and at **20 ppm or less** the difference must not exceed 2 ppm |
| 13.2 | **Bias**: if the mean difference is greater than the absolute value of the confidence coefficient, the PEMS is biased and must be corrected with a bias factor |
| 13.3 | **Variance**: the F-value must not exceed the critical F at the 95 percent level |
| 13.4 | **Correlation**: r must be at least 0.8 |
| 12.1 | when the average emission is below 50 percent of the standard, the standard replaces the RM mean in the denominator |
| 12.3 | the statistical tests are waived at a level whose RM mean is below 10 ppm or 5 percent of the standard |
| 8.2.2, 8.2.3 | compliance PEMS: nine runs at each of three operating levels (low, mid, high); up to three runs per level may be rejected if nine remain; every level must pass |
| 9.1 | sensor evaluation daily; an RAA each quarter except the quarter with a RATA; a RATA once a year |

Note that the RATA's absolute band starts at 10 ppm and the RAA's at 20 ppm. That is the "20" the owner
remembers for the absolute difference.

## The mapping to THME

| PS-16 | THME |
| :--- | :--- |
| PEMS | the candidate: a quantized, recovered or exported model, or the same model on another runtime |
| reference method | the unquantized model on THME's reference path (or the CPU result) |
| a paired comparison run | one evaluation window (1,024 tokens): the candidate's loss and the reference's loss on the same window |
| `d_i`, `d`, `Sd`, `cc` | the paired per-window loss differences, their mean, spread and confidence term. The Epiphanes gate already works on these per-block differences; its anytime-valid upper bound `U` plays the role of `|d| + cc` without needing a fixed run count |
| 2.5 percent confidence coefficient | the gate's confidence level `eta` and its running intersection |
| RA limit (10 / 20 percent, absolute) | the gate's tolerance `tau`, chosen by band from the reference reading |
| bias test | whether the candidate's mean loss difference is larger than its own confidence term (a real shift, not noise) |
| F-test | whether the candidate's per-window loss variance differs from the reference's |
| correlation r >= 0.8 | correlation of candidate and reference per-window losses |
| operating levels (low, mid, high) | groups of windows by the reference's own loss, or by context length, each of which must pass |
| RAA (three runs, quarterly) | a short repeat check after a change, on a handful of windows |

One caution on units. PS-16's bands are in ppm of a pollutant. A perplexity is not a concentration, so
the numbers 100, 20 and 10 carry over only by agreement (the owner's gate v2 already switches at 20).
Perplexities also differ by model and corpus: 19.8 for SmolLM-135M, 114.3 for Pythia-14M.

## The committed results under PS-16's numbers

From `scripts/ps16_check.py`, which reads committed reports only. The relative limit is applied to the mean
difference alone; PS-16 adds the confidence coefficient, so this is slightly more lenient than PS-16.

| model | candidate | reference | candidate | difference | relative | gate v2 | PS-16 RATA band | RATA | PS-16 RAA band | RAA |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- | :--- | :--- |
| Pythia-14M | Q4_0 round-to-nearest | 114.29 | 193.00 | +78.72 | +68.9% | fail | high (limit 11.43) | fail | high (limit 11.43) | fail |
| Pythia-14M | Q4_0 AWQ (mean of 5 seeds) | 114.29 | 170.70 | +56.42 | +49.4% | fail | high (limit 11.43) | fail | high (limit 11.43) | fail |
| Pythia-14M | Q4_0 GPTQ | 114.29 | 180.55 | +66.26 | +58.0% | fail | high (limit 11.43) | fail | high (limit 11.43) | fail |
| SmolLM-135M | Q4_0 round-to-nearest | 19.80 | 23.70 | +3.90 | +19.7% | fail | mid (limit 3.96) | pass | low (limit 2.00) | fail |
| SmolLM-135M | Q4_0 AWQ (mean of 5 seeds) | 19.80 | 22.04 | +2.24 | +11.3% | fail | mid (limit 3.96) | pass | low (limit 2.00) | fail |
| SmolLM-135M | Q4_0 GPTQ | 19.80 | 22.23 | +2.43 | +12.3% | fail | mid (limit 3.96) | pass | low (limit 2.00) | fail |
| SmolLM-135M | Q4_0 + LoRA recovery (mean of 3 seeds) | 19.80 | 20.41 | +0.61 | +3.1% | fail | mid (limit 3.96) | pass | low (limit 2.00) | pass |
| SmolLM-135M | Q8_0 uniform export | 19.80 | 19.83 | +0.03 | +0.2% | pass | mid (limit 3.96) | pass | low (limit 2.00) | pass |
| Pythia-14M | Q8_0 uniform export | 114.29 | 115.12 | +0.83 | +0.7% | pass | high (limit 11.43) | pass | high (limit 11.43) | pass |

## Reading

- Above 100 (Pythia-14M) the 10 percent band reproduces gate v2 on every result: round-to-nearest, AWQ and
  GPTQ at +49 to +69 percent fail; the Q8_0 export at +0.7 percent passes.
- Below 100 (SmolLM-135M) the answer depends on where the absolute band starts. With the RATA wording
  (absolute below 10), SmolLM sits in the 20 percent band and even plain round-to-nearest Q4_0 (+19.7
  percent) passes; that is too lenient to say anything about model quality. With the audit wording
  (absolute at 20 or less) the limit is 2 and only the recovered model (+0.61) and Q8_0 pass; AWQ at
  +2.24 narrowly fails.
- PS-16 chose its thresholds for emissions, where the absolute band exists to stop a tiny reading from
  failing on a trivial relative difference. For perplexity the same logic gives the audit's 20.
- Gate v2 (absolute 0.45 below 20, 2.25 percent above) is stricter than PS-16 in both bands.

## What is not computed yet

- The full PS-16 statistic, `(|d| + cc) / mean`, needs per-window paired differences. The committed reports
  hold only means. Producing them is cheap (about 300 windows per model, a minute on the GPU) and is what
  the Epiphanes gate consumes anyway.
- The bias, variance and correlation tests, and the three operating levels.
- Whether 2 is the right absolute limit in perplexity units. PS-16's 2 ppm is an emission figure.

## Decisions needed from the owner

1. Which absolute threshold: the RATA's (10) or the audit's (20)? The audit's matches the owner's recollection
   and keeps the middle band from passing plain round-to-nearest.
2. The absolute limit in perplexity units (PS-16's 2 is a concentration), and whether it should scale with the
   reference.
3. Whether the gate should adopt PS-16's tests beyond the accuracy limit (bias, variance, correlation) and
   the three operating levels.
4. Whether a separate, tighter tolerance stays for run-to-run stability and for replication checks, as the
   earlier note argued (PS-16 itself has a sensor check every day apart from the accuracy tests).
