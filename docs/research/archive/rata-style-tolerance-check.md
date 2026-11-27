# Band tolerances shaped like a relative accuracy test (RATA): check against the committed results

27 November 2026. SUPERSEDED the same day by `docs/research/ps16-transposition.md`, which reads the actual US EPA Performance Specification 16 text (the bands below were guessed before that). Produced by `scripts/rata_style_check.py` from committed reports only.

## The rule that was checked

Owner description (the standard's text was not consulted): the reference reading decides the limit.
Above 100 the difference may be at most 10 percent of the reference; below about 20 the test is an
absolute difference. The middle band (20 to 100) is not stated; 20 percent relative is assumed here, and
the absolute limit is swept (0.45, the current gate's value, then 1.0 and 2.0) because it was not given.

## Result

| model | candidate | reference | candidate | difference | relative | gate v1 | gate v2 | band | rule-A=0.45 | A=1.0 | A=2.0 |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| Pythia-14M | Q4_0 round-to-nearest | 114.29 | 193.00 | +78.72 | +68.9% | fail | fail | high | fail | fail | fail |
| Pythia-14M | Q4_0 AWQ (mean of 5 seeds) | 114.29 | 170.70 | +56.42 | +49.4% | fail | fail | high | fail | fail | fail |
| Pythia-14M | Q4_0 GPTQ | 114.29 | 180.55 | +66.26 | +58.0% | fail | fail | high | fail | fail | fail |
| SmolLM-135M | Q4_0 round-to-nearest | 19.80 | 23.70 | +3.90 | +19.7% | fail | fail | low | fail | fail | fail |
| SmolLM-135M | Q4_0 AWQ (mean of 5 seeds) | 19.80 | 22.04 | +2.24 | +11.3% | fail | fail | low | fail | fail | fail |
| SmolLM-135M | Q4_0 GPTQ | 19.80 | 22.23 | +2.43 | +12.3% | fail | fail | low | fail | fail | fail |
| SmolLM-135M | Q4_0 + LoRA recovery (mean of 3 seeds) | 19.80 | 20.41 | +0.61 | +3.1% | fail | fail | low | fail | pass | pass |
| SmolLM-135M | Q8_0 uniform export | 19.80 | 19.83 | +0.03 | +0.2% | pass | pass | low | pass | pass | pass |
| Pythia-14M | Q8_0 uniform export | 114.29 | 115.12 | +0.83 | +0.7% | fail | pass | high | pass | pass | pass |

verdicts that differ between gate v2 and the band rule with A=0.45: 0

## Reading

- With the absolute limit at 0.45, the band rule gives the same verdict as gate v2 on all nine results.
  Gate v2 is not stricter than a 10 percent high band in any case we have: the only result between 2.25
  and 10 percent would be one that does not exist yet.
- The high band only changes anything if a candidate on a model with perplexity above 100 lands between
  +2.25 and +10 percent. Round-to-nearest, AWQ and GPTQ on Pythia-14M are +49 to +69 percent and fail either way.
- The absolute limit below 20 is what matters: the recovered SmolLM-135M (+0.61 over 19.80) fails at 0.45
  and passes at 1.0 or 2.0. That value is still to be set by the owner.
- The middle band (20 to 100) is not exercised by any committed full-test result. It would be, for example,
  SmolLM-135M at 256-token windows (about 30).
- Different purposes need different tolerances, and the figures above are about the third one only:
  1. Stability of a measurement (the drift protocol, 5 percent): the noise seen is far smaller (the three QAD
     seeds differ by 0.09 percent; deterministic CPU runs repeat exactly), so a wider band here would let
     unstable batches through without changing any verdict.
  2. Replication of a number (the QAD replication gate, 1 percent): the PyTorch result matched THME's to
     2e-6. A 10 percent band would accept a wrong replica, because round-to-nearest and AWQ differ by 7 percent.
  3. Acceptable quality loss (the gate): this is where a RATA-style band belongs. Whether +10 percent
     perplexity is acceptable for a language model is a judgement; an instrument's 10 percent relative accuracy
     does not carry over automatically.

## Decisions needed from the owner

1. The absolute limit for readings below 20 (0.45 today).
2. The relative limit for the middle band (20 to 100): 20 percent was assumed.
3. Whether the high band is 10 percent for quality loss, or only for agreement between runtimes.
4. Whether the gate should read these bands from its file (a band-capable gate v3, not built).
