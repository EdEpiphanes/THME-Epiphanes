# 0010. Quality gate v2: two regimes by baseline perplexity

Date: 2026-11-27

## Status

Accepted

## Context

Quality gate v1 allows one flat absolute margin of 0.45 perplexity. Pythia-14M has a baseline of about 114, where a uniform Q8_0 export changes perplexity by 0.83 (0.73 percent) and fails v1.

## Decision

A criterion may set `switch_at`:
- A baseline below `switch_at` uses the absolute margin only (`max_abs` 0.45).
- A baseline at or above `switch_at` uses the relative margin only (`max_rel` 0.0225 = 0.45 / 20, so the two margins meet at the switch 20.0).
- v1 files and their hash are unchanged (the field is skipped when unset).
- The default gate stays v1; v2 is chosen with `--gate configs/gates/thme-perplexity-v2.json` on `thme evaluate`.

## Consequences

- Evidence level: committed reports re-judged:
  - Smol-135M Q8_0 (19.80 to 19.83) passes under both.
  - Pythia-14M Q8_0 (114.29 to 115.12) fails v1 and passes v2.
  - The SmolLM recipe (+28.9) and Pythia Q4_0 fail under both.
- Open gap: the accuracy-rule analogy the owner recalled from emissions monitoring was not checked against the standard text.
