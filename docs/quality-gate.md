# Quality gates

A quality gate decides whether a candidate model is accepted against its source model. In THME a
gate is **versioned data**, not a number buried in a command, and every decision goes through it.

## A gate file

```json
{
  "name": "example-strict",
  "version": 1,
  "criteria": [
    { "metric": "perplexity", "lower_is_better": true, "max_abs": 0.45, "max_rel": 0.02 },
    { "metric": "cross_entropy_nats", "lower_is_better": true, "max_abs": 0.02 }
  ]
}
```

- **One criterion or many.** Every criterion must pass.
- **Absolute and relative margins.** `max_abs` is in the metric's own units; `max_rel` is a
  fraction of the baseline (0.02 = 2%). Give either or both; when both are given, both must hold.
- **Direction.** `lower_is_better` is `true` for losses and perplexity, `false` for accuracies.
- **Not measured is not passed.** A criterion whose metric was not measured fails as
  `not measured`. `thme evaluate` and `thme search` measure `perplexity` and `cross_entropy_nats`
  and refuse, before doing any work, a gate that needs another metric.

Gates in this repository live in [`configs/gates/`](../configs/gates/). The default,
[`thme-perplexity-v1.json`](../configs/gates/thme-perplexity-v1.json), is one perplexity criterion
with an absolute margin of 0.45: the gate behind every committed measured report.

## Using a gate

```bash
thme evaluate MODEL --recipe recipe.json --tokens tokens.u32 --gate configs/gates/example-strict.json
thme search   MODEL --tokens tokens.u32 --budget 0.5 --gate configs/gates/example-strict.json
```

Without `--gate`, the default gate is used with `--max-ppl-delta` as its margin (0.45 unless
given). The two options cannot be combined.

## What a report records

Every report carries the gate's name, version, criteria and **content hash** (SHA-256 of the
gate's canonical JSON), the verdict, and one outcome per criterion with the baseline, candidate,
absolute and relative degradation. Changing any threshold changes the hash, so a result is never
re-labelled under another gate.

`thme validate --report REPORT.json` rebuilds the recorded gate, checks its hash, and re-judges
every variant from its measured numbers. A stored verdict that disagrees with the gate, or a gate
edited after the run, is an error.
