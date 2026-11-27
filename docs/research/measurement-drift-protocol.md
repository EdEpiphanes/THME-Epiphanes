# Measurement drift protocol

27 November 2026. A protocol for telling real differences from measurement noise. Written before it
is applied to any new run; the rules below are fixed in advance and are not changed after a result
is seen. Status: protocol and helper written (`scripts/drift.py`); not yet applied to a committed report.

## The idea

Think of a gas sensor that should read 100 ppm. Average its readings over 30 minutes and call that Run 1;
do Runs 2 and 3. Check the sensor against a known reference before and after. The difference between runs
and between the before and after checks is drift, and the drift sets how much of any difference we are
willing to treat as noise.

For THME, a "reading" is a measured value (a perplexity, a tokens-per-second figure, an agreement
rate); a "run" is one repeat of the same measurement; the "reference" is a measurement whose true
value is known (the committed CPU baseline of the same model on the same token slice).

## Definitions

- **Nominal value** `N`: the known value of the reference check, taken from a committed report.
- **Run mean** `R_k`: the mean of the readings of run `k`, for `k = 1..n` (default `n = 3`).
- **Pre-run bias** `B_pre = (ref_before - N) / N`, and **post-run bias** `B_post = (ref_after - N) / N`:
  the reference check measured immediately before and after a batch of runs.
- **Drift during the batch** `D = B_post - B_pre`.
- **Spread** `S = (max R_k - min R_k) / mean(R_k)`: how far the run means disagree.
- **Tolerance** `T`: the allowed relative deviation, plus or minus. Defaults, chosen here and open to the
  owner: `5 %` for timings and throughput; `0.01 %` for a deterministic CPU perplexity (it should repeat
  exactly, so any deviation is a finding); `0.5 %` for a GPU perplexity (floating-point order differs).

## Rules

1. **Warm-up.** The first run of a batch is a warm-up and is kept and reported but not averaged, unless
   `n = 1`. (Caches, clocks and first-time compilation make it unrepresentative.)
2. **Order.** With two or more arms to compare, interleave them (A B B A, not A A B B) so a steady
   drift hits both equally.
3. **In tolerance** means: `|B_pre| <= T`, `|B_post| <= T` and `S <= 2T` (the run means stay inside a
   band of plus or minus `T` around their centre). Then the batch is accepted and the run means are the result.
4. **Out of tolerance** means the batch is flagged `unstable` and is not used for a verdict. The
   report must include an attribution (below) before anyone re-runs it.
5. **Guard band.** The observed drift is added to the gate's own margin on the unfavourable side. A
   candidate passes only if it passes at the unfavourable edge of the band; if the verdict differs between
   the two edges it is reported `Inconclusive` (the gate already has that third state), not `Pass`.
6. **Bias correction.** A corrected value (the raw value minus the bias interpolated between `B_pre` and
   `B_post`) may be reported next to the raw value. It is allowed only when the reference check is itself
   deterministic or repeats within its tolerance. Verdicts use the raw value unless a report registered
   the correction before its run.
7. **No quiet exclusion.** Dropping a run other than the warm-up needs a recorded reason; the dropped
   run stays in the report.

## Attribution when the tolerance is exceeded

For each run the harness records factors that can move a reading: run index, number of other heavy
processes running, GPU temperature, GPU clock, GPU power and memory used (sampled with `nvidia-smi`),
CPU load, and the time of day. The report lists the Pearson correlation of the reading with each factor
and the run-index trend. A correlation is a pointer for the next experiment, not a cause. A strong trend
with run index suggests warm-up or thermal drift; a correlation with other processes suggests contention.

## What this does not do

It does not make a bad measurement good: averaging `n` runs shrinks random noise by about `sqrt(n)` and
does nothing for a bias shared by every run. The reference check is there to expose that bias. It also
does not decide whether a difference is practically important; the gate does that.

## Where it applies first

- Timings of `thme chat`, `thme forward` and the compatibility report (tolerance 5 %).
- GPU perplexities of the compatibility report (tolerance 0.5 %): do CUDA, Vulkan and DX12 repeat?
- Seeded training runs (the distillation work): run-to-run spread of the loss at a fixed seed.
- Deterministic CPU results (GPTQ, Hadamard) are expected to repeat exactly; a second run shows that.
