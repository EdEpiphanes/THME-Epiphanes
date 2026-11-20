# Calibration data practice for activation-aware quantization

Research note, 2026-11-20. Desk research only: no models run, no downloads. Every claim carries a
URL. "Not verified" means only a secondary source or my own reasoning supports it. Ideas only; no
third-party code was copied.

## 1. Method -> default calibration set

| Method | Default calibration data | Size | Notes |
| :--- | :--- | :--- | :--- |
| GPTQ | C4 train | 128 random segments x 2048 tokens (262k tokens) | Authors call it generic web text, so "zero-shot"; they admit WikiText-2 results are "not fully zero-shot". [arXiv 2210.17323](https://arxiv.org/html/2210.17323v2) |
| AWQ | Pile subset | Main-result size not stated in the text I read; ablation uses 16 vs 192 sequences | Chosen "in order not to overfit to a specific downstream domain". [arXiv 2306.00978](https://arxiv.org/html/2306.00978v5) |
| SmoothQuant | Pile | 512 random sentences | No calibration-sensitivity study in the text read. [arXiv 2211.10438](https://arxiv.org/html/2211.10438) |
| QuaRot | WikiText-2 train (for GPTQ weights) | 128 x 2048 | RTN weights need no calibration set; no calibration ablation. [arXiv 2404.00456](https://arxiv.org/html/2404.00456) |
| SpinQuant | WikiText-2 (rotation learning) | 800 samples, 100 Cayley iterations | 128 samples gives the same ppl (6.2); C4 gives the same result as WikiText-2. [arXiv 2405.16406](https://arxiv.org/html/2405.16406) |
| QuIP# | RedPajama 1T (Hessians) | 6144 sequences at native context (2048 Llama 1, 4096 Llama 2) | [arXiv 2402.04396](https://arxiv.org/html/2402.04396) |
| AQLM | RedPajama subset | Llama 2: 8M tokens, seq length 4096 | [arXiv 2401.06118](https://arxiv.org/html/2401.06118) |
| llama.cpp imatrix | User file via `-f` (README example `wiki.train.raw`); repo default text from Bartowski's set (secondary source, not verified) | All chunks by default (`--chunks -1`) | [imatrix README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/imatrix/README.md) |
| ExLlamaV3 | "Bundled corpus mix" (contents not named in the doc) | `--cal_rows` default 250 x `--cal_cols` default 2048 = 512,000 tokens | Custom file via `--cal_data`. [convert.md](https://raw.githubusercontent.com/turboderp-org/exllamav3/master/doc/convert.md) |

Newer (2025 to 2026) papers, seen only through search summaries or abstracts, so not verified:
self-calibration with model-generated data ([arXiv 2410.17170](https://arxiv.org/abs/2410.17170),
abstract: "frequently outperforming even using real data"), and a 40-dataset study saying a
calibration set matching the test distribution does not guarantee the best result
([arXiv 2406.12928](https://arxiv.org/abs/2406.12928), abstract only).

## 2. Evidence: sample size, domain, seeds

**Sample size**
- AQLM, Llama 2 7B, ~2.3 bits, WikiText-2 ppl averaged over 3 runs: 128 seq 6.994; 256 6.584;
  512 6.455; 1024 6.353; 2048 6.297; 4096 6.267. The authors say GPTQ saturates around 256
  sequences while AQLM keeps gaining. [arXiv 2401.06118, App. E](https://arxiv.org/html/2401.06118)
  (a learned-codebook method, so it needs far more data than scale search).
- AWQ reaches better ppl with 16 sequences than GPTQ with 192 (OPT-6.7B, INT3-g128).
  [arXiv 2306.00978 s5.3](https://arxiv.org/html/2306.00978v5)
- SpinQuant: 800 -> 128 samples leaves WikiText-2 ppl at 6.2 (Llama-2 7B, W4A4KV4).
  [arXiv 2405.16406 Table 11](https://arxiv.org/html/2405.16406)
- LiftQuant (2-bit): gains from 1M to 8M tokens, diminishing beyond ~4M; CoreQ works with 16 to
  32 samples; TWLA flat with size. Search-summary only, not verified:
  [LiftQuant 2606.04050](https://arxiv.org/pdf/2606.04050), [CoreQ 2602.05902](https://arxiv.org/pdf/2602.05902),
  [TWLA 2606.13054](https://arxiv.org/pdf/2606.13054).
- GPTQ with very few samples can blow up numerically (ppl beyond 10^2 for n <= 4, DASH-Q; search
  summary, not verified): [arXiv 2604.13806](https://arxiv.org/pdf/2604.13806).

**Domain mismatch**
- AWQ, Pile subsets PubMed vs Enron: same-distribution calibration is best; across distributions
  AWQ ppl rises only 0.5 to 0.6, GPTQ rises 2.3 to 4.9. 1k eval samples, no overlap between
  calibration and evaluation splits. [arXiv 2306.00978](https://arxiv.org/html/2306.00978v5)
- SpinQuant, Llama-2 7B W4A4KV16: WikiText-2 calibration 64.1 avg acc / 5.9 ppl; C4 calibration
  64.3 / 5.9. [arXiv 2405.16406 Table 13](https://arxiv.org/html/2405.16406)
- LiftQuant (2-bit): WikiText-2 calibration is best on WikiText-2 ppl and slightly worse on C4 and
  zero-shot (not verified, search summary above). BitsMoE: math-only calibration hurt general
  accuracy (not verified): [arXiv 2606.00079](https://arxiv.org/pdf/2606.00079).
- Reading: scale-search methods (AWQ) are mildly domain-sensitive (~0.5 ppl at 7B scale);
  in-domain calibration gives an optimistic score on that domain. I found no source that measures
  this for models under 1B.

**Leakage / overlap**
- Standard practice: calibrate on train, score on a disjoint split. WikiText-2 train and test
  are disjoint article sets but the same domain and style, so the score favours WikiText
  calibration. [dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
- Separate issue: Pythia was trained on the Pile, which contains Wikipedia (from memory of the
  Pile paper, not verified here). The test text may already be in pretraining data. This is the
  same for every arm, so it does not bias the comparison, but absolute ppl is not "held-out".

**Seeds / variance**
- No paper I found reports AWQ seed standard deviations for small models. MatGPTQ and BayesQ
  average over 3 calibration seeds (search summary, not verified:
  [2602.03537](https://arxiv.org/pdf/2602.03537), [2511.08821](https://arxiv.org/pdf/2511.08821)).
  AQLM averages 3 runs for its size table. SpinQuant reports a ±0.4 acc std over rotation seeds,
  and random Hadamard rotations vary by up to ~6 points. [arXiv 2405.16406](https://arxiv.org/html/2405.16406)
- Conclusion: THME must measure its own variance. Do not assume it is small.

**Model-generated / random text (llama.cpp community, anecdotal, not verified)**
- One user: ~25k tokens of high-entropy pseudo-random text gave KLD_95 1.22 vs 1.29 for ~40k
  tokens of wikitext at 7B; the same post said purely random data did worse than wikitext but
  better than no imatrix. Single model, unreviewed:
  [Reddit thread](https://redlib.hyperreal.coffee/r/LocalLLaMA/comments/1ah3w8d/comment/kouw5aj),
  [froggeric notes](https://huggingface.co/datasets/froggeric/imatrix/blob/6bc8bf98402ccff0a35ab25c80f1e8dbf8276bcc/README.md?code=true).
  Do not cite as evidence; it is only a reason a mismatch control may score close to the real arm.

## 3. Recommendation: first measured AWQ test (Pythia-14M, SmolLM-135M, CPU, WikiText-2 test)

**Is 32,768 tokens from WikiText-2 train, seed 1234, adequate?**
Yes as the primary arm, with changes. 32,768 = 16 x 2048, the count AWQ used in its
16-sequence claim (a 7B model). Hidden widths are small (Pythia-14M d=128, SmolLM-135M d=576;
check against each config), so mean |x| per channel gets thousands of tokens. The risk is rare
outlier tokens that 32k tokens may miss; that is my inference, not a sourced fact. Changes:
1. Use the model's training context as window length if CPU allows; otherwise record the shorter
   length. Cut non-overlapping windows from random offsets in the concatenated train split,
   tokenised with the model's own tokenizer.
2. Add a **size check** at 8,192 / 32,768 / 131,072 tokens, same seed. If test ppl differs by less
   than the seed spread (below), 32,768 is enough; if not, make the larger size primary.
3. Add a **second calibration source** (C4 or other non-Wikipedia text; licences in section 5)
   as an out-of-domain arm. Report both. Without it the WikiText-calibrated score is an in-domain,
   optimistic number.
4. Seed 1234 stays as the headline seed, but only as one of the seeds below.

**Seed-variance check**
- Seeds {1234, 1, 2, 3, 4}: 5 minimum, 10 if each run is cheap on CPU. Each seed draws different
  windows; nothing else changes.
- Report per model: mean, sample std dev, min and max of test ppl (and delta vs RTN and fp16).
- Pre-register the rule before running: call AWQ better than RTN only if the mean gap exceeds
  2 x the pooled std dev and every seed agrees in sign. A single-seed win is not a claim.
- Keep the alpha grid fixed across seeds, then separately report the chosen alpha per seed, so
  seed noise and grid noise can be told apart.

**Random control A: real scale distribution, no information**
- For each linear layer, randomly permute the real per-channel scale vector across input channels
  (same multiset of values, wrong channel assignment). Apply the same quantizer and the same
  fold into the previous op. Use >= 5 permutation seeds.
- Alternative: draw from a log-normal fitted to log(s) of that layer. Pick one variant as
  primary and name it.
- Expectation: real scales beat control A by more than the seed spread. If not, the gain is not
  coming from activation information.

**Random control B: statistics from shuffled or mismatched text**
- B1 shuffled: the same 32,768 train tokens with token order permuted (keeps unigram counts,
  destroys context).
- B2 mismatched: same token count from another domain (code, non-English, or the C4 arm); as an
  extreme stress arm, uniformly random token ids (not a baseline).
- Same pipeline, grid and seeds. A small gap between real and B arms means the model is
  insensitive to calibration at this scale. That is a finding; report it.

**Also include:** RTN with no scaling (floor) and fp16 (ceiling). Without them the controls have
no frame.

## 3b. Provenance to record in the report

llama.cpp records only `quantize.imatrix.file`, `quantize.imatrix.dataset` (first dataset only,
directory stripped, capped at 127 chars), `quantize.imatrix.entries_count` and
`quantize.imatrix.chunks_count`; listing several datasets is a TODO in the source
([quantize.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/quantize.cpp);
introduced by PR #6658, per a secondary page, not verified). ExLlamaV3 documents rows and columns
but I found no provenance recording (not verified). Neither stores a content hash, tokeniser or
seed, so THME should record more:
- dataset name, exact subset and split, source URL, revision, licence;
- SHA-256 of the raw calibration text and of the token id array;
- tokeniser name and hash, special-token handling, BOS/EOS policy;
- window length, number of windows, total tokens, selection method, offsets, seed;
- statistic collected (e.g. mean |x| per channel), dtype, tokens that contributed;
- method config (alpha grid, group size, bits), tool version, git commit;
- model id and revision hash, evaluation split and its hash, overlap check result;
- number of seeds, per-seed scores, summary stats, control definitions and seeds;
- hardware and thread count (CPU), and a "measured" vs "pending" flag per AGENTS.md rule 2.

## 4. Caveats

- Nearly all numbers above are for 7B+ models at 2 to 4 bits; none are for 14M to 135M models.
  Tiny models have few channels and little redundancy, so effects may be larger or smaller. Treat
  this note as a design guide, not a prediction.
- Several 2025 to 2026 sources were seen only as search summaries or abstracts (marked). Re-read
  before quoting them in a report. The AWQ and SmoothQuant main-result sample sizes were not
  confirmed.
- WikiText-2 perplexity is a narrow score, and calibrating on the same domain flatters it. Add a
  non-Wikipedia perplexity slice if possible.
- Pythia's Pile includes Wikipedia, so test text may have been seen in pretraining; SmolLM corpus
  overlap is unchecked.
- "2 x std dev with 5 seeds" is a pragmatic threshold, not a significance test. State exact
  numbers and do not over-claim.
- Gains from scaling at W8 or W4-g128 on tiny models may sit inside noise. A null result is a
  valid outcome.

## 5. Licences (THME policy: permissive; not legal advice; mostly not verified)

- WikiText-2: the card text says CC BY-SA 4.0, while its tags show cc-by-sa-3.0 and gfdl
  (inconsistent). Share-alike, so keep it out of the repo: fetch at run time and record the hash.
  [card](https://huggingface.co/datasets/Salesforce/wikitext)
- C4: ODC-By (from memory, not verified; check the card first).
- The Pile: mixed component licences (from memory, not verified); do not redistribute.
- Pythia and SmolLM: Apache-2.0 (from memory, not verified; check model cards).
- llama.cpp (MIT), AWQ (MIT), GPTQ (Apache-2.0), SpinQuant: from memory, not verified. Only
  ideas were taken; do not copy code. Check each LICENSE before any code reuse.
- Papers on arXiv are cited, not copied.
