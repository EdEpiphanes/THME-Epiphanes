# Datasets and models used in benchmarking

> **Notice.** THME does not own, host or distribute any of the datasets or models listed on this
> page, and none of them is included in this repository. Each belongs to its respective owners and
> is available to the public under the licence stated by its original source. THME downloads them
> from the original (or a named mirror) source onto the user's own machine at run time and uses
> them only to measure and compare the quality and speed of models, so that anyone who doubts a
> published result can recreate it. Licences and terms are quoted as the sources state them on the
> date given; check the source before you use or redistribute anything. Nothing on this page is
> legal advice.

Last checked: 20 November 2026.

## Used today

| Dataset | Used for | Original source | Licence as stated by the source |
| :--- | :--- | :--- | :--- |
| WikiText-2 (raw v1) | Perplexity scoring on the test split; calibration statistics from the train split | <https://huggingface.co/datasets/Salesforce/wikitext> (paper: Merity et al., "Pointer Sentinel Mixture Models", arXiv 1609.07843) | CC BY-SA 3.0 and GFDL (the dataset card lists both; other pages label it CC BY-SA 4.0) |

How THME gets WikiText-2: `scripts/prepare_eval_tokens.py` downloads the raw zip from a mirror
(<https://huggingface.co/datasets/ggml-org/ci>, file `wikitext-2-raw-v1.zip`) because it is a
single stable file. The mirror is not the original source. Every report records the SHA-256 of the
raw text used, so a copy fetched from the original source can be checked against it:

| Split | File in the zip | SHA-256 of the raw file (recorded in the reports) |
| :--- | :--- | :--- |
| test | `wikitext-2-raw/wiki.test.raw` | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| train | `wikitext-2-raw/wiki.train.raw` | `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57` |

| Model | Source | Licence as stated by the source | Revision checked |
| :--- | :--- | :--- | :--- |
| Pythia-14M | <https://huggingface.co/EleutherAI/pythia-14m> | Apache-2.0 | `cf967c0a9a` |
| SmolLM-135M | <https://huggingface.co/HuggingFaceTB/SmolLM-135M> | Apache-2.0 | `1d461723ee` |
| SmolLM2-1.7B | <https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B> | Apache-2.0 | `effd688a12` |

The licence of a model's weights does not settle the rights to its training data; see each model
card.

## Planned, not used yet

The benchmark gate (v0.2.8) is planned to use public benchmarks, each fetched by the user from its
own source. Licences are those shown on the dataset cards; none of these is used in a published
result yet, and the list changes only through the benchmark registry.

| Benchmark | Source | Licence as stated on the card |
| :--- | :--- | :--- |
| MMLU | <https://huggingface.co/datasets/cais/mmlu> | MIT |
| GSM8K | <https://huggingface.co/datasets/openai/gsm8k> | MIT |
| HumanEval | <https://huggingface.co/datasets/openai/openai_humaneval> | MIT |
| IFEval | <https://huggingface.co/datasets/google/IFEval> | Apache-2.0 |
| ARC (Easy and Challenge) | <https://huggingface.co/datasets/allenai/ai2_arc> | CC BY-SA 4.0 |
| HellaSwag | <https://huggingface.co/datasets/Rowan/hellaswag> | MIT (stated in the card text; to be confirmed from the repository) |

## What THME stores and what it does not

- Stored in this repository: scores, counts, hashes, item identifiers and the commands that
  produced them.
- Never stored: dataset files, benchmark questions or answers, prompts built from them, or token
  files made from them (token files are produced on the user's machine and ignored by git).
- Gated benchmarks that forbid showing their items stay out of every report and document.

## Recreating a result

1. Download the model from the source above.
2. Run `scripts/prepare_eval_tokens.py` (or fetch WikiText-2 from the original source and compare
   the SHA-256 values above).
3. Run the command printed in the report you want to check; the report records the model, token
   file hash, device, driver, build and settings.
4. Compare the numbers. Differences caused by a different device or a loaded machine are recorded
   in each report.
