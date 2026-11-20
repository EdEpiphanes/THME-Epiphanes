#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
THME evaluation corpus preparation.

Downloads the WikiText-2 (raw) corpus and a model's tokenizer, tokenizes one split with the
reference Hugging Face `tokenizers` implementation, and writes:

  <out-dir>/wikitext2_<split>.<model>[<tags>].u32   little-endian u32 token ids, no header
  <out-dir>/wikitext2_<split>.<model>[<tags>].json  provenance (sources, hashes, sizes, selection)

Default: the whole test split, the scoring corpus of the perplexity gate.

Calibration statistics (THME-2204) come from the train split, never the test split:

  --split train --calibration-windows 16 --window-len 2048 --seed 1234
      16 random non-overlapping windows of 2048 tokens, written in ascending offset order
  ... --shuffle        the same tokens with their order permuted (control B1)
  ... --random-ids --vocab-size V
      the same number of uniform random ids below V (stress arm B2, not a baseline)

The token file is the input to the measured perplexity gate and to `thme calibrate`:

  cargo run --release --example production_verification -- \
      --model-path models/SmolLM-135M --tokens target/eval/wikitext2_test.SmolLM-135M.u32

Requires: pip install tokenizers
"""

import argparse
import hashlib
import io
import json
import os
import random
import struct
import sys
import urllib.request
import zipfile

WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
SPLITS = ("train", "test")
BENCHMARK_ID = "wikitext2-raw"
# Version of this preparation protocol; bump it when the selection or the sidecar changes.
BENCHMARK_VERSION = "1"


def fetch(url, dest):
    if os.path.exists(dest):
        print(f"  cached: {dest}")
        return
    print(f"  downloading: {url}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_windows(total, count, length, seed):
    """Start offsets of `count` non-overlapping windows of `length` tokens, ascending."""
    if count < 1 or length < 1 or count * length > total:
        raise SystemExit(f"error: cannot place {count} windows of {length} tokens in {total} tokens")
    rng = random.Random(f"windows-{seed}")
    starts = total - length + 1
    for _ in range(10000):
        offsets = sorted(rng.sample(range(starts), count))
        if all(b - a >= length for a, b in zip(offsets, offsets[1:])):
            return offsets
    raise SystemExit("error: cannot place non-overlapping windows; use fewer or shorter windows")


def main():
    parser = argparse.ArgumentParser(description="Tokenize WikiText-2 for the THME perplexity gate and calibration")
    parser.add_argument("--repo", default="HuggingFaceTB/SmolLM-135M", help="Hugging Face repo providing tokenizer.json")
    parser.add_argument("--tokenizer", default=None, help="Local tokenizer.json (skips the tokenizer download)")
    parser.add_argument("--out-dir", default="target/eval", help="Output directory")
    parser.add_argument("--split", choices=SPLITS, default="test",
                        help="WikiText-2 split to tokenize (calibration statistics use train)")
    parser.add_argument("--calibration-windows", type=int, default=0,
                        help="Sample this many non-overlapping windows; 0 keeps the whole split")
    parser.add_argument("--window-len", type=int, default=2048, help="Tokens per sampled window")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for window offsets, shuffle and random ids")
    parser.add_argument("--shuffle", action="store_true",
                        help="Permute the order of the sampled tokens (needs --calibration-windows)")
    parser.add_argument("--random-ids", action="store_true",
                        help="Replace the tokens with uniform random ids (stress arm)")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Id range for --random-ids: the model's vocab_size from config.json")
    args = parser.parse_args()

    if args.shuffle and not args.calibration_windows:
        parser.error("--shuffle permutes sampled windows; add --calibration-windows")
    if args.random_ids and args.vocab_size is None:
        parser.error("--random-ids needs --vocab-size (the model's vocab_size)")
    if args.vocab_size is not None and not args.random_ids:
        parser.error("--vocab-size only applies with --random-ids")

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("error: the 'tokenizers' package is required (pip install tokenizers)", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = args.repo.split("/")[-1]
    member = f"wikitext-2-raw/wiki.{args.split}.raw"

    zip_path = os.path.join(args.out_dir, "wikitext-2-raw-v1.zip")
    fetch(WIKITEXT_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(member)
    text = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline="").read()

    tokenizer_path = args.tokenizer
    tokenizer_source = "local file"
    if tokenizer_path is None:
        tokenizer_path = os.path.join(args.out_dir, f"{model_tag}.tokenizer.json")
        tokenizer_source = f"https://huggingface.co/{args.repo}/resolve/main/tokenizer.json"
        fetch(tokenizer_source, tokenizer_path)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    split_ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids = split_ids

    selection = None
    if args.calibration_windows:
        offsets = select_windows(len(split_ids), args.calibration_windows, args.window_len, args.seed)
        ids = [t for o in offsets for t in split_ids[o : o + args.window_len]]
        selection = {
            "method": "random non-overlapping windows over the split, written in ascending offset order",
            "rng": f"random.Random('windows-{args.seed}').sample",
            "windows": args.calibration_windows,
            "window_len": args.window_len,
            "seed": args.seed,
            "offsets": offsets,
        }

    order = "window"
    if args.shuffle:
        random.Random(f"shuffle-{args.seed}").shuffle(ids)
        order = "shuffled"

    random_ids = None
    if args.random_ids:
        rng = random.Random(f"random-ids-{args.seed}")
        ids = [rng.randrange(args.vocab_size) for _ in ids]
        random_ids = {"seed": args.seed, "vocab_size": args.vocab_size}

    tag = ""
    if selection:
        tag += f".cal{args.calibration_windows}x{args.window_len}.s{args.seed}"
    if args.shuffle:
        tag += ".shuffled"
    if args.random_ids:
        tag += f".randomids.v{args.vocab_size}"
    stem = os.path.join(args.out_dir, f"wikitext2_{args.split}.{model_tag}{tag}")

    token_path = stem + ".u32"
    with open(token_path, "wb") as f:
        f.write(b"".join(struct.pack("<I", t) for t in ids))

    provenance = {
        "corpus": f"WikiText-2 (raw) {args.split} split",
        "benchmark_id": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "split": args.split,
        "corpus_source": WIKITEXT_URL,
        "corpus_member": member,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_bytes": len(raw),
        "split_token_count": len(split_ids),
        "tokenizer_repo": args.repo,
        "tokenizer_source": tokenizer_source,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "special_tokens_added": False,
        "selection": selection,
        "order": order,
        "random_ids": random_ids,
        "token_count": len(ids),
        "token_file_bytes": 4 * len(ids),
        "token_file_sha256": None,
    }
    provenance["token_file_sha256"] = sha256_file(token_path)
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(ids)} tokens to {token_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
