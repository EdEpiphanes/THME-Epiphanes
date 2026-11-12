#!/usr/bin/env python3
"""
THME evaluation corpus preparation.

Downloads the WikiText-2 (raw) test split and a model's tokenizer, tokenizes the
split with the reference Hugging Face `tokenizers` implementation, and writes:

  <out-dir>/wikitext2_test.<model>.u32    little-endian u32 token ids, no header
  <out-dir>/wikitext2_test.<model>.json   provenance (sources, hashes, token count)

The token file is the input to the measured perplexity gate:

  cargo run --release --example production_verification -- \
      --model-path models/SmolLM-135M --tokens target/eval/wikitext2_test.SmolLM-135M.u32

Requires: pip install tokenizers
"""

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import urllib.request
import zipfile

WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT_MEMBER = "wikitext-2-raw/wiki.test.raw"


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


def main():
    parser = argparse.ArgumentParser(description="Tokenize WikiText-2 test for the THME perplexity gate")
    parser.add_argument("--repo", default="HuggingFaceTB/SmolLM-135M", help="Hugging Face repo providing tokenizer.json")
    parser.add_argument("--tokenizer", default=None, help="Local tokenizer.json (skips the tokenizer download)")
    parser.add_argument("--out-dir", default="target/eval", help="Output directory")
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("error: the 'tokenizers' package is required (pip install tokenizers)", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = args.repo.split("/")[-1]

    zip_path = os.path.join(args.out_dir, "wikitext-2-raw-v1.zip")
    fetch(WIKITEXT_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(WIKITEXT_MEMBER)
    text = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline="").read()

    tokenizer_path = args.tokenizer
    tokenizer_source = "local file"
    if tokenizer_path is None:
        tokenizer_path = os.path.join(args.out_dir, f"{model_tag}.tokenizer.json")
        tokenizer_source = f"https://huggingface.co/{args.repo}/resolve/main/tokenizer.json"
        fetch(tokenizer_source, tokenizer_path)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    ids = tokenizer.encode(text, add_special_tokens=False).ids

    token_path = os.path.join(args.out_dir, f"wikitext2_test.{model_tag}.u32")
    with open(token_path, "wb") as f:
        f.write(struct.pack(f"<{len(ids)}I", *ids))

    provenance = {
        "corpus": "WikiText-2 (raw) test split",
        "corpus_source": WIKITEXT_URL,
        "corpus_member": WIKITEXT_MEMBER,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_bytes": len(raw),
        "tokenizer_repo": args.repo,
        "tokenizer_source": tokenizer_source,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "special_tokens_added": False,
        "token_count": len(ids),
        "token_file_sha256": sha256_file(token_path),
    }
    with open(os.path.splitext(token_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(ids)} tokens to {token_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
