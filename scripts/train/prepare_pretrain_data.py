#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Turns a text file into the flat id files that scripts/train/pretrain.py reads (THME-3110).

  tokens  Pythia tokenizer ids as uint16 (vocabulary 50304 fits), documents joined as they are in the text
  bytes   raw UTF-8 bytes as uint8

  python scripts/train/prepare_pretrain_data.py --mode tokens --tokenizer models/pythia-14m/tokenizer.json \
      --text target/testbed/wikitext103_train.txt --out target/testbed/train.u16
"""

import argparse
import hashlib

import numpy as np

CHUNK = 1 << 22  # characters per tokenizer call


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def encode_text(text, mode, tok=None):
    if mode == "bytes":
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return np.asarray(tok.encode(text).ids, dtype=np.uint16)


def convert(text_path, out, mode, tokenizer=None):
    tok = None
    if mode == "tokens":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer)
    total = 0
    with open(text_path, encoding="utf-8") as f, open(out, "wb") as o:
        buf = ""
        while True:
            piece = f.read(CHUNK)
            if not piece:
                break
            buf += piece
            cut = buf.rfind("\n") + 1  # split at a line end so no word is cut
            if cut == 0:
                continue
            ids = encode_text(buf[:cut], mode, tok)
            o.write(ids.tobytes())
            total += len(ids)
            buf = buf[cut:]
        if buf:
            ids = encode_text(buf, mode, tok)
            o.write(ids.tobytes())
            total += len(ids)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tokens", "bytes"), required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = convert(a.text, a.out, a.mode, a.tokenizer)
    print(f"{a.out}: {n} ids, sha256 {sha256(a.out)}")


if __name__ == "__main__":
    main()
