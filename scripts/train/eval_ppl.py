#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Perplexity of a Hugging Face causal LM over a THME u32 token file, with `thme evaluate`'s rule
(src/inference/perplexity.rs): consecutive non-overlapping windows of `window` tokens, a final
chunk of one token dropped, one forward pass per window, the first token of each window not
scored, perplexity = exp(mean negative log-likelihood over all scored tokens).

  python eval_ppl.py --model-dir <hf dir> --tokens file.u32 --window 1024 [--device cpu|cuda]
"""

import argparse
import json
import math
import struct

import torch


def read_token_file(path):
    """Little-endian u32 ids, no header (the file `thme evaluate` reads)."""
    with open(path, "rb") as f:
        data = f.read()
    if not data or len(data) % 4:
        raise ValueError(f"Token file {path} must hold a whole, non-zero number of u32 ids")
    return list(struct.unpack(f"<{len(data) // 4}I", data))


@torch.no_grad()
def evaluate_ppl(model, tokens, window, max_tokens=None, device="cpu"):
    if window < 2:
        raise ValueError(f"Context length must be at least 2, received {window}")
    used = len(tokens) if max_tokens is None else min(max_tokens, len(tokens))
    chunks = [tokens[s : min(s + window, used)] for s in range(0, used, window)]
    chunks = [c for c in chunks if len(c) >= 2]
    if not chunks:
        raise ValueError(f"Corpus has too few tokens ({used}) to score anything")
    was_training = model.training
    model.eval()
    total, scored = 0.0, 0
    for chunk in chunks:
        ids = torch.tensor([chunk], device=device)
        logp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
        nll = -logp.gather(1, ids[0, 1:, None]).squeeze(1)
        total += nll.double().sum().item()
        scored += nll.numel()
    model.train(was_training)
    if not math.isfinite(total):
        raise ValueError("Forward pass produced a non-finite loss")
    ce = total / scored
    return {
        "perplexity": math.exp(ce),
        "cross_entropy_nats": ce,
        "scored_tokens": scored,
        "corpus_tokens_used": used,
        "windows": len(chunks),
        "context_len": window,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32).to(a.device)
    print(json.dumps(evaluate_ppl(model, read_token_file(a.tokens), a.window, a.max_tokens, a.device)))


if __name__ == "__main__":
    main()
