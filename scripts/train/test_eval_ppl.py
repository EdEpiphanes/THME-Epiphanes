#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for eval_ppl.py against a brute-force per-window computation written here, in plain
Python floats, from THME's documented rule (src/inference/perplexity.rs)."""

import math
import os
import struct
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_ppl  # noqa: E402
from tiny import tiny_llama  # noqa: E402


def brute_force(model, tokens, window, max_tokens=None):
    """Windows are consecutive chunks; a chunk of one token is dropped; the first token of a
    chunk is not scored; perplexity is exp(mean NLL) over every scored token."""
    used = tokens if max_tokens is None else tokens[:max_tokens]
    nll, windows = [], 0
    for start in range(0, len(used), window):
        chunk = used[start : start + window]
        if len(chunk) < 2:
            continue
        windows += 1
        with torch.no_grad():
            logits = model(torch.tensor([chunk])).logits[0].double().tolist()
        for t in range(len(chunk) - 1):
            row = logits[t]
            top = max(row)
            lse = top + math.log(sum(math.exp(v - top) for v in row))
            nll.append(lse - row[chunk[t + 1]])
    return math.exp(sum(nll) / len(nll)), len(nll), windows


class EvalPpl(unittest.TestCase):
    def setUp(self):
        self.model = tiny_llama(seed=1)
        g = torch.Generator().manual_seed(7)
        self.tokens = torch.randint(0, 128, (50,), generator=g).tolist()

    def check(self, tokens, window, max_tokens=None):
        got = eval_ppl.evaluate_ppl(self.model, tokens, window, max_tokens=max_tokens)
        ppl, scored, windows = brute_force(self.model, tokens, window, max_tokens)
        self.assertEqual((got["scored_tokens"], got["windows"]), (scored, windows))
        self.assertAlmostEqual(got["perplexity"] / ppl, 1.0, places=5)
        return got

    def test_final_window_of_two_is_scored(self):
        got = self.check(self.tokens, 16)  # 16, 16, 16, 2
        self.assertEqual((got["windows"], got["scored_tokens"]), (4, 46))

    def test_final_window_of_one_token_is_dropped(self):
        got = self.check(self.tokens[:49], 16)  # 16, 16, 16, 1
        self.assertEqual((got["windows"], got["scored_tokens"]), (3, 45))

    def test_max_tokens_truncates_first(self):
        got = self.check(self.tokens, 8, max_tokens=20)  # 8, 8, 4
        self.assertEqual(
            (got["windows"], got["scored_tokens"], got["corpus_tokens_used"]), (3, 17, 20)
        )

    def test_zero_weights_give_perplexity_of_the_vocabulary(self):
        model = tiny_llama(seed=1)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        got = eval_ppl.evaluate_ppl(model, self.tokens, 16)
        self.assertAlmostEqual(got["perplexity"], 128.0, places=3)

    def test_reads_thme_token_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.u32")
            with open(path, "wb") as f:
                f.write(struct.pack("<4I", 1, 2, 3, 4000000000))
            self.assertEqual(eval_ppl.read_token_file(path), [1, 2, 3, 4000000000])
            with open(path, "wb") as f:
                f.write(b"\x01\x00\x00")
            with self.assertRaises(ValueError):
                eval_ppl.read_token_file(path)

    def test_too_few_tokens_and_bad_window_raise(self):
        with self.assertRaises(ValueError):
            eval_ppl.evaluate_ppl(self.model, [1], 16)
        with self.assertRaises(ValueError):
            eval_ppl.evaluate_ppl(self.model, self.tokens, 1)


if __name__ == "__main__":
    unittest.main()
