# SPDX-License-Identifier: AGPL-3.0-only
import argparse
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

import prepare_pretrain_data as prep
import pretrain


def tiny_dir(root, vocab):
    cfg = GPTNeoXConfig(vocab_size=vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=64, rotary_pct=0.25)
    d = os.path.join(root, "model")
    cfg.save_pretrained(d)
    return d


def args(root, arm, data, val, steps=60):
    return argparse.Namespace(arm=arm, model_dir=os.path.join(root, "model"), data=data, val=val,
                              out=os.path.join(root, "out-" + arm), steps=steps, ckpt_every=30, batch=8, seq=16,
                              lr=3e-3, warmup=5, weight_decay=0.0, val_windows=4, seed=7, cpu=True, micro_batch=0)


class PretrainTests(unittest.TestCase):
    def test_byte_arm_learns_a_repeating_pattern_and_the_checkpoint_reloads(self):
        with tempfile.TemporaryDirectory() as root:
            tiny_dir(root, 256)
            data = os.path.join(root, "d.u8")
            np.tile(np.arange(10, 20, dtype=np.uint8), 400).tofile(data)
            rep = pretrain.run(args(root, "byte", data, data))
            first, last = rep["checkpoints"][0]["val_loss"], rep["checkpoints"][-1]["val_loss"]
            self.assertLess(last, 0.5 * first)  # a ten-symbol cycle is easy; random start is about ln 256
            self.assertAlmostEqual(first, float(np.log(256)), delta=0.8)
            d = os.path.join(root, "out-byte", rep["checkpoints"][-1]["dir"])
            m = GPTNeoXForCausalLM.from_pretrained(d)
            self.assertEqual(m.config.vocab_size, 256)
            x = torch.tensor([[10, 11, 12, 13]])
            self.assertEqual(int(m(x).logits[0, -1].argmax()), 14)

    def test_same_seed_gives_the_same_losses(self):
        with tempfile.TemporaryDirectory() as root:
            tiny_dir(root, 256)
            data = os.path.join(root, "d.u8")
            np.random.default_rng(0).integers(0, 40, 4000).astype(np.uint8).tofile(data)
            a = pretrain.run(args(root, "byte", data, data, steps=30))
            b = pretrain.run(args(root, "byte", data, data, steps=30))
            self.assertEqual(a["train_loss_every_100"], b["train_loss_every_100"])
            self.assertEqual([c["val_loss"] for c in a["checkpoints"]], [c["val_loss"] for c in b["checkpoints"]])

    def test_prepare_bytes_matches_utf8(self):
        with tempfile.TemporaryDirectory() as root:
            t, o = os.path.join(root, "t.txt"), os.path.join(root, "o.u8")
            with open(t, "w", encoding="utf-8", newline="\n") as f:
                f.write("héllo\nworld\n")
            n = prep.convert(t, o, "bytes")
            self.assertEqual(n, len("héllo\nworld\n".encode("utf-8")))
            self.assertEqual(open(o, "rb").read(), "héllo\nworld\n".encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
