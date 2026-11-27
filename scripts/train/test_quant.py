#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for quant.py: the torch Q4_0 round trip against THME's own Rust function (the fixture is
written by `cargo run --example dump_q4_0`) and the list of tensors THME quantizes."""

import json
import os
import struct
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quant  # noqa: E402

FIXTURE = os.path.join(HERE, "..", "..", "tests", "fixtures", "q4_0_roundtrip.json")


def bits_to_tensor(bits, shape):
    n = len(bits)
    return torch.tensor(struct.unpack(f"<{n}f", struct.pack(f"<{n}I", *bits))).reshape(shape)


class Q4_0AgainstThme(unittest.TestCase):
    def test_round_trip_equals_the_rust_fixture_bit_for_bit(self):
        with open(FIXTURE) as f:
            fx = json.load(f)
        x = bits_to_tensor(fx["input_bits"], fx["shape"])
        want = bits_to_tensor(fx["output_bits"], fx["shape"])
        got = quant.q4_0_roundtrip(x)
        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)))
        self.assertFalse(torch.equal(got, x))  # the fixture really quantizes

    def test_hand_worked_block(self):
        # Block v[i] = i - 10, i = 0..31, so v runs -10..21.
        # Largest magnitude is +21, so scale = 21 / -8 = -2.625 (exact in FP16), inv = 1/scale.
        # code(v) = trunc(v * inv + 8.5) clamped to 0..15, decoded as scale * (code - 8):
        #   v = 21 : 21*inv = -8.0 (to f32 rounding) +8.5 = 0.5 -> code 0  -> -2.625 * -8 = 21
        #   v = 20 : -7.619 + 8.5 = 0.881 -> code 0 -> 21 (the extreme side saturates)
        #   v = 3  : -1.143 + 8.5 = 7.357 -> code 7 -> -2.625 * -1 = 2.625
        #   v = 1  : -0.381 + 8.5 = 8.119 -> code 8 -> 0
        #   v = 0  : 8.5 -> code 8 -> 0
        #   v = -10: 3.810 + 8.5 = 12.31 -> code 12 -> -2.625 * 4 = -10.5
        x = torch.arange(32, dtype=torch.float32) - 10.0
        y = quant.q4_0_roundtrip(x)
        for value, want in [(21, 21.0), (20, 21.0), (3, 2.625), (1, 0.0), (0, 0.0), (-10, -10.5)]:
            self.assertEqual(y[value + 10].item(), want, f"v={value}")

    def test_all_zero_block_gives_zeros(self):
        self.assertTrue(torch.count_nonzero(quant.q4_0_roundtrip(torch.zeros(64))) == 0)

    def test_rejects_wrong_size_and_scale_out_of_fp16_range(self):
        with self.assertRaises(ValueError):
            quant.q4_0_roundtrip(torch.zeros(33))
        with self.assertRaises(ValueError):
            quant.q4_0_roundtrip(torch.full((32,), 1e9))

    def test_keeps_shape_and_does_not_touch_the_input(self):
        x = torch.randn(4, 64)
        before = x.clone()
        self.assertEqual(quant.q4_0_roundtrip(x).shape, x.shape)
        self.assertTrue(torch.equal(x, before))


class Q4_0Names(unittest.TestCase):
    def test_smollm_style_names(self):
        d = {
            "model.embed_tokens.weight": torch.zeros(8, 4),
            "lm_head.weight": torch.zeros(8, 4),
            "model.norm.weight": torch.zeros(4),
            "model.layers.0.input_layernorm.weight": torch.zeros(4),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
            "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 4),
            "model.layers.12.mlp.gate_proj.weight": torch.zeros(4, 4),
            "model.layers.0.self_attn.q_proj.bias": torch.zeros(4),
        }
        self.assertEqual(
            quant.q4_0_param_names(d),
            [
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.12.mlp.gate_proj.weight",
            ],
        )

    def test_tiny_llama_has_seven_matrices_per_layer(self):
        from tiny import tiny_llama

        names = quant.q4_0_param_names(tiny_llama().state_dict())
        self.assertEqual(len(names), 14)
        self.assertTrue(all("layers." in n and n.endswith("_proj.weight") for n in names))


if __name__ == "__main__":
    unittest.main()
