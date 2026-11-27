#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for qad_lora.py on a tiny random LLaMA on the CPU (2 layers, hidden 64, vocab 128).
They check the machinery. They say nothing about SmolLM-135M."""

import copy
import json
import os
import struct
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qad_lora  # noqa: E402
import quant  # noqa: E402
from tiny import tiny_llama  # noqa: E402


def write_tokens(path, n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 128, (n,), generator=g).tolist()
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{n}I", *ids))


class QadLora(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.tmp.name
        cls.model_dir = os.path.join(t, "model")
        tiny_llama(seed=3).save_pretrained(cls.model_dir)
        cls.train = os.path.join(t, "train.u32")
        cls.eval = os.path.join(t, "eval.u32")
        write_tokens(cls.train, 16 * 24, seed=11)
        write_tokens(cls.eval, 100, seed=12)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def args(self, name, **kw):
        a = dict(
            model_dir=self.model_dir,
            train_tokens=self.train,
            eval_tokens=self.eval,
            out_dir=os.path.join(self.tmp.name, name),
            seed=5,
            steps=20,
            batch_size=4,
            window=16,
            rank=4,
            alpha=8.0,
            lr=3e-3,
            device="cpu",
            teacher_dtype="float32",
        )
        a.update(kw)
        return type("A", (), a)()

    def log(self, a):
        with open(os.path.join(a.out_dir, "train_log.json")) as f:
            return json.load(f)

    def test_student_has_exactly_the_thme_q4_0_tensors_rounded(self):
        teacher = tiny_llama(seed=3)
        student, names = qad_lora.build_student(self.model_dir, "cpu")
        self.assertEqual(len(names), 14)
        t, s = dict(teacher.named_parameters()), dict(student.named_parameters())
        for n in t:
            if n in names:
                self.assertTrue(torch.equal(s[n], quant.q4_0_roundtrip(t[n])), n)
                self.assertFalse(torch.equal(s[n], t[n]), n)
            else:
                self.assertTrue(torch.equal(s[n], t[n]), n)

    def test_zero_adapter_loss_equals_plain_kl_of_the_quantized_model(self):
        a = self.args("zero", steps=1)
        qad_lora.run(a)
        first = self.log(a)[0]["loss"]
        teacher = qad_lora.build_teacher(self.model_dir, "cpu", torch.float32)
        student, _ = qad_lora.build_student(self.model_dir, "cpu")
        windows = qad_lora.windows_from_tokens(qad_lora.read_u32(self.train), a.window)
        idx = qad_lora.batch_indices(a.seed, len(windows), a.batch_size, 1)[0]
        ids = windows[idx]
        with torch.no_grad():
            want = qad_lora.kl_loss(teacher(ids).logits, student(ids).logits).item()
        self.assertGreater(want, 0.0)
        self.assertAlmostEqual(first, want, delta=1e-6)

    def test_kl_matches_a_hand_computed_value(self):
        # teacher p = (0.5, 0.5), student q = (0.25, 0.75): KL = 0.5 ln 2 + 0.5 ln(2/3)
        t = torch.log(torch.tensor([[[0.5, 0.5]]]))
        s = torch.log(torch.tensor([[[0.25, 0.75]]]))
        want = 0.5 * torch.log(torch.tensor(2.0)) + 0.5 * torch.log(torch.tensor(2.0 / 3.0))
        self.assertAlmostEqual(qad_lora.kl_loss(t, s).item(), want.item(), places=6)
        self.assertAlmostEqual(qad_lora.kl_loss(t, t).item(), 0.0, places=7)

    def test_twenty_steps_lower_the_loss(self):
        a = self.args("train", batch_size=24)  # every window each step: low noise
        result = qad_lora.run(a)
        losses = [e["loss"] for e in self.log(a)]
        self.assertEqual(len(losses), 20)
        self.assertLess(sum(losses[-5:]) / 5, 0.5 * losses[0])
        self.assertLess(result["trainable_params"], result["total_params"])
        # every trainable tensor is a LoRA matrix of the seven roles: 7 roles x 2 layers x (A, B)
        self.assertEqual(result["trainable_tensors"], 28)

    def test_outputs_and_result_fields(self):
        a = self.args("outputs", steps=2)
        result = qad_lora.run(a)
        adapter = os.path.join(a.out_dir, "adapter", "adapter_model.safetensors")
        self.assertEqual(result["adapter_bytes"], os.path.getsize(adapter))
        self.assertTrue(os.path.exists(os.path.join(a.out_dir, "merged", "model.safetensors")))
        for key in ("wall_seconds", "eval_ppl", "eval_ppl_teacher", "eval_ppl_student_q4_0"):
            self.assertGreater(result[key], 0.0, key)
        with open(os.path.join(a.out_dir, "result.json")) as f:
            self.assertEqual(json.load(f)["adapter_bytes"], result["adapter_bytes"])
        merged = qad_lora.load_merged(os.path.join(a.out_dir, "merged"))
        self.assertEqual(next(merged.parameters()).dtype, torch.float16)

    def test_merged_logits_equal_adapter_logits(self):
        student, _ = qad_lora.build_student(self.model_dir, "cpu")
        ids = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            base_logits = student(ids).logits  # before the adapter is injected into the module
        peft_model = qad_lora.add_lora(student, rank=4, alpha=8.0, seed=1)
        with torch.no_grad():
            for n, p in peft_model.named_parameters():
                if "lora_B" in n:
                    p.normal_(0.0, 0.02)
        peft_model.eval()
        with torch.no_grad():
            adapter_logits = peft_model(ids).logits
            merged = qad_lora.merge(copy.deepcopy(peft_model))
            merged_logits = merged(ids).logits
        self.assertFalse(torch.allclose(adapter_logits, base_logits, atol=1e-4))
        self.assertLess((adapter_logits - merged_logits).abs().max().item(), 1e-5)

    def test_run_is_bit_reproducible_for_a_fixed_seed(self):
        a, b = self.args("rep_a", steps=6), self.args("rep_b", steps=6)
        qad_lora.run(a)
        qad_lora.run(b)
        self.assertEqual(self.log(a), self.log(b))
        path = lambda x: os.path.join(x.out_dir, "adapter", "adapter_model.safetensors")
        with open(path(a), "rb") as fa, open(path(b), "rb") as fb:
            self.assertEqual(fa.read(), fb.read())
        c = self.args("rep_c", steps=6, seed=6)
        qad_lora.run(c)
        self.assertNotEqual(self.log(a), self.log(c))


if __name__ == "__main__":
    unittest.main()
