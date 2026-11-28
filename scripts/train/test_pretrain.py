# SPDX-License-Identifier: AGPL-3.0-only
import argparse
import json
import math
import os
import tempfile
import unittest

import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM

import prepare_pretrain_data as prep
import pretrain


def tiny_dir(root, vocab):
    cfg = GPTNeoXConfig(vocab_size=vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=64, rotary_pct=0.25)
    d = os.path.join(root, "model")
    cfg.save_pretrained(d)
    return d


def tiny_llama_dir(root, vocab=64, subdir="llama-model"):
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=88,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    d = os.path.join(root, subdir)
    cfg.save_pretrained(d)
    return d


def args(root, arm, data, val, steps=60):
    return argparse.Namespace(arm=arm, model_dir=os.path.join(root, "model"), data=data, val=val,
                              out=os.path.join(root, "out-" + arm), steps=steps, ckpt_every=30, batch=8, seq=16,
                              lr=3e-3, warmup=5, weight_decay=0.0, val_windows=4, seed=7, cpu=True, micro_batch=0)


def llama_args(root, arm, model_dir, data, val, steps=10, ckpt_every=10, out_name=None, **extra):
    ns = argparse.Namespace(
        arm=arm,
        model_dir=model_dir,
        data=data,
        val=val,
        out=os.path.join(root, out_name or ("out-" + arm)),
        steps=steps,
        ckpt_every=ckpt_every,
        batch=8,
        seq=16,
        lr=3e-3,
        warmup=5,
        weight_decay=0.0,
        val_windows=4,
        seed=7,
        cpu=True,
        micro_batch=0,
    )
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


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

    def test_loop_arm_export_has_6_layers_equal_repeats_and_matches_in_memory_logits(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = tiny_llama_dir(root, vocab=64)
            data = os.path.join(root, "d.u16")
            np.random.default_rng(0).integers(0, 64, 4000, dtype=np.uint16).tofile(data)

            rep = pretrain.run(llama_args(root, "loop", model_dir, data, data, steps=6, ckpt_every=6))
            self.assertEqual(rep["layer_schedule"], [0, 1, 1, 1, 1, 2])
            ckpt_dir = os.path.join(root, "out-loop", rep["checkpoints"][-1]["dir"])
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "model.safetensors")))
            exported = LlamaForCausalLM.from_pretrained(ckpt_dir)
            self.assertEqual(exported.config.num_hidden_layers, 6)
            self.assertEqual(len(exported.model.layers), 6)
            ref = dict(exported.model.layers[1].named_parameters())
            for pos in (2, 3, 4):
                for name, param in exported.model.layers[pos].named_parameters():
                    self.assertTrue(torch.equal(ref[name], param), f"mismatch at layer {pos}.{name}")
            self.assertFalse(torch.equal(exported.model.layers[0].self_attn.q_proj.weight,
                                         exported.model.layers[1].self_attn.q_proj.weight))
            self.assertFalse(torch.equal(exported.model.layers[5].self_attn.q_proj.weight,
                                         exported.model.layers[1].self_attn.q_proj.weight))

            in_mem = pretrain.build_model("loop", model_dir, seed=7)
            in_mem.train()
            opt = torch.optim.AdamW(pretrain.unique_params(in_mem), lr=3e-3)
            ids = pretrain.load_ids(data, "loop")
            rng = np.random.default_rng(7)
            for _ in range(4):
                x, y = pretrain.batch(ids, rng, 8, 16, "cpu")
                opt.zero_grad(set_to_none=True)
                logits = in_mem(x).logits.float()
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                loss.backward()
                opt.step()
            in_mem_dir = pretrain.save(in_mem, os.path.join(root, "in-mem-out"), 4, model_dir, "loop")
            reloaded = LlamaForCausalLM.from_pretrained(in_mem_dir)
            in_mem.eval()
            reloaded.eval()
            fixed_x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
            with torch.no_grad():
                self.assertTrue(torch.allclose(reloaded(fixed_x).logits, in_mem(fixed_x).logits, atol=1e-5))
            del ids  # release the memory map so Windows can delete the temporary folder

    def test_loop_arm_has_fewer_unique_params_than_llama_arm(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = tiny_llama_dir(root, vocab=64)
            data = os.path.join(root, "d.u16")
            np.random.default_rng(0).integers(0, 64, 4000, dtype=np.uint16).tofile(data)
            llama_rep = pretrain.run(llama_args(root, "llama", model_dir, data, data, steps=2, ckpt_every=2))
            loop_rep = pretrain.run(llama_args(root, "loop", model_dir, data, data, steps=2, ckpt_every=2))
            self.assertLess(loop_rep["unique_parameters"], llama_rep["parameters"])
            self.assertEqual(loop_rep["parameters"], loop_rep["unique_parameters"])

    def test_same_seed_gives_identical_losses_for_loop_arm(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = tiny_llama_dir(root, vocab=64)
            data = os.path.join(root, "d.u16")
            np.random.default_rng(0).integers(0, 64, 4000, dtype=np.uint16).tofile(data)
            a = pretrain.run(llama_args(root, "loop", model_dir, data, data, steps=20, ckpt_every=10, out_name="out-a"))
            b = pretrain.run(llama_args(root, "loop", model_dir, data, data, steps=20, ckpt_every=10, out_name="out-b"))
            self.assertEqual(a["layer_schedule"], b["layer_schedule"])
            self.assertEqual([c["val_loss"] for c in a["checkpoints"]], [c["val_loss"] for c in b["checkpoints"]])

    def test_distillation_records_teacher_and_zero_initial_kl(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = tiny_llama_dir(root, vocab=64)
            data = os.path.join(root, "d.u16")
            np.random.default_rng(0).integers(0, 64, 4000, dtype=np.uint16).tofile(data)

            teacher_dir = os.path.join(root, "teacher")
            pretrain.build_model("llama", model_dir, seed=99).save_pretrained(teacher_dir, safe_serialization=True)
            rep = pretrain.run(llama_args(
                root, "llama", model_dir, data, data, steps=6, ckpt_every=6, micro_batch=4,
                distill_from=teacher_dir, distill_weight=0.5, distill_temp=2.0,
            ))
            self.assertEqual(rep["distill_from"], "teacher")
            self.assertEqual(rep["distill_weight"], 0.5)
            self.assertEqual(rep["distill_temp"], 2.0)
            self.assertTrue(math.isfinite(rep["first_train_loss"]))
            self.assertTrue(math.isfinite(rep["last_train_loss"]))
            with open(os.path.join(root, "out-llama", "report.json"), encoding="utf-8") as f:
                disk_rep = json.load(f)
            self.assertEqual(disk_rep["distill_from"], "teacher")

            same_teacher_dir = os.path.join(root, "same-teacher")
            pretrain.build_model("llama", model_dir, seed=7).save_pretrained(same_teacher_dir, safe_serialization=True)
            rep0 = pretrain.run(llama_args(
                root, "llama", model_dir, data, data, steps=2, ckpt_every=2, out_name="out-zero-kl",
                micro_batch=4, distill_from=same_teacher_dir, distill_weight=1.0, distill_temp=1.5,
            ))
            self.assertAlmostEqual(rep0["first_train_loss"], 0.0, places=5)

    def test_llama_arm_learns_a_repeating_pattern_and_reloads(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = tiny_llama_dir(root, vocab=64)
            data = os.path.join(root, "d.u16")
            np.tile(np.arange(10, 20, dtype=np.uint16), 400).tofile(data)
            rep = pretrain.run(llama_args(root, "llama", model_dir, data, data, steps=60, ckpt_every=30))
            first, last = rep["checkpoints"][0]["val_loss"], rep["checkpoints"][-1]["val_loss"]
            self.assertLess(last, 0.5 * first)
            self.assertAlmostEqual(first, float(np.log(64)), delta=0.8)
            d = os.path.join(root, "out-llama", rep["checkpoints"][-1]["dir"])
            m = LlamaForCausalLM.from_pretrained(d)
            self.assertEqual(m.config.vocab_size, 64)
            self.assertEqual(m.config.num_hidden_layers, 6)
            x = torch.tensor([[10, 11, 12, 13]])
            self.assertEqual(int(m(x).logits[0, -1].argmax()), 14)


if __name__ == "__main__":
    unittest.main()
