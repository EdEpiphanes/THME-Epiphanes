#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Offline tests for scripts/prepare_eval_tokens.py (THME-2204).

Uses a synthetic WikiText-style zip and a toy word-level tokenizer, so no network and no
model is needed. Run with: python -m unittest scripts/test_prepare_eval_tokens.py
"""

import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "prepare_eval_tokens.py"
MODEL = "toy-model"


def read_ids(path):
    data = path.read_bytes()
    return [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class PrepareEvalTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer, models, pre_tokenizers

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        rng = random.Random(7)
        cls.train = " ".join(f"w{rng.randrange(97)}" for _ in range(6000)) + "\n"
        cls.test = " ".join(f"w{rng.randrange(97)}" for _ in range(900)) + "\n"
        vocab = {f"w{i}": i for i in range(97)}
        vocab["<unk>"] = 97
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        cls.tokenizer_path = root / "tokenizer.json"
        tokenizer.save(str(cls.tokenizer_path))
        cls.tokenizer = tokenizer
        cls.out = root / "out"
        cls.out.mkdir()
        # The script reuses a cached zip, so nothing is downloaded.
        with zipfile.ZipFile(cls.out / "wikitext-2-raw-v1.zip", "w") as archive:
            archive.writestr("wikitext-2-raw/wiki.train.raw", cls.train)
            archive.writestr("wikitext-2-raw/wiki.test.raw", cls.test)
        cls.train_ids = tokenizer.encode(cls.train, add_special_tokens=False).ids
        cls.test_ids = tokenizer.encode(cls.test, add_special_tokens=False).ids

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_script(self, *extra):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo",
                f"org/{MODEL}",
                "--tokenizer",
                str(self.tokenizer_path),
                "--out-dir",
                str(self.out),
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        return result

    def run_ok(self, *extra):
        result = self.run_script(*extra)
        self.assertEqual(result.returncode, 0, result.stderr)

    def sidecar(self, stem):
        return json.loads((self.out / f"{stem}.json").read_text(encoding="utf-8"))

    def test_test_split_keeps_the_legacy_name_and_records_the_split(self):
        self.run_ok()
        stem = f"wikitext2_test.{MODEL}"
        self.assertEqual(read_ids(self.out / f"{stem}.u32"), self.test_ids)
        meta = self.sidecar(stem)
        self.assertEqual(meta["split"], "test")
        self.assertEqual(meta["corpus_member"], "wikitext-2-raw/wiki.test.raw")
        self.assertEqual(meta["corpus_sha256"], sha256(self.test.encode("utf-8")))
        self.assertEqual(meta["corpus_bytes"], len(self.test.encode("utf-8")))
        self.assertEqual(meta["split_token_count"], len(self.test_ids))
        self.assertEqual(meta["token_count"], len(self.test_ids))
        self.assertEqual(meta["benchmark_id"], "wikitext2-raw")
        self.assertIsNone(meta["selection"])

    def test_calibration_windows_come_from_train_and_reproduce(self):
        extra = ["--split", "train", "--calibration-windows", "4", "--window-len", "128"]
        stem = f"wikitext2_train.{MODEL}.cal4x128.s1234"
        self.run_ok(*extra, "--seed", "1234")
        path = self.out / f"{stem}.u32"
        first_hash = sha256(path.read_bytes())
        meta = self.sidecar(stem)
        offsets = meta["selection"]["offsets"]
        self.assertEqual(len(offsets), 4)
        self.assertEqual(offsets, sorted(offsets))
        self.assertTrue(all(b - a >= 128 for a, b in zip(offsets, offsets[1:])))
        expected = [t for o in offsets for t in self.train_ids[o : o + 128]]
        self.assertEqual(read_ids(path), expected)
        self.assertEqual(meta["split_token_count"], len(self.train_ids))
        self.assertEqual(meta["token_count"], 512)
        self.assertEqual(meta["selection"]["seed"], 1234)

        # Same seed reproduces the file; another seed draws other windows.
        self.run_ok(*extra, "--seed", "1234")
        self.assertEqual(sha256(path.read_bytes()), first_hash)
        self.run_ok(*extra, "--seed", "2")
        other = self.sidecar(f"wikitext2_train.{MODEL}.cal4x128.s2")
        self.assertNotEqual(other["selection"]["offsets"], offsets)

    def test_shuffle_keeps_the_tokens_and_changes_the_order(self):
        base = ["--split", "train", "--calibration-windows", "4", "--window-len", "128"]
        self.run_ok(*base, "--seed", "1234")
        plain = read_ids(self.out / f"wikitext2_train.{MODEL}.cal4x128.s1234.u32")
        self.run_ok(*base, "--seed", "1234", "--shuffle")
        stem = f"wikitext2_train.{MODEL}.cal4x128.s1234.shuffled"
        shuffled = read_ids(self.out / f"{stem}.u32")
        self.assertEqual(sorted(shuffled), sorted(plain))
        self.assertNotEqual(shuffled, plain)
        meta = self.sidecar(stem)
        self.assertEqual(meta["order"], "shuffled")
        self.assertEqual(meta["selection"]["offsets"], self.sidecar(f"wikitext2_train.{MODEL}.cal4x128.s1234")["selection"]["offsets"])

    def test_random_ids_stay_in_the_vocabulary_and_are_marked(self):
        base = ["--split", "train", "--calibration-windows", "4", "--window-len", "128", "--seed", "1234"]
        self.assertNotEqual(self.run_script(*base, "--random-ids").returncode, 0)
        self.run_ok(*base, "--random-ids", "--vocab-size", "98")
        stem = f"wikitext2_train.{MODEL}.cal4x128.s1234.randomids.v98"
        ids = read_ids(self.out / f"{stem}.u32")
        self.assertEqual(len(ids), 512)
        self.assertTrue(all(0 <= t < 98 for t in ids))
        self.assertNotEqual(ids, self.train_ids[:512])
        meta = self.sidecar(stem)
        self.assertEqual(meta["random_ids"], {"seed": 1234, "vocab_size": 98})

    def test_impossible_window_counts_are_refused(self):
        result = self.run_script("--split", "train", "--calibration-windows", "100000", "--window-len", "128")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot place", result.stderr)


if __name__ == "__main__":
    unittest.main()
