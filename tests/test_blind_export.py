# -*- coding: utf-8 -*-
"""盲评导出：打乱可复现、格式正确。"""
import os
import random
import tempfile
import unittest

from toneforge.eval import blind_export

PAIRS = [{"id": f"q{i:02d}", "response_a": f"A{i:02d}", "response_b": f"B{i:02d}"}
         for i in range(30)]


class TestShuffle(unittest.TestCase):
    def test_shuffle_pair_deterministic_per_rng_stream(self):
        r1, r2 = random.Random(99), random.Random(99)
        self.assertEqual(blind_export.shuffle_pair("a", "b", r1),
                         blind_export.shuffle_pair("a", "b", r2))

    def test_shuffle_balanced(self):
        rng = random.Random(0)
        swapped = sum(1 for _ in range(2000)
                      if blind_export.shuffle_pair("a", "b", rng) == ["b", "a"])
        self.assertTrue(800 <= swapped <= 1200)


class TestExport(unittest.TestCase):
    def test_same_seed_identical_file(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "b1.csv")
            p2 = os.path.join(td, "b2.csv")
            n = blind_export.export_blind_csv(PAIRS, p1, seed=13)
            blind_export.export_blind_csv(PAIRS, p2, seed=13)
            self.assertEqual(n, 30)
            with open(p1, "rb") as f1, open(p2, "rb") as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_different_seed_changes_order(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "b1.csv")
            p2 = os.path.join(td, "b2.csv")
            blind_export.export_blind_csv(PAIRS, p1, seed=13)
            blind_export.export_blind_csv(PAIRS, p2, seed=2024)
            with open(p1, "rb") as f1, open(p2, "rb") as f2:
                self.assertNotEqual(f1.read(), f2.read())

    def test_header_and_empty_preference(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "b.csv")
            blind_export.export_blind_csv(PAIRS, path, seed=1)
            with open(path, "r", encoding="utf-8-sig") as f:
                lines = f.read().splitlines()
            self.assertEqual(lines[0], "item_id,response_1,response_2,preference,comment")
            self.assertEqual(len(lines), 31)
            for line in lines[1:]:
                cols = line.split(",")
                self.assertEqual(cols[3], "")  # preference 留空待人工填写
                self.assertEqual({cols[1], cols[2]},
                                 {f"A{cols[0][1:]}", f"B{cols[0][1:]}"})

    def test_load_pairs_missing_field(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "p.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"id": "x", "response_a": "a"}\n')
            with self.assertRaises(ValueError):
                blind_export.load_pairs(path)


if __name__ == "__main__":
    unittest.main()
