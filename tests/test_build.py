# -*- coding: utf-8 -*-
"""数据构建流水线单元测试：确定性、条数、混入比例、切分不重叠、负样本可拦截。"""
import json
import os
import unittest

from toneforge.data import build, schema
from toneforge.eval import format_check, rule_metrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEEDS_DIR = os.path.join(ROOT, "data", "seeds")
GENERAL = os.path.join(ROOT, "data", "general_samples.jsonl")


def _dump(result):
    return json.dumps([result["train"], result["val"], result["test"]],
                      ensure_ascii=False, sort_keys=True)


class TestBuildDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build.build_dataset(SEEDS_DIR, GENERAL, seed=42)
        cls.all_ids = set()
        for split in ("train", "val", "test"):
            cls.all_ids.update(r["id"] for r in cls.result[split])

    def test_determinism_same_seed(self):
        other = build.build_dataset(SEEDS_DIR, GENERAL, seed=42)
        self.assertEqual(_dump(self.result), _dump(other))

    def test_different_seed_changes_selection(self):
        other = build.build_dataset(SEEDS_DIR, GENERAL, seed=7)
        self.assertNotEqual(_dump(self.result), _dump(other))

    def test_total_size_in_range(self):
        st = self.result["stats"]
        self.assertGreaterEqual(st["total"], 800)
        self.assertLessEqual(st["total"], 2000)
        self.assertEqual(st["total"], st["train"] + st["val"] + st["test"])

    def test_general_ratio_about_10_percent(self):
        ratio = self.result["stats"]["general_ratio"]
        self.assertAlmostEqual(ratio, 0.10, delta=0.01)

    def test_splits_disjoint_and_schema_valid(self):
        counts = [len([r["id"] for r in self.result[s]]) for s in ("train", "val", "test")]
        total = sum(len(self.result[s]) for s in ("train", "val", "test"))
        self.assertEqual(sum(counts), total)  # id 全局唯一 → 切分不重叠
        problems = schema.validate_records(
            self.result["train"] + self.result["val"] + self.result["test"])
        self.assertEqual(problems, [])

    def test_negatives_only_in_test(self):
        train_ids = {r["id"] for r in self.result["train"]}
        val_ids = {r["id"] for r in self.result["val"]}
        negs = [r for r in self.result["test"] if r["meta"]["source"] == "negative"]
        self.assertEqual(len(negs), 60)
        for r in negs:
            self.assertNotIn(r["id"], train_ids | val_ids)

    def test_positives_fully_compliant_three_part(self):
        pos = [r for r in self.result["train"] if r["meta"]["source"] in ("seed", "variant")]
        self.assertEqual(format_check.compliance_rate(pos), 1.0)

    def test_negatives_are_caught(self):
        negs = [r for r in self.result["test"] if r["meta"]["source"] == "negative"]
        for r in negs:
            kind = r["meta"]["negative_type"]
            res = format_check.check_record(r)
            if kind == "missing_empathy":
                self.assertFalse(res["has_empathy"], r["id"])
                self.assertFalse(res["ok"], r["id"])
            elif kind == "format_error":
                self.assertFalse(res["ok"], r["id"])
            elif kind == "fabricated_policy":
                # 三段式格式完好（结构校验放行），由政策落地性规则拦截
                self.assertTrue(res["ok"], r["id"])
                answer = r["messages"][1]["content"]
                ok, fab = rule_metrics.policy_grounding(
                    answer, build.POLICY_CLAUSE_REGISTRY)
                self.assertFalse(ok, r["id"])
                self.assertTrue(fab, r["id"])

    def test_variant_diversity_not_parroting(self):
        pos = [r for r in self.result["train"] if r["meta"]["source"] in ("seed", "variant")]
        users = {r["messages"][0]["content"] for r in pos}
        assistants = {r["messages"][1]["content"] for r in pos}
        self.assertGreater(len(users) / len(pos), 0.75)
        self.assertGreater(len(assistants) / len(pos), 0.85)

    def test_slot_consistency_clause_days(self):
        # 条款与时效耦合：《七天无理由退货》样本不得出现 15/30 天时效表述
        for r in self.result["train"]:
            if r["meta"]["source"] not in ("seed", "variant"):
                continue
            ans = r["messages"][1]["content"]
            if "《七天无理由退货》" in ans:
                self.assertNotIn("15天内", ans, r["id"])
                self.assertNotIn("30天内", ans, r["id"])


if __name__ == "__main__":
    unittest.main()
