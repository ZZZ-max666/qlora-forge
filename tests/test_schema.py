# -*- coding: utf-8 -*-
"""schema 校验器单元测试。"""
import json
import os
import tempfile
import unittest

from toneforge.data import schema


def good_record(rid="r1"):
    return {
        "id": rid,
        "messages": [
            {"role": "user", "content": "能退货吗？"},
            {"role": "assistant", "content": "理解您的心情。\n\n根据《七天无理由退货》…\n\n请在订单页点击申请。", "loss": True},
        ],
    }


class TestValidateRecord(unittest.TestCase):
    def test_good_record(self):
        self.assertEqual(schema.validate_record(good_record()), [])

    def test_missing_or_bad_id(self):
        rec = good_record()
        del rec["id"]
        self.assertTrue(any("id" in e for e in schema.validate_record(rec)))
        rec["id"] = "  "
        self.assertTrue(any("id" in e for e in schema.validate_record(rec)))

    def test_role_must_alternate(self):
        rec = good_record()
        rec["messages"].insert(1, {"role": "user", "content": "在吗？"})
        errs = schema.validate_record(rec)
        self.assertTrue(any("交替" in e for e in errs))

    def test_must_start_user_end_assistant(self):
        rec = good_record()
        rec["messages"] = [{"role": "assistant", "content": "您好", "loss": True},
                           {"role": "user", "content": "hi"}]
        errs = schema.validate_record(rec)
        self.assertTrue(any("以 user 轮开始" in e for e in errs))
        self.assertTrue(any("以 assistant 轮结束" in e for e in errs))

    def test_system_prefix_allowed(self):
        rec = good_record()
        rec["messages"].insert(0, {"role": "system", "content": "你是客服助手"})
        self.assertEqual(schema.validate_record(rec), [])

    def test_loss_only_on_assistant_and_bool(self):
        rec = good_record()
        rec["messages"][0]["loss"] = True
        self.assertTrue(any("仅 assistant" in e for e in schema.validate_record(rec)))
        rec2 = good_record()
        rec2["messages"][1]["loss"] = "yes"
        self.assertTrue(any("布尔" in e for e in schema.validate_record(rec2)))

    def test_all_loss_false_rejected(self):
        rec = good_record()
        rec["messages"][1]["loss"] = False
        self.assertTrue(any("无训练目标" in e for e in schema.validate_record(rec)))

    def test_empty_content_rejected(self):
        rec = good_record()
        rec["messages"][1]["content"] = "   "
        self.assertTrue(any("非空字符串" in e for e in schema.validate_record(rec)))

    def test_duplicate_ids(self):
        problems = schema.validate_records([good_record("a"), good_record("a")])
        self.assertEqual(len(problems), 1)
        self.assertTrue(any("重复" in e for _, errs in problems for e in errs))


class TestFileLevel(unittest.TestCase):
    def test_validate_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "d.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(good_record("a"), ensure_ascii=False) + "\n")
                f.write(json.dumps({"id": "b", "messages": []}, ensure_ascii=False) + "\n")
            good, errors = schema.validate_file(path)
            self.assertEqual([r["id"] for r in good], ["a"])
            self.assertTrue(any("第 2 行" in e for e in errors))

    def test_bad_json_line_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "d.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not-json}\n")
            with self.assertRaises(ValueError):
                schema.load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
