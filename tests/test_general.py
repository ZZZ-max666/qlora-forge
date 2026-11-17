# -*- coding: utf-8 -*-
"""通用数据接入模块单元测试。"""
import os
import unittest

from toneforge.data import general, schema

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENERAL_PATH = os.path.join(ROOT, "data", "general_samples.jsonl")


class TestGeneral(unittest.TestCase):
    def test_builtin_samples_load_and_valid(self):
        rows = general.load_general(GENERAL_PATH)
        self.assertGreaterEqual(len(rows), 60)
        records = [general.to_messages(r) for r in rows]
        self.assertEqual(schema.validate_records(records), [])

    def test_ids_unique(self):
        rows = general.load_general(GENERAL_PATH)
        ids = [r["id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_to_messages_shape(self):
        rec = general.to_messages({"id": "g1", "category": "math",
                                   "user": "1+1?", "assistant": "2"})
        self.assertEqual(rec["meta"]["source"], "general")
        self.assertEqual(rec["messages"][1]["role"], "assistant")
        self.assertTrue(rec["messages"][1]["loss"])

    def test_fetch_external_unknown_source_raises(self):
        with self.assertRaises(KeyError):
            general.fetch_external("does-not-exist")

    def test_fetch_external_is_skeleton(self):
        # 已登记来源也只打印指引并退出，绝不静默下载大文件
        name = next(iter(general.EXTERNAL_SOURCES))
        with self.assertRaises(SystemExit):
            general.fetch_external(name)


if __name__ == "__main__":
    unittest.main()
