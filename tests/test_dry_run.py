# -*- coding: utf-8 -*-
"""qlora --dry-run：无 GPU/无第三方依赖环境下必须成功。"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from toneforge.data import schema
from toneforge.train import qlora

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write_valid_jsonl(path, n=3):
    recs = []
    for i in range(n):
        recs.append({
            "id": f"dry-{i}",
            "messages": [
                {"role": "user", "content": f"问题{i}"},
                {"role": "assistant", "content": "理解您。\n\n根据《七天无理由退货》可退。\n\n请在订单页点击申请。",
                 "loss": True},
            ],
            "meta": {"source": "variant", "negative_type": "none"},
        })
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs


class TestDryRun(unittest.TestCase):
    def test_smoke_config_dry_run_ok(self):
        cfg = qlora.load_config(os.path.join(ROOT, "configs", "smoke_qwen05b.json"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = qlora.dry_run(cfg, base_dir=ROOT)
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("dry-run", out)
        self.assertIn("预计峰值显存", out)
        self.assertNotIn("Downloading", out)  # 不加载模型

    def test_dry_run_with_built_data(self):
        cfg = qlora.load_config(os.path.join(ROOT, "configs", "smoke_qwen05b.json"))
        with tempfile.TemporaryDirectory() as td:
            _write_valid_jsonl(os.path.join(td, "train.jsonl"))
            _write_valid_jsonl(os.path.join(td, "val.jsonl"), n=1)
            cfg["data"] = {"train": "train.jsonl", "val": "val.jsonl"}
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = qlora.dry_run(cfg, base_dir=td)
            self.assertEqual(code, 0)
            self.assertIn("3 条", buf.getvalue())

    def test_dry_run_invalid_data_fails(self):
        cfg = qlora.load_config(os.path.join(ROOT, "configs", "smoke_qwen05b.json"))
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "train.jsonl")
            with open(bad, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "x", "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"}]}, ensure_ascii=False) + "\n")
            _write_valid_jsonl(os.path.join(td, "val.jsonl"), n=1)
            cfg["data"] = {"train": "train.jsonl", "val": "val.jsonl"}
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = qlora.dry_run(cfg, base_dir=td)
            self.assertEqual(code, 1)

    def test_dry_run_invalid_config_fails(self):
        cfg = qlora.load_config(os.path.join(ROOT, "configs", "smoke_qwen05b.json"))
        cfg["method"] = "dream"
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = qlora.dry_run(cfg, base_dir=ROOT)
        self.assertEqual(code, 1)
        self.assertIn("配置错误", buf.getvalue())

    def test_main_entry_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = qlora.main(["--config", os.path.join(ROOT, "configs", "smoke_qwen05b.json"),
                               "--dry-run", "--base-dir", ROOT])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
