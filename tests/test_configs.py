# -*- coding: utf-8 -*-
"""configs/*.json 合法性与字段完备性测试。"""
import json
import os
import unittest

from toneforge.train import qlora, vram

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = ["configs/smoke_qwen05b.json", "configs/prod_qwen7b.json"]


class TestConfigs(unittest.TestCase):
    def test_all_configs_valid(self):
        for rel in CONFIGS:
            path = os.path.join(ROOT, rel)
            with self.subTest(rel):
                cfg = qlora.load_config(path)
                self.assertEqual(qlora.validate_config(cfg), [])

    def test_smoke_fits_3060(self):
        cfg = qlora.load_config(os.path.join(ROOT, CONFIGS[0]))
        est = vram.estimate_from_config(cfg)
        self.assertLess(est["total_gb"], 6.0)
        self.assertTrue(cfg["train"]["gradient_checkpointing"])
        self.assertLessEqual(cfg["train"]["seq_len"], 1024)

    def test_prod_uses_nf4_double_quant(self):
        cfg = qlora.load_config(os.path.join(ROOT, CONFIGS[1]))
        q = cfg["quant"]
        self.assertTrue(q["load_in_4bit"])
        self.assertEqual(q["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(q["bnb_4bit_use_double_quant"])
        self.assertEqual(cfg["lora"]["r"], 16)
        self.assertEqual(cfg["lora"]["alpha"], 32)
        # 注意力 + FFN 目标模块
        tm = set(cfg["lora"]["target_modules"])
        self.assertTrue({"q_proj", "k_proj", "v_proj", "o_proj"} <= tm)
        self.assertTrue({"gate_proj", "up_proj", "down_proj"} <= tm)

    def test_invalid_configs_rejected(self):
        bad = {"model": "x", "model_profile": "qwen2.5-0.5b", "method": "qlora"}
        errs = qlora.validate_config(bad)
        self.assertTrue(errs)

        cfg = qlora.load_config(os.path.join(ROOT, CONFIGS[0]))
        cfg["quant"]["bnb_4bit_quant_type"] = "int8"
        self.assertTrue(any("nf4" in e for e in qlora.validate_config(cfg)))

        cfg = qlora.load_config(os.path.join(ROOT, CONFIGS[0]))
        cfg["lora"]["r"] = 0
        self.assertTrue(any("lora.r" in e for e in qlora.validate_config(cfg)))

        cfg = qlora.load_config(os.path.join(ROOT, CONFIGS[0]))
        cfg["model_profile"] = "unknown-9b"
        self.assertTrue(any("model_profile" in e for e in qlora.validate_config(cfg)))

    def test_required_sections(self):
        cfg = qlora.load_config(os.path.join(ROOT, CONFIGS[0]))
        for section in qlora.REQUIRED_SECTIONS:
            self.assertIn(section, cfg)


if __name__ == "__main__":
    unittest.main()
