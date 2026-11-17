# -*- coding: utf-8 -*-
"""显存估算器数值单元测试。"""
import json
import os
import unittest

from toneforge.train import vram

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestLoraParamCount(unittest.TestCase):
    def test_attention_ffn_gt_attention_only(self):
        p = vram.MODEL_PROFILES["qwen2.5-7b"]
        attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
        both = vram.DEFAULT_TARGET_MODULES
        self.assertGreater(vram.lora_param_count(p, 16, both),
                           vram.lora_param_count(p, 16, attn))

    def test_7b_r16_all_linear_in_20_60M(self):
        p = vram.MODEL_PROFILES["qwen2.5-7b"]
        n = vram.lora_param_count(p, 16, vram.DEFAULT_TARGET_MODULES)
        self.assertTrue(20e6 <= n <= 60e6, n)

    def test_scales_with_r(self):
        p = vram.MODEL_PROFILES["qwen2.5-7b"]
        n8 = vram.lora_param_count(p, 8, vram.DEFAULT_TARGET_MODULES)
        n16 = vram.lora_param_count(p, 16, vram.DEFAULT_TARGET_MODULES)
        self.assertEqual(n16, 2 * n8)

    def test_unknown_profile_raises(self):
        with self.assertRaises(KeyError):
            vram.get_profile("nope")


class TestEstimate(unittest.TestCase):
    def test_smoke_05b_under_6gb(self):
        est = vram.estimate("qwen2.5-0.5b", seq_len=1024, batch_size=2,
                            gradient_checkpointing=True)
        self.assertLess(est["total_gb"], 6.0)

    def test_7b_qlora_2048_in_10_16(self):
        est = vram.estimate("qwen2.5-7b", seq_len=2048, batch_size=1,
                            gradient_checkpointing=True)
        self.assertTrue(10.0 <= est["total_gb"] <= 16.0, est["total_gb"])

    def test_7b_qlora_4096_in_10_16(self):
        est = vram.estimate("qwen2.5-7b", seq_len=4096, batch_size=1,
                            gradient_checkpointing=True)
        self.assertTrue(10.0 <= est["total_gb"] <= 16.0, est["total_gb"])

    def test_monotonic_in_seq_batch_r(self):
        base = vram.estimate("qwen2.5-1.5b", seq_len=1024)["total_gb"]
        self.assertGreater(vram.estimate("qwen2.5-1.5b", seq_len=2048)["total_gb"], base)
        self.assertGreater(vram.estimate("qwen2.5-1.5b", seq_len=1024, batch_size=4)["total_gb"], base)
        self.assertGreater(vram.estimate("qwen2.5-1.5b", seq_len=1024, lora_r=64)["total_gb"], base)

    def test_checkpointing_reduces(self):
        on = vram.estimate("qwen2.5-7b", seq_len=2048, gradient_checkpointing=True)["total_gb"]
        off = vram.estimate("qwen2.5-7b", seq_len=2048, gradient_checkpointing=False)["total_gb"]
        self.assertLess(on, off)

    def test_full_ft_much_bigger(self):
        q = vram.estimate("qwen2.5-7b", seq_len=1024, method="qlora")["total_gb"]
        f = vram.estimate("qwen2.5-7b", seq_len=1024, method="full")["total_gb"]
        self.assertGreater(f, 8 * q)

    def test_kv_cache_optional(self):
        est = vram.estimate("qwen2.5-7b", seq_len=2048, include_kv=True)
        self.assertGreater(est["kv_cache_gb"], 0.0)

    def test_trainable_ratio_small_for_lora(self):
        est = vram.estimate("qwen2.5-7b", seq_len=2048)
        self.assertLess(est["trainable_ratio"], 0.02)

    def test_estimate_from_smoke_config(self):
        with open(os.path.join(ROOT, "configs", "smoke_qwen05b.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        est = vram.estimate_from_config(cfg)
        self.assertLess(est["total_gb"], 6.0)

    def test_format_table_rows(self):
        rows = [vram.estimate("qwen2.5-0.5b", seq_len=1024)]
        table = vram.format_table(rows)
        self.assertIn("qwen2.5-0.5b", table)
        self.assertEqual(len(table.splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
