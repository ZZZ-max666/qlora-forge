# -*- coding: utf-8 -*-
"""三段式结构校验器单元测试。"""
import unittest

from toneforge.eval import format_check

GOOD = (
    "非常理解您遇到尺码不合时的烦躁，别担心，我来帮您处理。\n\n"
    "根据平台《七天无理由退货》，商品未经使用、吊牌完好，签收后7天内可提交退货申请。\n\n"
    "您可以在App「我的订单」中点击「申请退货」，审核通过后快递免费上门取件。"
)
BLOB_GOOD = ("非常理解您的着急。依据平台《48小时发货保障》，订单超时即可催办。"
             "我马上向承运快递发起核查工单，24小时内给您结果。")
NO_EMPATHY = (
    "根据平台《七天无理由退货》，签收后7天内可退。\n\n"
    "请在订单页点击「申请退货」上传照片。"
)
NO_POLICY = (
    "您的心情我完全理解，别担心。\n\n"
    "请在订单页点击「申请退货」，审核通过后快递上门取件。"
)
WRONG_ORDER = (
    "理解您的心情，这事我来跟进。\n\n"
    "请在订单页点击「申请退货」并上传照片，快递会上门取件。\n\n"
    "依据平台《七天无理由退货》，签收后7天内均可申请。"
)


class TestCheckText(unittest.TestCase):
    def test_good(self):
        res = format_check.check_text(GOOD)
        self.assertTrue(res["ok"])
        self.assertTrue(res["has_empathy"] and res["has_policy"] and res["has_action"])

    def test_blob_good(self):
        self.assertTrue(format_check.check_text(BLOB_GOOD)["ok"])

    def test_missing_empathy(self):
        res = format_check.check_text(NO_EMPATHY)
        self.assertFalse(res["has_empathy"])
        self.assertFalse(res["ok"])

    def test_missing_policy(self):
        res = format_check.check_text(NO_POLICY)
        self.assertFalse(res["has_policy"])
        self.assertFalse(res["ok"])

    def test_action_before_policy_rejected(self):
        res = format_check.check_text(WRONG_ORDER)
        self.assertTrue(res["has_empathy"])
        self.assertTrue(res["has_policy"])
        self.assertFalse(res["has_action"])  # 动作必须位于政策引用之后
        self.assertFalse(res["ok"])

    def test_empty(self):
        self.assertFalse(format_check.check_text("")["ok"])

    def test_compliance_rate(self):
        recs = [{"messages": [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": c}]}
                for c in (GOOD, NO_EMPATHY, NO_POLICY, WRONG_ORDER)]
        self.assertAlmostEqual(format_check.compliance_rate(recs), 0.25)

    def test_evaluate_skips_general(self):
        tone = {"id": "t", "messages": [{"role": "user", "content": "q"},
                                        {"role": "assistant", "content": GOOD}],
                "meta": {"source": "variant", "negative_type": "none"}}
        gen = {"id": "g", "messages": [{"role": "user", "content": "q"},
                                       {"role": "assistant", "content": "391。"}],
               "meta": {"source": "general", "negative_type": "none"}}
        rep = format_check.evaluate([tone, gen])
        self.assertEqual(rep["total"], 1)
        self.assertEqual(rep["ok"], 1.0)


if __name__ == "__main__":
    unittest.main()
