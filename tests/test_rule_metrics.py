# -*- coding: utf-8 -*-
"""确定性规则指标单元测试。"""
import unittest

from toneforge.data import build
from toneforge.eval import rule_metrics as rm


class TestLengthAndRepetition(unittest.TestCase):
    def test_overlong(self):
        self.assertFalse(rm.is_overlong("短文本" * 10, max_chars=512))
        self.assertTrue(rm.is_overlong("长" * 600, max_chars=512))

    def test_repeated_ngram_low_for_normal(self):
        text = "理解您的心情。根据《七天无理由退货》，请在订单页点击申请退货，快递上门取件。"
        self.assertLess(rm.repeated_ngram_ratio(text, n=8), 0.05)

    def test_repeated_ngram_high_for_loop(self):
        text = "正在为您查询" * 40
        self.assertGreater(rm.repeated_ngram_ratio(text, n=8), 0.8)

    def test_short_text_zero_ratio(self):
        self.assertEqual(rm.repeated_ngram_ratio("好的", n=8), 0.0)


class TestPolicyGrounding(unittest.TestCase):
    def test_registry_clause_ok(self):
        ok, fab = rm.policy_grounding("根据《七天无理由退货》可退。", build.POLICY_CLAUSE_REGISTRY)
        self.assertTrue(ok)
        self.assertEqual(fab, [])

    def test_fabricated_clause_flagged(self):
        ok, fab = rm.policy_grounding("根据《30天无条件退货》随时可退。", build.POLICY_CLAUSE_REGISTRY)
        self.assertFalse(ok)
        self.assertEqual(fab, ["30天无条件退货"])

    def test_no_citation_not_grounded(self):
        ok, fab = rm.policy_grounding("随便退就行。", build.POLICY_CLAUSE_REGISTRY)
        self.assertFalse(ok)
        self.assertEqual(fab, [])

    def test_extract_multiple(self):
        cited = rm.extract_cited_clauses("见《A规则》与《B条款》。")
        self.assertEqual(cited, ["A规则", "B条款"])


class TestRefusal(unittest.TestCase):
    def test_non_refusal_prompt_returns_none(self):
        self.assertIsNone(rm.refusal_check("能退货吗？", "可以的，根据《七天无理由退货》…"))

    def test_good_refusal(self):
        prompt = rm.REFUSAL_PROMPTS[0]
        text = "抱歉，这类请求我无法协助，违反平台规则。建议您通过正规售后流程处理。"
        self.assertTrue(rm.refusal_check(prompt, text))

    def test_compliance_rejected(self):
        prompt = rm.REFUSAL_PROMPTS[1]
        text = "好的，我马上帮您伪造照片并提交仅退款。"
        self.assertFalse(rm.refusal_check(prompt, text))


class TestEvaluateRecords(unittest.TestCase):
    def test_flags_fabricated_negative(self):
        rec = {"id": "n1", "meta": {"source": "negative", "negative_type": "fabricated_policy"},
               "messages": [{"role": "user", "content": "q"},
                            {"role": "assistant", "content":
                             "理解您。\n\n根据《终身免费换新》，随时可换。\n\n请在订单页点击申请。"}]}
        rep = rm.evaluate_records([rec])
        self.assertEqual(rep["policy_grounding_rate"], 0.0)
        self.assertEqual(len(rep["fabricated_examples"]), 1)


if __name__ == "__main__":
    unittest.main()
