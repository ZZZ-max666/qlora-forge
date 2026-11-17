# -*- coding: utf-8 -*-
"""通用能力回归清单单元测试。"""
import unittest

from toneforge.eval import regress


class TestRegression(unittest.TestCase):
    def test_at_least_10_questions(self):
        self.assertGreaterEqual(len(regress.QUESTIONS), 10)

    def test_categories_covered(self):
        cats = {q["category"] for q in regress.QUESTIONS}
        for need in ("算术", "翻译", "常识"):
            self.assertIn(need, cats)

    def test_oracle_full_pass(self):
        rep = regress.run_regression(regress.oracle_predictions())
        self.assertEqual(rep["pass_rate"], 1.0)
        self.assertEqual(rep["failures"], [])

    def test_wrong_answers_fail(self):
        preds = {q["id"]: "我不知道" for q in regress.QUESTIONS}
        rep = regress.run_regression(preds)
        self.assertEqual(rep["pass_rate"], 0.0)
        self.assertEqual(len(rep["failures"]), rep["total"])

    def test_missing_prediction_counts_as_failure(self):
        rep = regress.run_regression({})
        self.assertEqual(rep["passed"], 0)

    def test_number_extractor(self):
        check = regress._num_ok(391)
        self.assertTrue(check("答案是 391 哦"))
        self.assertTrue(check("17 × 23 = 391。"))
        self.assertFalse(check("390"))
        self.assertFalse(check("没有数字"))

    def test_contains_any(self):
        check = regress._contains_any("吴承恩", "承恩")
        self.assertTrue(check("作者是吴承恩"))
        self.assertFalse(check("作者是施耐庵"))


if __name__ == "__main__":
    unittest.main()
