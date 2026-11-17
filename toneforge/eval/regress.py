# -*- coding: utf-8 -*-
"""通用能力回归清单（纯标准库）。

风格微调最大的隐性风险是"学会了客服腔，忘了算数/翻译/常识"。
本模块内置 12 道确定性小测（算术/翻译/常识/逻辑），每题带可机器判定的
检查器；对模型批量输出跑一遍即可得到 pass 率，训练前后各跑一次
即可量化通用能力回归程度。

用法：
    python -m toneforge.eval.regress --oracle                 # 自检判分器（应 100%）
    python -m toneforge.eval.regress --predictions preds.jsonl # 评模型输出 {id, output}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Callable, Dict, List, Optional


def _num_ok(expected: float):
    """在文本中提取数字并匹配期望值（取最后一个命中 1e-6 容差的数）。"""
    def check(text: str) -> bool:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return any(abs(float(x) - expected) < 1e-4 for x in nums)
    return check


def _contains_any(*keys: str):
    def check(text: str) -> bool:
        return any(k in text for k in keys)
    return check


# 每题：id / 类别 / 问题 / 判分器 / 参考答案（oracle）
QUESTIONS: List[Dict[str, Any]] = [
    {"id": "reg-math-01", "category": "算术", "prompt": "17 乘以 23 等于多少？",
     "check": _num_ok(391), "gold": "17 × 23 = 391。"},
    {"id": "reg-math-02", "category": "算术", "prompt": "计算 128 + 256 - 64。",
     "check": _num_ok(320), "gold": "128 + 256 - 64 = 320。"},
    {"id": "reg-math-03", "category": "算术", "prompt": "240 的 15% 是多少？",
     "check": _num_ok(36), "gold": "240 × 15% = 36。"},
    {"id": "reg-math-04", "category": "算术", "prompt": "7 的 6 次方是多少？",
     "check": _num_ok(117649), "gold": "7^6 = 117649。"},
    {"id": "reg-tr-01", "category": "翻译", "prompt": "把「包裹正在派送途中」翻译成英文。",
     "check": _contains_any("on the way", "in transit", "being delivered", "out for delivery"),
     "gold": "The parcel is out for delivery."},
    {"id": "reg-tr-02", "category": "翻译", "prompt": "将「Please keep the receipt」译成中文。",
     "check": _contains_any("保留", "保存", "留好"), "gold": "请保留好凭证/收据。"},
    {"id": "reg-cs-01", "category": "常识", "prompt": "一年有几个季度？",
     "check": _num_ok(4), "gold": "一年有4个季度。"},
    {"id": "reg-cs-02", "category": "常识", "prompt": "水的化学式是什么？",
     "check": _contains_any("H2O", "H₂O"), "gold": "水的化学式是 H2O。"},
    {"id": "reg-cs-03", "category": "常识", "prompt": "《西游记》的作者是谁？",
     "check": _contains_any("吴承恩"), "gold": "《西游记》作者是吴承恩。"},
    {"id": "reg-cs-04", "category": "常识", "prompt": "1 小时等于多少秒？",
     "check": _num_ok(3600), "gold": "1 小时 = 3600 秒。"},
    {"id": "reg-lg-01", "category": "逻辑", "prompt": "数列 3, 6, 12, 24 的下一项是什么？",
     "check": _num_ok(48), "gold": "等比数列，公比 2，下一项是 48。"},
    {"id": "reg-lg-02", "category": "逻辑", "prompt": "所有鸟都会飞吗？企鹅是鸟，企鹅会飞吗？",
     "check": _contains_any("不会", "不能飞", "否"), "gold": "并非所有鸟都会飞；企鹅是鸟但不会飞。"},
]


def oracle_predictions() -> Dict[str, str]:
    return {q["id"]: q["gold"] for q in QUESTIONS}


def load_predictions(path: str) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            preds[obj["id"]] = obj.get("output", "")
    return preds


def run_regression(predictions: Dict[str, str]) -> Dict[str, Any]:
    passed, failures = 0, []
    for q in QUESTIONS:
        out = predictions.get(q["id"])
        ok = out is not None and q["check"](out)
        passed += 1 if ok else 0
        if not ok:
            failures.append(q["id"])
    total = len(QUESTIONS)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "failures": failures,
        "by_category": {
            cat: {
                "n": sum(1 for q in QUESTIONS if q["category"] == cat),
                "passed": sum(1 for q in QUESTIONS if q["category"] == cat
                              and predictions.get(q["id"]) is not None
                              and q["check"](predictions[q["id"]])),
            }
            for cat in sorted({q["category"] for q in QUESTIONS})
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="通用能力回归小测")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--oracle", action="store_true", help="用内置参考答案自检判分器")
    group.add_argument("--predictions", help="模型输出 JSONL（{id, output}）")
    args = parser.parse_args(argv)

    if args.oracle:
        preds = oracle_predictions()
    else:
        preds = load_predictions(args.predictions)
    rep = run_regression(preds)
    print("通用能力回归报告")
    print(f"  pass 率 : {rep['passed']}/{rep['total']} = {rep['pass_rate']:.1%}")
    print(f"  分类别  : {rep['by_category']}")
    if rep["failures"]:
        print(f"  未通过  : {rep['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
