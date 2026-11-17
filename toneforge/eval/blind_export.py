# -*- coding: utf-8 -*-
"""人工盲评导出（纯标准库）。

把两个模型（如 基线 vs 微调后）对同一批问题的回答成对导出为 CSV，
逐行随机打乱左右顺序并隐藏来源标签，评分人只看到 回复1/回复2，
避免"先入为主"的位置与身份偏置。固定 seed 时打乱结果可复现。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from typing import Any, Dict, List, Optional

CSV_HEADER = ["item_id", "response_1", "response_2", "preference", "comment"]


def load_pairs(path: str) -> List[Dict[str, str]]:
    """读取 {id, response_a, response_b} 逐行 JSON。"""
    pairs: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key in ("id", "response_a", "response_b"):
                if not isinstance(obj.get(key), str) or not obj[key].strip():
                    raise ValueError(f"{path} 第 {lineno} 行缺少字段 {key!r}")
            pairs.append(obj)
    return pairs


def shuffle_pair(a: str, b: str, rng: random.Random) -> List[str]:
    return [a, b] if rng.random() < 0.5 else [b, a]


def export_blind_csv(pairs: List[Dict[str, str]], out_path: str, seed: int = 13) -> int:
    """导出盲评 CSV（utf-8-sig 便于 Excel 直接打开），返回行数。"""
    rng = random.Random(seed)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for p in pairs:
            r1, r2 = shuffle_pair(p["response_a"], p["response_b"], rng)
            writer.writerow([p["id"], r1, r2, "", ""])
    return len(pairs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="导出人工盲评 CSV")
    parser.add_argument("--input", required=True, help="成对输出 JSONL（id/response_a/response_b）")
    parser.add_argument("--out", default="outputs/eval/blind_review.csv")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    pairs = load_pairs(args.input)
    n = export_blind_csv(pairs, args.out, seed=args.seed)
    print(f"已导出盲评表 {args.out}（{n} 行，seed={args.seed}）")
    print("评分说明：preference 列填 1 / 2 / tie；comment 列可注明三段式与政策引用问题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
