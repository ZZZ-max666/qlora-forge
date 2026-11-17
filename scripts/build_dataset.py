# -*- coding: utf-8 -*-
"""全数据流水线入口：种子 → 指令集 → train/val/test 切分 + 统计报告。

用法（仓库根目录）：
    python scripts/build_dataset.py
    python scripts/build_dataset.py --seed 7 --out outputs/dataset_exp
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from toneforge.data.build import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
