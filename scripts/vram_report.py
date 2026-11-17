# -*- coding: utf-8 -*-
"""显存估算表：0.5B / 1.5B / 7B × 序列长度 的 QLoRA 峰值显存估算。

用法（仓库根目录）：
    python scripts/vram_report.py            # markdown 表格
    python scripts/vram_report.py --full     # 附全参微调对照
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from toneforge.train import vram  # noqa: E402


def build_rows(with_full: bool = False):
    rows = []
    for profile in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-7b"):
        for seq_len in (512, 1024, 2048, 4096):
            rows.append(vram.estimate(profile, seq_len=seq_len, batch_size=1,
                                      lora_r=16, gradient_checkpointing=True))
    if with_full:
        for profile in ("qwen2.5-0.5b", "qwen2.5-7b"):
            rows.append(vram.estimate(profile, seq_len=1024, batch_size=1,
                                      method="full", gradient_checkpointing=True))
    return rows


def main() -> int:
    argv = sys.argv[1:]
    with_full = "--full" in argv
    rows = build_rows(with_full)
    print("# toneforge 显存估算表（QLoRA：NF4 双量化 + AdamW，batch=1, r=16, 注意力+FFN 目标模块）")
    print()
    print(vram.format_table(rows))
    print()
    print("说明：估算公式与常数含义见 README「显存估算」一节；数值为训练峰值近似，")
    print("实际显存受框架缓存与分配器碎片影响，建议预留 10% 余量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
