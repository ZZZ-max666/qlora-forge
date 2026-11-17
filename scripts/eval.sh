#!/usr/bin/env bash
# 一键评估（纯标准库规则指标；模型生成结果需先落盘为 JSONL）
# 用法：bash scripts/eval.sh [生成结果.jsonl，默认 outputs/dataset/test.jsonl]
set -euo pipefail
cd "$(dirname "$0")/.."

PRED="${1:-outputs/dataset/test.jsonl}"

echo "== 1/3 三段式结构遵循率 =="
python -m toneforge.eval.format_check "$PRED"

echo "== 2/3 确定性规则指标（政策落地/超长/复读） =="
python -m toneforge.eval.rule_metrics "$PRED"

echo "== 3/3 通用能力回归（--oracle 为判分器自检；评模型请加 --predictions） =="
python -m toneforge.eval.regress --oracle

echo "如需人工盲评：python -m toneforge.eval.blind_export --input <成对输出.jsonl>"
