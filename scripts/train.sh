#!/usr/bin/env bash
# 一键训练（Linux/WSL）：构建数据 → dry-run 校验 → QLoRA 训练
# 用法：bash scripts/train.sh [configs/smoke_qwen05b.json]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/smoke_qwen05b.json}"

echo "== 1/3 构建指令集 =="
python scripts/build_dataset.py

echo "== 2/3 dry-run 校验配置与显存估算 =="
python -m toneforge.train.qlora --config "$CONFIG" --dry-run

echo "== 3/3 启动 QLoRA 训练（需 CUDA + requirements.txt 训练依赖） =="
python -m toneforge.train.qlora --config "$CONFIG"

echo "完成。适配器位于配置中的 output_dir。"
