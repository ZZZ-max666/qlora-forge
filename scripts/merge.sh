#!/usr/bin/env bash
# 一键合并 + 可选 GPTQ 量化（Linux/WSL）
# 用法：bash scripts/merge.sh <基座模型> <适配器目录> <合并输出目录> [gptq输出目录]
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${1:?用法: merge.sh <base> <adapter> <merged_out> [gptq_out]}"
ADAPTER="${2:?缺少适配器目录}"
MERGED="${3:?缺少合并输出目录}"
GPTQ_OUT="${4:-}"

echo "== LoRA 合并为完整权重 =="
python -m toneforge.deploy.merge --base "$BASE" --adapter "$ADAPTER" --out "$MERGED"

if [ -n "$GPTQ_OUT" ]; then
  echo "== GPTQ INT4 量化（仅 Linux） =="
  python -m toneforge.deploy.quant --model "$MERGED" --out "$GPTQ_OUT"
fi

echo "完成。可用 vLLM 部署：python -m toneforge.deploy.serve_vllm --serve --model $MERGED"
