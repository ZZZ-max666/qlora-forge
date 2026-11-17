# RTX 3060 6GB 复现指南（从 clone 到评估结果）

> 训练与量化步骤推荐在 **WSL2 (Ubuntu 22.04)** 或任意 Linux GPU 机器上执行：
> bitsandbytes 与 auto-gptq 对 Windows 原生环境支持有限。
> 数据构建、显存估算、单元测试、dry-run、规则评估均为纯标准库，Windows 原生 Python 3.12 可直接运行。

## 0. 环境准备（WSL2）

```bash
# 确认 Windows 侧已装 NVIDIA 驱动（无需在 WSL 里装驱动）
nvidia-smi

sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. 构建数据集（任何平台）

```bash
python scripts/build_dataset.py
# 输出 outputs/dataset/{train,val,test}.jsonl + stats.json
# 期望：总量 900 条，通用占比 10.0%，正样本三段式遵循率 100%
```

## 2. 训练前校验（任何平台，无需 GPU）

```bash
python -m toneforge.train.qlora --config configs/smoke_qwen05b.json --dry-run
# 期望：配置合法、数据校验通过、预计峰值显存 ≈ 2.8 GB（< 6 GB，3060 可跑）
```

## 3. 冒烟训练（GPU）

```bash
bash scripts/train.sh configs/smoke_qwen05b.json
# Qwen2.5-0.5B-Instruct + QLoRA，672 条 × 1 epoch，有效 batch 16，约 42 steps
# 单卡 3060 6G 预计 < 15 分钟（含首次下载基座权重）
```

## 4. 合并与评估

```bash
bash scripts/merge.sh Qwen/Qwen2.5-0.5B-Instruct \
    outputs/checkpoints/smoke outputs/merged/smoke

# 用合并后模型对 test 集的 user 轮批量生成，落盘为
# outputs/eval/preds.jsonl（{id, output} 逐行 JSON）后：
bash scripts/eval.sh outputs/dataset/test.jsonl
python -m toneforge.eval.regress --predictions outputs/eval/preds.jsonl
python -m toneforge.eval.format_check outputs/eval/preds.jsonl
```

盲评导出（对比基线与微调模型）：

```bash
python -m toneforge.eval.blind_export --input outputs/eval/pairs.jsonl \
    --out outputs/eval/blind_review.csv --seed 13
```

## 5. 部署（可选）

```bash
python -m toneforge.deploy.serve_vllm --serve --model outputs/merged/smoke --port 8000
python -m toneforge.deploy.serve_vllm --bench --url http://127.0.0.1:8000 --rounds 32
```

## 6. 显存不足时的降级选项（按代价从小到大）

1. `train.seq_len` 1024 → 512（激活近似线性下降）；
2. `train.batch_size` → 1，同时 `grad_accum` ×2 保持有效 batch；
3. 确认 `gradient_checkpointing: true`（估算表见 README，约省 2-4 GB 激活）；
4. 优化器换 `paged_adamw_8bit`（本仓库默认），或 `optim="adafactor"` 进一步省优化器状态；
5. `lora.target_modules` 收缩为仅注意力四投影（可训练参数约降一半）；
6. 换更小基座（7B → 1.5B → 0.5B）；
7. 以上都不够：只跑 dry-run + 数据/评估链路（本仓库 CPU 可完成的部分）。

## 7. 7B 生产配置说明

`configs/prod_qwen7b.json`（seq 2048 / batch 1 / accum 16 / r 16 / NF4）估算峰值
约 10.1 GB，3060 6G **无法**直接训练；需 12 GB 以上显存（如 3060 12G 关闭
seq 至 1024 勉强、4060 Ti 16G / A100 舒适）。先用
`python -m toneforge.train.qlora --config configs/prod_qwen7b.json --dry-run`
核对估算再上卡。

> 本仓库不公布任何训练指标结论：冒烟配置只验证链路可跑通，
> 风格收敛与否请以 `format_check` 遵循率与盲评结果为准。
