# -*- coding: utf-8 -*-
"""GPTQ INT4 量化（auto-gptq）。

输入是 merge.py 产出的完整 bf16 模型；GPTQ 用少量校准样本逐层求解
量化误差最优舍入，得到可独立部署的 INT4 权重。
注意：auto-gptq 官方仅提供 Linux 预编译轮子，Windows 原生环境请先在
WSL/Linux 中执行（本仓库的训练与量化步骤均推荐 Linux/WSL）。

用法：
    python -m toneforge.deploy.quant --model outputs/merged/smoke \
        --out outputs/gptq/smoke-int4 --samples 128 --seqlen 2048
"""
from __future__ import annotations

import argparse
import platform
import sys
from typing import Optional


def quantize_gptq(model_dir: str, out_dir: str, samples: int = 128,
                  seqlen: int = 2048) -> None:  # pragma: no cover - 需 Linux+GPU
    import torch
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    print(f"[quant] 平台={platform.system()}；auto-gptq 推荐 Linux/WSL 运行")
    quant_cfg = BaseQuantizeConfig(
        bits=4, group_size=128, desc_act=False, model_name_or_path=model_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoGPTQForCausalLM.from_pretrained(model_dir, quant_cfg, trust_remote_code=True)

    # 校准样本：用客服风格验证集文本（贴近目标分布的校准能明显降低量化损失）
    from toneforge.data import schema
    calib_texts = []
    try:
        for rec in schema.load_jsonl("outputs/dataset/val.jsonl")[:samples]:
            u = rec["messages"][0]["content"]
            a = rec["messages"][1]["content"]
            calib_texts.append(tokenizer(f"用户：{u}\n客服：{a}", truncation=True,
                                         max_length=seqlen))
    except FileNotFoundError:
        print("[quant] 未找到 outputs/dataset/val.jsonl，回退为随机校准样本；"
              "建议先运行 scripts/build_dataset.py 以获得目标分布校准。")
        calib_texts = [tokenizer("客服话术校准样本 " * (seqlen // 8),
                                 truncation=True, max_length=seqlen) for _ in range(samples)]

    model.quantize(calib_texts, batch_size=1)
    model.save_quantized(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[quant] GPTQ INT4 权重已保存到 {out_dir}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="GPTQ INT4 量化")
    parser.add_argument("--model", required=True, help="合并后的完整模型目录")
    parser.add_argument("--out", required=True, help="量化输出目录")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    args = parser.parse_args(argv)

    if platform.system() != "Linux":
        print(f"[quant] 当前平台 {platform.system()}：auto-gptq 仅提供 Linux 轮子，"
              "请在 WSL/Linux 中执行本脚本。", file=sys.stderr)
        return 2
    try:
        import auto_gptq  # noqa: F401
    except ImportError:
        print("[quant] 请先安装 auto-gptq（见 requirements.txt 说明）", file=sys.stderr)
        return 2
    quantize_gptq(args.model, args.out, args.samples, args.seqlen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
