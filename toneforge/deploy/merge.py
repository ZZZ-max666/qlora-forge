# -*- coding: utf-8 -*-
"""LoRA 适配器合并：基座 + adapter → 完整权重（safetensors）。

合并后模型不再依赖 peft 运行时，可直接被 vLLM / transformers 加载；
注意：QLoRA 训练时基座是 NF4 量化态，合并前需以 bf16 重新加载基座，
LoRA 增量按训练时 dtype 计算，合并结果精度为 bf16（非量化模型）。

用法：
    python -m toneforge.deploy.merge --base Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/checkpoints/smoke --out outputs/merged/smoke

依赖 torch/transformers/peft，惰性导入；本机无 GPU 时请勿执行。
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional


def merge_lora(base: str, adapter: str, out_dir: str, dtype: str = "bfloat16") -> None:  # pragma: no cover
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch_dtype = getattr(torch, dtype)
    print(f"[merge] 加载基座 {base} ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch_dtype,
                                                 device_map="auto", trust_remote_code=True)
    print(f"[merge] 挂载适配器 {adapter} 并合并 ...")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()

    print(f"[merge] 保存完整权重到 {out_dir} (safetensors) ...")
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    print("[merge] 完成。下一步可执行 quant.py 做 GPTQ 或直接 vLLM 部署。")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器为完整权重")
    parser.add_argument("--base", required=True, help="基座模型名或本地路径")
    parser.add_argument("--adapter", required=True, help="PeftModel 适配器目录")
    parser.add_argument("--out", required=True, help="合并输出目录")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args(argv)
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("merge.py 需要 torch/transformers/peft，请在 GPU 环境安装后运行（见 requirements.txt）",
              file=sys.stderr)
        return 2
    merge_lora(args.base, args.adapter, args.out, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
