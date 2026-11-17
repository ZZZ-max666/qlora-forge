# -*- coding: utf-8 -*-
"""QLoRA/全参微调显存估算器（纯标准库，可单测、可离线运行）。

估算模型（各项相加，单位 GB，按 1GB = 1e9 字节）：

1. 冻结权重 weights
   - QLoRA：NF4 双量化 ≈ 0.52 字节/参数（4bit 的 0.5 + 双量化元数据开销约 4%）
   - 全参 BF16：2 字节/参数
2. LoRA 可训练参数 lora_params
   对每个目标模块 (in, out)：r × (in + out)，逐层求和。
3. 梯度 grads：仅可训练参数产生梯度，BF16 → 2 字节/参数；
   全参微调时 grads = 全部参数 × 2。
4. 优化器状态 optimizer：AdamW 主权重(fp32=4) + 一阶/二阶动量(fp32=8)
   → 12 字节/可训练参数（QLoRA 只覆盖 LoRA 参数，这正是 QLoRA 省显存的核心）。
5. 激活 activations：反向传播需缓存的中间张量。
   经验公式 act = B × S × H × L × 2字节 × C，
   C 为"每层激活单元数"，按业界公开经验值校准：
   不启用梯度检查点取 C=20（QKV/注意力输出/FFN 中间层等多份
   [B,S,H] 张量全程驻留）；启用梯度检查点取 C=10（层输入驻留 +
   逐层重算瞬时峰值 + 标签/采样缓冲）。
6. KV cache（仅推理/生成场景计入，训练默认不计）：
   kv = 2(K与V) × B × S × kv_dim × L × 2字节。
7. 运行时开销 overhead：CUDA 上下文、分配器碎片、通信缓冲与安全余量，
   固定取 1.5 GB。

公式与常数的设计依据、与业界经验值的对照见 README「显存估算」一节。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# 模型结构档案（参数量/hidden/层数/注意力头/KV头/FFN 中间维）
MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "qwen2.5-0.5b": {"params": 494_000_000, "hidden": 896, "layers": 24,
                     "heads": 14, "kv_heads": 2, "ffn": 4864},
    "qwen2.5-1.5b": {"params": 1_540_000_000, "hidden": 1536, "layers": 28,
                     "heads": 12, "kv_heads": 2, "ffn": 8960},
    "qwen2.5-7b": {"params": 7_620_000_000, "hidden": 3584, "layers": 28,
                   "heads": 28, "kv_heads": 4, "ffn": 18944},
}

BYTES_NF4 = 0.52          # NF4 双量化：0.5 字节 + 约 4% 元数据
BYTES_BF16 = 2.0
GRAD_BYTES = 2.0          # 可训练参数梯度（BF16）
OPT_BYTES = 12.0          # AdamW: fp32 主权重 + 两个动量
# 激活单元常数按业界公开经验值反推校准（详见 README「显存估算」）：
# 7B QLoRA @seq2048/batch1 实测普遍落在 10-16GB，取 C_ACT_CKPT=10 时公式输出约 10GB。
C_ACT = 20.0              # 无梯度检查点：每层缓存的多份 [B,S,H] 张量
C_ACT_CKPT = 10.0         # 梯度检查点：层输入驻留 + 逐层重算瞬时峰值 + 标签/采样缓冲
OVERHEAD_GB = 1.5         # CUDA 上下文/分配器碎片/通信缓冲（含 10% 安全余量）

ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
FFN_MODULES = {"gate_proj", "up_proj", "down_proj"}

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


def get_profile(name: str) -> Dict[str, Any]:
    if name not in MODEL_PROFILES:
        raise KeyError(f"未知模型档案 {name!r}，可选：{sorted(MODEL_PROFILES)}")
    return MODEL_PROFILES[name]


def lora_param_count(profile: Dict[str, Any], r: int,
                     target_modules: List[str]) -> int:
    """LoRA 可训练参数量：Σ_模块 r×(in+out) × 层数。"""
    h = profile["hidden"]
    ffn = profile["ffn"]
    head_dim = h // profile["heads"]
    kv_dim = profile["kv_heads"] * head_dim
    dims = {
        "q_proj": (h, h), "k_proj": (h, kv_dim), "v_proj": (h, kv_dim),
        "o_proj": (h, h), "gate_proj": (h, ffn), "up_proj": (h, ffn),
        "down_proj": (ffn, h),
    }
    per_layer = 0
    for mod in set(target_modules):
        if mod in dims:
            din, dout = dims[mod]
            per_layer += r * (din + dout)
    return per_layer * profile["layers"]


def estimate(profile_name: str, seq_len: int, batch_size: int = 1,
             lora_r: int = 16, target_modules: Optional[List[str]] = None,
             method: str = "qlora", gradient_checkpointing: bool = True,
             include_kv: bool = False) -> Dict[str, Any]:
    """返回各分项显存（GB）与合计。method ∈ {"qlora", "full"}。"""
    p = get_profile(profile_name)
    if method not in ("qlora", "full"):
        raise ValueError(f"未知训练方法 {method!r}")
    target_modules = target_modules or DEFAULT_TARGET_MODULES

    gb = 1e9
    params = p["params"]

    # 1) 冻结/全量权重
    if method == "qlora":
        weights = params * BYTES_NF4 / gb
    else:
        weights = params * BYTES_BF16 / gb

    # 2) 可训练参数与梯度、优化器
    trainable = lora_param_count(p, lora_r, target_modules) if method == "qlora" else params
    grads = trainable * GRAD_BYTES / gb
    optimizer = trainable * OPT_BYTES / gb

    # 3) 激活
    c = C_ACT_CKPT if gradient_checkpointing else C_ACT
    activations = batch_size * seq_len * p["hidden"] * p["layers"] * BYTES_BF16 * c / gb

    # 4) KV cache（推理/生成）
    kv_cache = 0.0
    if include_kv:
        head_dim = p["hidden"] // p["heads"]
        kv_dim = p["kv_heads"] * head_dim
        kv_cache = 2 * batch_size * seq_len * kv_dim * p["layers"] * BYTES_BF16 / gb

    overhead = OVERHEAD_GB
    total = weights + grads + optimizer + activations + kv_cache + overhead
    return {
        "profile": profile_name, "method": method,
        "seq_len": seq_len, "batch_size": batch_size,
        "lora_r": lora_r if method == "qlora" else 0,
        "gradient_checkpointing": gradient_checkpointing,
        "trainable_params": trainable,
        "trainable_ratio": trainable / params,
        "weights_gb": round(weights, 3),
        "grads_gb": round(grads, 3),
        "optimizer_gb": round(optimizer, 3),
        "activations_gb": round(activations, 3),
        "kv_cache_gb": round(kv_cache, 3),
        "overhead_gb": round(overhead, 3),
        "total_gb": round(total, 2),
    }


def estimate_from_config(cfg: Dict[str, Any], include_kv: bool = False) -> Dict[str, Any]:
    """从训练配置 JSON（见 configs/*.json）直接产出估算。"""
    train = cfg["train"]
    lora = cfg.get("lora", {})
    return estimate(
        cfg["model_profile"],
        seq_len=int(train["seq_len"]),
        batch_size=int(train["batch_size"]),
        lora_r=int(lora.get("r", 16)),
        target_modules=lora.get("target_modules"),
        method=cfg.get("method", "qlora"),
        gradient_checkpointing=bool(train.get("gradient_checkpointing", True)),
        include_kv=include_kv,
    )


def format_table(rows: List[Dict[str, Any]]) -> str:
    """把估算结果列表渲染为 markdown 表格（README 与 vram_report 共用）。"""
    header = ("| 模型 | 方法 | seq_len | batch | r | 梯度检查点 | 权重 | 梯度 | 优化器 | 激活 | 开销 | 合计(GB) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [header[0], header[1]]
    for r in rows:
        lines.append(
            f"| {r['profile']} | {r['method']} | {r['seq_len']} | {r['batch_size']} "
            f"| {r['lora_r'] or '-'} | {'是' if r['gradient_checkpointing'] else '否'} "
            f"| {r['weights_gb']:.2f} | {r['grads_gb']:.2f} | {r['optimizer_gb']:.2f} "
            f"| {r['activations_gb']:.2f} | {r['overhead_gb']:.2f} | **{r['total_gb']:.2f}** |"
        )
    return "\n".join(lines)
