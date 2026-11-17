# -*- coding: utf-8 -*-
"""通用指令数据接入（纯标准库可运行）。

设计原则：
1. 仓库内置一份小规模通用样本 `data/general_samples.jsonl`，保证流水线
   离线可复现；
2. 若需要真实开源指令集（数万条级），通过 `EXTERNAL_SOURCES` 登记的来源
   自行下载——本模块只提供下载骨架与来源/License 提示，不在仓库中
   硬编码或打包大文件；
3. 通用数据按约 10% 比例混入指令集，用于缓解风格微调导致的
   通用能力回归（见 README 设计决策 FAQ）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# 内置样本的默认路径（相对仓库根）
DEFAULT_GENERAL_PATH = os.path.join("data", "general_samples.jsonl")

# 外部真实指令集来源登记（骨架）：
# 注意：接入前务必核对上游 License（多为 Apache-2.0 / CC-BY / ODC-BY 等），
# 并在发布衍生数据集时遵守署名与相同方式共享条款。
EXTERNAL_SOURCES: Dict[str, Dict[str, str]] = {
    "sample-adapt": {
        "hint": "示例条目：请替换为你确认过 License 的开源指令集",
        "fetch": "huggingface-cli download <repo> --repo-type dataset --local-dir data/external/<name>",
        "license": "以官方仓库声明为准，商用前必须逐项确认",
    },
}


def load_general(path: str = DEFAULT_GENERAL_PATH) -> List[Dict[str, Any]]:
    """读取内置通用样本（{id, category, user, assistant} 扁平格式）。"""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {lineno} 行 JSON 解析失败: {exc}") from exc
            for key in ("id", "user", "assistant"):
                if not isinstance(obj.get(key), str) or not obj[key].strip():
                    raise ValueError(f"{path} 第 {lineno} 行缺少必需字段 {key!r}")
            records.append(obj)
    return records


def to_messages(record: Dict[str, Any]) -> Dict[str, Any]:
    """把扁平通用样本转为 ShareGPT messages 记录。"""
    return {
        "id": record["id"],
        "messages": [
            {"role": "user", "content": record["user"]},
            {"role": "assistant", "content": record["assistant"], "loss": True},
        ],
        "meta": {
            "source": "general",
            "category": record.get("category", "general"),
            "negative_type": "none",
            "difficulty": "easy",
            "scenario": "general",
            "variant_of": None,
        },
    }


def fetch_external(name: str, dest_dir: str = "data/external") -> None:
    """外部指令集下载骨架：打印来源与 License 提示，不做静默网络拉取。

    真实接入时，请按 EXTERNAL_SOURCES[name]["fetch"] 手动执行下载命令，
    再把落地文件转换为 {id, category, user, assistant} 扁平格式。
    """
    if name not in EXTERNAL_SOURCES:
        raise KeyError(f"未登记的来源 {name!r}，请先在 EXTERNAL_SOURCES 中补充来源与 License 说明")
    src = EXTERNAL_SOURCES[name]
    print(f"[来源] {src['hint']}")
    print(f"[下载] {src['fetch']}")
    print(f"[许可] {src['license']}")
    print(f"[落地] 请将转换后的 JSONL 放入 {os.path.join(dest_dir, name)}")
    raise SystemExit(
        "fetch_external 仅为操作指引骨架：为避免把许可与体积不可控的大文件带入仓库，"
        "请按上述命令手动下载并完成格式转换。"
    )
