# -*- coding: utf-8 -*-
"""ShareGPT messages 风格 JSONL 的 schema 校验器（纯标准库）。

数据规范（详见 docs/pipeline.md）：
- 每条记录是一个 JSON 对象，必需字段：
    - "id":       非空字符串，数据集内唯一；
    - "messages": 非空数组，元素为 {"role": ..., "content": ...}；
- role 取值：system / user / assistant；
- 消息顺序：可选以一条 system 开头，随后必须以 user 开始，
  user 与 assistant 严格交替，且最后一条必须是 assistant；
- loss 标记：仅允许出现在 assistant 消息上，值为布尔；
  缺省视为 true（即该轮参与训练）。整个会话至少有一条
  assistant 消息的 loss 有效值为 true，否则没有训练目标；
- "meta": 可选对象，建议携带 source / scenario / difficulty 等溯源信息。
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

VALID_ROLES = ("system", "user", "assistant")


def validate_messages(messages: Any) -> List[str]:
    """校验 messages 数组，返回错误列表（空列表表示通过）。"""
    errors: List[str] = []
    if not isinstance(messages, list) or not messages:
        return ["messages 必须是非空数组"]

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"messages[{idx}] 必须是对象")
            continue
        role = msg.get("role")
        if role not in VALID_ROLES:
            errors.append(f"messages[{idx}].role 非法: {role!r}")
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"messages[{idx}].content 必须是非空字符串")
        if "loss" in msg:
            if role != "assistant":
                errors.append(f"messages[{idx}]: 仅 assistant 消息允许携带 loss 标记")
            elif not isinstance(msg["loss"], bool):
                errors.append(f"messages[{idx}].loss 必须是布尔值")

    # 角色序列检查：system(可选) 之后 user 开头、交替、assistant 结尾
    seq = [m.get("role") for m in messages if isinstance(m, dict) and m.get("role") in VALID_ROLES]
    if seq and seq[0] == "system":
        seq = seq[1:]
    if not seq:
        errors.append("缺少 user/assistant 对话轮次")
        return errors
    if seq[0] != "user":
        errors.append("对话必须以 user 轮开始")
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            errors.append(f"角色未交替：第 {i} 条与第 {i-1} 条同为 {seq[i]}")
    if seq[-1] != "assistant":
        errors.append("对话必须以 assistant 轮结束")

    assistant_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    if assistant_msgs and not any(m.get("loss", True) is True for m in assistant_msgs):
        errors.append("所有 assistant 消息的 loss 均为 false，无训练目标")
    return errors


def validate_record(rec: Any) -> List[str]:
    """校验单条记录，返回错误列表（空列表表示通过）。"""
    if not isinstance(rec, dict):
        return ["记录必须是 JSON 对象"]
    errors: List[str] = []
    rid = rec.get("id")
    if not isinstance(rid, str) or not rid.strip():
        errors.append("id 必须是非空字符串")
    errors.extend(validate_messages(rec.get("messages")))
    if "meta" in rec and not isinstance(rec["meta"], dict):
        errors.append("meta 必须是对象")
    return errors


def validate_records(records: Iterable[Any]) -> List[Tuple[int, List[str]]]:
    """批量校验，返回 [(行号(0基), 错误列表)]；同时检查 id 唯一性。"""
    problems: List[Tuple[int, List[str]]] = []
    seen: Dict[str, int] = {}
    for i, rec in enumerate(records):
        errs = validate_record(rec)
        if isinstance(rec, dict):
            rid = rec.get("id")
            if isinstance(rid, str) and rid:
                if rid in seen:
                    errs.append(f"id 重复：与第 {seen[rid]} 行冲突")
                else:
                    seen[rid] = i
        if errs:
            problems.append((i, errs))
    return problems


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 JSONL 文件；行号从 1 开始报错。"""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {lineno} 行 JSON 解析失败: {exc}") from exc
    return records


def validate_file(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """校验整个 JSONL 文件，返回 (合法记录列表, 人类可读错误列表)。"""
    records = load_jsonl(path)
    problems = validate_records(records)
    bad_lines = {i for i, _ in problems}
    errors = []
    for i, errs in problems:
        for e in errs:
            errors.append(f"{path} 第 {i + 1} 行: {e}")
    good = [r for i, r in enumerate(records) if i not in bad_lines]
    return good, errors
