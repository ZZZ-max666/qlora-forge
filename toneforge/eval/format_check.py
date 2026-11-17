# -*- coding: utf-8 -*-
"""三段式话术结构校验器（纯标准库）。

客服规范回复必须依次包含：
  1. 共情句   —— 先回应情绪（理解/抱歉/别担心…），位于回复开头段；
  2. 政策依据 —— 引用具体条款（《…》或"根据/依据…规定"）；
  3. 动作指引 —— 下一步做什么 + 在哪做（订单页/App/工单/入口…），
     且必须出现在政策依据之后。

判定采用 关键词 + 正则 + 段落顺序 三类确定性规则，输出遵循率，
可直接用于：
- 训练集质检（种子/变体应接近 100% 通过）；
- 坏例夹具验证（build.py 构造的负样本必须被拦截）；
- 模型输出自动评估（对生成结果跑同一套规则）。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional

try:
    from toneforge.data import schema
except ImportError:  # 兼容以脚本方式直接运行
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from toneforge.data import schema

# 共情关键词（开头段命中其一即可）
EMPATHY_RE = re.compile(
    r"理解|抱歉|不好意思|对不起|感谢|别担心|您放心|放心|明白|收到"
    r"|您的心情|您的着急|您的感受|换作是我|换谁|向您致歉|您先别急|我接手"
    r"|给您添麻烦|不找借口|不该让您"
)
# 政策引用：书名号条款，或 引导词+平台，或显式"政策依据/规定"
POLICY_RE = re.compile(
    r"《[^》]{2,24}》"
    r"|(?:根据|依据|按照|遵照)平台"
    r"|(?:政策|规则|条款)(?:依据|规定|约定)"
    r"|平台规定"
)
# 动作动词 与 动作场景词 需同时命中（动作段位于政策引用之后）
ACTION_VERB_RE = re.compile(
    r"点击|打开|上传|提供|提交|发起|申请|登记|回复|保留|拨打|发送|拍照"
    r"|催办|转接|推送|建单|备注|安排|查询|联系|同步|留意|等待|通知|告知|升级|办理|代您|为您"
)
ACTION_CTX_RE = re.compile(
    r"「|」|App|APP|订单页|商品页|详情页|页面|订单号|工单|入口|链接|消息中心|邮箱"
    r"|短信|站内消息|电话|回电|专员|答复|个工作日|小时内|天内|上门|账户|余额"
    r"|原支付渠道|发货|寄回|取件|补发|补偿|赔付|退款|报销|时效|进度|结果|地址|凭证"
)


def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def check_text(text: str) -> Dict[str, Any]:
    """对单条 assistant 回复做三段式判定。"""
    paras = _paragraphs(text)
    if not paras:
        return {"has_empathy": False, "has_policy": False, "has_action": False,
                "order_ok": False, "ok": False}

    # 共情：首段命中（自由生成无分段时退化为前 80 字窗口）
    emp_scope = paras[0] if len(paras) > 1 or "\n\n" in text else text[:80]
    has_empathy = bool(EMPATHY_RE.search(emp_scope))

    if len(paras) > 1:
        # 政策：全篇找首个命中段落
        pol_idx = next((i for i, p in enumerate(paras) if POLICY_RE.search(p)), None)
        has_policy = pol_idx is not None
        # 动作：动词+场景同段命中，且必须位于政策引用之后（三段式顺序要求）
        act_idx = None
        if pol_idx is not None:
            act_idx = next((i for i in range(pol_idx + 1, len(paras))
                            if ACTION_VERB_RE.search(paras[i]) and ACTION_CTX_RE.search(paras[i])), None)
        has_action = act_idx is not None
    else:
        # 单段自由文本（模型不分段生成时）：按字符位置判定 政策在前、动作在后
        m = POLICY_RE.search(text)
        has_policy = m is not None
        has_action = False
        if m is not None:
            rest = text[m.end():]
            has_action = bool(ACTION_VERB_RE.search(rest) and ACTION_CTX_RE.search(rest))

    order_ok = bool(has_empathy and has_policy and has_action)
    return {
        "has_empathy": has_empathy,
        "has_policy": has_policy,
        "has_action": has_action,
        "order_ok": order_ok,
        "ok": order_ok,
    }


def check_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """校验一条 ShareGPT 记录的最后一段 assistant 回复。"""
    msgs = record.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return check_text(m.get("content", ""))
    return {"has_empathy": False, "has_policy": False, "has_action": False,
            "order_ok": False, "ok": False}


def compliance_rate(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if check_record(r)["ok"]) / len(records)


def evaluate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总：整体遵循率 + 各分项命中率 + 按 negative_type 分组。

    三段式是客服风格规范，通用能力样本（meta.source == "general"）
    不适用该规范，统计时剔除。"""
    records = [r for r in records if r.get("meta", {}).get("source") != "general"]
    n = len(records)
    flags = {"has_empathy": 0, "has_policy": 0, "has_action": 0, "order_ok": 0, "ok": 0}
    groups: Dict[str, List[bool]] = {}
    for r in records:
        res = check_record(r)
        for k in flags:
            flags[k] += 1 if res[k] else 0
        key = r.get("meta", {}).get("negative_type", "none")
        groups.setdefault(key, []).append(res["ok"])
    report = {"total": n}
    report.update({k: round(v / n, 4) if n else 0.0 for k, v in flags.items()})
    report["by_negative_type"] = {
        k: {"n": len(v), "ok_rate": round(sum(v) / len(v), 4)}
        for k, v in sorted(groups.items())
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="三段式话术结构校验")
    parser.add_argument("jsonl", help="待校验数据集/生成结果 JSONL")
    args = parser.parse_args(argv)

    records, errors = schema.validate_file(args.jsonl)
    if errors:
        print(f"[schema] {len(errors)} 处不合法，示例：")
        for e in errors[:5]:
            print("  -", e)
    rep = evaluate(records)
    print("三段式遵循率报告")
    print(f"  总条数        : {rep['total']}")
    print(f"  共情句命中    : {rep['has_empathy']:.1%}")
    print(f"  政策依据命中  : {rep['has_policy']:.1%}")
    print(f"  动作指引命中  : {rep['has_action']:.1%}")
    print(f"  顺序正确      : {rep['order_ok']:.1%}")
    print(f"  整体遵循率    : {rep['ok']:.1%}")
    print(f"  分组(negative_type): {rep['by_negative_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
