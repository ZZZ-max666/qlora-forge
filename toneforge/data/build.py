# -*- coding: utf-8 -*-
"""指令集构建器：合成种子对话 → 客服话术风格指令微调数据集（纯标准库）。

流水线（固定 seed 完全可复现）：
1. 读取 data/seeds/*.jsonl 的种子（模板 tpl + 槽位 slots 形式，
   渲染后即一条规范的"共情→政策依据→下一步动作"三段式客服对话）；
2. 每条种子扩展 4 个变体：
   - v1 easy   ：槽位重采样（商品/时长/金额/原因/条款/快递 全量替换，
                 条款与时效保持耦合一致）；
   - v2 medium ：保留槽位，追加约束条件句 + 针对性回应句；
   - v3 hard   ：槽位重采样 + 情绪前缀 + 复合追问 + 补充解答；
   - v4 easy   ：同义表达替换（SWAP_PAIRS 逐条按概率改写措辞）；
3. 负样本构造：缺共情 / 编造政策 / 格式错误（动作先于政策、缺动作），
   全部进入 test 集，供 toneforge.eval 校验规则能否识别坏例；
4. 混入约 10% 通用指令数据（toneforge.data.general）；
5. 80/10/10 切分 train/val/test（正样本与通用样本参与切分，互不重叠）。

与"简单复读"的区别：变体同时改变 场景参数、用户诉求结构、措辞分布 与
回复的论证内容，且条款-时效-商品三者语义耦合，不会输出自相矛盾的样本。
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from . import general as general_mod
from . import schema

# ---------------------------------------------------------------------------
# 槽位池：与 data/seeds 生成口径一致，用于变体时的槽位重采样
# ---------------------------------------------------------------------------
PRODUCTS = ["女式冲锋衣", "跑步鞋", "蓝牙耳机", "14寸笔记本电脑", "面霜套装", "儿童积木",
            "不锈钢保温杯", "羽绒服", "智能手表", "护眼台灯", "陶瓷餐具四件套", "机械键盘",
            "瑜伽垫", "登机行李箱", "纯棉T恤", "电动牙刷", "羊绒围巾", "空气炸锅",
            "羽毛球拍", "手机壳"]
AMOUNTS = [89, 129, 159, 199, 239, 299, 359, 429, 499, 599, 699, 899, 1099, 1299, 1599, 2099]
REASONS = ["尺码不合适", "颜色与图片色差大", "开机没有反应", "屏幕有一道划痕", "材质和描述不符",
           "充电发热严重", "遥控器连不上", "到货太迟错过了使用时间", "扣子松了", "气味太重",
           "盖子盖不严", "功能和详情页写的不一样"]
COURIERS = ["中通", "圆通", "申通", "顺丰", "百世", "韵达", "EMS", "极兔"]

SCENARIO_CLAUSES: Dict[str, List[str]] = {
    "return_exchange": ["七天无理由退货", "十五天换货保障", "质量问题免费退换", "三十天质保", "服饰类尺码调换保障"],
    "logistics_push": ["48小时发货保障", "超时赔付标准", "遗失损毁赔付规则", "物流信息24小时更新要求", "运费险赔付规则"],
    "complaint_soothe": ["投诉处理时效承诺", "商家违规积分管理规则", "平台服务补偿标准", "消费先行赔付规则", "虚假描述认定与处理规范"],
    "invoice_query": ["电子发票开具规则", "开票时效承诺", "发票红冲与重开流程", "商品发票信息规范要求", "部分退款发票金额处理规则"],
    "price_protect": ["十五天价保服务", "差价补偿规则", "大促价保范围说明", "价保申请条件", "价保审核时效承诺"],
}
SCENARIO_DAYS: Dict[str, List[int]] = {
    "return_exchange": [7, 15, 30],
    "logistics_push": [2, 3, 4, 5, 7],
    "complaint_soothe": [3, 5, 7],
    "invoice_query": [3, 7, 15],
    "price_protect": [7, 15, 30],
}
# 条款→时效耦合：退换货条款名称本身含时效，days 槽必须与之一致
CLAUSE_TO_DAYS: Dict[str, int] = {
    "七天无理由退货": 7, "十五天换货保障": 15, "质量问题免费退换": 15,
    "三十天质保": 30, "服饰类尺码调换保障": 30,
}

# 政策条款登记表：真实存在的条款名（评估中用于"编造政策"检测）
POLICY_CLAUSE_REGISTRY: List[str] = sorted({c for v in SCENARIO_CLAUSES.values() for c in v})

# 编造条款：仅用于构造负样本，绝不出现在登记表里
FABRICATED_CLAUSES = ["30天无条件退货", "终身免费换新", "不满意全额秒退且无需寄回商品", "假一赔十倍极速到账"]

# 同义表达替换对（v4 变体的措辞扰动，均为语义安全的改写）
SWAP_PAIRS = [
    ("别担心", "您放心"), ("马上", "立即"), ("帮您", "协助您"), ("这就", "即刻"),
    ("根据平台", "依据平台"), ("上传", "提供"), ("通知您", "告知您"),
    ("审核通过后", "通过审核后"), ("请您", "麻烦您"), ("我为您", "这边为您"),
    ("点击", "打开"), ("短信", "站内消息"), ("非常理解", "完全理解"),
]

# 中等难度：用户追加约束 + 客服针对性回应
MEDIUM_USER_SUFFIX: Dict[str, List[str]] = {
    "return_exchange": ["补充说明：商品吊牌未剪、原包装齐全。", "另外，我是用优惠券下的单，退款会退券吗？", "这单是代家人签收的，不影响退货吧？"],
    "logistics_push": ["补充：商家页面标注了48小时发货。", "这是给客户寄的样品，时间很紧。", "包裹显示已到达本市，但一直没派送。"],
    "complaint_soothe": ["我保留了全部聊天记录和开箱视频。", "之前客服给我的工单编号我还在。", "同类问题这不是第一次发生了。"],
    "invoice_query": ["报销截止日是本月底，比较急。", "发票抬头是我们公司的全称。", "之前开过一次，信息是对的，这次要补第二张。"],
    "price_protect": ["降价截图我已经保存好了。", "我买的是带赠品的套装。", "现在挂的是官方立减，不是我个人的券。"],
}
MEDIUM_EXTRA: Dict[str, List[str]] = {
    "return_exchange": ["您提到的商品与包装完好，正好满足《{clause}》中保持商品完好的条件。", "使用优惠券的订单，券的处理按活动规则执行，不影响本次退货受理。", "代签收不影响退货，由下单账号发起申请即可。"],
    "logistics_push": ["订单页面承诺的发货时效与《{clause}》口径一致，超时即可按标准处理。", "理解时效压力，我已在工单备注加急，优先核查这单。", "包裹已到本市仍未派送属于末端延误，核查会一并覆盖派送环节。"],
    "complaint_soothe": ["您保存的聊天与视频凭证非常关键，平台介入审核会直接采信。", "我已关联此前的沟通记录，这次合并处理，不需要您重复描述。", "重复发生的问题会被纳入商家违规记录，这一点请您放心。"],
    "invoice_query": ["理解报销时效压力，这一单我标记为加急开票并盯审核进度。", "专票需要抬头、税号、开户行与注册地址齐全，请您一并提供。", "补开第二张不影响第一张效力，按《{clause}》提交即可。"],
    "price_protect": ["有比价截图即可走人工审核，截图请包含商品页与价格时间。", "套装订单的价保以主商品实付为准，赠品不影响差价计算。", "官方立减属于商家降价，在价保支持范围内。"],
}

# 困难样本：情绪前缀 + 复合追问 + 补充解答
HARD_PREFIX = ["我已经打过三次电话都没解决，", "这个问题拖了快一周了，", "说实话我很失望，", "请认真对待这件事，"]
HARD_QUESTION: Dict[str, List[str]] = {
    "return_exchange": ["另外我想知道退款具体几天到账？", "还有，换货的话新商品什么时候能发出？", "如果验收说商品有问题，我会收到什么通知？"],
    "logistics_push": ["如果这单确认丢了，赔偿怎么算？", "核查期间我可以先申请退款吗？", "补发的话还走{courier}吗？"],
    "complaint_soothe": ["你们打算怎么给商家处罚？", "处理期间我需要做什么吗？", "如果结果我不满意，还能往哪里申诉？"],
    "invoice_query": ["重开的发票会覆盖原来那张吗？", "纸质票还能寄给我吗？", "税号填错会影响报销吗？"],
    "price_protect": ["补差是退到余额还是原支付渠道？", "价保通过后我原来的订单会变吗？", "如果审核不通过，依据会告诉我吗？"],
}
HARD_EXTRA: Dict[str, List[str]] = {
    "return_exchange": ["关于到账时间：仓库验收通过后1-7个工作日按原支付渠道退回。", "关于换货时效：新商品在原包裹退回签收后48小时内发出。", "若验收有异议，平台会在24小时内电话与您确认，不会直接驳回。"],
    "logistics_push": ["若确认遗失，按《{clause}》商家需在{days}天内补发或退款。", "核查期间您可以正常发起退款，两者不冲突。", "补发承运方以商家安排为准，我会备注优先选择时效稳定的快递。"],
    "complaint_soothe": ["商家若被核实违规，将按《{clause}》记分并限制流量。", "处理期间您无需操作，保留商品现状即可。", "对结果不满意，可在办结后7天内申请复核。"],
    "invoice_query": ["重开成功后，原发票自动红冲作废，以新票为准。", "纸质票在电子票开出后另行寄出，物流单号会同步给您。", "税号有误会导致报销失败，所以红冲重开是必要的。"],
    "price_protect": ["补差金额原路退回，不影响订单其他权益。", "订单本身不变，价保只是追加一笔差额退款。", "审核不通过会明确标注驳回理由，可补充凭证再次提交。"],
}

NEGATIVE_KINDS = ["missing_empathy", "fabricated_policy", "format_error"]


# ---------------------------------------------------------------------------
# 种子读取与渲染
# ---------------------------------------------------------------------------
def load_seeds(seeds_dir: str) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []
    for fn in sorted(os.listdir(seeds_dir)):
        if not fn.endswith(".jsonl"):
            continue
        path = os.path.join(seeds_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} 第 {lineno} 行 JSON 解析失败: {exc}") from exc
                for key in ("id", "scenario", "tpl", "slots"):
                    if key not in obj:
                        raise ValueError(f"{path} 第 {lineno} 行缺少字段 {key!r}")
                seeds.append(obj)
    seeds.sort(key=lambda s: s["id"])
    return seeds


def _fill(text: str, slots: Dict[str, str]) -> str:
    return text.format(**slots)


def make_record(rid: str, scenario: str, user: str, segments: List[str],
                source: str, difficulty: str, negative_type: str = "none",
                variant_of: Optional[str] = None) -> Dict[str, Any]:
    """把三段式组件渲染为 ShareGPT messages 记录。"""
    assistant = "\n\n".join(seg for seg in segments if seg)
    return {
        "id": rid,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant, "loss": True},
        ],
        "meta": {
            "source": source,
            "scenario": scenario,
            "difficulty": difficulty,
            "negative_type": negative_type,
            "variant_of": variant_of,
        },
    }


def _apply_swaps(text: str, rng: random.Random, p: float = 0.5) -> str:
    for old, new in SWAP_PAIRS:
        if old in text and rng.random() < p:
            text = text.replace(old, new)
    return text


def _resample_slots(seed: Dict[str, Any], rng: random.Random) -> Dict[str, str]:
    """槽位重采样：与种子值不同，且维持 条款↔时效 耦合。"""
    sc = seed["scenario"]
    slots = dict(seed["slots"])
    old_clause = slots["clause"]
    clauses = SCENARIO_CLAUSES[sc]
    new_clause = rng.choice([c for c in clauses if c != old_clause])
    slots["clause"] = new_clause
    if new_clause in CLAUSE_TO_DAYS:
        slots["days"] = str(CLAUSE_TO_DAYS[new_clause])
    else:
        slots["days"] = str(rng.choice(SCENARIO_DAYS[sc]))
    slots["product"] = rng.choice([x for x in PRODUCTS if x != slots["product"]])
    slots["reason"] = rng.choice([x for x in REASONS if x != slots["reason"]])
    slots["amount"] = str(rng.choice([x for x in AMOUNTS if str(x) != slots["amount"]]))
    if "courier" in slots:
        slots["courier"] = rng.choice([x for x in COURIERS if x != slots["courier"]])
    return slots


def expand_seed(seed: Dict[str, Any], rng: random.Random, n_variants: int = 4) -> List[Dict[str, Any]]:
    """一条种子 → 1 条规范种子样本 + n_variants 条变体样本。"""
    sc = seed["scenario"]
    tpl, slots0 = seed["tpl"], seed["slots"]

    canonical = make_record(
        seed["id"], sc,
        _fill(tpl["user"], slots0),
        [_fill(tpl["empathy"], slots0), _fill(tpl["policy"], slots0), _fill(tpl["action"], slots0)],
        source="seed", difficulty="easy",
    )
    out = [canonical]
    for v in range(1, n_variants + 1):
        slots = _resample_slots(seed, rng) if v in (1, 3) else dict(slots0)
        e, p, a = (_fill(tpl[k], slots) for k in ("empathy", "policy", "action"))
        user = _fill(tpl["user"], slots)
        extras: List[str] = []
        difficulty = "easy"
        if v == 2:
            difficulty = "medium"
            user += rng.choice(MEDIUM_USER_SUFFIX[sc])
            extras.append(_fill(rng.choice(MEDIUM_EXTRA[sc]), slots))
        elif v == 3:
            difficulty = "hard"
            user = rng.choice(HARD_PREFIX) + user + rng.choice(HARD_QUESTION[sc])
            extras.append(_fill(rng.choice(HARD_EXTRA[sc]), slots))
            e, p, a = _apply_swaps(e, rng), _apply_swaps(p, rng), _apply_swaps(a, rng)
        elif v == 4:
            e, p, a = _apply_swaps(e, rng), _apply_swaps(p, rng), _apply_swaps(a, rng)
            user = _apply_swaps(user, rng)
        out.append(make_record(
            f"{seed['id']}-v{v}", sc, user, [e, p, a] + extras,
            source="variant", difficulty=difficulty, variant_of=seed["id"],
        ))
    return out


def make_negative(seed: Dict[str, Any], rng: random.Random, kind: str) -> Dict[str, Any]:
    """构造坏例：缺共情 / 编造政策 / 格式错误（顺序颠倒或缺动作）。"""
    sc = seed["scenario"]
    tpl, slots = seed["tpl"], seed["slots"]
    user = _fill(tpl["user"], slots)
    e, p, a = (_fill(tpl[k], slots) for k in ("empathy", "policy", "action"))
    if kind == "missing_empathy":
        segs = [p, a]
    elif kind == "fabricated_policy":
        fab = rng.choice(FABRICATED_CLAUSES)
        p_bad = (f"根据平台《{fab}》，签收后{slots['days']}天内不满意可无条件全额退货，"
                 f"运费由平台承担，且无需任何审核。")
        segs = [e, p_bad, a]
    elif kind == "format_error":
        if rng.random() < 0.5:
            segs = [e, a, p]          # 动作先于政策依据
        else:
            segs = [e, p]             # 缺下一步动作
    else:
        raise ValueError(f"未知负样本类型: {kind}")
    return make_record(
        f"{seed['id']}-neg-{kind}", sc, user, segs,
        source="negative", difficulty="negative", negative_type=kind, variant_of=seed["id"],
    )


# ---------------------------------------------------------------------------
# 数据集组装
# ---------------------------------------------------------------------------
def build_dataset(seeds_dir: str, general_path: str, seed: int = 42,
                  n_variants: int = 4, general_ratio: float = 0.10,
                  negatives_per_scenario: int = 12) -> Dict[str, Any]:
    rng = random.Random(seed)
    seeds = load_seeds(seeds_dir)

    positives: List[Dict[str, Any]] = []
    for s in seeds:
        positives.extend(expand_seed(s, rng, n_variants))

    negatives: List[Dict[str, Any]] = []
    by_sc: Dict[str, List[Dict[str, Any]]] = {}
    for s in seeds:
        by_sc.setdefault(s["scenario"], []).append(s)
    for sc in sorted(by_sc):
        pool = by_sc[sc][:negatives_per_scenario]
        for i, s in enumerate(pool):
            negatives.append(make_negative(s, rng, NEGATIVE_KINDS[i % len(NEGATIVE_KINDS)]))

    # 混入通用数据：gen / total ≈ general_ratio
    n_spec = len(positives) + len(negatives)
    n_general = int(round(n_spec * general_ratio / (1.0 - general_ratio)))
    pool = general_mod.load_general(general_path)
    if n_general <= len(pool):
        picked = rng.sample(pool, n_general)
    else:  # 池子不足时循环取用并加轮次后缀，保证 id 唯一
        picked = []
        rounds = (n_general + len(pool) - 1) // len(pool)
        for r in range(rounds):
            take = pool if r == rounds - 1 else rng.sample(pool, len(pool))
            for g in take:
                g2 = dict(g)
                if r > 0:
                    g2["id"] = f"{g['id']}-r{r}"
                picked.append(g2)
            if len(picked) >= n_general:
                break
        picked = picked[:n_general]
    generals = [general_mod.to_messages(g) for g in picked]

    # 切分：正样本 + 通用样本按 80/10/10；负样本全部进 test（作为规则评估的坏例夹具）
    mix = positives + generals
    idx = list(range(len(mix)))
    rng.shuffle(idx)
    n_train = int(len(mix) * 0.8)
    n_val = int(len(mix) * 0.1)
    train = [mix[i] for i in idx[:n_train]]
    val = [mix[i] for i in idx[n_train:n_train + n_val]]
    test = [mix[i] for i in idx[n_train + n_val:]] + negatives

    result = {
        "train": train, "val": val, "test": test,
        "stats": _stats(positives, negatives, generals, train, val, test, seeds),
    }
    return result


def _stats(positives, negatives, generals, train, val, test, seeds) -> Dict[str, Any]:
    def count_by(records, key):
        out: Dict[str, int] = {}
        for r in records:
            v = r["meta"][key]
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items()))

    total = len(train) + len(val) + len(test)
    return {
        "n_seeds": len(seeds),
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "n_general": len(generals),
        "total": total,
        "train": len(train), "val": len(val), "test": len(test),
        "general_ratio": round(len(generals) / total, 4) if total else 0.0,
        "by_scenario": count_by(positives + negatives, "scenario"),
        "by_difficulty": count_by(positives, "difficulty"),
        "by_negative_type": count_by(negatives, "negative_type"),
        "unique_user_turns": len({r["messages"][0]["content"] for r in positives}),
        "unique_assistant_turns": len({r["messages"][1]["content"] for r in positives}),
    }


def write_dataset(result: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        path = os.path.join(out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in result[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(result["stats"], f, ensure_ascii=False, indent=2)


def print_report(result: Dict[str, Any]) -> None:
    st = result["stats"]
    print("=" * 60)
    print("toneforge 数据集构建报告")
    print("=" * 60)
    print(f"种子对话          : {st['n_seeds']}")
    print(f"风格正样本        : {st['n_positives']}  (种子直出 + 变体扩充)")
    print(f"负样本(坏例夹具)  : {st['n_negatives']}  {st['by_negative_type']}")
    print(f"通用样本          : {st['n_general']}  (占比 {st['general_ratio']:.1%})")
    print(f"总量              : {st['total']}  train/val/test = {st['train']}/{st['val']}/{st['test']}")
    print(f"场景分布          : {st['by_scenario']}")
    print(f"难度分布(正样本)  : {st['by_difficulty']}")
    print(f"正样本去重        : user {st['unique_user_turns']}/{st['n_positives']}, "
          f"assistant {st['unique_assistant_turns']}/{st['n_positives']}")
    print("=" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建客服话术风格指令数据集（纯标准库）")
    parser.add_argument("--seeds-dir", default="data/seeds")
    parser.add_argument("--general", default="data/general_samples.jsonl")
    parser.add_argument("--out", default=os.path.join("outputs", "dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--general-ratio", type=float, default=0.10)
    args = parser.parse_args(argv)

    result = build_dataset(args.seeds_dir, args.general, seed=args.seed,
                           n_variants=args.variants, general_ratio=args.general_ratio)

    # 输出前自检：所有记录必须通过 schema 校验
    problems = schema.validate_records(result["train"] + result["val"] + result["test"])
    if problems:
        for line_no, errs in problems[:20]:
            print(f"[schema] 第 {line_no + 1} 行: {errs}")
        raise SystemExit(f"schema 校验失败：{len(problems)} 条记录不合法")

    write_dataset(result, args.out)
    print_report(result)
    print(f"已写入 {args.out}/train.jsonl, val.jsonl, test.jsonl, stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
