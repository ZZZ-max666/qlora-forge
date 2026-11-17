# toneforge 流水线设计说明

## 1. 数据规范

### 1.1 训练数据：ShareGPT messages JSONL

每行一个 JSON 对象（`toneforge/data/schema.py` 校验）：

```json
{
  "id": "return_exchange-001-v2",
  "messages": [
    {"role": "user", "content": "我7天前下单的…"},
    {"role": "assistant", "content": "共情段\n\n政策段\n\n动作段", "loss": true}
  ],
  "meta": {
    "source": "seed | variant | negative | general",
    "scenario": "return_exchange | logistics_push | complaint_soothe | invoice_query | price_protect | general",
    "difficulty": "easy | medium | hard | negative",
    "negative_type": "none | missing_empathy | fabricated_policy | format_error",
    "variant_of": "return_exchange-001"
  }
}
```

校验规则：

- `id` 非空且数据集内唯一；
- 消息序列：可选一条 `system` 开头，随后 `user` 开始、与 `assistant` 严格交替、以 `assistant` 结束；
- `content` 为非空字符串；
- `loss` 标记只允许出现在 `assistant` 消息上且必须是布尔值；缺省视为 `true`。
  整个会话至少一条 `assistant` 的 `loss` 为 `true`，否则视为无训练目标；
- `meta` 可选，但强烈建议携带，评估模块按它分组统计。

### 1.2 种子格式：模板 + 槽位

`data/seeds/*.jsonl` 每行是一条种子的**结构化模板**（渲染后即一段规范的三段式对话）：

```json
{
  "id": "return_exchange-001",
  "scenario": "return_exchange",
  "tpl": {"user": "我{days}天前下单的{product}，{reason}，可以退货吗？",
          "empathy": "…", "policy": "…《{clause}》…{days}天内…", "action": "…"},
  "slots": {"product": "女式冲锋衣", "days": "7", "amount": "699",
            "clause": "七天无理由退货", "reason": "尺码不合适"}
}
```

三段式规范：**共情 → 政策依据 → 下一步动作**。政策段必须引用《条款名》，
动作段必须包含"做什么 + 在哪做"。

> ⚠️ 全部种子与通用样本均为**演示用合成数据**，不含任何真实客户信息。

## 2. 扩充策略（`toneforge/data/build.py`）

150 条种子 → 900 条指令集（固定 `--seed` 完全可复现）：

| 变体 | 难度 | 手段 |
|---|---|---|
| 种子直出 | easy | 模板渲染 |
| v1 | easy | 槽位重采样（商品/金额/原因/条款/快递），条款↔时效耦合保持一致 |
| v2 | medium | 用户追加约束条件句 + 客服针对性回应句 |
| v3 | hard | 情绪前缀 + 复合追问 + 补充解答 + 槽位重采样 |
| v4 | easy | 同义表达替换（13 组语义安全的措辞改写对） |

负样本（每种 20 条，全部进 test 集作为规则评估的坏例夹具）：

- `missing_empathy`：删除共情段；
- `fabricated_policy`：引用登记条款表之外的编造条款（格式完好、内容违规）；
- `format_error`：动作段先于政策段，或缺失动作段。

通用数据按 `gen / total ≈ 10%` 混入（`toneforge/data/general.py`，内置 104 条
算术/翻译/常识/逻辑/编程/写作样本；真实开源指令集通过 `EXTERNAL_SOURCES`
骨架自行接入，注意核对上游 License）。

切分：正样本+通用样本 80/10/10 随机切分（不重叠），负样本全部进 test。

## 3. 超参说明

| 参数 | smoke (0.5B) | prod (7B) | 说明 |
|---|---|---|---|
| 量化 | NF4 双量化 | NF4 双量化 | 权重 0.52 字节/参数 |
| lora.r / alpha | 16 / 32 | 16 / 32 | alpha=2r 为常用起点 |
| lora.dropout | 0.05 | 0.05 | 小数据集防过拟合 |
| target_modules | 注意力+FFN 全线性层 | 同左 | QLoRA 论文结论：全线性层优于仅注意力 |
| batch × grad_accum | 2 × 8 = 16 | 1 × 16 = 16 | 有效 batch 对齐 |
| lr | 2e-4 | 1e-4 | 小模型可用更大学习率 |
| seq_len | 1024 | 2048 | 覆盖三段式回复长度 |
| epochs | 1 | 3 | 冒烟只求跑通 |
| 优化器 | paged_adamw_8bit | 同左 | 分页防 OOM |

## 4. 评估体系

| 层 | 模块 | 性质 |
|---|---|---|
| 结构遵循 | `eval/format_check.py` | 三段式规则+关键词+正则，输出遵循率；正样本应≈100%，负样本必须被拦截 |
| 内容合规 | `eval/rule_metrics.py` | 政策落地性（条款须存在于登记表）、超长、重复 n-gram、拒答正确性 |
| 通用回归 | `eval/regress.py` | 12 道内置确定性小测，训练前后 pass 率对比 |
| 主观质量 | `eval/blind_export.py` | 双模型输出打乱导出盲评 CSV，固定 seed 可复现 |

分工示例：`fabricated_policy` 负样本**通过**结构校验（格式完好）但被政策
落地性规则拦截——这正是"格式校验"与"内容校验"必须分开的原因。

## 5. 部署链路

训练产出 LoRA 适配器 → `deploy/merge.py` 合并为完整权重（bf16 safetensors）
→ 可选 `deploy/quant.py` GPTQ INT4（仅 Linux）→ `deploy/serve_vllm.py`
vLLM OpenAI 兼容服务 + 标准库压测客户端（吞吐/延迟分位数）。
