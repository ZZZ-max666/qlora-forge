# -*- coding: utf-8 -*-
"""toneforge：客服话术风格 QLoRA 指令微调流水线。

模块分层：
- toneforge.data   数据构建（纯标准库）
- toneforge.train  训练入口与显存估算（训练依赖惰性导入，估算纯标准库）
- toneforge.deploy 合并 / 量化 / vLLM 部署
- toneforge.eval   确定性规则评估（纯标准库）
"""

__version__ = "0.1.0"
