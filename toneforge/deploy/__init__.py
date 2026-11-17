# -*- coding: utf-8 -*-
"""部署子包：LoRA 合并、GPTQ 量化、vLLM 服务与压测。

GPU/第三方依赖均为惰性导入；serve_vllm.py 的压测客户端为纯标准库，
可在无 GPU 环境单测（tests/test_serve_client.py）。"""
