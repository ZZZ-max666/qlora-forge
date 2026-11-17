# -*- coding: utf-8 -*-
"""vLLM 部署脚本 + 压测客户端。

- build_launch_args()：把部署配置转为 vLLM OpenAI 兼容服务启动参数（纯函数，可单测）；
- launch()：在当前机器上拉起 vllm.entrypoints.openai.api_server（需 Linux+GPU+vllm）；
- bench()：纯标准库（urllib）压测客户端，统计吞吐(req/s、tokens/s)与延迟分位数，
  可在无 GPU 环境用本地 mock 服务单测。

用法：
    python -m toneforge.deploy.serve_vllm --serve --model outputs/merged/smoke
    python -m toneforge.deploy.serve_vllm --bench --url http://127.0.0.1:8000 --rounds 32
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_PROMPTS = [
    "用户：我买的跑步鞋尺码不合适，吊牌还没剪，可以退货吗？",
    "用户：包裹三天没物流更新了，帮我催一下。",
    "用户：刚买就降价了100元，能补差价吗？",
    "用户：发票开错了抬头，怎么重新开？",
]


def build_launch_args(model: str, host: str = "127.0.0.1", port: int = 8000,
                      max_model_len: int = 4096, gpu_memory_utilization: float = 0.85,
                      quantization: Optional[str] = None,
                      tensor_parallel: int = 1) -> List[str]:
    """构造 `python -m vllm.entrypoints.openai.api_server ...` 参数列表。"""
    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel),
    ]
    if quantization:
        args += ["--quantization", quantization]
    return args


def launch(cfg: Dict[str, Any]) -> None:  # pragma: no cover - 需 Linux+GPU
    if platform.system() != "Linux":
        print(f"[serve] vLLM 仅支持 Linux，当前 {platform.system()}；请在 WSL/服务器运行。",
              file=sys.stderr)
        raise SystemExit(2)
    cmd = build_launch_args(**cfg)
    print("[serve] 启动命令：", " ".join(cmd))
    subprocess.call(cmd)


def percentile(values: List[float], q: float) -> float:
    """最近邻法百分位（q ∈ [0,100]）。"""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))
    return s[idx]


def bench(url: str, prompts: Optional[List[str]] = None, rounds: int = 16,
          timeout: float = 60.0, max_tokens: int = 256) -> Dict[str, Any]:
    """串行压测 OpenAI 兼容 /v1/chat/completions 接口，返回吞吐与延迟统计。"""
    prompts = prompts or DEFAULT_PROMPTS
    latencies: List[float] = []
    completion_tokens = 0
    failures = 0
    t0 = time.perf_counter()
    for i in range(rounds):
        payload = {
            "messages": [{"role": "user", "content": prompts[i % len(prompts)]}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            latencies.append(time.perf_counter() - start)
            completion_tokens += int(body.get("usage", {}).get("completion_tokens", 0))
        except Exception as exc:  # 网络/服务异常计入失败，不中断压测
            failures += 1
            print(f"[bench] 第 {i + 1} 次请求失败: {exc}", file=sys.stderr)
    wall = time.perf_counter() - t0
    ok = len(latencies)
    return {
        "rounds": rounds,
        "success": ok,
        "failures": failures,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(ok / wall, 3) if wall > 0 else 0.0,
        "throughput_tps": round(completion_tokens / wall, 2) if wall > 0 else 0.0,
        "latency_avg": round(sum(latencies) / ok, 4) if ok else 0.0,
        "latency_p50": round(percentile(latencies, 50), 4),
        "latency_p95": round(percentile(latencies, 95), 4),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM 启动 / 压测")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--serve", action="store_true", help="启动 vLLM 服务（Linux+GPU）")
    group.add_argument("--bench", action="store_true", help="压测已运行的服务")
    parser.add_argument("--model", default="outputs/merged/smoke")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--quantization", default=None, help="如 gptq（量化目录部署时指定）")
    parser.add_argument("--rounds", type=int, default=16)
    args = parser.parse_args(argv)

    if args.serve:
        launch(dict(model=args.model, host=args.host, port=args.port,
                    max_model_len=args.max_model_len, quantization=args.quantization))
    else:
        rep = bench(args.url, rounds=args.rounds)
        print("压测报告（串行请求，反映单并发时延与吞吐上限）")
        for k, v in rep.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
