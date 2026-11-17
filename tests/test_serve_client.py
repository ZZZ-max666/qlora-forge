# -*- coding: utf-8 -*-
"""vLLM 启动参数构造 + 压测客户端（对本地 mock 服务跑真实 HTTP）。"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from toneforge.deploy import serve_vllm


class _MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        assert body["messages"][0]["role"] == "user"
        resp = {
            "choices": [{"message": {"role": "assistant", "content": "好的"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # 静默
        pass


class TestBuildLaunchArgs(unittest.TestCase):
    def test_basic(self):
        args = serve_vllm.build_launch_args("outputs/merged/smoke", port=8123)
        self.assertIn("--model", args)
        self.assertIn("outputs/merged/smoke", args)
        self.assertIn("8123", args)
        self.assertIn("vllm.entrypoints.openai.api_server", args)

    def test_quantization_flag(self):
        args = serve_vllm.build_launch_args("m", quantization="gptq")
        i = args.index("--quantization")
        self.assertEqual(args[i + 1], "gptq")

    def test_no_quantization_by_default(self):
        self.assertNotIn("--quantization", serve_vllm.build_launch_args("m"))


class TestPercentile(unittest.TestCase):
    def test_basic(self):
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(serve_vllm.percentile(vals, 50), 5)
        self.assertEqual(serve_vllm.percentile(vals, 95), 10)
        self.assertEqual(serve_vllm.percentile([], 50), 0.0)


class TestBenchClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_bench_success_stats(self):
        rep = serve_vllm.bench(self.url, rounds=6)
        self.assertEqual(rep["success"], 6)
        self.assertEqual(rep["failures"], 0)
        self.assertGreater(rep["throughput_rps"], 0)
        self.assertAlmostEqual(rep["throughput_tps"], rep["throughput_rps"] * 8, delta=1.0)
        self.assertGreater(rep["latency_p50"], 0)
        self.assertLessEqual(rep["latency_p50"], rep["latency_p95"])

    def test_bench_connection_error_counted(self):
        rep = serve_vllm.bench("http://127.0.0.1:1", rounds=2, timeout=2)
        self.assertEqual(rep["success"], 0)
        self.assertEqual(rep["failures"], 2)


if __name__ == "__main__":
    unittest.main()
