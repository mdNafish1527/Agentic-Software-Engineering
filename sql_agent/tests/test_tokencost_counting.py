"""Tests for token counting, cost, overhead and the Ollama reply parser (no real model needed)."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from tokencost.backend import BackendError, FakeBackend, OllamaBackend
from tokencost.counting import cache_check, cost_usd, count_call, estimate_overhead


class CostTests(unittest.TestCase):
    def test_price_per_million_tokens(self):
        self.assertAlmostEqual(cost_usd(1_000_000, 0), 0.15)
        self.assertAlmostEqual(cost_usd(0, 1_000_000), 0.60)

    def test_mixed_and_zero(self):
        self.assertAlmostEqual(cost_usd(200, 100), 200 * 0.15 / 1e6 + 100 * 0.60 / 1e6)
        self.assertEqual(cost_usd(0, 0), 0)


class CountCallTests(unittest.TestCase):
    def test_counts_come_from_backend_and_cost_is_computed(self):
        b = FakeBackend(template_overhead=20, answer_words=5)
        r = count_call(b, "one two three")
        self.assertEqual(r["input_tokens"], 23)          # 20 overhead + 3 words
        self.assertEqual(r["output_tokens"], 5)
        self.assertAlmostEqual(r["cost_usd"], cost_usd(23, 5))

    def test_fresh_unloads_before_each_call(self):
        b = FakeBackend()
        count_call(b, "a", fresh=True)
        count_call(b, "b", fresh=True)
        self.assertEqual(b.unloads, 2)

    def test_not_fresh_keeps_cache(self):
        b = FakeBackend()
        count_call(b, "a", fresh=False)
        self.assertEqual(b.unloads, 0)


class OverheadTests(unittest.TestCase):
    def test_overhead_estimate_recovers_template_size(self):
        r = estimate_overhead(FakeBackend(template_overhead=20))
        self.assertEqual(r["overhead_tokens"], 20)
        self.assertEqual(r["tokens_per_extra_word"], 1)

    def test_cache_check_detects_a_reused_prefix(self):
        class Cached(FakeBackend):
            def chat(self, user_text, system=""):
                reply = super().chat(user_text, system)
                if self.calls[-2:-1] and self.calls[-2][1] == user_text:
                    reply.prompt_tokens = 1          # simulate Ollama reusing the cached prompt
                return reply
        self.assertTrue(cache_check(Cached())["cache_reduced_second_call"])
        self.assertFalse(cache_check(FakeBackend())["cache_reduced_second_call"])


class StubOllama(BaseHTTPRequestHandler):
    omit_counts = False
    paths = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubOllama.paths.append((self.path, body))
        if self.path == "/api/generate":
            reply = {"done": True}
        else:
            reply = {"message": {"role": "assistant", "content": f"ok {body['options']['temperature']}"},
                     "prompt_eval_count": 33, "eval_count": 4,
                     "total_duration": 3_000_000_000, "load_duration": 1_000_000_000, "done_reason": "stop"}
            if StubOllama.omit_counts:
                del reply["eval_count"]
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class OllamaBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), StubOllama)
        cls.host = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        StubOllama.paths.clear()
        StubOllama.omit_counts = False

    def test_reads_ollama_counts_and_excludes_load_time_from_latency(self):
        reply = OllamaBackend(host=self.host).chat("hello")
        self.assertEqual((reply.prompt_tokens, reply.completion_tokens), (33, 4))
        self.assertAlmostEqual(reply.latency_s, 2.0)     # 3.0 total minus 1.0 load
        self.assertAlmostEqual(reply.load_s, 1.0)
        self.assertEqual(reply.done_reason, "stop")

    def test_request_uses_fixed_options_and_system_prompt(self):
        OllamaBackend(host=self.host).chat("hello", system="SYS")
        path, body = StubOllama.paths[-1]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["seed"], 42)
        self.assertEqual(body["options"]["num_predict"], 256)
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYS"})

    def test_unload_posts_keep_alive_zero(self):
        OllamaBackend(host=self.host).unload()
        path, body = StubOllama.paths[-1]
        self.assertEqual(path, "/api/generate")
        self.assertEqual(body["keep_alive"], 0)

    def test_missing_token_count_is_an_error_not_a_zero(self):
        StubOllama.omit_counts = True
        with self.assertRaises(BackendError):
            OllamaBackend(host=self.host).chat("hello")

    def test_unreachable_server_is_a_backend_error(self):
        with self.assertRaises(BackendError):
            OllamaBackend(host="http://127.0.0.1:9", timeout=2).chat("hi")


if __name__ == "__main__":
    unittest.main()
