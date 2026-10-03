from __future__ import annotations

import dataclasses
import json
import threading
import unittest
import urllib.error
import urllib.request

from tests.test_phase30_qwen_gpu import gpu_fixture, gpu_metadata
from vinf.errors import ExecutorUnavailableError
from vinf.gguf.tokenizer import QwenTokenizer, SpecialTokens

# 40-token vocabulary matching the fixture model: specials + single byte-level characters.
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
CHARS = list("abcdefghijklmnopqrstuvwxyz") + ["Ġ", "Ċ", ".", "?", "A", "B", "C", "!", "'"]
VOCAB = SPECIALS + CHARS
assert len(VOCAB) == 40


def tiny_tokenizer() -> QwenTokenizer:
    return QwenTokenizer(
        tokens=tuple(VOCAB),
        merges=(),
        token_types=tuple([3, 3, 3, 4, 4] + [1] * len(CHARS)),
        special_tokens=SpecialTokens(bos_token_id=0, eos_token_id=2, padding_token_id=None, im_start_id=1, im_end_id=2),
        model="gpt2",
        pre="qwen35",
        chat_template=None,
    )


def backend_or_skip(test, speculative: int = 0):
    from vinf.qwen_backend import QwenBackend

    meta = dataclasses.replace(gpu_metadata(), max_position_embeddings=64)
    try:
        from vinf.qwen_gpu import QwenGpuExecutor

        ex = QwenGpuExecutor(gpu_fixture(meta), meta, max_context=64, mtp=speculative > 0,
                             snapshot_tokens=speculative + 1 if speculative else 0)
    except (ExecutorUnavailableError, RuntimeError) as exc:
        test.skipTest(str(exc))
    decoder = None
    if speculative:
        from vinf.qwen_speculative import QwenSpeculativeDecoder

        decoder = QwenSpeculativeDecoder(ex, speculative)
    return QwenBackend(ex, tiny_tokenizer(), decoder=decoder, model_name="tiny-qwen")


class Server:
    def __init__(self, backend, **kwargs) -> None:
        from vinf.server import OpenAIService, make_server

        self.service = OpenAIService(backend, log=lambda msg: None, **kwargs)
        self.httpd = make_server(self.service, "127.0.0.1", 0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, method: str, path: str, body=None, headers=None, raw: bytes | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.headers.get("Content-Type", ""), err.read().decode()

    def json(self, method, path, body=None, headers=None):
        status, _, text = self.request(method, path, body, headers)
        return status, json.loads(text)

    def stream(self, path, body):
        status, ctype, text = self.request("POST", path, {**body, "stream": True})
        self_events = [line[len("data: "):] for line in text.split("\n") if line.startswith("data: ")]
        return status, ctype, self_events


CHAT = {"model": "tiny-qwen", "messages": [{"role": "user", "content": "hi there"}],
        "chat_template_kwargs": {"enable_thinking": False}}


class Phase31hServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = backend_or_skip(self)
        self.srv = Server(self.backend)

    def tearDown(self) -> None:
        self.srv.close()

    def expected(self, messages, max_tokens, thinking=False):
        tok = self.backend.tokenizer
        prompt = tok.encode(tok.apply_chat_template(messages, enable_thinking=thinking))
        tokens, _ = self.backend.generate(prompt, max_tokens)
        body = [t for t in tokens if t not in self.backend.stop_token_ids]
        return prompt, tokens, tok.decode(body)

    def test_health_and_models(self) -> None:
        self.assertEqual(self.srv.json("GET", "/health"), (200, {"status": "ok"}))
        status, body = self.srv.json("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][0]["id"], "tiny-qwen")

    def test_chat_completion_matches_backend_greedy(self) -> None:
        prompt, tokens, text = self.expected(CHAT["messages"], 12)
        status, body = self.srv.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 12})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], text.lstrip("\n"))
        self.assertEqual(body["usage"]["prompt_tokens"], len(prompt))
        self.assertEqual(body["usage"]["completion_tokens"], len(tokens))
        expected_finish = "stop" if tokens[-1] in self.backend.stop_token_ids else "length"
        self.assertEqual(body["choices"][0]["finish_reason"], expected_finish)

    def test_streaming_chat_equals_non_streaming(self) -> None:
        _, full = self.srv.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 12})
        status, ctype, events = self.srv.stream("/v1/chat/completions",
                                                {**CHAT, "max_tokens": 12, "stream_options": {"include_usage": True}})
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", ctype)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        self.assertEqual(text, full["choices"][0]["message"]["content"])
        finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
        self.assertEqual(finals[-1]["choices"][0]["finish_reason"], full["choices"][0]["finish_reason"])
        self.assertEqual(chunks[-1]["usage"], full["usage"])

    def test_max_tokens_and_stop_strings(self) -> None:
        status, body = self.srv.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 3})
        tokens = self.expected(CHAT["messages"], 3)[1]
        if tokens[-1] not in self.backend.stop_token_ids:
            self.assertEqual(body["choices"][0]["finish_reason"], "length")
            self.assertEqual(body["usage"]["completion_tokens"], 3)
        _, _, text = self.expected(CHAT["messages"], 20)
        content = text.lstrip("\n")
        if len(content) >= 4:
            stop = content[2:4]
            status, body = self.srv.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 20, "stop": stop})
            self.assertEqual(body["choices"][0]["message"]["content"], content[: content.find(stop)])
            self.assertEqual(body["choices"][0]["finish_reason"], "stop")
            _, _, events = self.srv.stream("/v1/chat/completions", {**CHAT, "max_tokens": 20, "stop": stop})
            streamed = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
            self.assertEqual(streamed, content[: content.find(stop)])

    def test_completions_endpoint(self) -> None:
        tok = self.backend.tokenizer
        tokens, _ = self.backend.generate(tok.encode("hello world"), 8)
        text = tok.decode([t for t in tokens if t not in self.backend.stop_token_ids])
        status, body = self.srv.json("POST", "/v1/completions", {"prompt": "hello world", "max_tokens": 8})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["choices"][0]["text"], text)
        _, _, events = self.srv.stream("/v1/completions", {"prompt": "hello world", "max_tokens": 8})
        self.assertEqual("".join(json.loads(e)["choices"][0]["text"] for e in events[:-1]), text)

    def test_errors_are_openai_formatted(self) -> None:
        status, body = self.srv.json("POST", "/v1/chat/completions", {**CHAT, "n": 2})
        self.assertEqual(status, 400)
        self.assertEqual(set(body["error"]), {"message", "type", "param", "code"})
        self.assertEqual(body["error"]["param"], "n")
        status, _, text = self.srv.request("POST", "/v1/chat/completions", raw=b"{nope")
        self.assertEqual(status, 400)
        self.assertEqual(self.srv.json("GET", "/v1/nothing")[0], 404)
        status, body = self.srv.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 10_000})
        self.assertEqual(status, 200)  # clipped to the room left in the context
        long_chat = {**CHAT, "messages": [{"role": "user", "content": "a" * 100}]}
        status, body = self.srv.json("POST", "/v1/chat/completions", long_chat)
        self.assertEqual((status, body["error"]["code"]), (400, "context_length_exceeded"))
        self.assertEqual(self.srv.json("POST", "/chat/completions", {**CHAT, "max_tokens": 2})[0], 200)
        self.assertEqual(self.srv.json("GET", "/models")[0], 200)
        status, body = self.srv.json("POST", "/v1/chat/completions", {"messages": [{"role": "robot", "content": "x"}]})
        self.assertEqual(status, 400)

    def test_sampling_parameters_are_ignored_or_rejected(self) -> None:
        _, greedy = self.srv.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 6})
        status, body = self.srv.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 6, "temperature": 0.7})
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"], greedy["choices"][0]["message"])
        strict = Server(self.backend, strict_sampling=True)
        try:
            status, body = strict.json("POST", "/v1/chat/completions", {**CHAT, "temperature": 0.7})
            self.assertEqual((status, body["error"]["param"]), (400, "temperature"))
            self.assertEqual(strict.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 4, "temperature": 0})[0], 200)
        finally:
            strict.close()

    def test_api_key(self) -> None:
        keyed = Server(self.backend, api_key="secret")
        try:
            self.assertEqual(keyed.json("GET", "/v1/models")[0], 401)
            self.assertEqual(keyed.json("GET", "/v1/models", headers={"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(keyed.json("GET", "/health")[0], 200)
        finally:
            keyed.close()

    def test_loading_state(self) -> None:
        from vinf.server import OpenAIService, make_server

        service = OpenAIService(None, log=lambda m: None)
        httpd = make_server(service, "127.0.0.1", 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(base + "/health", timeout=10)
            self.assertEqual(json.loads(ctx.exception.read()), {"status": "loading"})
            service.set_backend(self.backend)
            with urllib.request.urlopen(base + "/health", timeout=10) as resp:
                self.assertEqual(json.loads(resp.read()), {"status": "ok"})
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_speculative_backend_serves_identical_text(self) -> None:
        spec = Server(backend_or_skip(self, speculative=3))
        try:
            _, a = self.srv.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 16})
            _, b = spec.json("POST", "/v1/chat/completions", {**CHAT, "max_tokens": 16})
            self.assertEqual(a["choices"][0]["message"], b["choices"][0]["message"])
        finally:
            spec.close()


class Phase31hTextStreamTests(unittest.TestCase):
    def ids(self, text: str) -> list[int]:
        return tiny_tokenizer().encode(text)

    def run_stream(self, tokens, stops=(), reasoning=False):
        from vinf.server import TextStream, _Cancelled

        ts = TextStream(tiny_tokenizer(), frozenset({2}), list(stops), reasoning=reasoning)
        out = []
        try:
            for t in tokens:
                out.extend(ts.feed(t))
        except _Cancelled as stop:
            out.extend(stop.args[0])
        if ts.finish_reason is None:
            out.extend(ts._flush())
        return out, ts.finish_reason

    def test_reasoning_split(self) -> None:
        out, finish = self.run_stream(self.ids("let me think</think>\n\nanswer") + [2], reasoning=True)
        self.assertEqual("".join(t for f, t in out if f == "reasoning_content"), "let me think")
        self.assertEqual("".join(t for f, t in out if f == "content"), "answer")
        self.assertEqual(finish, "stop")

    def test_stop_prefix_is_held_back(self) -> None:
        from vinf.server import TextStream

        ts = TextStream(tiny_tokenizer(), frozenset({2}), ["xyz"], reasoning=False)
        self.assertEqual(ts.feed(self.ids("a")[0]), [("content", "a")])
        self.assertEqual(ts.feed(self.ids("x")[0]), [])
        self.assertEqual(ts.feed(self.ids("y")[0]), [])
        self.assertEqual(ts.feed(self.ids("q")[0]), [("content", "xyq")])
        out, finish = self.run_stream(self.ids("abxyzcd"), stops=["xyz"])
        self.assertEqual("".join(t for _, t in out), "ab")
        self.assertEqual(finish, "stop")


if __name__ == "__main__":
    unittest.main()


# ---- tool calling (scripted backend: no GPU needed) ------------------------------------------------

class CharTokenizer:
    """One token per character (id = code point); id 0 ends the turn."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


class ScriptedBackend:
    model_name = "scripted"
    max_context = 100_000
    stop_token_ids = frozenset({0})

    def __init__(self) -> None:
        self.tokenizer = CharTokenizer()
        self.reply = ""
        self.prompts: list[str] = []

    def generate(self, prompt, max_new, on_token=None):
        self.prompts.append(self.tokenizer.decode(prompt))
        out = []
        for t in self.tokenizer.encode(self.reply)[:max_new] + [0]:
            out.append(t)
            if on_token:
                on_token(t)
            if t == 0:
                break
        return out, None


WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
CALL = ("I will check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")


class Phase31hToolCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ScriptedBackend()
        self.srv = Server(self.backend)

    def tearDown(self) -> None:
        self.srv.close()

    def body(self, **extra):
        return {"model": "scripted", "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": WEATHER, **extra}

    def test_tools_rendered_with_gguf_template(self) -> None:
        from vinf.chat_template import render_chat

        self.backend.reply = "</think>\n\nSunny."
        status, body = self.srv.json("POST", "/v1/chat/completions", self.body())
        self.assertEqual(status, 200, body)
        self.assertEqual(self.backend.prompts[-1],
                         render_chat([{"role": "user", "content": "weather in Paris?"}], tools=WEATHER))
        self.assertIn("<tools>", self.backend.prompts[-1])
        self.assertEqual(body["choices"][0]["message"]["content"], "Sunny.")
        self.assertNotIn("tool_calls", body["choices"][0]["message"])

    def test_non_streaming_tool_call(self) -> None:
        self.backend.reply = "Thinking it over.</think>\n\n" + CALL
        status, body = self.srv.json("POST", "/v1/chat/completions", self.body())
        self.assertEqual(status, 200, body)
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "I will check.")
        self.assertEqual(choice["message"]["reasoning_content"], "Thinking it over.")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual((call["type"], call["function"]["name"]), ("function", "get_weather"))
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris", "days": 3})

    def test_streaming_tool_call(self) -> None:
        self.backend.reply = "</think>\n\n" + CALL
        status, _, events = self.srv.stream("/chat/completions", self.body())
        self.assertEqual(status, 200)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        self.assertEqual(content, "I will check.")
        self.assertNotIn("<tool", content)
        calls = [d for c in chunks for d in c["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["index"], 0)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Paris", "days": 3})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_lookalike_text_is_not_swallowed(self) -> None:
        self.backend.reply = "</think>\n\na <tool and <tool_call> unfinished  "
        _, _, events = self.srv.stream("/v1/chat/completions", self.body())
        chunks = [json.loads(e) for e in events[:-1]]
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        self.assertEqual(content, "a <tool and <tool_call> unfinished  ")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_tool_history_round_trip(self) -> None:
        from vinf.chat_template import render_chat

        self.backend.reply = "</think>\n\nIt is sunny."
        messages = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 3}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "sunny, 21C"}]},
        ]
        status, body = self.srv.json("POST", "/v1/chat/completions", {"messages": messages, "tools": WEATHER})
        self.assertEqual(status, 200, body)
        expected = render_chat([
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]},
            {"role": "tool", "content": "sunny, 21C"},
        ], tools=WEATHER)
        self.assertEqual(self.backend.prompts[-1], expected)
        self.assertIn("<tool_response>\nsunny, 21C\n</tool_response>", expected)
        bad = [*messages[:2], {"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": "{oops"}}]}]
        self.assertEqual(self.srv.json("POST", "/v1/chat/completions", {"messages": bad})[0], 400)

    def test_tool_choice_none_hides_tools(self) -> None:
        self.backend.reply = "ok"
        self.srv.json("POST", "/v1/chat/completions", self.body(tool_choice="none"))
        self.assertNotIn("<tools>", self.backend.prompts[-1])


class Phase31hLoadingTests(unittest.TestCase):
    def test_requests_wait_for_loading(self) -> None:
        from vinf.server import OpenAIService, make_server

        service = OpenAIService(None, log=lambda m: None)
        httpd = make_server(service, "127.0.0.1", 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        backend = ScriptedBackend()
        backend.reply = "</think>\n\nhello"
        threading.Timer(0.5, service.set_backend, args=(backend,)).start()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/chat/completions",
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                self.assertEqual(json.loads(resp.read())["choices"][0]["message"]["content"], "hello")
        finally:
            httpd.shutdown()
            httpd.server_close()
