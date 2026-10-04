"""OpenAI-compatible HTTP server for the qwen35 engine (standard library only).

Endpoints (the /v1 prefix is optional):
  GET  /health                 {"status": "loading" | "ok" | "error"}
  GET  /v1/models              the loaded model
  POST /v1/chat/completions    chat (the model's GGUF chat template) with tool calling, streaming via server-sent events
  POST /v1/completions         raw text completion, streaming via server-sent events

Concurrent requests are decoded together by the continuous-batching scheduler (vinf.scheduler) when the
backend supports it (--max-seqs > 1); otherwise they wait in line.
Sampling parameters are accepted and decoded greedily (the engine is greedy-only for now) unless
the server runs with --strict-sampling, which rejects them.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vinf.chat_template import TOOL_CALL_START, parse_tool_calls, render_chat
from vinf.errors import UnsupportedModelError, VinfError

THINK_END = "</think>"


class ApiError(Exception):
    def __init__(self, status: int, message: str, *, param: str | None = None, code: str | None = None,
                 kind: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status, self.message, self.param, self.code, self.kind = status, message, param, code, kind

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.kind, "param": self.param, "code": self.code}}


class _Cancelled(Exception):
    """Raised inside the token callback to stop generation (stop string hit or client gone)."""


class TextStream:
    """Turns generated token ids into text deltas: holds back incomplete UTF-8 characters and any tail
    that could still become a stop string; splits reasoning (before </think>) from content."""

    def __init__(self, tokenizer, stop_ids: frozenset[int], stops: list[str], *, reasoning: bool,
                 tools: bool = False) -> None:
        self.tokenizer, self.stop_ids, self.stops = tokenizer, stop_ids, [s for s in stops if s]
        self.reasoning = reasoning  # True while generating inside <think> ... </think>
        self.tokens: list[int] = []
        self.emitted = 0  # characters of the decoded text already emitted
        self.finish_reason: str | None = None
        self.content_started = False
        # With tools, content that could begin a <tool_call> (and trailing whitespace before one) is held in
        # `pending`; from the first <tool_call> on, everything goes to `call_text` for parse_tool_calls.
        self.tools = tools
        self.pending = ""
        self.call_text: str | None = None
        self.call_space = ""  # whitespace dropped before <tool_call>, restored if the call is malformed

    def _holdback(self, text: str) -> int:
        """Characters at the end of text that must wait (partial UTF-8, possible stop-string prefix)."""
        hold = 1 if text.endswith("�") else 0
        for stop in self.stops + ([THINK_END] if self.reasoning else []):
            for n in range(min(len(stop) - 1, len(text)), 0, -1):
                if text.endswith(stop[:n]):
                    hold = max(hold, n)
                    break
        return hold

    def feed(self, token: int) -> list[tuple[str, str]]:
        """Returns [(field, text)] deltas, field in {"reasoning_content", "content"}; raises _Cancelled on a stop string."""
        if token in self.stop_ids:
            self.finish_reason = "stop"
            return self._flush()
        self.tokens.append(token)
        text = self.tokenizer.decode(self.tokens)
        for stop in self.stops:
            idx = text.find(stop, max(0, self.emitted - len(stop) + 1))
            if idx >= 0:
                deltas = self._emit(text[:idx])
                self.finish_reason = "stop"
                raise _Cancelled(deltas)
        return self._emit(text[: len(text) - self._holdback(text)])

    def _flush(self) -> list[tuple[str, str]]:
        text = self.tokenizer.decode(self.tokens)
        return self._emit(text.rstrip("�"))

    def _emit(self, upto: str) -> list[tuple[str, str]]:
        new = upto[self.emitted:]
        self.emitted = max(self.emitted, len(upto))
        if not new:
            return []
        out: list[tuple[str, str]] = []
        if self.reasoning:
            idx = new.find(THINK_END)
            if idx < 0:
                return [("reasoning_content", new)]
            if idx:
                out.append(("reasoning_content", new[:idx]))
            self.reasoning = False
            new = new[idx + len(THINK_END):]
        if not self.content_started:
            new = new.lstrip("\n")
            if not new:
                return out
            self.content_started = True
        if self.tools:
            return out + self._tool_filter(new)
        out.append(("content", new))
        return out

    def _tool_filter(self, new: str) -> list[tuple[str, str]]:
        if self.call_text is not None:
            self.call_text += new
            return []
        text = self.pending + new
        idx = text.find(TOOL_CALL_START)
        if idx >= 0:
            head = text[:idx].rstrip()
            self.pending, self.call_text, self.call_space = "", text[idx:], text[len(head):idx]
            return [("content", head)] if head else []
        hold = 0
        for n in range(min(len(TOOL_CALL_START) - 1, len(text)), 0, -1):
            if text.endswith(TOOL_CALL_START[:n]):
                hold = n
                break
        rest = text[: len(text) - hold]
        hold += len(rest) - len(rest.rstrip())
        self.pending = text[len(text) - hold:] if hold else ""
        head = text[: len(text) - hold]
        return [("content", head)] if head else []

    def finish(self) -> list[tuple[str, str]]:
        """Release held-back content once generation is over (call text stays in call_text)."""
        pending, self.pending = self.pending, ""
        return [("content", pending)] if pending and self.call_text is None else []


def _now() -> int:
    return int(time.time())


def _content_text(content, index: int) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                raise ApiError(400, "only text message content is supported", param=f"messages[{index}].content")
            parts.append(str(part.get("text", "")))
        return "".join(parts)
    raise ApiError(400, "message content must be a string or a list of text parts", param=f"messages[{index}].content")


def _tool_arguments(args, index: int, n: int):
    if isinstance(args, str):
        if not args.strip():
            return {}
        try:
            args = json.loads(args)
        except json.JSONDecodeError as exc:
            raise ApiError(400, f"tool call arguments are not valid JSON: {exc}",
                           param=f"messages[{index}].tool_calls[{n}].function.arguments") from exc
    if args is not None and not isinstance(args, dict):
        raise ApiError(400, "tool call arguments must be a JSON object",
                       param=f"messages[{index}].tool_calls[{n}].function.arguments")
    return args or {}


def _chat_messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "messages must be a non-empty list", param="messages")
    chat = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise ApiError(400, "each message needs a role of system, developer, user, assistant, or tool",
                           param=f"messages[{i}].role")
        msg = {"role": m["role"], "content": _content_text(m.get("content"), i)}
        if m["role"] == "assistant":
            reasoning = m.get("reasoning_content", m.get("reasoning"))
            if isinstance(reasoning, str):
                msg["reasoning_content"] = reasoning
            calls = []
            for n, call in enumerate(m.get("tool_calls") or []):
                fn = call.get("function") if isinstance(call, dict) else None
                if not isinstance(fn, dict) or not fn.get("name"):
                    raise ApiError(400, "tool calls need function.name", param=f"messages[{i}].tool_calls[{n}]")
                calls.append({"name": fn["name"], "arguments": _tool_arguments(fn.get("arguments"), i, n)})
            if calls:
                msg["tool_calls"] = calls
        chat.append(msg)
    return chat


def _tool_list(req: dict) -> list[dict] | None:
    tools = req.get("tools")
    if tools is None and req.get("functions"):  # legacy OpenAI "functions" field
        tools = [{"type": "function", "function": f} for f in req["functions"]]
    if not tools or req.get("tool_choice") == "none":
        return None
    if not isinstance(tools, list) or not all(
            isinstance(t, dict) and isinstance(t.get("function", t), dict) and t.get("function", t).get("name")
            for t in tools):
        raise ApiError(400, "tools must be a list of {type: function, function: {name, parameters}}", param="tools")
    return tools


def tool_call_objects(calls: list[dict]) -> list[dict]:
    return [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
            for c in calls]


class OpenAIService:
    """Request validation and generation; independent of the HTTP transport (also used by tests)."""

    def __init__(self, backend=None, *, model_name: str | None = None, api_key: str | None = None,
                 strict_sampling: bool = False, default_thinking: bool = True, log=print) -> None:
        self.backend = backend
        self.model_name = model_name or (backend.model_name if backend else "qwen")
        self.api_key = api_key
        self.strict_sampling = strict_sampling
        self.default_thinking = default_thinking
        self.status = "ok" if backend else "loading"
        self.lock = threading.Lock()  # one generation at a time on the single model instance
        self.ready = threading.Event()  # set once loading finished (or failed)
        if backend:
            self.ready.set()
        self.log = log
        self._warned_sampling = False

    def set_backend(self, backend) -> None:
        self.backend = backend
        if self.model_name in (None, "qwen"):
            self.model_name = backend.model_name
        self.status = "ok"
        self.ready.set()

    def set_failed(self) -> None:
        self.status = "error"
        self.ready.set()

    # ---- validation --------------------------------------------------------------------------

    def _common(self, req: dict) -> tuple[int | None, list[str], bool, bool]:
        if not isinstance(req, dict):
            raise ApiError(400, "request body must be a JSON object")
        if self.status == "loading":  # hold requests that arrive during startup instead of failing them
            self.log("request waiting for the model to finish loading")
            self.ready.wait(timeout=1800)
        if self.status != "ok":
            raise ApiError(503, f"model is {self.status}", kind="server_error", code="model_not_ready")
        if req.get("n", 1) != 1:
            raise ApiError(400, "only n=1 is supported", param="n")
        if req.get("logprobs") or req.get("top_logprobs"):
            raise ApiError(400, "logprobs are not supported", param="logprobs")
        sampled = [k for k in ("temperature", "top_p", "top_k", "min_p")
                   if req.get(k) is not None and req.get(k) != {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0}[k]]
        if sampled:
            if self.strict_sampling:
                raise ApiError(400, "sampling is not supported yet (greedy decoding only)", param=sampled[0])
            if not self._warned_sampling:
                self.log(f"note: sampling parameters {sampled} are ignored; decoding greedily")
                self._warned_sampling = True
        max_tokens = req.get("max_completion_tokens", req.get("max_tokens"))
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1):
            raise ApiError(400, "max_tokens must be a positive integer", param="max_tokens")
        stop = req.get("stop")
        stops = [stop] if isinstance(stop, str) else list(stop or [])
        if len(stops) > 4 or not all(isinstance(s, str) for s in stops):
            raise ApiError(400, "stop must be a string or a list of up to 4 strings", param="stop")
        stream = bool(req.get("stream", False))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage", False))
        return max_tokens, stops, stream, include_usage

    def _budget(self, prompt_tokens: list[int], max_tokens: int | None) -> int:
        """Tokens to generate: max_tokens clipped to the room left in the context (clients such as agentic
        tools send their own large output limits)."""
        room = self.backend.max_context - len(prompt_tokens)
        if room < 1:
            raise ApiError(400, f"prompt is {len(prompt_tokens)} tokens; the context holds {self.backend.max_context} "
                                "(start the server with a larger --max-context)",
                           param="messages", code="context_length_exceeded")
        return min(max_tokens, room) if max_tokens is not None else room

    def prepare_chat(self, req: dict):
        max_tokens, stops, stream, include_usage = self._common(req)
        fmt = (req.get("response_format") or {}).get("type", "text")
        if fmt != "text":
            raise ApiError(400, f"response_format {fmt!r} is not supported", param="response_format")
        chat = _chat_messages(req.get("messages"))
        tools = _tool_list(req)
        thinking = self.default_thinking
        effort = req.get("reasoning_effort")
        if effort is not None:
            thinking = effort not in ("none", "minimal")
            effort = effort if effort in ("low", "medium", "high", "xhigh") else None
        kwargs = req.get("chat_template_kwargs") or {}
        if "enable_thinking" in kwargs:
            thinking = bool(kwargs["enable_thinking"])
        effort = kwargs.get("reasoning_effort", effort)
        try:
            text = render_chat(chat, tools=tools, add_generation_prompt=True, enable_thinking=thinking,
                               reasoning_effort=effort, preserve_thinking=kwargs.get("preserve_thinking"))
        except UnsupportedModelError as exc:
            raise ApiError(400, str(exc), param="messages") from exc
        prompt = self.backend.tokenizer.encode(text)
        return prompt, self._budget(prompt, max_tokens), stops, stream, include_usage, thinking, tools

    def prepare_completion(self, req: dict):
        max_tokens, stops, stream, include_usage = self._common(req)
        prompt = req.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt:
            raise ApiError(400, "prompt must be a non-empty string", param="prompt")
        if req.get("echo"):
            raise ApiError(400, "echo is not supported", param="echo")
        ids = self.backend.tokenizer.encode(prompt)
        return ids, self._budget(ids, 16 if req.get("max_tokens") is None else req.get("max_tokens")), stops, stream, include_usage

    # ---- generation --------------------------------------------------------------------------

    def run(self, prompt: list[int], max_new: int, stops: list[str], *, reasoning: bool, on_delta,
            tools: list[dict] | None = None) -> dict:
        """Generate; on_delta(field, text) receives text deltas. Returns finish reason, usage, and with tools
        the parsed tool calls (OpenAI objects; finish reason "tool_calls")."""
        ts = TextStream(self.backend.tokenizer, self.backend.stop_token_ids, stops, reasoning=reasoning,
                        tools=bool(tools))
        produced = [0]
        times: dict[str, float] = {}

        def on_token(token: int) -> None:
            if not produced[0]:
                times["first"] = time.perf_counter()
                took = times["first"] - times["start"]
                self.log(f"prompt processed: {len(prompt)} tokens in {took:.1f}s ({len(prompt) / took:.0f} tok/s)")
            produced[0] += 1
            try:
                deltas = ts.feed(token)
            except _Cancelled as stop:
                for field, text in stop.args[0]:
                    on_delta(field, text)
                raise
            for field, text in deltas:
                on_delta(field, text)

        if hasattr(self.backend, "submit"):  # scheduler: runs alongside other requests
            times["start"] = time.perf_counter()
            job = self.backend.submit(prompt, max_new)
            self.log(f"request: {len(prompt)} prompt tokens, up to {max_new} new tokens "
                     f"({self.backend.scheduler().active()} active)")
            tokens = None
            try:
                for kind, value in job:
                    if kind == "token":
                        on_token(value)
                tokens = job.out
            except _Cancelled:
                job.cancel()
            except BaseException:
                job.cancel()
                raise
            if job.cached_tokens:
                self.log(f"prefix cache: reused {job.cached_tokens} of {len(prompt)} prompt tokens")
        else:
            if self.lock.locked():
                self.log("request queued behind the running generation")
            with self.lock:
                times["start"] = time.perf_counter()
                self.log(f"request: {len(prompt)} prompt tokens, up to {max_new} new tokens")
                try:
                    tokens, _ = self.backend.generate(prompt, max_new, on_token=on_token)
                except _Cancelled:
                    tokens = None
        if ts.finish_reason is None:
            for field, text in ts._flush():
                on_delta(field, text)
            ts.finish_reason = "length"
        for field, text in ts.finish():
            on_delta(field, text)
        calls = []
        if ts.call_text is not None:
            leftover, parsed = parse_tool_calls(ts.call_text, tools)
            if parsed:
                calls = tool_call_objects(parsed)
                ts.finish_reason = "tool_calls"
            else:  # not a well-formed call: hand the text back as content
                on_delta("content", ts.call_space + leftover)
        completion = produced[0] if tokens is None else len(tokens)
        if "first" in times and completion > 1:
            took = time.perf_counter() - times["first"]
            self.log(f"generated {completion} tokens, {(completion - 1) / max(took, 1e-9):.1f} tok/s, "
                     f"finish={ts.finish_reason}")
        return {
            "finish_reason": ts.finish_reason,
            "tool_calls": calls,
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": completion,
                      "total_tokens": len(prompt) + completion},
        }


class _Handler(BaseHTTPRequestHandler):
    service: OpenAIService  # set on the subclass created per server
    server_version = "vinf"

    def log_message(self, fmt, *args) -> None:  # route through the service logger
        self.service.log(f"{self.address_string()} {fmt % args}")

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        key = self.service.api_key
        if not key:
            return True
        if self.headers.get("Authorization", "") == f"Bearer {key}":
            return True
        self._json(401, ApiError(401, "invalid or missing API key", kind="authentication_error",
                                 code="invalid_api_key").body())
        return False

    def _route(self) -> str:
        """Request path with the optional /v1 prefix restored (clients differ in what base URL they expect)."""
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/models", "/chat/completions", "/completions"):
            path = "/v1" + path
        return path

    def do_GET(self) -> None:  # noqa: N802
        path = self._route()
        if path == "/health":
            status = self.service.status
            self._json(200 if status == "ok" else 503, {"status": status})
            return
        if not self._authorized():
            return
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": self.service.model_name, "object": "model", "created": _now(), "owned_by": "vinf"}]})
            return
        self._json(404, ApiError(404, f"unknown path {self.path}", code="not_found").body())

    def do_POST(self) -> None:  # noqa: N802
        path = self._route()
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._json(404, ApiError(404, f"unknown path {self.path}", code="not_found").body())
            return
        if not self._authorized():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, ApiError(400, "request body is not valid JSON").body())
            return
        try:
            if path == "/v1/chat/completions":
                self._chat(req)
            else:
                self._completion(req)
        except ApiError as err:
            self._json(err.status, err.body())
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (VinfError, Exception) as exc:  # noqa: BLE001 - report as an OpenAI server error
            self.service.log(traceback.format_exc())
            try:
                self._json(500, ApiError(500, str(exc), kind="server_error").body())
            except (BrokenPipeError, ConnectionResetError):
                pass

    # ---- server-sent events -----------------------------------------------------------------

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, payload) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        try:
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise _Cancelled([]) from exc  # client went away: stop generating

    def _chat(self, req: dict) -> None:
        svc = self.service
        prompt, max_new, stops, stream, include_usage, thinking, tools = svc.prepare_chat(req)
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", _now()

        def chunk(delta: dict, finish=None) -> dict:
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": svc.model_name,
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}

        if stream:
            self._sse_start()
            self._sse(chunk({"role": "assistant", "content": ""}))
            try:
                result = svc.run(prompt, max_new, stops, reasoning=thinking, tools=tools,
                                 on_delta=lambda field, text: self._sse(chunk({field: text})))
            except _Cancelled:
                return
            for n, call in enumerate(result["tool_calls"]):
                self._sse(chunk({"tool_calls": [{"index": n, **call}]}))
            self._sse(chunk({}, result["finish_reason"]))
            if include_usage:
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": svc.model_name, "choices": [], "usage": result["usage"]})
            self._sse("[DONE]")
            return
        parts = {"content": [], "reasoning_content": []}
        result = svc.run(prompt, max_new, stops, reasoning=thinking, tools=tools,
                         on_delta=lambda f, t: parts[f].append(t))
        message = {"role": "assistant", "content": "".join(parts["content"])}
        if thinking:
            message["reasoning_content"] = "".join(parts["reasoning_content"])
        if result["tool_calls"]:
            message["tool_calls"] = result["tool_calls"]
            message["content"] = message["content"] or None
        self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": svc.model_name,
                         "choices": [{"index": 0, "message": message, "logprobs": None,
                                      "finish_reason": result["finish_reason"]}],
                         "usage": result["usage"]})

    def _completion(self, req: dict) -> None:
        svc = self.service
        prompt, max_new, stops, stream, include_usage = svc.prepare_completion(req)
        cid, created = f"cmpl-{uuid.uuid4().hex}", _now()

        def chunk(text: str, finish=None) -> dict:
            return {"id": cid, "object": "text_completion", "created": created, "model": svc.model_name,
                    "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}

        if stream:
            self._sse_start()
            try:
                result = svc.run(prompt, max_new, stops, reasoning=False,
                                 on_delta=lambda field, text: self._sse(chunk(text)))
            except _Cancelled:
                return
            self._sse(chunk("", result["finish_reason"]))
            if include_usage:
                self._sse({"id": cid, "object": "text_completion", "created": created, "model": svc.model_name,
                           "choices": [], "usage": result["usage"]})
            self._sse("[DONE]")
            return
        parts: list[str] = []
        result = svc.run(prompt, max_new, stops, reasoning=False, on_delta=lambda f, t: parts.append(t))
        self._json(200, {"id": cid, "object": "text_completion", "created": created, "model": svc.model_name,
                         "choices": [{"index": 0, "text": "".join(parts), "logprobs": None,
                                      "finish_reason": result["finish_reason"]}],
                         "usage": result["usage"]})


def make_server(service: OpenAIService, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(args) -> int:
    """CLI entry (`--serve`): listen immediately, load the model, then answer requests until interrupted."""
    from vinf.qwen_backend import QwenBackend

    log = lambda msg: print(msg, flush=True)  # noqa: E731
    service = OpenAIService(model_name=args.served_model_name, api_key=args.api_key,
                            strict_sampling=args.strict_sampling, default_thinking=not args.no_think, log=log)
    server = make_server(service, args.host, args.port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log(f"listening on http://{args.host}:{args.port} (OpenAI-compatible API; requests wait until the model is loaded)")
    try:
        service.set_backend(QwenBackend.load(args, log=log))
    except Exception:
        service.set_failed()
        server.shutdown()
        raise
    log(f"ready: model {service.model_name!r} at http://{args.host}:{args.port}/v1")
    try:
        thread.join()
    except KeyboardInterrupt:
        log("shutting down")
        server.shutdown()
    return 0
