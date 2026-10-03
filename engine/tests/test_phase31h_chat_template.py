"""Phase 31h: the Python chat template port must match the GGUF's Jinja template byte for byte."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from vinf.chat_template import parse_tool_calls, render_chat
from vinf.errors import UnsupportedModelError

GGUF = Path(os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk. Ünïcode ok.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "flags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path"],
            },
        },
    },
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
]

CONVERSATIONS = {
    "plain": [{"role": "user", "content": "  Hello  "}],
    "system": [
        {"role": "system", "content": "You are helpful."},
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ],
    "parts": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
    "multi_turn": [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two", "reasoning_content": " thought "},
        {"role": "user", "content": "three"},
    ],
    "tool_history": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": "Reading.",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.txt", "offset": 3, "flags": ["x"]}}},
                {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls\n-la"}}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file body"},
        {"role": "tool", "tool_call_id": "c2", "content": "x y"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "pwd"}}}]},
        {"role": "tool", "tool_call_id": "c3", "content": "/root"},
    ],
}


def _gguf_template() -> str | None:
    if not GGUF.is_file():
        return None
    from vinf.gguf.parser import load_gguf

    return load_gguf(GGUF).metadata_value("tokenizer.chat_template")


class ChatTemplateParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import jinja2
        except ImportError:
            raise unittest.SkipTest("jinja2 not installed")
        source = _gguf_template()
        if source is None:
            raise unittest.SkipTest(f"model not found: {GGUF}")

        def raise_exception(message):
            raise jinja2.TemplateError(message)

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
        env.globals["raise_exception"] = raise_exception
        cls.template = env.from_string(source)

    def check(self, messages, **kwargs):
        expected = self.template.render(messages=messages, add_generation_prompt=True, **kwargs)
        self.assertEqual(render_chat(messages, **kwargs), expected)

    def test_conversations(self):
        for name, messages in CONVERSATIONS.items():
            for tools in (None, TOOLS):
                for thinking in (None, True, False):
                    for effort in (None, "low", "medium", "high"):
                        kwargs = {"tools": tools, "enable_thinking": thinking, "reasoning_effort": effort}
                        kwargs = {k: v for k, v in kwargs.items() if v is not None}
                        with self.subTest(name=name, **{k: str(v)[:20] for k, v in kwargs.items()}):
                            self.check(messages, **kwargs)

    def test_preserve_thinking_false(self):
        self.check(CONVERSATIONS["multi_turn"], preserve_thinking=False)


class ToolCallParseTest(unittest.TestCase):
    def test_parse_typed_arguments(self):
        text = (
            "Let me look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/a b.py\n</parameter>\n"
            "<parameter=offset>\n12\n</parameter>\n<parameter=flags>\n[\"r\"]\n</parameter>\n</function>\n</tool_call>"
            "\n<tool_call>\n<function=bash>\n<parameter=command>\n123\n</parameter>\n</function>\n</tool_call>"
        )
        content, calls = parse_tool_calls(text, TOOLS)
        self.assertEqual(content, "Let me look.")
        self.assertEqual(calls[0], {"name": "read_file", "arguments": {"path": "src/a b.py", "offset": 12, "flags": ["r"]}})
        self.assertEqual(calls[1], {"name": "bash", "arguments": {"command": "123"}})

    def test_round_trip_through_template(self):
        args = {"path": "x\ny", "offset": 0, "flags": []}
        rendered = render_chat(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": args}}]}],
            add_generation_prompt=False,
        )
        body = rendered.split("</think>\n\n", 1)[1]
        self.assertEqual(parse_tool_calls(body, TOOLS)[1], [{"name": "read_file", "arguments": args}])

    def test_no_call(self):
        self.assertEqual(parse_tool_calls("just text <tool_call> broken", TOOLS), ("just text <tool_call> broken", []))

    def test_string_arguments_rejected(self):
        with self.assertRaises(UnsupportedModelError):
            render_chat([{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{\"a\":1}"}}]}])


if __name__ == "__main__":
    unittest.main()
