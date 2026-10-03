"""Qwen3.5 / Qwen3.8 (qwen35) chat formatting and tool-call parsing.

`render_chat` is a line-for-line port of the chat template shipped in the Qwen3.8-27B GGUF
(`tokenizer.chat_template`, Unsloth revision): leading system/developer messages are merged,
reasoning-effort instructions are injected when thinking is on, tools are listed in the system
prompt, assistant history keeps its <think> blocks, and tool results become <tool_response> turns.
`tests/test_phase31h_chat_template.py` checks it against the GGUF's Jinja template.

Tool calls use the template's XML form:
    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>
"""

from __future__ import annotations

import json
import re
from typing import Any

from vinf.errors import UnsupportedModelError

TOOL_CALL_START = "<tool_call>"

_TOOL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)
_REASONING = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    "medium": "",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
    "without unnecessary elaboration.",
}


def tojson(value: Any) -> str:
    """The template's `tojson` filter as Hugging Face chat templating defines it."""
    return json.dumps(value, ensure_ascii=False)


def _render_content(content: Any, is_system: bool = False) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        out = []
        for item in content:
            if not isinstance(item, dict):
                raise UnsupportedModelError("Unexpected item type in content.")
            if "image" in item or "image_url" in item or item.get("type") == "image":
                raise UnsupportedModelError("images are not supported by this engine")
            if "video" in item or item.get("type") == "video":
                raise UnsupportedModelError("videos are not supported by this engine")
            if "text" in item:
                out.append(str(item["text"]))
            else:
                raise UnsupportedModelError("Unexpected item type in content.")
        return "".join(out)
    raise UnsupportedModelError("Unexpected content type.")


def render_chat(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    preserve_thinking: bool | None = None,
) -> str:
    """Render messages exactly like the Qwen3.8 GGUF chat template."""
    if not messages:
        raise UnsupportedModelError("No messages provided.")
    out: list[str] = []
    num_sys, sys_parts = 0, []
    for index, message in enumerate(messages):
        if num_sys == index and message.get("role") in ("system", "developer"):
            text = _render_content(message.get("content"), True).strip()
            if text:
                sys_parts.append(text)
            num_sys += 1
    merged_system = "\n".join(sys_parts)
    instructions = ""
    if enable_thinking is None or enable_thinking is True:
        effort = reasoning_effort if reasoning_effort is not None else "xhigh"
        if effort == "high":
            effort = "xhigh"
        if effort not in _REASONING:
            raise UnsupportedModelError(
                f"Unexpected reasoning effort {reasoning_effort}. Supported types are xhigh (default), medium, and low."
            )
        instructions = _REASONING[effort]
    if tools:
        out.append("<|im_start|>system\n")
        if instructions:
            out.append(instructions + "\n\n")
        out.append("# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for tool in tools:
            out.append("\n" + tojson(tool))
        out.append("\n</tools>")
        out.append(_TOOL_INSTRUCTIONS)
        if merged_system:
            out.append("\n\n" + merged_system)
        out.append("<|im_end|>\n")
    elif merged_system:
        out.append("<|im_start|>system\n" + (instructions + "\n\n" if instructions else "") + merged_system + "<|im_end|>\n")
    elif instructions:
        out.append("<|im_start|>system\n" + instructions + "<|im_end|>\n")

    last_query_index = len(messages) - 1
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user":
            content = _render_content(message.get("content")).strip()
            if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
                last_query_index = index
                break

    for index, message in enumerate(messages):
        if index < num_sys:
            continue
        role = message.get("role")
        content = _render_content(message.get("content")).strip()
        if role in ("system", "developer"):
            raise UnsupportedModelError("System message must be at the beginning.")
        if role == "user":
            out.append("<|im_start|>user\n" + content + "<|im_end|>\n")
        elif role == "assistant":
            reasoning = message.get("reasoning_content")
            reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
            if preserve_thinking is None or preserve_thinking is True or index > last_query_index:
                out.append("<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content)
            else:
                out.append("<|im_start|>assistant\n" + content)
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for n, call in enumerate(calls):
                    if isinstance(call, dict) and "function" in call:
                        call = call["function"]
                    name = call.get("name") if isinstance(call, dict) else None
                    if name is None:
                        raise UnsupportedModelError("Tool call is missing a function name.")
                    if n == 0:
                        out.append(("\n\n" if content.strip() else "") + "<tool_call>\n<function=" + str(name) + ">\n")
                    else:
                        out.append("\n<tool_call>\n<function=" + str(name) + ">\n")
                    args = call.get("arguments")
                    if isinstance(args, dict):
                        for key, value in args.items():
                            out.append("<parameter=" + str(key) + ">\n")
                            out.append(value if isinstance(value, str) else tojson(value))
                            out.append("\n</parameter>\n")
                    elif isinstance(args, str):
                        if args.strip():
                            raise UnsupportedModelError(
                                f'Tool call arguments for function "{name}" were passed as a JSON string. '
                                "Parse them into an object before calling apply_chat_template."
                            )
                    elif args is not None:
                        raise UnsupportedModelError(
                            f'Tool call arguments for function "{name}" must be an object/mapping or a JSON string.'
                        )
                    out.append("</function>\n</tool_call>")
            out.append("<|im_end|>\n")
        elif role == "tool":
            if index > 0 and messages[index - 1].get("role") != "tool":
                out.append("<|im_start|>user")
            out.append("\n<tool_response>\n" + content + "\n</tool_response>")
            if index == len(messages) - 1 or messages[index + 1].get("role") != "tool":
                out.append("<|im_end|>\n")
        else:
            raise UnsupportedModelError("Unexpected message role.")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
        out.append("<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n")
    return "".join(out)


_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _schema_types(tools: list[dict] | None, name: str) -> dict[str, Any]:
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if fn.get("name") == name:
            return (fn.get("parameters") or {}).get("properties") or {}
    return {}


def _convert(value: str, schema: dict) -> Any:
    kind = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    if kind == "string":
        return value
    if kind in ("integer", "number", "boolean", "object", "array", "null"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            if kind == "boolean" and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            return value
    try:  # no schema: keep JSON literals typed, anything else as text
        parsed = json.loads(value)
        return parsed if not isinstance(parsed, str) else value
    except json.JSONDecodeError:
        return value


def parse_tool_calls(text: str, tools: list[dict] | None) -> tuple[str, list[dict]]:
    """Split model output into (content, [{"name", "arguments": dict}]) following the template's format.
    Text without complete <tool_call> blocks is returned unchanged as content."""
    start = text.find(TOOL_CALL_START)
    if start < 0:
        return text, []
    calls = []
    for match in _CALL_RE.finditer(text, start):
        name = match.group(1).strip()
        props = _schema_types(tools, name)
        args = {key.strip(): _convert(value, props.get(key.strip(), {})) for key, value in _PARAM_RE.findall(match.group(2))}
        calls.append({"name": name, "arguments": args})
    if not calls:
        return text, []
    return text[:start].rstrip(), calls
