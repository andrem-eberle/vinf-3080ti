"""Run a qwen35 GGUF model on the GPU executor.

    python3.12 -m vinf.qwen_cli --prompt "The capital of France is" --max-new-tokens 16
    python3.12 -m vinf.qwen_cli --chat "What is 2+2?" --max-new-tokens 64
    python3.12 -m vinf.qwen_cli --report-only
"""

from __future__ import annotations

import os

import argparse
import sys

from vinf.errors import ConfigurationError, VinfError
from vinf.gguf.mapper import metadata_from_gguf
from vinf.gguf.parser import load_gguf
from vinf.gguf.qwen import require_qwen_inference_ready
from vinf.gguf.tokenizer import ChatMessage

DEFAULT_MODEL = os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3.12 -m vinf.qwen_cli", description="qwen35 GPU inference")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--prompt", help="raw completion prompt")
    text.add_argument("--chat", help="user message, formatted with the Qwen chat template")
    parser.add_argument("--no-think", action="store_true", help="chat: emit an empty <think> block")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--first-token-only", action="store_true", help="bring-up: prefill and emit one token")
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--head-order", choices=("grouped", "tiled"), default="tiled",
                        help="SSM value-head order; llama.cpp GGUF conversions are tiled")
    parser.add_argument("--safety-mib", type=int, default=256, help="VRAM left unallocated (raise if loading runs out of memory)")
    parser.add_argument("--prefill-batch", type=int, default=256,
                        help="prompt tokens per pass; each streamed weight crosses PCIe once per pass")
    parser.add_argument("--kv-dtype", choices=("f16", "f32"), default="f16",
                        help="attention KV cache precision (f16 halves its VRAM; the megakernel uses f32)")
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="concurrent sequences, up to 64 (default 4 with --serve, else 1); they share the KV pool")
    parser.add_argument("--ssm-dtype", choices=("f32", "f16"), default="f32",
                        help="SSM recurrent state precision on the GPU (f16 halves the ~150 MB per sequence)")
    parser.add_argument("--verify-rows", type=int, default=64,
                        help="max rows of a multi-sequence verification pass (drafts per sequence shrink to fit)")
    parser.add_argument("--kv-pool-tokens", type=int, default=None,
                        help="KV cache tokens shared by all sequences (default: --max-context)")
    parser.add_argument("--prefix-cache-mib", type=int, default=8192,
                        help="host memory for prompt prefix checkpoints (0 disables); reuses earlier turns of a conversation")
    parser.add_argument("--report-only", action="store_true", help="print memory/residency plan and exit")
    parser.add_argument("--profile", action="store_true", help="per-op GPU timing (synchronizes after each op)")
    parser.add_argument("--executor", choices=("per-op", "megakernel"), default="per-op",
                        help="per-op kernels (debug/fallback) or one fused megakernel launch per token")
    parser.add_argument("--streaming", choices=("dma", "sm"), default="dma",
                        help="megakernel weight streaming: copy engine (dma) or loader SMs (sm)")
    parser.add_argument("--loader-sms", type=int, default=8, help="megakernel --streaming sm: loader SM count")
    parser.add_argument("--stream-slots", type=int, default=3, help="staging slots for streamed weights (prefetch depth)")
    parser.add_argument("--drafter", choices=("mtp", "dflash"), default="mtp",
                        help="speculative drafter: the model's MTP block, or a DFlash/DFlash 2 checkpoint (--dflash)")
    parser.add_argument("--dflash", metavar="DIR", help="DFlash draft checkpoint directory (config.json + safetensors)")
    parser.add_argument("--dflash-allow-unimplemented", action="store_true",
                        help="load DFlash 2 checkpoints whose conv/selector tensors are not implemented yet")
    parser.add_argument("--dflash-bf16", action="store_true", help="keep DFlash weights unquantized (F32) instead of Q8_0")
    parser.add_argument("--placement", choices=("hybrid", "stream"), default="stream",
                        help="stream: copy non-resident weights to the GPU per pass (fastest here: PCIe ~= RAM bandwidth); "
                             "hybrid: compute layers that do not fit in VRAM on the CPU (better with fast RAM)")
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU layer threads (default: physical cores)")
    serve = parser.add_argument_group("HTTP server (OpenAI-compatible API)")
    serve.add_argument("--serve", action="store_true", help="serve the model over HTTP with the OpenAI API")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="listen port (default 8000)")
    serve.add_argument("--api-key", help="require 'Authorization: Bearer <key>' on API requests")
    serve.add_argument("--served-model-name", help="model id reported by /v1/models (default: GGUF general.name)")
    serve.add_argument("--strict-sampling", action="store_true",
                       help="reject requests asking for temperature/top_p/top_k sampling (default: decode greedily)")
    parser.add_argument("--speculative", type=int, default=3, metavar="K",
                        help="greedy speculative decoding with the model's MTP block drafting K tokens per step "
                             "(output identical to plain greedy; 0 disables)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except VinfError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    if args.serve:
        from vinf.server import serve

        return serve(args)
    if args.report_only:
        from vinf.qwen_gpu import QwenGpuExecutor

        gguf = load_gguf(args.model)
        require_qwen_inference_ready(gguf)
        executor = QwenGpuExecutor.plan_only(
            gguf, metadata_from_gguf(gguf), max_context=args.max_context, safety_bytes=args.safety_mib * 1024**2
        )
        print(executor.report())
        return 0
    from vinf.qwen_backend import QwenBackend

    if args.profile and args.executor == "megakernel":
        raise ConfigurationError("--profile applies to the per-op executor only")
    backend = QwenBackend.load(args, log=lambda msg: print(msg, flush=True))
    tokenizer = backend.tokenizer
    if args.chat is not None:
        text = tokenizer.apply_chat_template(
            [ChatMessage("user", args.chat)], enable_thinking=not args.no_think
        )
    else:
        text = args.prompt if args.prompt is not None else "The capital of France is"
    prompt_tokens = tokenizer.encode(text)
    print(f"prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
    max_new = 1 if args.first_token_only else args.max_new_tokens
    profiler = backend.executor.enable_profiling() if args.profile and backend.decoder is None else None
    print("output: " + text, end="", flush=True)
    emit = lambda token: print(tokenizer.decode([token]), end="", flush=True)  # noqa: E731
    tokens, stats = backend.generate(prompt_tokens, max_new, on_token=emit)
    print()
    print(f"generated ids: {tokens}")
    spec = stats.speculative
    if spec is not None:
        hist = ", ".join(f"{a}:{n}" for a, n in sorted(spec.accepted_histogram.items()))
        print(
            f"prefill {spec.prompt_tokens} tokens in {spec.prefill_seconds:.2f}s; speculative k={backend.decoder.k}: "
            f"{spec.steps} steps, {spec.generated_tokens - 1} tokens in {spec.decode_seconds:.2f}s "
            f"({spec.decode_tokens_per_second:.2f} tok/s); acceptance {spec.acceptance_rate:.0%}, "
            f"{spec.tokens_per_step:.2f} tokens/step; draft {spec.draft_seconds:.2f}s verify {spec.verify_seconds:.2f}s; "
            f"accepted histogram {{{hist}}}"
        )
        return 0
    plain = stats.plain
    print(
        f"prefill {plain.prompt_tokens} tokens in {plain.prefill_seconds:.2f}s; "
        f"decode {len(plain.step_seconds)} steps in {plain.decode_seconds:.2f}s "
        f"({plain.decode_tokens_per_second:.2f} tok/s); streamed {plain.streamed_bytes / 1024**3:.2f} GiB"
    )
    if profiler is not None:
        print(profiler.report(plain.prompt_tokens + len(plain.step_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
