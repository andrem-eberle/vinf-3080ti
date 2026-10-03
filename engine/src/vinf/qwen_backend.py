"""Loaded qwen35 model + tokenizer + decoding strategy, shared by the CLI and the HTTP server."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from vinf.errors import ConfigurationError
from vinf.gguf.mapper import metadata_from_gguf
from vinf.gguf.parser import load_gguf
from vinf.gguf.qwen import require_qwen_inference_ready
from vinf.gguf.tokenizer import QwenTokenizer, load_qwen_tokenizer


@dataclass(slots=True)
class GenerationStats:
    prompt_tokens: int
    completion_tokens: int
    prefill_seconds: float
    decode_seconds: float
    speculative: object | None = None  # SpeculativeStats when a speculative decoder ran
    plain: object | None = None  # QwenGenerationStats otherwise
    cached_tokens: int = 0  # prompt tokens reused from the prefix cache


class QwenBackend:
    """Wraps an executor (per-op or megakernel) and an optional speculative decoder."""

    def __init__(self, executor, tokenizer: QwenTokenizer, *, decoder=None, model_name: str = "qwen", base=None,
                 prefix_cache_bytes: int = 0) -> None:
        self.executor = executor
        self.base = base if base is not None else executor  # QwenGpuExecutor (for reports / limits)
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.model_name = model_name
        ids = {tokenizer.special_tokens.eos_token_id}
        if tokenizer.special_tokens.im_end_id is not None:
            ids.add(tokenizer.special_tokens.im_end_id)
        self.stop_token_ids = frozenset(ids)
        self.prefix_cache = None
        if prefix_cache_bytes > 0 and self._prefix_cache_supported():
            from vinf.prefix_cache import PrefixCache

            self.prefix_cache = PrefixCache(self.base, max_bytes=prefix_cache_bytes)

    def _prefix_cache_supported(self) -> bool:
        """Per-op executor, plain greedy or MTP speculation (DFlash keeps drafter context of its own)."""
        if self.executor is not self.base:
            return False
        if self.decoder is None:
            return True
        from vinf.qwen_speculative import MtpDrafter

        return isinstance(self.decoder.drafter, MtpDrafter)

    @property
    def max_context(self) -> int:
        return self.base.max_context

    def generate(self, prompt_tokens: list[int], max_new_tokens: int, *, on_token=None) -> tuple[list[int], GenerationStats]:
        """Greedy generation; on_token(token) is called for every emitted token (may raise to cancel)."""
        if self.decoder is not None:
            tokens, spec = self.decoder.generate(
                prompt_tokens, max_new_tokens, stop_token_ids=self.stop_token_ids, on_token=on_token,
                prefix_cache=self.prefix_cache,
            )
            return tokens, GenerationStats(len(prompt_tokens), len(tokens), spec.prefill_seconds, spec.decode_seconds,
                                           speculative=spec, cached_tokens=spec.cached_tokens)
        kwargs = {"prefix_cache": self.prefix_cache} if self.prefix_cache is not None else {}
        tokens, stats = self.executor.generate_greedy(
            prompt_tokens, max_new_tokens, stop_token_ids=self.stop_token_ids, on_token=on_token, **kwargs
        )
        return tokens, GenerationStats(len(prompt_tokens), len(tokens), stats.prefill_seconds, stats.decode_seconds,
                                       plain=stats, cached_tokens=getattr(stats, "cached_tokens", 0))

    @classmethod
    def load(cls, args: argparse.Namespace, *, log=print) -> "QwenBackend":
        """Load a GGUF model with the CLI options (model, placement, executor, speculative, DFlash)."""
        from vinf.qwen_gpu import QwenGpuExecutor

        gguf = load_gguf(args.model)
        report = require_qwen_inference_ready(gguf)
        metadata = metadata_from_gguf(gguf)
        tokenizer = load_qwen_tokenizer(gguf)
        log(f"model: {args.model}")
        log(f"arch={report.architecture} layers={metadata.num_hidden_layers} hidden={metadata.hidden_size} vocab={metadata.vocab_size}")
        last = [0.0]

        def progress(done: int, total: int, entry) -> None:
            now = time.perf_counter()
            if done == total or now - last[0] > 5.0:
                last[0] = now
                log(f"  loading weights {done}/{total}")

        t0 = time.perf_counter()
        speculative = args.speculative
        if args.executor == "megakernel" and speculative > 0:
            log("note: --speculative runs on the per-op executor; megakernel decodes without speculation")
            speculative = 0
        dflash_ck = None
        if speculative > 0 and args.drafter == "dflash":
            from vinf.dflash import DFlashCheckpoint

            if not args.dflash:
                raise ConfigurationError("--drafter dflash needs --dflash DIR")
            dflash_ck = DFlashCheckpoint(args.dflash, allow_unimplemented=args.dflash_allow_unimplemented)
            c = dflash_ck.config
            log(f"DFlash {c.version} drafter: {c.num_layers} layers, block {c.block_size}, target layers {list(c.target_layer_ids)}"
                + (f"; ignoring unimplemented tensors: {len(dflash_ck.unimplemented)}" if dflash_ck.unimplemented else ""))
        executor = QwenGpuExecutor(
            gguf,
            metadata,
            max_context=args.max_context,
            head_order=args.head_order,
            safety_bytes=args.safety_mib * 1024**2,
            stream_slots=args.stream_slots,
            mtp=speculative > 0 and dflash_ck is None,
            capture_layers=dflash_ck.config.target_layer_ids if dflash_ck else (),
            reserve_extra_bytes=dflash_ck.gpu_bytes(args.max_context, quantize=not args.dflash_bf16) if dflash_ck else 0,
            placement="stream" if args.executor == "megakernel" else args.placement,
            cpu_threads=args.cpu_threads,
            snapshot_tokens=speculative + 1 if speculative > 0 else 0,
            max_batch=max(8, speculative + 1),
            prefill_batch=args.prefill_batch,
            progress=progress,
        )
        log(executor.report())
        log(f"load time {time.perf_counter() - t0:.1f}s; device weights {executor.rt.device_bytes() / 1024**3:.2f} GiB, "
            f"pinned host {executor.rt.pinned_bytes() / 1024**3:.2f} GiB")
        name = getattr(args, "served_model_name", None) or str(gguf.metadata_value("general.name") or Path(args.model).stem)
        if args.executor == "megakernel":
            from vinf.qwen_megakernel import QwenMegakernelExecutor

            t1 = time.perf_counter()
            mk = QwenMegakernelExecutor(executor, streaming=args.streaming, loader_sms=args.loader_sms)
            prog = mk.program
            log(f"megakernel: {prog.instruction_count} instructions on {prog.compute_sms} compute + "
                f"{prog.loader_sms} loader SMs, queue {prog.queue_len}, {prog.num_counters} counters, "
                f"smem {prog.smem_bytes} B (built in {time.perf_counter() - t1:.1f}s)")
            log(executor.vram_report())
            return cls(mk, tokenizer, model_name=name, base=executor)
        log(executor.vram_report())
        decoder = None
        if speculative > 0:
            from vinf.qwen_speculative import QwenSpeculativeDecoder

            drafter = None
            if dflash_ck is not None:
                from vinf.dflash import DFlashDrafter

                drafter = DFlashDrafter(executor, dflash_ck, quantize=not args.dflash_bf16)
            decoder = QwenSpeculativeDecoder(executor, speculative, drafter=drafter)
        backend = cls(executor, tokenizer, decoder=decoder, model_name=name,
                      prefix_cache_bytes=args.prefix_cache_mib * 1024**2)
        if backend.prefix_cache is not None:
            log(f"prefix cache: up to {args.prefix_cache_mib} MiB host memory "
                f"(KV {executor.kv_bytes_per_token() / 1024:.0f} KiB/token, "
                f"recurrent state {executor.recurrent_bytes() / 1024**2:.0f} MiB per checkpoint)")
        return backend
