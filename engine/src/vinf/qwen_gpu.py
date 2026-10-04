"""GPU decode executor for qwen35 hybrid models.

CPU responsibilities: layer schedule, op ordering, token embedding row lookup,
stop conditions. GPU responsibilities: every projection, norm, attention, SSM step,
and the LM head argmax. Weights are GPU-resident when they fit the live VRAM budget
and are otherwise streamed from pinned host memory per use.
"""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field

from vinf.errors import ConfigurationError, InsufficientMemoryError, UnsupportedModelError
from vinf.gguf.parser import GGUFFile
from vinf.gguf.qwen_tensors import (
    MTP_TENSORS,
    load_qwen_token_embeddings,
    qwen_mtp_layer_index,
    qwen_layer_schedule,
    qwen_tensor_coverage,
)
from vinf.gguf.residency import (
    QwenDecodeResidencyPlan,
    TensorResidency,
    plan_qwen_decode_residency,
)
from vinf.models.metadata import ModelMetadata
from vinf.qwen_ops import (
    QwenLayerKind,
    qwen_linear_attention_config_from_metadata,
    qwen_rope_config_from_gguf,
)

FULL_ATTENTION_WEIGHTS = (
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
)
LINEAR_ATTENTION_WEIGHTS = (
    "attn_norm.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_beta.weight",
    "ssm_alpha.weight",
    "ssm_a",
    "ssm_dt.bias",
    "ssm_conv1d.weight",
    "ssm_norm.weight",
    "ssm_out.weight",
)
MLP_WEIGHTS = ("post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
# Matvec weights in the exact per-token order the executor calls qmv (drives stream prefetch).
FULL_ATTENTION_QMV = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")
LINEAR_ATTENTION_QMV = ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_out.weight")
MLP_QMV = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
# Value-head -> key-head order for SSM layers; GGUF files from llama.cpp are tiled.
HEAD_ORDERS = {"grouped": 0, "tiled": 1}
# Must match kMaxTokens in csrc/megakernel/qwen_runtime.cu.
MAX_BATCH = 8
# Concurrent sequences per executor (kMaxArgmaxRows in the runtime bounds the rows of one LM-head pass).
MAX_SEQS = 64


class KvPoolExhausted(InsufficientMemoryError):
    """No free KV pages (or sequence slots) for the requested context."""


@dataclass(eq=False)
class Sequence:
    """One decoding sequence: a state slot (SSM/conv state, drafter rows) and its KV pages.

    tokens are the ids at positions [0, position) whose KV/state are live."""

    slot: int
    pages: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    busy: bool = False  # owned by a running request (idle sequences may be reused or evicted)
    last_used: int = 0

    @property
    def position(self) -> int:
        return len(self.tokens)


def qwen_qmv_order(schedule: tuple[QwenLayerKind, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for layer_idx, kind in enumerate(schedule):
        attn = FULL_ATTENTION_QMV if kind is QwenLayerKind.FULL_ATTENTION else LINEAR_ATTENTION_QMV
        names.extend(f"blk.{layer_idx}.{suffix}" for suffix in attn + MLP_QMV)
    names.append("output.weight")
    return tuple(names)


def qwen_decode_weight_names(schedule: tuple[QwenLayerKind, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for layer_idx, kind in enumerate(schedule):
        suffixes = FULL_ATTENTION_WEIGHTS if kind is QwenLayerKind.FULL_ATTENTION else LINEAR_ATTENTION_WEIGHTS
        names.extend(f"blk.{layer_idx}.{suffix}" for suffix in suffixes + MLP_WEIGHTS)
    return tuple(names)


@dataclass(frozen=True, slots=True)
class QwenGpuShapes:
    hidden: int
    intermediate: int
    vocab: int
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    freq_base: float
    key_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int
    eps: float

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_heads * self.key_head_dim + self.value_heads * self.value_head_dim

    @property
    def value_dim(self) -> int:
        return self.value_heads * self.value_head_dim


@dataclass(slots=True)
class QwenGenerationStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    streamed_bytes: int = 0
    cached_tokens: int = 0  # prompt tokens restored from the prefix cache instead of computed
    step_seconds: list[float] = field(default_factory=list)

    @property
    def decode_tokens_per_second(self) -> float:
        return self.generated_tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0


class ProfilingRuntime:
    """Runtime proxy that synchronizes after each device op and accumulates wall time per op.

    Synchronizing serializes the GPU, so totals approximate per-op cost rather than overlap.
    qmv time is split by weight residency (resident vs streamed).
    """

    TIMED = frozenset({
        "qmv", "rmsnorm", "gated_rmsnorm", "add", "silu_mul", "sigmoid_mul", "split_gated_q",
        "rope", "kv_append", "attention", "conv_update", "gated_delta", "argmax", "write",
        "rope_seg", "kv_append_seg", "attention_seg", "conv_update_seg", "gated_delta_seg",
    })

    def __init__(self, runtime, streamed_names: frozenset[str]) -> None:
        self._inner = runtime
        self._streamed = streamed_names
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if name not in self.TIMED:
            return attr

        def timed(*args):
            self._inner.synchronize()
            t0 = time.perf_counter()
            result = attr(*args)
            self._inner.synchronize()
            key = name
            if name == "qmv":
                key = "qmv streamed" if args[0] in self._streamed else "qmv resident"
            self.seconds[key] = self.seconds.get(key, 0.0) + time.perf_counter() - t0
            self.calls[key] = self.calls.get(key, 0) + 1
            return result

        return timed

    def report(self, tokens: int) -> str:
        total = sum(self.seconds.values())
        lines = [f"profile over {tokens} forwarded tokens ({total / max(1, tokens) * 1000:.1f} ms/token synchronized):"]
        for key, sec in sorted(self.seconds.items(), key=lambda item: -item[1]):
            lines.append(
                f"  {key:16s} {sec / max(1, tokens) * 1000:8.2f} ms/token {100 * sec / total:5.1f}%  "
                f"({self.calls[key] // max(1, tokens)} calls/token)"
            )
        return "\n".join(lines)


class QwenGpuExecutor:
    def __init__(
        self,
        gguf: GGUFFile,
        metadata: ModelMetadata,
        *,
        max_context: int = 2048,
        head_order: str = "tiled",
        free_vram_bytes: int | None = None,
        safety_bytes: int = 256 * 1024**2,
        stream_slots: int = 3,
        max_batch: int = 8,
        prefill_batch: int = 256,
        kv_dtype: str = "f16",
        max_seqs: int = 1,
        kv_pool_tokens: int | None = None,
        page_size: int = 256,
        ssm_dtype: str = "f32",
        verify_rows: int = 64,
        batch_kernels: bool | None = None,
        snapshot_tokens: int = 0,
        mtp: bool = False,
        placement: str = "hybrid",
        cpu_threads: int = 0,
        capture_layers: tuple[int, ...] = (),
        reserve_extra_bytes: int = 0,
        runtime=None,
        progress=None,
    ) -> None:
        if placement not in ("hybrid", "stream"):
            raise ConfigurationError("placement must be 'hybrid' (CPU computes layers that do not fit) or 'stream'")
        if not 1 <= max_batch <= MAX_BATCH:
            raise ConfigurationError(f"max_batch must be in [1, {MAX_BATCH}]")
        if not 1 <= max_seqs <= MAX_SEQS:
            raise ConfigurationError(f"max_seqs must be in [1, {MAX_SEQS}]")
        if ssm_dtype not in ("f16", "f32"):
            raise ConfigurationError("ssm_dtype must be 'f16' or 'f32'")
        if kv_dtype not in ("f16", "f32"):
            raise ConfigurationError("kv_dtype must be 'f16' or 'f32'")
        if prefill_batch < 1:
            raise ConfigurationError("prefill_batch must be positive")
        if not 0 <= snapshot_tokens <= max_batch:
            raise ConfigurationError("snapshot_tokens must be in [0, max_batch]")
        if head_order not in HEAD_ORDERS:
            raise ConfigurationError(f"head_order must be one of {sorted(HEAD_ORDERS)}")
        if max_context <= 0 or max_context > metadata.max_position_embeddings:
            raise ConfigurationError(
                f"max_context must be in [1, {metadata.max_position_embeddings}], got {max_context}"
            )
        coverage = qwen_tensor_coverage(gguf, metadata)
        if not coverage.complete:
            raise UnsupportedModelError("missing Qwen tensors: " + ", ".join(coverage.missing_tensors))
        if runtime is None:
            from vinf.cuda.qwen_runtime import CudaWeightRuntime

            runtime = CudaWeightRuntime()
        self.gguf = gguf
        self.metadata = metadata
        self.rt = runtime
        self.max_context = max_context
        self.head_order = HEAD_ORDERS[head_order]
        self.stream_slots = stream_slots
        self.max_batch = max_batch
        # GPU attention KV caches in fp16 halve their VRAM (attention math stays fp32); CPU layers keep fp32.
        self.kv_dtype = kv_dtype
        # Sequences: max_seqs state slots; attention KV lives in a shared pool of pages ([page][kv_head]
        # [page_size][head_dim]) handed out per sequence, so concurrent sequences share the pool. CPU layers
        # (hybrid) and the megakernel use one page spanning the whole context (= the contiguous layout).
        self.max_seqs = max_seqs
        self.page_size = max(1, min(page_size, max_context))
        self.pages_per_seq = -(-max_context // self.page_size)
        pool = max_context if kv_pool_tokens is None else max(kv_pool_tokens, self.page_size)
        self.kv_pages = -(-pool // self.page_size)
        self.reclaim = None  # optional callback(pages_needed) -> bool that frees idle sequences
        # Rows of a verification pass: every sequence's next input plus its drafts (capped; with many
        # sequences the scheduler drafts fewer tokens per sequence).
        self.verify_rows = max(max_batch, max_seqs, min(verify_rows, max_seqs * snapshot_tokens))
        # SSM recurrent state precision on the GPU (fp16 halves the per-sequence state; math stays fp32).
        self.ssm_dtype = ssm_dtype
        # Several sequences: every pass uses the prompt kernels (tensor-core GEMM, tiled attention), whose
        # per-row results do not depend on how many rows share the pass, so a sequence's output does not
        # depend on its neighbours and large batches stay cheap.
        self.batch_kernels = max_seqs > 1 if batch_kernels is None else batch_kernels
        # Rows per prompt-processing pass: each streamed weight crosses PCIe once per pass, so wide passes
        # make prefill compute-bound instead of PCIe-bound. DFlash feature capture keeps max_batch rows.
        self.prefill_batch = max(max_batch, prefill_batch) if not capture_layers else max_batch
        self.snapshot_tokens = snapshot_tokens
        self.last_ntok = 0
        self._verify: list[tuple] = []  # (sequence, first row, rows) of the last verification pass
        self.schedule = qwen_layer_schedule(gguf, metadata)
        self.mtp_layer = qwen_mtp_layer_index(gguf, metadata) if mtp else None
        self.mtp_concat = "eh"  # eh_proj input order: [enorm(embedding), hnorm(hidden)]
        # Decoder layers whose output hidden rows are recorded per pass (DFlash target features),
        # concatenated per token into "cap_feat" as [ntok][len(capture_layers) * hidden].
        self.capture_layers = tuple(capture_layers)
        if any(not 0 <= i < len(qwen_layer_schedule(gguf, metadata)) for i in self.capture_layers):
            raise ConfigurationError(f"capture_layers {self.capture_layers} outside the decoder stack")
        self.reserve_extra_bytes = reserve_extra_bytes
        self.placement = placement
        self.shapes = self._shapes()
        self.cpu_layers: frozenset[int] = frozenset()
        self.cpu = None
        if free_vram_bytes is None:
            free_vram_bytes = self.rt.mem_info()[0]
        if placement == "hybrid":
            self.cpu_layers = self._choose_cpu_layers(free_vram_bytes, safety_bytes)
        self.plan = self._plan(free_vram_bytes, safety_bytes)
        while placement == "hybrid" and self.plan.streamed_bytes:
            # Keep weights where they are computed: push the first streamed layer (and all after it) to the CPU.
            streamed_layers = {int(n.split(".")[1]) for n in self.plan.names(TensorResidency.PINNED_STREAM) if n.startswith("blk.")}
            first = min(streamed_layers - set(self.cpu_layers)) if streamed_layers - set(self.cpu_layers) else None
            if first is None:
                raise InsufficientMemoryError("hybrid placement could not keep the output head / MTP block on the GPU")
            self.cpu_layers = frozenset(range(first, len(self.schedule)))
            self.plan = self._plan(free_vram_bytes, safety_bytes)
        if self.cpu_layers:
            # CPU layers hold one sequence in contiguous caches: one KV page spanning the context.
            if self.max_seqs > 1:
                raise ConfigurationError("the model does not fit the GPU with hybrid placement and several "
                                         "sequences; CPU layers run a single sequence (use --max-seqs 1)")
            if self.page_size != max_context:
                self.page_size, self.pages_per_seq, self.kv_pages = max_context, 1, 1
                self.plan = self._plan(free_vram_bytes, safety_bytes)
            from vinf.cpu_qwen import CpuQwenRuntime

            self.cpu = CpuQwenRuntime(cpu_threads)
        self._load_weights(progress)
        self._alloc_buffers()
        self.free_slots = list(range(self.max_seqs))
        self.free_pages = list(range(self.kv_pages))
        self.sequences: list[Sequence] = []
        self.seq = self.new_sequence()  # the active sequence of the single-sequence API

    @classmethod
    def plan_only(cls, gguf: GGUFFile, metadata: ModelMetadata, *, max_context: int = 2048,
                  free_vram_bytes: int | None = None, safety_bytes: int = 512 * 1024**2) -> "QwenGpuExecutor":
        """Residency/memory planning without uploading weights or allocating buffers."""
        self = cls.__new__(cls)
        self.gguf, self.metadata, self.max_context, self.head_order = gguf, metadata, max_context, 0
        self.stream_slots = 3
        self.max_batch, self.snapshot_tokens, self.last_ntok = 8, 0, 0
        self.prefill_batch = 256
        self.kv_dtype = "f16"
        self.max_seqs, self.page_size, self.pages_per_seq = 1, max_context, 1
        self.kv_pages, self.verify_rows = 1, 8
        self.ssm_dtype, self.batch_kernels = "f32", False
        self.mtp_layer = None
        self.placement, self.cpu_layers, self.cpu = "stream", frozenset(), None
        self.capture_layers, self.reserve_extra_bytes = (), 0
        if free_vram_bytes is None:
            from vinf.cuda.qwen_runtime import CudaWeightRuntime

            self.rt = CudaWeightRuntime()
        else:
            self.rt = None
        self.schedule = qwen_layer_schedule(gguf, metadata)
        self.shapes = self._shapes()
        if free_vram_bytes is None:
            free_vram_bytes = self.rt.mem_info()[0]
        self.placement = "hybrid"
        self.cpu_layers = self._choose_cpu_layers(free_vram_bytes, safety_bytes)
        self.plan = self._plan(free_vram_bytes, safety_bytes)
        self.seq = None
        return self

    def report(self) -> str:
        full = sum(kind is QwenLayerKind.FULL_ATTENTION for kind in self.schedule)
        lines = [
            f"decoder layers {len(self.schedule)}: {full} full-attention, {len(self.schedule) - full} linear-attention (SSM)",
            self.plan.report(),
        ]
        if self.cpu_layers:
            cpu_bytes = sum(self.gguf.tensors[n].nbytes for i in self.cpu_layers for n in self._layer_weight_names(i))
            lines.append(
                f"CPU-computed layers  {cpu_bytes / 1024**3:7.2f} GiB ({len(self.cpu_layers)} layers: "
                f"{min(self.cpu_layers)}..{max(self.cpu_layers)}, weights read in place from mmap)"
            )
        return "\n".join(lines)

    # ---- setup -------------------------------------------------------------------------

    def _shapes(self) -> QwenGpuShapes:
        meta, gguf = self.metadata, self.gguf
        rope = qwen_rope_config_from_gguf(gguf, meta)
        eps = gguf.metadata_value(f"{meta.architecture.value}.attention.layer_norm_rms_epsilon", 1e-6)
        has_linear = QwenLayerKind.LINEAR_ATTENTION in self.schedule
        lin = qwen_linear_attention_config_from_metadata(gguf, meta) if has_linear else None
        return QwenGpuShapes(
            hidden=meta.hidden_size,
            intermediate=meta.intermediate_size,
            vocab=meta.vocab_size,
            heads=meta.num_attention_heads,
            kv_heads=meta.num_kv_heads,
            head_dim=meta.head_dim,
            rotary_dim=rope.rotated_dims,
            freq_base=rope.freq_base,
            key_heads=lin.key_heads if lin else 0,
            value_heads=lin.value_heads if lin else 0,
            key_head_dim=lin.key_head_dim if lin else 0,
            value_head_dim=lin.value_head_dim if lin else 0,
            conv_kernel=lin.conv_kernel if lin else 0,
            eps=float(eps),
        )

    def _buffer_specs(self) -> dict[str, int]:
        """Activation buffers, sized for max_batch token rows ([ntok][...] layout)."""
        s = self.shapes
        specs = {
            "h": s.hidden,
            "xn": s.hidden,
            "proj": s.hidden,
            "mlp_gate": s.intermediate,
            "mlp_up": s.intermediate,
            "logits": s.vocab,
        }
        if QwenLayerKind.FULL_ATTENTION in self.schedule:
            q = s.heads * s.head_dim
            kv = s.kv_heads * s.head_dim
            specs.update({"q_raw": 2 * q, "q": q, "q_gate": q, "k": kv, "v": kv, "attn": q})
        if QwenLayerKind.LINEAR_ATTENTION in self.schedule:
            specs.update(
                {
                    "lin_qkv": s.conv_dim,
                    "lin_conv": s.conv_dim,
                    "lin_z": s.value_dim,
                    "lin_beta": s.value_heads,
                    "lin_alpha": s.value_heads,
                    "lin_core": s.value_dim,
                    "lin_out": s.value_dim,
                }
            )
        if self.mtp_layer is not None:
            specs.update({"mtp_e": s.hidden, "mtp_hid": s.hidden, "mtp_tmp": s.hidden, "mtp_cat": 2 * s.hidden,
                          "mtp_h": s.hidden, "mtp_out": s.hidden, "spec_pend": s.hidden})
        if self.capture_layers:
            specs["cap_feat"] = len(self.capture_layers) * s.hidden
        specs = {name: n * self.prefill_batch for name, n in specs.items()}
        specs["logits"] = s.vocab * self.verify_rows  # LM head rows: verification / multi-sequence passes only
        if "spec_pend" in specs:  # MTP rows pending per sequence slot
            specs["spec_pend"] = s.hidden * self.max_batch * self.max_seqs
            specs["mtp_last"] = s.hidden * self.max_seqs
        specs["page_table"] = self.max_seqs * self.pages_per_seq
        if self.snapshot_tokens > 0:
            # Inputs of the SSM layers in a verification pass (conv input, beta, alpha rows):
            # rollback replays the accepted rows from the untouched state.
            for i, kind in enumerate(self.schedule):
                if kind is QwenLayerKind.LINEAR_ATTENTION:
                    specs[f"hq.{i}"] = s.conv_dim * self.verify_rows
                    specs[f"hb.{i}"] = s.value_heads * self.verify_rows
                    specs[f"ha.{i}"] = s.value_heads * self.verify_rows
        specs["h_last"] = s.hidden
        return specs

    def _state_specs(self) -> dict[str, int]:
        s = self.shapes
        specs: dict[str, int] = {}
        for layer_idx, kind in enumerate(self.schedule):
            if kind is QwenLayerKind.FULL_ATTENTION:
                cache = self.kv_pages * s.kv_heads * self.page_size * s.head_dim
                specs[f"kc.{layer_idx}"] = cache
                specs[f"vc.{layer_idx}"] = cache
            else:
                conv = s.conv_dim * s.conv_kernel
                ssm = s.value_heads * s.key_head_dim * s.value_head_dim
                specs[f"conv.{layer_idx}"] = conv * self.max_seqs
                specs[f"ssm.{layer_idx}"] = ssm * self.max_seqs
                if self.snapshot_tokens > 1 and layer_idx in self.cpu_layers:
                    # CPU layers: states after tokens 0..n-2 of a verification pass, for rollback. GPU layers
                    # keep the pass inputs instead and replay the accepted rows (see commit).
                    specs[f"conv_snap.{layer_idx}"] = conv * (self.snapshot_tokens - 1)
                    specs[f"ssm_snap.{layer_idx}"] = ssm * (self.snapshot_tokens - 1)
        if self.mtp_layer is not None:
            cache = self.kv_pages * s.kv_heads * self.page_size * s.head_dim
            specs[f"kc.{self.mtp_layer}"] = cache
            specs[f"vc.{self.mtp_layer}"] = cache
        return specs

    def _state_elem(self, name: str) -> int:
        """Bytes per element of a GPU state buffer (fp16 attention KV caches)."""
        if name.startswith(("kc.", "vc.")):
            return 2 if self.kv_dtype == "f16" else 4
        if name.startswith("ssm."):
            return 2 if self.ssm_dtype == "f16" else 4
        return 4

    def _layer_of(self, name: str) -> int | None:
        return int(name.split(".")[1]) if name.split(".")[1].isdigit() else None

    def _gpu_state_specs(self) -> dict[str, int]:
        return {n: v for n, v in self._state_specs().items() if self._layer_of(n) not in self.cpu_layers}

    def _cpu_state_specs(self) -> dict[str, int]:
        return {n: v for n, v in self._state_specs().items() if self._layer_of(n) in self.cpu_layers}

    def _layer_weight_names(self, layer_idx: int) -> tuple[str, ...]:
        kind = self.schedule[layer_idx]
        suffixes = FULL_ATTENTION_WEIGHTS if kind is QwenLayerKind.FULL_ATTENTION else LINEAR_ATTENTION_WEIGHTS
        return tuple(f"blk.{layer_idx}.{suffix}" for suffix in suffixes + MLP_WEIGHTS)

    def _choose_cpu_layers(self, free_vram_bytes: int, safety_bytes: int) -> frozenset[int]:
        """Whole layers on the GPU in order while they fit (weights + their own state); the rest on the CPU."""
        states = self._state_specs()
        fixed = sum(self._buffer_specs().values()) * 4 + 32 * 1024**2 + safety_bytes + self.reserve_extra_bytes
        fixed += sum(self.gguf.tensors[n].nbytes for n in ("output.weight", "output_norm.weight") + self._mtp_weight_names())
        fixed += sum(v * self._state_elem(n) for n, v in states.items() if self._layer_of(n) == self.mtp_layer)
        budget = free_vram_bytes - fixed
        if budget < 0:
            raise InsufficientMemoryError(
                f"insufficient VRAM: {free_vram_bytes / 1024**3:.2f} GiB free cannot hold the output head, "
                "activations, and safety margin"
            )
        used = 0
        for layer_idx in range(len(self.schedule)):
            cost = sum(self.gguf.tensors[n].nbytes for n in self._layer_weight_names(layer_idx))
            cost += sum(v * self._state_elem(n) for n, v in states.items() if self._layer_of(n) == layer_idx)
            if used + cost > budget:
                return frozenset(range(layer_idx, len(self.schedule)))
            used += cost
        return frozenset()

    def _plan(self, free_vram_bytes: int | None, safety_bytes: int) -> QwenDecodeResidencyPlan:
        if free_vram_bytes is None:
            free_vram_bytes = self.rt.mem_info()[0]
        states = self._gpu_state_specs()
        kv = sum(n * self._state_elem(name) for name, n in states.items() if name.startswith(("kc.", "vc.")))
        ssm = sum(n * self._state_elem(name) for name, n in states.items() if name.startswith(("conv", "ssm")))
        # + kernel/runtime slack + memory reserved for an external drafter (e.g. DFlash weights)
        activations = sum(self._buffer_specs().values()) * 4 + 32 * 1024**2 + self.reserve_extra_bytes
        gpu_layers = [i for i in range(len(self.schedule)) if i not in self.cpu_layers]
        return plan_qwen_decode_residency(
            self.gguf,
            tuple(n for i in gpu_layers for n in self._layer_weight_names(i)),
            free_vram_bytes=free_vram_bytes,
            kv_cache_bytes=kv,
            ssm_state_bytes=ssm,
            activation_bytes=activations,
            safety_bytes=safety_bytes,
            max_context=self.max_context,
            stream_slots=self.stream_slots,
            priority_names=self._mtp_weight_names(),
        )

    def _mtp_weight_names(self) -> tuple[str, ...]:
        if self.mtp_layer is None:
            return ()
        p = f"blk.{self.mtp_layer}."
        return tuple(p + suffix for suffix in FULL_ATTENTION_WEIGHTS + MLP_WEIGHTS + MTP_TENSORS)

    def _load_weights(self, progress) -> None:
        entries = [e for e in self.plan.entries if e.residency is not TensorResidency.CPU_MMAP]
        # Fail before any upload if a tensor type has no CUDA kernel.
        from vinf.cuda.qwen_runtime import CUDA_MATVEC_TYPES

        unsupported = sorted(
            {self.gguf.tensors[e.name].tensor_type.name for e in entries}
            - {t.name for t in CUDA_MATVEC_TYPES}
        )
        if unsupported:
            raise UnsupportedModelError("no CUDA kernel for tensor types: " + ", ".join(unsupported))
        resident = [e for e in entries if e.residency is TensorResidency.GPU]
        self.rt.set_arena(sum((e.nbytes + 16 + 255) // 256 * 256 for e in resident))
        for idx, entry in enumerate(entries):
            self.rt.upload_gguf_tensor(
                self.gguf, entry.name, resident=entry.residency is TensorResidency.GPU
            )
            if progress is not None:
                progress(idx + 1, len(entries), entry)
        if self.cpu is not None:
            for layer_idx in sorted(self.cpu_layers):
                for name in self._layer_weight_names(layer_idx):
                    self.cpu.add_gguf_tensor(self.gguf, name)
        streamed = set(self.plan.names(TensorResidency.PINNED_STREAM))
        order = [name for name in qwen_qmv_order(self.schedule) if name in streamed]
        if set(order) != streamed:
            raise UnsupportedModelError(f"streamed tensors outside the qmv order: {sorted(streamed - set(order))}")
        self.rt.set_stream_order(order, self.stream_slots)

    def _alloc_buffers(self) -> None:
        if self.ssm_dtype == "f16" and QwenLayerKind.LINEAR_ATTENTION in self.schedule:
            s = self.shapes
            if s.key_head_dim not in (32, 64, 128, 256) or s.value_head_dim % 32:
                raise ConfigurationError("fp16 SSM state needs key_head_dim in {32, 64, 128, 256} and value_head_dim % 32 == 0")
        for name, n in self._buffer_specs().items():
            self.rt.alloc(name, n)
        for name, n in self._gpu_state_specs().items():
            if self._state_elem(name) == 4:
                self.rt.alloc(name, n)
            else:
                self.rt.alloc(name, n, self._state_elem(name))
        if self.cpu is not None:
            for name, n in {**self._buffer_specs(), **self._cpu_state_specs()}.items():
                self.cpu.alloc(name, n)
        self.rt.set_paging("page_table", self.page_size, self.pages_per_seq, self.max_seqs)
        self.prompt_mode(False)

    def prompt_mode(self, on: bool) -> None:
        """Prompt passes use the tensor-core GEMM and tiled attention; with several sequences every pass does."""
        gemm = getattr(self.rt, "set_gemm_min_rows", None)
        if gemm is not None:
            gemm(1 if on or self.batch_kernels else 0)

    # ---- sequences ------------------------------------------------------------------------------

    @property
    def position(self) -> int:
        return self.seq.position if self.seq is not None else 0

    @property
    def tokens(self) -> list[int]:
        return self.seq.tokens

    def new_sequence(self) -> Sequence:
        """A fresh sequence in a free slot (state zeroed, no pages yet)."""
        if not self.free_slots:
            raise KvPoolExhausted(f"all {self.max_seqs} sequence slots are in use")
        seq = Sequence(self.free_slots.pop(0))
        self.sequences.append(seq)
        self._zero_slot(seq.slot)
        return seq

    def release_sequence(self, seq: Sequence) -> None:
        """Free a sequence's slot and KV pages."""
        if seq not in self.sequences:
            return
        self.sequences.remove(seq)
        self.free_pages.extend(seq.pages)
        seq.pages, seq.tokens = [], []
        self.free_slots.append(seq.slot)
        if self.seq is seq:
            self.seq = None

    def activate(self, seq: Sequence) -> None:
        """Make seq the target of the single-sequence API (forward_tokens, prefill, rollback, ...)."""
        self.seq = seq

    def ensure_pages(self, seq: Sequence, length: int) -> None:
        """Give seq enough KV pages for positions [0, length)."""
        if length > self.max_context:
            raise ConfigurationError(f"context is full (max_context={self.max_context})")
        need = -(-length // self.page_size) - len(seq.pages)
        if need <= 0:
            return
        while len(self.free_pages) < need:
            if self.reclaim is None or not self.reclaim(need - len(self.free_pages)):
                raise KvPoolExhausted(
                    f"KV pool exhausted: need {need} more pages of {self.page_size} tokens, "
                    f"{len(self.free_pages)} free of {self.kv_pages}")
        seq.pages.extend(self.free_pages[:need])
        del self.free_pages[:need]
        self.rt.write("page_table", array("i", seq.pages).tobytes(), seq.slot * self.pages_per_seq)

    def copy_kv(self, src: Sequence, dst: Sequence, length: int) -> None:
        """Copy the attention KV rows [0, length) of src into dst's pages (device to device)."""
        if self.cpu_layers:
            raise ConfigurationError("KV copies between sequences need every attention layer on the GPU")
        self.ensure_pages(dst, length)
        s = self.shapes
        page_elems = s.kv_heads * self.page_size * s.head_dim
        for layer in self._kv_layers():
            for kind in ("kc", "vc"):
                name = f"{kind}.{layer}"
                for i in range(-(-length // self.page_size)):
                    self.rt.copy(name, dst.pages[i] * page_elems, name, src.pages[i] * page_elems, page_elems)

    def free_page_count(self) -> int:
        return len(self.free_pages)

    def _zero_slot(self, slot: int) -> None:
        specs = self._state_specs()
        for name in self._recurrent_names():
            layer = self._layer_of(name)
            if layer in self.cpu_layers:
                self.cpu.zero(name)
            else:
                per = specs[name] // self.max_seqs
                self.rt.zero_range(name, slot * per, per)

    # ---- decode ------------------------------------------------------------------------

    def vram_report(self) -> str:
        """Runtime-owned VRAM vs. device usage; the remainder is CUDA context, other processes, and fragmentation."""
        free, total = self.rt.mem_info()
        weights, buffers, staging = self.rt.device_bytes(), self.rt.buffer_bytes(), self.rt.staging_bytes()
        ours = weights + buffers + staging
        used = total - free
        mib = 1024**2
        return (
            f"VRAM used {used / mib:.0f} MiB of {total / mib:.0f} MiB: weights {weights / mib:.0f}, "
            f"buffers/caches {buffers / mib:.0f}, staging {staging / mib:.0f}, "
            f"other (context, other processes, fragmentation) {(used - ours) / mib:.0f}; free {free / mib:.0f} MiB"
        )

    def enable_profiling(self) -> ProfilingRuntime:
        streamed = frozenset(self.plan.names(TensorResidency.PINNED_STREAM))
        self.rt = ProfilingRuntime(self.rt, streamed)
        return self.rt

    def _rt(self, layer_idx: int):
        return self.cpu if layer_idx in self.cpu_layers else self.rt

    def reset(self) -> None:
        """Restart the active sequence at position 0 (keeps its slot and pages)."""
        if self.seq is None:
            self.seq = self.new_sequence()
        self._zero_slot(self.seq.slot)
        if self.cpu is not None:
            for name in self._cpu_state_specs():
                self.cpu.zero(name)
        self.seq.tokens = []
        self.last_ntok = 0

    def _full_attention(self, layer_idx: int, position: int, n: int = 1, h: str = "h") -> None:
        rt, s, p = self._rt(layer_idx), self.shapes, f"blk.{layer_idx}."
        rt.rmsnorm(h, p + "attn_norm.weight", "xn", s.hidden, s.eps, n)
        rt.qmv(p + "attn_q.weight", "xn", "q_raw", n)
        rt.qmv(p + "attn_k.weight", "xn", "k", n)
        rt.qmv(p + "attn_v.weight", "xn", "v", n)
        rt.split_gated_q("q_raw", "q", "q_gate", s.heads * n, s.head_dim)
        rt.rmsnorm("q", p + "attn_q_norm.weight", "q", s.head_dim, s.eps, s.heads * n)
        rt.rmsnorm("k", p + "attn_k_norm.weight", "k", s.head_dim, s.eps, s.kv_heads * n)
        kc, vc = f"kc.{layer_idx}", f"vc.{layer_idx}"
        if rt is self.rt:  # paged KV, rows laid out by the current segments (set_segments)
            rt.rope_seg("q", s.heads, s.head_dim, s.rotary_dim, s.freq_base)
            rt.rope_seg("k", s.kv_heads, s.head_dim, s.rotary_dim, s.freq_base)
            rt.kv_append_seg("k", "v", kc, vc, s.kv_heads, s.head_dim)
            rt.attention_seg("q", kc, vc, "attn", s.heads, s.kv_heads, s.head_dim)
        else:  # CPU layers: single sequence, contiguous cache
            rt.rope("q", s.heads, s.head_dim, s.rotary_dim, position, s.freq_base, n)
            rt.rope("k", s.kv_heads, s.head_dim, s.rotary_dim, position, s.freq_base, n)
            rt.kv_append("k", "v", kc, vc, s.kv_heads, self.max_context, s.head_dim, position, n)
            rt.attention("q", kc, vc, "attn", s.heads, s.kv_heads, s.head_dim, self.max_context, position + 1, n)
        rt.sigmoid_mul("attn", "q_gate", "attn", s.heads * s.head_dim * n)
        rt.qmv(p + "attn_output.weight", "attn", "proj", n)
        rt.add(h, "proj", h, s.hidden * n)

    def _linear_attention(self, layer_idx: int, n: int = 1, snapshot: bool = False) -> None:
        rt, s, p = self._rt(layer_idx), self.shapes, f"blk.{layer_idx}."
        rt.rmsnorm("h", p + "attn_norm.weight", "xn", s.hidden, s.eps, n)
        rt.qmv(p + "attn_qkv.weight", "xn", "lin_qkv", n)
        rt.qmv(p + "attn_gate.weight", "xn", "lin_z", n)
        rt.qmv(p + "ssm_beta.weight", "xn", "lin_beta", n)
        rt.qmv(p + "ssm_alpha.weight", "xn", "lin_alpha", n)
        conv_snap = f"conv_snap.{layer_idx}" if snapshot else None
        ssm_snap = f"ssm_snap.{layer_idx}" if snapshot else None
        gd = ("lin_conv", "lin_beta", "lin_alpha", p + "ssm_a", p + "ssm_dt.bias", f"ssm.{layer_idx}", "lin_core",
              s.key_heads, s.value_heads, s.key_head_dim, s.value_head_dim, s.eps, self.head_order)
        if rt is self.rt and snapshot:  # verification: state untouched, inputs kept for the commit replay
            rt.conv_update_seg("lin_qkv", f"conv.{layer_idx}", p + "ssm_conv1d.weight", "lin_conv", s.conv_dim,
                               s.conv_kernel, None, 0)
            rt.gated_delta_seg(*gd, None, 0)
            rt.copy(f"hq.{layer_idx}", 0, "lin_qkv", 0, n * s.conv_dim)
            rt.copy(f"hb.{layer_idx}", 0, "lin_beta", 0, n * s.value_heads)
            rt.copy(f"ha.{layer_idx}", 0, "lin_alpha", 0, n * s.value_heads)
        elif rt is self.rt:  # state slot per segment
            rt.conv_update_seg("lin_qkv", f"conv.{layer_idx}", p + "ssm_conv1d.weight", "lin_conv", s.conv_dim,
                               s.conv_kernel)
            rt.gated_delta_seg(*gd)
        else:
            rt.conv_update("lin_qkv", f"conv.{layer_idx}", p + "ssm_conv1d.weight", "lin_conv", s.conv_dim,
                           s.conv_kernel, n, conv_snap)
            rt.gated_delta(*gd, n, ssm_snap)
        rt.gated_rmsnorm("lin_core", p + "ssm_norm.weight", "lin_z", "lin_out", s.value_head_dim, s.eps, s.value_heads * n)
        rt.qmv(p + "ssm_out.weight", "lin_out", "proj", n)
        rt.add("h", "proj", "h", s.hidden * n)

    def _mlp(self, layer_idx: int, n: int = 1, h: str = "h") -> None:
        rt, s, p = self._rt(layer_idx), self.shapes, f"blk.{layer_idx}."
        rt.rmsnorm(h, p + "post_attention_norm.weight", "xn", s.hidden, s.eps, n)
        rt.qmv(p + "ffn_gate.weight", "xn", "mlp_gate", n)
        rt.qmv(p + "ffn_up.weight", "xn", "mlp_up", n)
        rt.silu_mul("mlp_gate", "mlp_up", "mlp_gate", s.intermediate * n)
        rt.qmv(p + "ffn_down.weight", "mlp_gate", "proj", n)
        rt.add(h, "proj", h, s.hidden * n)

    def forward_tokens(self, token_ids: list[int], *, snapshot: bool = False) -> None:
        """Run len(token_ids) <= max_batch tokens at positions [position, position + n) through every
        decoder layer, reading each weight once. Logits are not computed (see greedy_rows / greedy_next).
        snapshot=True records SSM/conv state after every token for `rollback`."""
        n = len(token_ids)
        if not 1 <= n <= self.prefill_batch:
            raise ConfigurationError(f"token batch must be in [1, {self.prefill_batch}], got {n}")
        if self.position + n > self.max_context:
            raise ConfigurationError(f"context is full (max_context={self.max_context})")
        if snapshot and n > self.snapshot_tokens:
            raise ConfigurationError(f"snapshot needs snapshot_tokens >= {n} (have {self.snapshot_tokens})")
        seq = self.seq
        self.ensure_pages(seq, seq.position + n)
        self.rt.set_segments([(seq.slot, seq.position, n)])
        self._forward_rows(list(token_ids), self.position, snapshot)
        self._verify = [(seq, 0, n)] if snapshot else []
        seq.tokens.extend(token_ids)
        self.last_ntok = n

    def forward_verify(self, seqs: list[Sequence], rows: list[list[int]]) -> None:
        """Verification pass for several sequences at once: rows[i] = sequence i's next input followed by
        its drafts. SSM/conv state is left untouched until commit(); KV is appended for every row.
        greedy_rows() then gives every row's target (rows of sequence i are contiguous, in order)."""
        n = sum(len(r) for r in rows)
        if not seqs or len(seqs) != len(rows) or any(not r for r in rows):
            raise ConfigurationError("forward_verify needs one non-empty row list per sequence")
        if n > self.verify_rows or any(len(r) > self.snapshot_tokens for r in rows):
            raise ConfigurationError(f"verification pass of {n} rows exceeds verify_rows={self.verify_rows} "
                                     f"or snapshot_tokens={self.snapshot_tokens} per sequence")
        if self.cpu_layers and len(seqs) > 1:
            raise ConfigurationError("CPU layers run a single sequence")
        for seq, r in zip(seqs, rows):
            self.ensure_pages(seq, seq.position + len(r))
        segs, verify, row0 = [], [], 0
        for seq, r in zip(seqs, rows):
            segs.append((seq.slot, seq.position, len(r)))
            verify.append((seq, row0, len(r)))
            row0 += len(r)
        self.rt.set_segments(segs)
        flat = [t for r in rows for t in r]
        self._forward_rows(flat, seqs[0].position, True)
        for seq, r in zip(seqs, rows):
            seq.tokens.extend(r)
        self._verify = verify
        self.last_ntok = n

    def commit(self, keeps: list[int]) -> None:
        """After a verification pass, keep the first keeps[i] rows of sequence i: advance its SSM/conv state
        by replaying those rows from the stored pass inputs, and drop the rest (KV rows past the kept
        positions are ignored and later overwritten)."""
        if len(keeps) != len(self._verify):
            raise ConfigurationError("commit needs one keep count per sequence of the last verification pass")
        for (seq, row0, n), keep in zip(self._verify, keeps):
            if not 1 <= keep <= n:
                raise ConfigurationError(f"keep must be in [1, {n}]")
        s = self.shapes
        gpu_linear = [i for i, k in enumerate(self.schedule)
                      if k is QwenLayerKind.LINEAR_ATTENTION and i not in self.cpu_layers]
        if gpu_linear:
            self.rt.set_segments([(seq.slot, seq.position, keep, row0)
                                  for (seq, row0, _), keep in zip(self._verify, keeps)])
            for i in gpu_linear:
                p = f"blk.{i}."
                self.rt.conv_update_seg(f"hq.{i}", f"conv.{i}", p + "ssm_conv1d.weight", "lin_conv", s.conv_dim,
                                        s.conv_kernel)
                self.rt.gated_delta_seg("lin_conv", f"hb.{i}", f"ha.{i}", p + "ssm_a", p + "ssm_dt.bias", f"ssm.{i}",
                                        "lin_core", s.key_heads, s.value_heads, s.key_head_dim, s.value_head_dim,
                                        s.eps, self.head_order)
        if self.cpu_layers:  # single sequence: restore the CPU layers' snapshots
            (seq, _, n), keep = self._verify[0], keeps[0]
            conv = s.conv_dim * s.conv_kernel
            ssm = s.value_heads * s.key_head_dim * s.value_head_dim
            if keep < n:
                for i in sorted(self.cpu_layers):
                    if self.schedule[i] is QwenLayerKind.LINEAR_ATTENTION:
                        self.cpu.copy(f"conv.{i}", 0, f"conv_snap.{i}", (keep - 1) * conv, conv)
                        self.cpu.copy(f"ssm.{i}", 0, f"ssm_snap.{i}", (keep - 1) * ssm, ssm)
        for (seq, _, n), keep in zip(self._verify, keeps):
            del seq.tokens[len(seq.tokens) - (n - keep):]
        self._verify = []

    def forward_multi(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """One pass over one new token for each sequence (reading every weight once for all of them).
        Row r of the pass belongs to seqs[r]; follow with greedy_rows() for each sequence's next token."""
        n = len(seqs)
        if not 1 <= n <= self.verify_rows or len(token_ids) != n:
            raise ConfigurationError(f"multi-sequence pass needs 1..{self.verify_rows} sequences, one token each")
        if self.cpu_layers and n > 1:
            raise ConfigurationError("CPU layers run a single sequence")
        for seq in seqs:
            self.ensure_pages(seq, seq.position + 1)
        self.rt.set_segments([(seq.slot, seq.position, 1) for seq in seqs])
        self._forward_rows(list(token_ids), seqs[0].position, False)
        for seq, token in zip(seqs, token_ids):
            seq.tokens.append(token)
        self.last_ntok = n

    def _forward_rows(self, token_ids: list[int], position: int, snapshot: bool) -> None:
        """Run the rows laid out by the current segments through every decoder layer (position is used
        by CPU layers, which hold a single sequence)."""
        n = len(token_ids)
        rows = load_qwen_token_embeddings(self.gguf, token_ids)
        hn = n * self.shapes.hidden
        here = self._rt(0)
        here.write("h", array("f", [value for row in rows for value in row]).tobytes())
        for layer_idx, kind in enumerate(self.schedule):
            there = self._rt(layer_idx)
            if there is not here:  # hidden rows cross the PCIe bus only at a GPU/CPU boundary
                there.write("h", here.read("h", hn))
                here = there
            if kind is QwenLayerKind.FULL_ATTENTION:
                self._full_attention(layer_idx, position, n)
            else:
                self._linear_attention(layer_idx, n, snapshot)
            self._mlp(layer_idx, n)
            if layer_idx in self.capture_layers:
                self._capture(layer_idx, here, n)
        if here is not self.rt:  # LM head / MTP live on the GPU
            self.rt.write("h", here.read("h", hn))

    def _capture(self, layer_idx: int, rt, n: int) -> None:
        """Copy this layer's output rows into "cap_feat" at the layer's column slot."""
        h, slots = self.shapes.hidden, len(self.capture_layers)
        col = self.capture_layers.index(layer_idx)
        for t in range(n):
            dst = (t * slots + col) * h
            if rt is self.rt:
                self.rt.copy("cap_feat", dst, "h", t * h, h)
            else:
                self.rt.write("cap_feat", rt.read("h", h, t * h), dst)

    def forward_token(self, token_id: int) -> None:
        """Run one token through every decoder layer at the current position (logits not computed)."""
        self.forward_tokens([token_id])

    def rollback(self, keep: int) -> None:
        """After a verification pass of n tokens (forward_tokens(..., snapshot=True)), keep only the first
        `keep` (1..n)."""
        if not self._verify:
            raise ConfigurationError("rollback needs a preceding verification pass")
        self.commit([keep])
        self.last_ntok = keep

    def _lm_head(self) -> None:
        """LM head on the last forwarded token row."""
        s = self.shapes
        self.rt.copy("h_last", 0, "h", (self.last_ntok - 1) * s.hidden, s.hidden)
        self.rt.rmsnorm("h_last", "output_norm.weight", "xn", s.hidden, s.eps)
        self.rt.qmv("output.weight", "xn", "logits")

    def greedy_next(self) -> int:
        """Argmax of the LM head for the last forwarded token; only the token id leaves the GPU."""
        self._lm_head()
        return self.rt.argmax("logits", self.shapes.vocab)

    def greedy_rows(self) -> list[int]:
        """Argmax for every row of the last multi-token pass (one LM-head weight read).
        Leaves the post-norm hidden rows in "xn" ([n][hidden])."""
        s, n = self.shapes, self.last_ntok
        self.rt.rmsnorm("h", "output_norm.weight", "xn", s.hidden, s.eps, n)
        self.rt.qmv("output.weight", "xn", "logits", n)
        return self.rt.argmax_rows("logits", s.vocab, n)

    def logits(self) -> list[float]:
        """Explicit full-vocab transfer (debugging / parity tests only)."""
        self._lm_head()
        return self.rt.read_floats("logits", self.shapes.vocab)

    def hidden(self) -> list[float]:
        """Pre-norm hidden state of the last forwarded token."""
        s = self.shapes
        return self.rt.read_floats("h", self.last_ntok * s.hidden)[-s.hidden:]

    # ---- MTP draft block -------------------------------------------------------------------

    def post_norm_rows(self) -> str:
        """Final-norm the last pass's hidden rows into "xn" (no LM head); returns the buffer name."""
        s = self.shapes
        self.rt.rmsnorm("h", "output_norm.weight", "xn", s.hidden, s.eps, self.last_ntok)
        return "xn"

    def mtp_load_hidden(self, src: str, src_row: int, n: int, dst_row: int = 0) -> None:
        """Copy n post-norm hidden rows into the MTP hidden input buffer."""
        h = self.shapes.hidden
        self.rt.copy("mtp_hid", dst_row * h, src, src_row * h, n * h)

    def mtp_forward(self, next_tokens: list[int], position: int) -> int:
        """Run the MTP block over n rows of the active sequence at positions [position, position + n). Row t
        combines the embedding of next_tokens[t] with hidden row t of "mtp_hid" (post-norm target hidden of
        position + t, or the previous MTP output when drafting recursively). Post-norm outputs land in
        "mtp_out"; returns the greedy draft from the last row."""
        return self.mtp_forward_multi([(self.seq, position, list(next_tokens))])[0]

    def mtp_forward_multi(self, items: list[tuple]) -> list[int]:
        """MTP block over several sequences in one pass: items = [(sequence, position, next_tokens)], whose
        hidden rows are stacked in "mtp_hid" in the same order. Returns each item's greedy draft (its last
        row); post-norm outputs are stacked in "mtp_out"."""
        if self.mtp_layer is None:
            raise ConfigurationError("executor was created without mtp=True")
        n = sum(len(toks) for _, _, toks in items)
        if not items or not 1 <= n <= self.prefill_batch or len(items) > self.max_seqs:
            raise ConfigurationError("invalid MTP batch")
        for seq, position, toks in items:
            if not toks or position + len(toks) > self.max_context:
                raise ConfigurationError("invalid MTP position")
            self.ensure_pages(seq, position + len(toks))
        rt, s, p = self.rt, self.shapes, f"blk.{self.mtp_layer}."
        h = s.hidden
        rt.set_segments([(seq.slot, position, len(toks)) for seq, position, toks in items])
        rows = load_qwen_token_embeddings(self.gguf, [t for _, _, toks in items for t in toks])
        rt.write("mtp_e", array("f", [v for row in rows for v in row]).tobytes())
        rt.rmsnorm("mtp_e", p + "nextn.enorm.weight", "mtp_tmp", h, s.eps, n)
        rt.rmsnorm("mtp_hid", p + "nextn.hnorm.weight", "mtp_out", h, s.eps, n)  # mtp_out as scratch
        if self.mtp_concat == "eh":
            rt.concat_rows("mtp_cat", "mtp_tmp", "mtp_out", n, h)
        else:
            rt.concat_rows("mtp_cat", "mtp_out", "mtp_tmp", n, h)
        rt.qmv(p + "nextn.eh_proj.weight", "mtp_cat", "mtp_h", n)
        self._full_attention(self.mtp_layer, items[0][1], n, h="mtp_h")
        self._mlp(self.mtp_layer, n, h="mtp_h")
        rt.rmsnorm("mtp_h", p + "nextn.shared_head_norm.weight", "mtp_out", h, s.eps, n)
        row = 0
        for i, (_, _, toks) in enumerate(items):
            row += len(toks)
            rt.copy("mtp_last", i * h, "mtp_out", (row - 1) * h, h)
        rt.qmv("output.weight", "mtp_last", "logits", len(items))
        return rt.argmax_rows("logits", s.vocab, len(items))

    def prefill(self, prompt_tokens: list[int], *, start: int = 0, before_last=None, observe=None) -> None:
        """Forward prompt_tokens[start:] (positions [start, len)) in passes of prefill_batch rows.
        before_last() runs before the final pass (a prefix-cache checkpoint point);
        observe(chunk_start, chunk) runs after every pass (speculative drafters)."""
        if start != self.position:
            raise ConfigurationError(f"prefill from {start} but the executor is at position {self.position}")
        chunks = [(i, prompt_tokens[i : i + self.prefill_batch])
                  for i in range(start, len(prompt_tokens), self.prefill_batch)]
        # Prompt passes use the tensor-core GEMM for every pass size, so the computed state does not
        # depend on where a prompt is split (prefix-cache resumes match a fresh prefill).
        self.prompt_mode(True)
        try:
            for idx, (i, chunk) in enumerate(chunks):
                if idx == len(chunks) - 1 and before_last is not None:
                    before_last()
                self.forward_tokens(chunk)
                if observe is not None:
                    observe(i, chunk)
        finally:
            self.prompt_mode(False)

    # ---- state save / restore (prefix cache) ----------------------------------------------------

    def _kv_layers(self) -> list[int]:
        layers = [i for i, kind in enumerate(self.schedule) if kind is QwenLayerKind.FULL_ATTENTION]
        return layers + ([self.mtp_layer] if self.mtp_layer is not None else [])

    def _recurrent_names(self) -> list[str]:
        return [f"{kind}.{i}" for i, k in enumerate(self.schedule) if k is QwenLayerKind.LINEAR_ATTENTION
                for kind in ("conv", "ssm")]

    def kv_bytes_per_token(self) -> int:
        s = self.shapes
        return sum(2 * s.kv_heads * s.head_dim * (4 if layer in self.cpu_layers else self._state_elem("kc."))
                   for layer in self._kv_layers())

    def recurrent_bytes(self) -> int:
        specs = self._state_specs()  # per sequence (the specs cover every slot)
        return sum(specs[n] // self.max_seqs * self._state_elem(n) for n in self._recurrent_names())

    def save_recurrent(self) -> dict[str, bytes]:
        """Host copy of the active sequence's SSM/conv state (the part that cannot be rewound)."""
        specs, out = self._state_specs(), {}
        for name in self._recurrent_names():
            rt = self._rt(self._layer_of(name))
            per = specs[name] // self.max_seqs
            out[name] = rt.read(name, per, (self.seq.slot if rt is self.rt else 0) * per)
        return out

    def load_recurrent(self, state: dict[str, bytes]) -> None:
        specs = self._state_specs()
        for name, data in state.items():
            rt = self._rt(self._layer_of(name))
            per = specs[name] // self.max_seqs
            rt.write(name, data, (self.seq.slot if rt is self.rt else 0) * per)

    def save_kv(self, length: int) -> dict[str, list[bytes]]:
        """Host copy of the active sequence's attention KV rows [0, length): the pages covering them
        (GPU pools) or per-head row ranges (contiguous CPU caches)."""
        s, out = self.shapes, {}
        page_elems = s.kv_heads * self.page_size * s.head_dim
        npages = -(-length // self.page_size)
        for layer in self._kv_layers():
            rt = self._rt(layer)
            for kind in ("kc", "vc"):
                name = f"{kind}.{layer}"
                if rt is self.rt:
                    out[name] = [rt.read(name, page_elems, pg * page_elems) for pg in self.seq.pages[:npages]]
                else:
                    out[name] = [rt.read(name, length * s.head_dim, h * self.max_context * s.head_dim)
                                 for h in range(s.kv_heads)]
        return out

    def load_kv(self, kv: dict[str, list[bytes]], length: int) -> None:
        s = self.shapes
        page_elems = s.kv_heads * self.page_size * s.head_dim
        self.ensure_pages(self.seq, length)
        for name, parts in kv.items():
            rt = self._rt(self._layer_of(name))
            for i, data in enumerate(parts):
                if rt is self.rt:
                    rt.write(name, data, self.seq.pages[i] * page_elems)
                else:
                    rt.write(name, data, i * self.max_context * s.head_dim)

    def resume(self, tokens: list[int], recurrent: dict[str, bytes], kv: dict[str, list[bytes]] | None) -> None:
        """Make `tokens` the active sequence's live state: recurrent state from a checkpoint, KV rows
        from the host copy (or already in the sequence's pages when kv is None)."""
        self.load_recurrent(recurrent)
        if kv is not None:
            self.load_kv(kv, len(tokens))
        self.seq.tokens = list(tokens)
        self.last_ntok = 0

    def generate_greedy(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        *,
        stop_token_ids: frozenset[int] = frozenset(),
        on_token=None,
        prefix_cache=None,
    ) -> tuple[list[int], QwenGenerationStats]:
        if not prompt_tokens:
            raise ConfigurationError("prompt_tokens must not be empty")
        if len(prompt_tokens) + max_new_tokens > self.max_context:
            raise ConfigurationError(
                f"prompt ({len(prompt_tokens)}) + max_new_tokens ({max_new_tokens}) exceeds max_context {self.max_context}"
            )
        stats = QwenGenerationStats(prompt_tokens=len(prompt_tokens))
        streamed_start = self.rt.streamed_bytes()
        t0 = time.perf_counter()
        start = prefix_cache.begin(prompt_tokens) if prefix_cache is not None else 0
        if prefix_cache is None:
            self.reset()
        self.prefill(prompt_tokens, start=start,
                     before_last=(lambda: prefix_cache.checkpoint(prompt_tokens)) if prefix_cache is not None else None)
        stats.cached_tokens = start
        token = self.greedy_next()
        stats.prefill_seconds = time.perf_counter() - t0
        out = [token]
        if on_token is not None:
            on_token(token)
        while len(out) < max_new_tokens and token not in stop_token_ids:
            t1 = time.perf_counter()
            self.forward_token(token)
            token = self.greedy_next()
            stats.step_seconds.append(time.perf_counter() - t1)
            out.append(token)
            if on_token is not None:
                on_token(token)
        stats.decode_seconds = sum(stats.step_seconds)
        stats.generated_tokens = len(out)
        stats.streamed_bytes = self.rt.streamed_bytes() - streamed_start
        return out, stats
