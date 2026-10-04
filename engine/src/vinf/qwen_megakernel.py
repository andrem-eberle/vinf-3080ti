"""qwen35 fused megakernel executor (Phase 31b).

Builds per-SM instruction queues for one decode token from the qwen35 layer schedule and
runs them in a single cooperative launch. Reuses a `QwenGpuExecutor` for weights, buffers,
KV cache, and SSM state, so the per-op path remains available as the debug/fallback path.

Scheduling invariants:
- Instructions are generated in one global topological order and appended to SM queues in
  that order, so the earliest unfinished instruction is always runnable (no deadlock).
- Each producer group signals one counter; consumers wait for (counter, number of producers).
- Streamed weights fill K ring slots; item i uses slot i % K, and the fill of item i + K
  waits until every consumer of item i has signalled the slot's consumed counter.
  streaming="dma" (default): the copy engine fills slots from a host-enqueued copy plan
  (stream wait/write-value ops on the same counters); all SMs compute.
  streaming="sm": LOAD instructions on dedicated loader SMs (slower: SM reads of
  host-mapped memory top out well below DMA bandwidth).
"""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field

from vinf.errors import ConfigurationError, DecodeError, ExecutorUnavailableError, UnsupportedModelError
from vinf.gguf.qwen_tensors import load_qwen_token_embeddings
from vinf.gguf.residency import TensorResidency
from vinf.qwen_gpu import HEAD_ORDERS, QwenGenerationStats, QwenGpuExecutor, qwen_qmv_order
from vinf.qwen_ops import QwenLayerKind
from vinf.runtime.instructions import (
    INTS_PER_INSTRUCTION,
    QMV_FLAG_ADD,
    QMV_FLAG_ARGMAX,
    QMV_FLAG_NORM,
    QMV_FLAG_SILU_MUL,
    Instruction,
    NoOp,
    QwenArgmax,
    QwenAttnHead,
    QwenLoad,
    QwenQmv,
    QwenSsmGroup,
)


# Defaults of VINF_MK_WARPS / VINF_MK_WARP_BUF in csrc/megakernel/qwen_megakernel.cu; the
# executor reads the compiled values from device_info().
MK_WARPS = 16
MK_WARP_BUF_BYTES = 1024
# GGUF types the megakernel matvec stages as int8 x (dp4a); others stage float x.
INT8_PATH_TYPES = frozenset({8, 11, 12, 13, 14, 20, 21, 23})


def _load_module():
    try:
        from vinf import _cuda_qwen_megakernel
    except ImportError as exc:
        raise ExecutorUnavailableError(
            "vinf._cuda_qwen_megakernel is not built; run `make cuda-qwen-megakernel`"
        ) from exc
    return _cuda_qwen_megakernel


@dataclass(slots=True)
class QwenMegakernelProgram:
    queues: list[list[Instruction]]
    num_counters: int
    num_partials: int
    tensor_names: list[str]
    buffer_names: list[str]
    stream_order: list[str]
    smem_bytes: int
    compute_sms: int
    loader_sms: int
    counter_names: list[str] = field(default_factory=list)
    copy_plan: list[tuple[int, int, int, int, int, int]] = field(default_factory=list)

    @property
    def queue_len(self) -> int:
        return max(len(q) for q in self.queues)

    @property
    def instruction_count(self) -> int:
        return sum(len(q) for q in self.queues)

    def tensorize(self) -> bytes:
        words = array("i")
        qlen = self.queue_len
        for queue in self.queues:
            for ins in queue + [NoOp()] * (qlen - len(queue)):
                words.extend(ins.serialize())
        return words.tobytes()


class _ProgramBuilder:
    def __init__(
        self,
        ex: QwenGpuExecutor,
        num_sms: int,
        loader_sms: int,
        stream_slots: int,
        load_parts: int,
        dma: bool,
        warps: int = MK_WARPS,
        warp_buf_bytes: int = MK_WARP_BUF_BYTES,
    ) -> None:
        self.warps = warps
        self.warp_buf_bytes = warp_buf_bytes
        self.max_stage_bytes = 0
        self.dma = dma
        self.copy_plan: list[tuple[int, int, int, int, int, int]] = []
        self.ex = ex
        self.compute_sms = num_sms - loader_sms
        self.loader_sms = loader_sms
        if self.compute_sms <= 0:
            raise ConfigurationError("megakernel needs at least one compute SM")
        self.queues: list[list[Instruction]] = [[] for _ in range(num_sms)]
        self.load_cost = [0] * self.compute_sms
        self.rr = 0
        self.tensors: dict[str, int] = {}
        self.buffers: dict[str, int] = {}
        self.counter_names: list[str] = []
        self.num_partials = 0
        self.max_cols = 0
        streamed = set(ex.plan.names(TensorResidency.PINNED_STREAM))
        self.stream_order = [name for name in qwen_qmv_order(ex.schedule) if name in streamed]
        self.stream_index = {name: idx for idx, name in enumerate(self.stream_order)}
        self.slots = min(stream_slots, len(self.stream_order))
        self.load_parts = load_parts
        self.ready: dict[int, tuple[int, int]] = {}
        self.loader_cursor = 0
        if self.stream_order and not dma and loader_sms <= 0:
            raise ConfigurationError("streamed weights require at least one loader SM")

    # ---- tables -------------------------------------------------------------------------

    def tensor(self, name: str) -> int:
        if name not in self.tensors:
            self.tensors[name] = len(self.tensors)
        return self.tensors[name]

    def buffer(self, name: str) -> int:
        if name not in self.buffers:
            self.buffers[name] = len(self.buffers)
        return self.buffers[name]

    def counter(self, name: str) -> int:
        self.counter_names.append(name)
        return len(self.counter_names) - 1

    # ---- placement ----------------------------------------------------------------------

    def _place(self, ins: Instruction, cost: int, sm: int | None = None) -> None:
        # Round-robin across compute SMs: consecutive parts of one phase land on distinct SMs,
        # so a phase finishes in about one part-time instead of piling onto lightly loaded SMs.
        if sm is None:
            sm = self.rr % self.compute_sms
            self.rr += 1
        self.load_cost[sm] += cost
        self.queues[sm].append(ins)

    def _emit_load(self, item: int, wait: tuple[int, int]) -> None:
        name = self.stream_order[item]
        info = self.ex.rt.tensor_info(name)
        nbytes = info[5]
        ready = self.counter(f"ready:{name}")
        if self.dma:
            self.copy_plan.append((info[0], nbytes, item % self.slots, wait[0], wait[1], ready))
            self.ready[item] = (ready, 1)
            return
        parts = max(1, min(self.load_parts, nbytes // 4096))
        step = ((nbytes // parts) + 15) // 16 * 16
        emitted = 0
        for part in range(parts):
            start = part * step
            end = nbytes if part == parts - 1 else min(nbytes, (part + 1) * step)
            if start >= end:
                continue
            sm = self.compute_sms + (self.loader_cursor % self.loader_sms)
            self.loader_cursor += 1
            self.queues[sm].append(
                QwenLoad(
                    wait0_counter=wait[0],
                    wait0_target=wait[1],
                    signal=ready,
                    tensor=self.tensor(name),
                    slot=item % self.slots,
                    byte_start=start,
                    byte_end=end,
                )
            )
            emitted += 1
        self.ready[item] = (ready, emitted)

    def start(self) -> None:
        for item in range(self.slots):
            self._emit_load(item, (-1, 0))

    def qmv(
        self,
        weight: str,
        *,
        x: str,
        y: str,
        wait: tuple[int, int],
        signal: int,
        flags: int = 0,
        norm: str | None = None,
        x2: str | None = None,
        parts: int | None = None,
        argmax: bool = False,
    ) -> int:
        info = self.ex.rt.tensor_info(weight)
        rows, cols, row_bytes = info[2], info[3], info[4]
        self.max_cols = max(self.max_cols, cols)
        stage = cols + cols // 32 * 8 if info[1] in INT8_PATH_TYPES else cols * 4
        self.max_stage_bytes = max(self.max_stage_bytes, stage)
        if parts is None:
            parts = max(1, min(self.compute_sms, rows // 8))
        item = self.stream_index.get(weight)
        slot = -1
        wait1 = (-1, 0)
        consumed = -1
        if item is not None:
            slot = item % self.slots
            wait1 = self.ready[item]
            consumed = self.counter(f"consumed:{weight}")
        emitted = 0
        for part in range(parts):
            r0 = rows * part // parts
            r1 = rows * (part + 1) // parts
            if r0 >= r1:
                continue
            partial = -1
            if argmax:
                partial = self.num_partials
                self.num_partials += 1
            self._place(
                QwenQmv(
                    wait0_counter=wait[0],
                    wait0_target=wait[1],
                    wait1_counter=wait1[0],
                    wait1_target=wait1[1],
                    signal=signal,
                    tensor=self.tensor(weight),
                    slot=slot,
                    x=self.buffer(x),
                    y=self.buffer(y),
                    row_start=r0,
                    row_end=r1,
                    flags=flags | (QMV_FLAG_ARGMAX if argmax else 0),
                    norm=self.tensor(norm) if norm else -1,
                    partial=partial,
                    consumed_signal=consumed,
                    x2=self.buffer(x2) if x2 else -1,
                ),
                (r1 - r0) * row_bytes,
            )
            emitted += 1
        if item is not None and item + self.slots < len(self.stream_order):
            self._emit_load(item + self.slots, (consumed, emitted))
        return emitted

    # ---- layer program ------------------------------------------------------------------

    def build(self) -> QwenMegakernelProgram:
        ex, s = self.ex, self.ex.shapes
        self.start()
        prev = (-1, 0)
        for layer_idx, kind in enumerate(ex.schedule):
            p = f"blk.{layer_idx}."
            c_in = self.counter(f"L{layer_idx}:in")
            if kind is QwenLayerKind.FULL_ATTENTION:
                n = 0
                for weight, out in (("attn_q.weight", "q_raw"), ("attn_k.weight", "k"), ("attn_v.weight", "v")):
                    n += self.qmv(p + weight, x="h", y=out, wait=prev, signal=c_in, flags=QMV_FLAG_NORM, norm=p + "attn_norm.weight")
                c_mix = self.counter(f"L{layer_idx}:attn")
                for head in range(s.heads):
                    self._place(
                        QwenAttnHead(
                            wait0_counter=c_in,
                            wait0_target=n,
                            signal=c_mix,
                            kc=self.buffer(f"kc.{layer_idx}"),
                            vc=self.buffer(f"vc.{layer_idx}"),
                            head=head,
                            q_raw=self.buffer("q_raw"),
                            k=self.buffer("k"),
                            v=self.buffer("v"),
                            out=self.buffer("attn"),
                            q_norm=self.tensor(p + "attn_q_norm.weight"),
                            k_norm=self.tensor(p + "attn_k_norm.weight"),
                        ),
                        cost=s.head_dim * (ex.max_context + 1) * 8,
                    )
                mixed, out_weight, out_x = (c_mix, s.heads), p + "attn_output.weight", "attn"
            else:
                n = 0
                for weight, out in (
                    ("attn_qkv.weight", "lin_qkv"),
                    ("attn_gate.weight", "lin_z"),
                    ("ssm_beta.weight", "lin_beta"),
                    ("ssm_alpha.weight", "lin_alpha"),
                ):
                    n += self.qmv(p + weight, x="h", y=out, wait=prev, signal=c_in, flags=QMV_FLAG_NORM, norm=p + "attn_norm.weight")
                c_mix = self.counter(f"L{layer_idx}:ssm")
                rep = s.value_heads // s.key_heads
                for key_head in range(s.key_heads):
                    self._place(
                        QwenSsmGroup(
                            wait0_counter=c_in,
                            wait0_target=n,
                            signal=c_mix,
                            key_head=key_head,
                            qkv=self.buffer("lin_qkv"),
                            z=self.buffer("lin_z"),
                            beta=self.buffer("lin_beta"),
                            alpha=self.buffer("lin_alpha"),
                            conv_state=self.buffer(f"conv.{layer_idx}"),
                            ssm_state=self.buffer(f"ssm.{layer_idx}"),
                            out=self.buffer("lin_out"),
                            conv_w=self.tensor(p + "ssm_conv1d.weight"),
                            ssm_a=self.tensor(p + "ssm_a"),
                            dt_bias=self.tensor(p + "ssm_dt.bias"),
                            ssm_norm=self.tensor(p + "ssm_norm.weight"),
                        ),
                        cost=rep * s.key_head_dim * s.value_head_dim * 16,
                    )
                mixed, out_weight, out_x = (c_mix, s.key_heads), p + "ssm_out.weight", "lin_out"
            c_out = self.counter(f"L{layer_idx}:out")
            n = self.qmv(out_weight, x=out_x, y="h", wait=mixed, signal=c_out, flags=QMV_FLAG_ADD)
            c_mlp = self.counter(f"L{layer_idx}:mlp")
            m = self.qmv(p + "ffn_gate.weight", x="h", y="mlp_gate", wait=(c_out, n), signal=c_mlp,
                         flags=QMV_FLAG_NORM, norm=p + "post_attention_norm.weight")
            m += self.qmv(p + "ffn_up.weight", x="h", y="mlp_up", wait=(c_out, n), signal=c_mlp,
                          flags=QMV_FLAG_NORM, norm=p + "post_attention_norm.weight")
            c_down = self.counter(f"L{layer_idx}:down")
            d = self.qmv(p + "ffn_down.weight", x="mlp_gate", x2="mlp_up", y="h", wait=(c_mlp, m), signal=c_down,
                         flags=QMV_FLAG_SILU_MUL | QMV_FLAG_ADD)
            prev = (c_down, d)
        c_lm = self.counter("lm_head")
        n = self.qmv("output.weight", x="h", y="logits", wait=prev, signal=c_lm, flags=QMV_FLAG_NORM,
                     norm="output_norm.weight", argmax=True)
        self._place(QwenArgmax(wait0_counter=c_lm, wait0_target=n, partials=self.num_partials), cost=0)
        rep = s.value_heads // s.key_heads if s.key_heads else 0
        smem = max(
            self.warps * self.warp_buf_bytes + self.max_stage_bytes + 64,
            (4 * s.head_dim + ex.max_context + 32) * 4,
            (2 * s.key_head_dim + 2 * rep * s.value_head_dim) * 4,
        )
        return QwenMegakernelProgram(
            queues=self.queues,
            num_counters=len(self.counter_names),
            num_partials=self.num_partials,
            tensor_names=list(self.tensors),
            buffer_names=list(self.buffers),
            stream_order=self.stream_order,
            smem_bytes=smem,
            compute_sms=self.compute_sms,
            loader_sms=self.loader_sms,
            counter_names=self.counter_names,
            copy_plan=self.copy_plan,
        )


class QwenMegakernelExecutor:
    """One cooperative megakernel launch per token on top of a loaded `QwenGpuExecutor`."""

    def __init__(
        self,
        base: QwenGpuExecutor,
        *,
        streaming: str = "dma",
        loader_sms: int = 8,
        load_parts: int | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        if streaming not in ("dma", "sm"):
            raise ConfigurationError("streaming must be 'dma' or 'sm'")
        if base.cpu_layers:
            raise UnsupportedModelError("the megakernel needs every layer on the GPU; create the executor with placement='stream'")
        if getattr(base, "kv_dtype", "f32") != "f32":
            raise UnsupportedModelError("the megakernel reads fp32 KV caches; create the executor with kv_dtype='f32'")
        if base.head_order != HEAD_ORDERS["tiled"]:
            raise UnsupportedModelError("the megakernel implements the tiled SSM value-head order only")
        if base.max_seqs != 1 or base.page_size != base.max_context:
            raise UnsupportedModelError("the megakernel needs one sequence with a single KV page (page_size = max_context)")
        self.base = base
        base.ensure_pages(base.seq, base.max_context)  # one page = the contiguous cache layout the megakernel reads
        self.mk = _load_module().Megakernel(_iq3_s_grid_bytes())
        info = self.mk.device_info()
        if not info["cooperative"]:
            raise ExecutorUnavailableError("device does not support cooperative launch")
        num_sms = info["sms"]
        streamed = bool(base.plan.names(TensorResidency.PINNED_STREAM))
        dma = streaming == "dma"
        if dma and not info["stream_mem_ops"]:
            raise ExecutorUnavailableError("device lacks stream memory operations; use streaming='sm'")
        loaders = loader_sms if streamed and not dma else 0
        # The per-op prefetch ring is replaced by megakernel slots of the same count and size.
        base.rt.set_stream_order([], 0)
        builder = _ProgramBuilder(
            base, num_sms, loaders, base.stream_slots, load_parts or max(1, loaders), dma,
            warps=info["warps"], warp_buf_bytes=info["warp_buf_bytes"],
        )
        self.program = builder.build()
        if self.program.smem_bytes > info["max_smem_optin"] - 4096:
            raise UnsupportedModelError(
                f"megakernel needs {self.program.smem_bytes} bytes of shared memory; "
                f"device allows {info['max_smem_optin']}"
            )
        slot_bytes = max((base.rt.tensor_info(name)[5] for name in self.program.stream_order), default=0)
        # Batched prefill runs on the per-op path (shared buffers/caches); give it one staging buffer.
        base.rt.set_staging(slot_bytes)
        tensors = []
        for name in self.program.tensor_names:
            addr, typ, rows, cols, row_bytes, _nbytes, _resident = base.rt.tensor_info(name)
            tensors.append((addr, typ, rows, cols, row_bytes))
        buffers = [base.rt.buffer_ptr(name)[0] for name in self.program.buffer_names]
        s = base.shapes
        params = {
            "max_ctx": base.max_context,
            "heads": s.heads,
            "kv_heads": s.kv_heads,
            "hd": s.head_dim,
            "rot": s.rotary_dim,
            "freq_base": float(s.freq_base),
            "eps": float(s.eps),
            "key_heads": s.key_heads,
            "value_heads": s.value_heads,
            "kd": s.key_head_dim,
            "vd": s.value_head_dim,
            "conv_k": s.conv_kernel,
            "timeout_cycles": int(timeout_seconds * info["clock_khz"] * 1000),
        }
        self.mk.configure(
            self.program.tensorize(),
            num_sms,
            self.program.queue_len,
            tensors,
            buffers,
            builder.slots,
            slot_bytes,
            self.program.num_counters,
            self.program.num_partials,
            params,
            self.program.smem_bytes,
            self.program.copy_plan,
        )
        self.streamed_bytes_per_token = sum(base.rt.tensor_info(name)[5] for name in self.program.stream_order)

    @property
    def position(self) -> int:
        return self.base.position

    def reset(self) -> None:
        self.base.reset()

    def step(self, token_id: int) -> int:
        """Forward one token at the current position; returns the greedy next token."""
        base = self.base
        if base.position >= base.max_context:
            raise ConfigurationError(f"context is full (max_context={base.max_context})")
        embedding = load_qwen_token_embeddings(base.gguf, [token_id])[0]
        base.rt.write("h", array("f", embedding).tobytes())
        token, error, block, instr = self.mk.run(base.position)
        if error:
            reason = "dependency wait timed out" if error == 1 else "invalid opcode"
            raise DecodeError(f"megakernel aborted ({reason}) at SM {block}, queue entry {instr}")
        base.seq.tokens.append(token_id)
        return token

    def logits(self) -> list[float]:
        """Logits of the last step (explicit full-vocab transfer; debugging/parity only)."""
        return self.base.rt.read_floats("logits", self.base.shapes.vocab)

    def hidden(self) -> list[float]:
        return self.base.rt.read_floats("h", self.base.shapes.hidden)

    def generate_greedy(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        *,
        stop_token_ids: frozenset[int] = frozenset(),
        on_token=None,
    ) -> tuple[list[int], QwenGenerationStats]:
        base = self.base
        if not prompt_tokens:
            raise ConfigurationError("prompt_tokens must not be empty")
        if len(prompt_tokens) + max_new_tokens > base.max_context:
            raise ConfigurationError(
                f"prompt ({len(prompt_tokens)}) + max_new_tokens ({max_new_tokens}) exceeds max_context {base.max_context}"
            )
        self.reset()
        stats = QwenGenerationStats(prompt_tokens=len(prompt_tokens))
        t0 = time.perf_counter()
        # Batched prefill (per-op path, one weight read per chunk), then one launch per decoded token.
        base.prefill(prompt_tokens)
        token = base.greedy_next()
        stats.prefill_seconds = time.perf_counter() - t0
        out = [token]
        if on_token is not None:
            on_token(token)
        while len(out) < max_new_tokens and token not in stop_token_ids:
            t1 = time.perf_counter()
            token = self.step(token)
            stats.step_seconds.append(time.perf_counter() - t1)
            out.append(token)
            if on_token is not None:
                on_token(token)
        stats.decode_seconds = sum(stats.step_seconds)
        stats.generated_tokens = len(out)
        stats.streamed_bytes = self.streamed_bytes_per_token * (len(prompt_tokens) + len(stats.step_seconds))
        return out, stats


def _iq3_s_grid_bytes() -> bytes:
    from vinf.cuda.qwen_runtime import _iq3_s_grid_bytes as grid

    return grid()
