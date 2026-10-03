"""Prompt prefix cache for the per-op executor.

Agentic clients resend the whole conversation every turn, so each prompt usually extends an earlier one.
The cache keeps checkpoints of the sequence state taken just before the last prefill pass of each prompt
(at most prefill_batch tokens short of its end); a later prompt that starts with a checkpoint's tokens
resumes from it and only computes the rest.

A checkpoint holds the SSM/conv state (host copy; recurrent state cannot be rewound) and needs the
attention KV rows [0, L). Those stay on the device while the live sequence still starts with the
checkpoint's tokens; a host copy is taken only before a request is about to overwrite them (e.g. a
side request with a different prompt), and uploaded again when the checkpoint is resumed.

A checkpoint at L also records the prompt token at L: the MTP drafter's KV row L-1 was computed from
it, so a prompt matches a checkpoint only if it agrees on L + 1 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Checkpoint:
    tokens: tuple[int, ...]  # live sequence [0, L)
    key: tuple[int, ...]  # tokens + the prompt token at L
    recurrent: dict[str, bytes]
    kv: dict[str, list[bytes]] | None = None  # host copy of KV rows [0, len(tokens)), when taken
    last_used: int = 0
    nbytes: int = 0

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class PrefixCacheStats:
    requests: int = 0
    hits: int = 0
    reused_tokens: int = 0
    kv_saves: int = 0
    kv_loads: int = 0
    checkpoints: list[int] = field(default_factory=list)


class PrefixCache:
    def __init__(self, executor, *, max_bytes: int = 8 * 1024**3, max_entries: int = 8, min_tokens: int = 64) -> None:
        self.ex = executor
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.min_tokens = min_tokens  # shorter prompts are not worth a checkpoint
        self.entries: list[Checkpoint] = []
        self.prompt: list[int] = []
        self.clock = 0
        self.stats = PrefixCacheStats()
        self._kv_token = executor.kv_bytes_per_token()

    def _bytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    def _device_holds(self, cp: Checkpoint) -> bool:
        """The device KV rows [0, L) belong to cp: the live sequence starts with cp's tokens."""
        live = self.ex.tokens
        return len(live) > len(cp) and tuple(live[: len(cp.key)]) == cp.key

    def begin(self, prompt: list[int]) -> int:
        """Prepare the executor for `prompt`; returns the position prefill starts from (0 = no reuse)."""
        self.clock += 1
        self.stats.requests += 1
        self.prompt = list(prompt)
        best = None
        for cp in self.entries:
            if len(cp.key) <= len(prompt) and (best is None or len(cp) > len(best)) and tuple(prompt[: len(cp.key)]) == cp.key:
                best = cp
        start = len(best) if best is not None else 0
        # The device KV is about to be rewritten except for the rows the prompt shares with the resumed
        # checkpoint: keep host copies of the other checkpoints that still rely on device rows.
        for cp in self.entries:
            kept = len(cp) <= start and tuple(prompt[: len(cp.key)]) == cp.key
            if cp is not best and cp.kv is None and not kept and self._device_holds(cp):
                cp.kv = self.ex.save_kv(len(cp))
                cp.nbytes += len(cp) * self._kv_token
                self.stats.kv_saves += 1
        if best is None:
            self.ex.reset()
            self._evict()
            return 0
        on_device = self._device_holds(best)
        if not on_device and best.kv is None:  # cannot happen if every overwrite was preceded by a save
            self.entries.remove(best)
            self.ex.reset()
            return 0
        self.ex.resume(list(best.tokens), best.recurrent, None if on_device else best.kv)
        if not on_device:
            self.stats.kv_loads += 1
        best.last_used = self.clock
        self.stats.hits += 1
        self.stats.reused_tokens += start
        self._evict()
        return start

    def checkpoint(self) -> None:
        """Record the live state (called by prefill before its last pass)."""
        tokens = tuple(self.ex.tokens)
        if len(tokens) < self.min_tokens or len(tokens) >= len(self.prompt) or tuple(self.prompt[: len(tokens)]) != tokens:
            return
        key = tokens + (self.prompt[len(tokens)],)
        if any(cp.key == key for cp in self.entries):
            return
        recurrent = self.ex.save_recurrent()
        cp = Checkpoint(tokens, key, recurrent, last_used=self.clock,
                        nbytes=sum(len(v) for v in recurrent.values()))
        self.entries.append(cp)
        self.stats.checkpoints.append(len(tokens))
        self._evict()

    def _evict(self) -> None:
        while self.entries and (len(self.entries) > self.max_entries or self._bytes() > self.max_bytes):
            self.entries.remove(min(self.entries, key=lambda e: e.last_used))

    def clear(self) -> None:
        self.entries.clear()
