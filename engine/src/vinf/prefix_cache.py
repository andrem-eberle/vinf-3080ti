"""Prompt prefix cache for the per-op executor (multi-sequence).

Agentic clients resend the whole conversation every turn, so each prompt usually extends an earlier one.
The cache keeps checkpoints of a sequence's state taken just before the last prefill pass of each prompt
(at most prefill_batch tokens short of its end); a later prompt that starts with a checkpoint's tokens
resumes from it and only computes the rest.

A checkpoint holds the SSM/conv state (host copy; recurrent state cannot be rewound) and needs the
attention KV rows [0, L). Those stay in the pages of the sequence that produced them for as long as that
sequence's tokens still start with the checkpoint's; finished sequences stay resident (idle) so the next
turn of their conversation resumes in place. A host copy of the rows is taken only before they would be
lost (the sequence is reused for a diverging prompt, or evicted to free a slot or pages). A prompt that
matches a checkpoint held by a busy sequence (a concurrent request sharing a prefix) gets a device copy.

A checkpoint at L also records the prompt token at L: the MTP drafter's KV row L-1 was computed from it,
so a prompt matches a checkpoint only if it agrees on L + 1 tokens.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from vinf.qwen_gpu import KvPoolExhausted, Sequence


@dataclass(eq=False)
class Checkpoint:
    tokens: tuple[int, ...]  # sequence state [0, L)
    key: tuple[int, ...]  # tokens + the prompt token at L
    recurrent: dict[str, bytes]
    seq: Sequence | None = None  # sequence whose pages hold rows [0, L), while it does
    kv: dict[str, list[bytes]] | None = None  # host copy of KV rows [0, L), when taken
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
    kv_copies: int = 0
    evictions: int = 0
    checkpoints: list[int] = field(default_factory=list)


class PrefixCache:
    def __init__(self, executor, *, max_bytes: int = 8 * 1024**3, max_entries: int = 16, min_tokens: int = 64) -> None:
        self.ex = executor
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.min_tokens = min_tokens  # shorter prompts are not worth a checkpoint
        self.entries: list[Checkpoint] = []
        self.clock = 0
        self.stats = PrefixCacheStats()
        self._kv_token = executor.kv_bytes_per_token()
        executor.reclaim = self.reclaim

    # ---- helpers ------------------------------------------------------------------------------

    @contextmanager
    def _active(self, seq: Sequence):
        prev = self.ex.seq
        self.ex.activate(seq)
        try:
            yield
        finally:
            self.ex.activate(prev)

    def _bytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    def _holds(self, cp: Checkpoint) -> bool:
        """cp's KV rows are live in its sequence's pages."""
        seq = cp.seq
        if seq is None or seq not in self.ex.sequences:
            return False
        n = len(cp)
        if tuple(seq.tokens[:n]) != cp.tokens:
            return False
        return len(seq.tokens) == n or seq.tokens[n] == cp.key[-1]

    def _save_kv(self, cp: Checkpoint) -> None:
        if cp.kv is None and self._holds(cp):
            with self._active(cp.seq):
                cp.kv = self.ex.save_kv(len(cp))
            cp.nbytes += len(cp) * self._kv_token
            self.stats.kv_saves += 1
        cp.seq = None

    def _protect(self, seq: Sequence, keep_from: int, prompt: list[int], skip: Checkpoint | None = None) -> None:
        """seq's KV rows from keep_from on are about to be rewritten for `prompt`: host-copy checkpoints
        that need them."""
        for cp in self.entries:
            if cp is skip or cp.seq is not seq:
                continue
            kept = len(cp) <= keep_from and tuple(prompt[: len(cp.key)]) == cp.key
            if not kept:
                self._save_kv(cp)

    def evict_sequence(self, seq: Sequence) -> None:
        """Free an idle sequence (host-copying the KV rows its checkpoints still need)."""
        for cp in self.entries:
            if cp.seq is seq:
                self._save_kv(cp)
        self.ex.release_sequence(seq)
        self.stats.evictions += 1

    def _idle(self) -> list[Sequence]:
        return sorted((s for s in self.ex.sequences if not s.busy), key=lambda s: s.last_used)

    def reclaim(self, pages_needed: int) -> bool:
        """Executor callback when the KV pool is short: evict idle sequences (least recently used first)."""
        freed = False
        for seq in self._idle():  # busy sequences (running requests) are never evicted
            self.evict_sequence(seq)
            freed = True
            if self.ex.free_page_count() >= pages_needed:
                break
        return freed

    def _new_sequence(self, avoid: Sequence | None = None) -> Sequence:
        while True:
            try:
                return self.ex.new_sequence()
            except KvPoolExhausted:
                idle = [s for s in self._idle() if s is not avoid]
                if not idle:
                    raise
                self.evict_sequence(idle[0])

    # ---- requests -----------------------------------------------------------------------------

    def acquire(self, prompt: list[int]) -> tuple[Sequence, int]:
        """A busy, active sequence for `prompt` and the position prefill starts from (0 = no reuse)."""
        self.clock += 1
        self.stats.requests += 1
        best = None
        for cp in self.entries:
            if len(cp.key) <= len(prompt) and (best is None or len(cp) > len(best)) and tuple(prompt[: len(cp.key)]) == cp.key:
                best = cp
        if best is not None and not self._holds(best) and best.kv is None:
            self.entries.remove(best)  # its rows are gone
            best = None
        start = len(best) if best is not None else 0
        if best is not None and self._holds(best) and not best.seq.busy:
            seq = best.seq  # resume in place
            self._protect(seq, start, prompt, skip=best)
            with self._active(seq):
                self.ex.resume(list(best.tokens), best.recurrent, None)
        else:
            seq = self._new_sequence(avoid=best.seq if best is not None else None)
            self.ex.activate(seq)
            if best is None:
                self.ex.reset()
            elif self._holds(best):  # held by a busy sequence: copy its rows on the device
                self.ex.copy_kv(best.seq, seq, start)
                self.ex.resume(list(best.tokens), best.recurrent, None)
                self.stats.kv_copies += 1
            else:
                self.ex.resume(list(best.tokens), best.recurrent, best.kv)
                self.stats.kv_loads += 1
        if best is not None:
            best.last_used = self.clock
            self.stats.hits += 1
            self.stats.reused_tokens += start
        seq.busy = True
        seq.last_used = self.clock
        self.ex.activate(seq)
        self._evict_entries()
        return seq, start

    def begin(self, prompt: list[int]) -> int:
        """Single-request API: the previous request's sequence becomes idle; returns the start position."""
        for seq in self.ex.sequences:
            seq.busy = False
        _, start = self.acquire(prompt)
        return start

    def release(self, seq: Sequence) -> None:
        """The request using seq finished: it stays resident (idle) for the next turn."""
        seq.busy = False
        seq.last_used = self.clock

    def checkpoint(self, full: list[int] | None = None) -> None:
        """Record the active sequence's state; full = the token sequence it is processing (its token at
        the current position is the lookahead). Called before the last prefill pass, or to swap out."""
        seq = self.ex.seq
        tokens = tuple(seq.tokens)
        if full is None:
            return
        if len(tokens) < self.min_tokens or len(tokens) >= len(full) or tuple(full[: len(tokens)]) != tokens:
            return
        key = tokens + (full[len(tokens)],)
        for cp in self.entries:
            if cp.key == key:
                if not self._holds(cp):
                    cp.seq = seq if cp.kv is None else cp.seq
                cp.last_used = self.clock
                return
        recurrent = self.ex.save_recurrent()
        cp = Checkpoint(tokens, key, recurrent, seq=seq, last_used=self.clock,
                        nbytes=sum(len(v) for v in recurrent.values()))
        self.entries.append(cp)
        self.stats.checkpoints.append(len(tokens))
        self._evict_entries()

    def _evict_entries(self) -> None:
        while self.entries and (len(self.entries) > self.max_entries or self._bytes() > self.max_bytes):
            self.entries.remove(min(self.entries, key=lambda e: e.last_used))

    def clear(self) -> None:
        self.entries.clear()
