from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from vinf.executors.base import DraftProposal
from vinf.models.metadata import ModelMetadata
from vinf.runtime.kv_cache import LogicalKVCache
from vinf.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class DraftStats:
    calls: int = 0
    tokens_proposed: int = 0
    elapsed_seconds: float = 0.0

    @property
    def seconds_per_token(self) -> float:
        if self.tokens_proposed == 0:
            return 0.0
        return self.elapsed_seconds / self.tokens_proposed


@dataclass(slots=True)
class HeuristicDraftRunner:
    metadata: ModelMetadata
    kv_cache: LogicalKVCache
    strategy: str = "repeat_last"
    last_token_ids: tuple[int, ...] = ()
    last_probability_rows: tuple[tuple[float, ...], ...] = ()
    stats: DraftStats = field(default_factory=DraftStats)

    def __post_init__(self) -> None:
        if self.strategy != "repeat_last":
            raise ValueError("only repeat_last heuristic draft strategy is supported")
        if self.kv_cache.max_speculative_tokens <= 0:
            raise ValueError("draft kv_cache must reserve speculative capacity")

    def prefill(self, state: RuntimeState) -> None:
        if len(state.prompt_tokens) > self.kv_cache.max_seq_len:
            raise ValueError("prompt length exceeds draft kv cache max_seq_len")
        self.kv_cache.committed_len = len(state.prompt_tokens)
        self.kv_cache.spec_len = 0

    def propose(self, state: RuntimeState, gamma: int) -> DraftProposal:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if gamma > self.kv_cache.max_speculative_tokens:
            raise ValueError("gamma exceeds draft kv speculative capacity")
        started = perf_counter()
        token = self._next_token(state)
        tokens = tuple(token for _ in range(gamma))
        rows = tuple(self._one_hot_row(token) for _ in range(gamma))
        self.kv_cache.write_speculative(gamma)
        self.last_token_ids = tokens
        self.last_probability_rows = rows
        self.stats = DraftStats(
            calls=self.stats.calls + 1,
            tokens_proposed=self.stats.tokens_proposed + gamma,
            elapsed_seconds=self.stats.elapsed_seconds + (perf_counter() - started),
        )
        return DraftProposal(token_ids=tokens, probability_rows_ref=[list(row) for row in rows])

    def rollback_to(self, committed_len: int) -> None:
        self.kv_cache.rollback_to(committed_len)

    def discard_speculative(self) -> None:
        self.kv_cache.discard_speculative()

    def _next_token(self, state: RuntimeState) -> int:
        tokens = state.all_tokens
        if not tokens:
            return 0
        token = tokens[-1]
        if token < 0 or token >= self.metadata.vocab_size:
            raise ValueError("last token outside draft vocabulary")
        return token

    def _one_hot_row(self, token: int) -> tuple[float, ...]:
        return tuple(1.0 if idx == token else 0.0 for idx in range(self.metadata.vocab_size))


def validate_draft_tokenizer_compatibility(
    target: ModelMetadata, draft: ModelMetadata
) -> None:
    if target.tokenizer_id != draft.tokenizer_id:
        raise ValueError("target and draft tokenizer_id must match")
    if target.vocab_size != draft.vocab_size:
        raise ValueError("target and draft vocab_size must match")
