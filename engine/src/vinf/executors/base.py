from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vinf.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class DecodeStep:
    token_id: int
    logits_ref: object | None = None
    probabilities_ref: object | None = None


@dataclass(frozen=True, slots=True)
class DraftProposal:
    token_ids: tuple[int, ...]
    probability_rows_ref: object | None = None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    probability_rows_ref: object
    speculative_kv_ref: object | None = None


class TargetExecutor(Protocol):
    def prefill(self, state: RuntimeState) -> None:
        ...

    def decode_one(self, state: RuntimeState) -> DecodeStep:
        ...

    def verify_many(
        self, state: RuntimeState, draft_tokens: tuple[int, ...]
    ) -> VerificationResult:
        ...


class DraftExecutor(Protocol):
    def prefill(self, state: RuntimeState) -> None:
        ...

    def propose(self, state: RuntimeState, gamma: int) -> DraftProposal:
        ...

