from __future__ import annotations

from typing import Protocol

from vinf.executors.base import DraftProposal, VerificationResult
from vinf.runtime.state import RuntimeState
from vinf.speculative.sampler import SpeculativeDecision


class DraftRunner(Protocol):
    def propose(self, state: RuntimeState, gamma: int) -> DraftProposal:
        ...


class TargetVerifier(Protocol):
    def verify_many(
        self, state: RuntimeState, draft_tokens: tuple[int, ...]
    ) -> VerificationResult:
        ...


class KVCommitManager(Protocol):
    def apply(self, state: RuntimeState, decision: SpeculativeDecision) -> None:
        ...
