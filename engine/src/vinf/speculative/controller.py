from __future__ import annotations

from vinf.runtime.state import RuntimeState
from vinf.speculative.interfaces import DraftRunner, KVCommitManager, TargetVerifier
from vinf.speculative.sampler import SpeculativeSampler


class SpeculativeDecodeStrategy:
    def __init__(
        self,
        draft_runner: DraftRunner,
        verifier: TargetVerifier,
        sampler: SpeculativeSampler,
        kv_commit: KVCommitManager,
        gamma: int,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.draft_runner = draft_runner
        self.verifier = verifier
        self.sampler = sampler
        self.kv_commit = kv_commit
        self.gamma = gamma

    def step(self, state: RuntimeState) -> list[int]:
        proposal = self.draft_runner.propose(state, self.gamma)
        verification = self.verifier.verify_many(state, proposal.token_ids)
        decision = self.sampler.accept_or_correct(
            proposal.token_ids,
            proposal.probability_rows_ref,
            verification.probability_rows_ref,
        )
        self.kv_commit.apply(state, decision)
        return list(decision.emitted_tokens)
