from __future__ import annotations

from dataclasses import dataclass

from vinf.runtime.kv_cache import KVCacheSet
from vinf.runtime.state import RuntimeState
from vinf.speculative.sampler import SpeculativeDecision


@dataclass(frozen=True, slots=True)
class LogicalKVCommitManager:
    caches: KVCacheSet | None = None

    def apply(self, state: RuntimeState, decision: SpeculativeDecision) -> None:
        state.append_tokens(list(decision.emitted_tokens))
        if self.caches is None:
            return
        if decision.accepted_count:
            self.caches.target.commit_speculative(decision.accepted_count)
            if self.caches.draft is not None:
                self.caches.draft.commit_speculative(decision.accepted_count)
        self.caches.discard_speculative()
