from __future__ import annotations

from time import perf_counter

from vinf.config import GenerationConfig
from vinf.executors.base import DraftProposal, VerificationResult
from vinf.speculative import (
    CPUSpeculativeSampler,
    LogicalKVCommitManager,
    SpeculativeDecodeStrategy,
    SpeculativeHybridEngine,
)


def main() -> None:
    gamma = 2
    strategy = SpeculativeDecodeStrategy(
        Draft((0, 1), [[1.0, 0.0], [0.0, 1.0]]),
        Verifier([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        CPUSpeculativeSampler(seed=0),
        LogicalKVCommitManager(),
        gamma=gamma,
    )
    engine = SpeculativeHybridEngine(
        strategy,
        generation=GenerationConfig(max_new_tokens=96),
        disable_below_tokens_per_target=1.0,
    )
    started = perf_counter()
    result = engine.generate(prompt_tokens=[0])
    elapsed = perf_counter() - started
    print(f"generated_tokens={result.metrics.generated_tokens}")
    print(f"target_calls={result.metrics.target_calls}")
    print(f"tokens_per_target_call={result.metrics.tokens_per_target_call:.2f}")
    print(f"acceptance_rate={result.metrics.speculative_acceptance_rate:.2f}")
    print(f"elapsed_seconds={elapsed:.6f}")


class Draft:
    def __init__(self, tokens, rows) -> None:  # type: ignore[no-untyped-def]
        self.tokens = tokens
        self.rows = rows

    def propose(self, state, gamma: int) -> DraftProposal:  # type: ignore[no-untyped-def]
        _ = state, gamma
        return DraftProposal(token_ids=self.tokens, probability_rows_ref=self.rows)


class Verifier:
    def __init__(self, rows) -> None:  # type: ignore[no-untyped-def]
        self.rows = rows

    def verify_many(self, state, draft_tokens) -> VerificationResult:  # type: ignore[no-untyped-def]
        _ = state, draft_tokens
        return VerificationResult(probability_rows_ref=self.rows)


if __name__ == "__main__":
    main()
