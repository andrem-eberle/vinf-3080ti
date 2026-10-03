from __future__ import annotations

import unittest

from vinf.config import GenerationConfig
from vinf.executors.base import DraftProposal, VerificationResult
from vinf.runtime.state import RuntimeState
from vinf.speculative import (
    CPUSpeculativeSampler,
    LogicalKVCommitManager,
    SpeculativeDecodeStrategy,
    SpeculativeHybridEngine,
)


class Draft:
    def __init__(self, tokens: tuple[int, ...], rows: list[list[float]]) -> None:
        self.tokens = tokens
        self.rows = rows

    def propose(self, state: RuntimeState, gamma: int) -> DraftProposal:
        _ = state
        if gamma != len(self.tokens):
            raise ValueError("gamma mismatch")
        return DraftProposal(token_ids=self.tokens, probability_rows_ref=self.rows)


class Verifier:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.calls = 0

    def verify_many(
        self, state: RuntimeState, draft_tokens: tuple[int, ...]
    ) -> VerificationResult:
        _ = state, draft_tokens
        self.calls += 1
        return VerificationResult(probability_rows_ref=self.rows)


class Phase23SpeculativeHybridTests(unittest.TestCase):
    def test_hybrid_connects_draft_verifier_sampler_and_commit(self) -> None:
        strategy = SpeculativeDecodeStrategy(
            Draft((0, 1), [[1.0, 0.0], [0.0, 1.0]]),
            Verifier([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(),
            gamma=2,
        )
        engine = SpeculativeHybridEngine(
            strategy,
            generation=GenerationConfig(max_new_tokens=3),
            disable_below_tokens_per_target=1.0,
        )
        result = engine.generate(prompt_tokens=[9])
        self.assertEqual(result.state.output_tokens, [0, 1, 0])
        self.assertEqual(result.metrics.generated_tokens, 3)
        self.assertEqual(result.metrics.target_calls, 1)
        self.assertEqual(result.metrics.verification_calls, 1)
        self.assertEqual(result.metrics.draft_calls, 1)
        self.assertEqual(result.metrics.speculative_tokens_proposed, 2)
        self.assertEqual(result.metrics.speculative_tokens_accepted, 2)
        self.assertAlmostEqual(result.metrics.speculative_acceptance_rate, 1.0)
        self.assertAlmostEqual(result.metrics.tokens_per_target_call, 3.0)

    def test_configurable_gamma_reaches_strategy(self) -> None:
        strategy = SpeculativeDecodeStrategy(
            Draft((0,), [[1.0, 0.0]]),
            Verifier([[1.0, 0.0], [0.0, 1.0]]),
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(),
            gamma=1,
        )
        result = SpeculativeHybridEngine(
            strategy,
            generation=GenerationConfig(max_new_tokens=2),
            disable_below_tokens_per_target=1.0,
        ).generate(prompt_tokens=[])
        self.assertEqual(result.metrics.speculative_tokens_proposed, 1)

    def test_rejection_histogram_records_rejection_position(self) -> None:
        strategy = SpeculativeDecodeStrategy(
            Draft((0, 1), [[1.0, 0.0], [0.0, 1.0]]),
            Verifier([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]),
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(),
            gamma=2,
        )
        result = SpeculativeHybridEngine(
            strategy,
            generation=GenerationConfig(max_new_tokens=1),
            disable_below_tokens_per_target=1.0,
        ).generate(prompt_tokens=[])
        self.assertEqual(result.metrics.rejection_position_histogram, {0: 1})
        self.assertEqual(result.metrics.speculative_tokens_accepted, 0)

    def test_adaptive_disable_if_tokens_per_target_is_low(self) -> None:
        strategy = SpeculativeDecodeStrategy(
            Draft((0, 1), [[1.0, 0.0], [0.0, 1.0]]),
            Verifier([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]),
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(),
            gamma=2,
        )
        engine = SpeculativeHybridEngine(
            strategy,
            generation=GenerationConfig(max_new_tokens=5),
            disable_below_tokens_per_target=1.2,
            warmup_iterations=1,
        )
        result = engine.generate(prompt_tokens=[])
        self.assertTrue(result.disabled)
        self.assertEqual(result.state.stop_reason, "speculation_disabled")


if __name__ == "__main__":
    unittest.main()
