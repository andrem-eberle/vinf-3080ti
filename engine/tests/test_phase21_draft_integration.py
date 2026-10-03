from __future__ import annotations

import unittest

from vinf.executors.base import VerificationResult
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.runtime.kv_cache import KVCacheSet, LogicalKVCache
from vinf.runtime.state import RuntimeState
from vinf.speculative import (
    CPUSpeculativeSampler,
    HeuristicDraftRunner,
    LogicalKVCommitManager,
    SpeculativeDecodeStrategy,
)


def metadata() -> ModelMetadata:
    return ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_kv_heads=1,
        hidden_size=4,
        intermediate_size=8,
        head_dim=4,
        vocab_size=5,
        max_position_embeddings=16,
        dtype="fp16",
        tokenizer_id="tiny-tokenizer",
    )


class TargetVerifierFixture:
    def __init__(
        self,
        target_cache: LogicalKVCache,
        probability_rows: list[list[float]],
    ) -> None:
        self.target_cache = target_cache
        self.probability_rows = probability_rows
        self.seen_draft_tokens: tuple[int, ...] | None = None

    def verify_many(
        self, state: RuntimeState, draft_tokens: tuple[int, ...]
    ) -> VerificationResult:
        self.seen_draft_tokens = draft_tokens
        self.target_cache.write_speculative(len(draft_tokens))
        return VerificationResult(probability_rows_ref=self.probability_rows)


class Phase21DraftIntegrationTests(unittest.TestCase):
    def test_drafter_sampler_and_kv_commit_accept_all_integration(self) -> None:
        state = RuntimeState(prompt_tokens=[2])
        caches = KVCacheSet(
            target=LogicalKVCache("target", max_seq_len=16, max_speculative_tokens=2),
            draft=LogicalKVCache("draft", max_seq_len=16, max_speculative_tokens=2),
        )
        caches.target.append_committed(len(state.prompt_tokens))
        drafter = HeuristicDraftRunner(metadata(), caches.draft)
        drafter.prefill(state)
        verifier = TargetVerifierFixture(
            caches.target,
            probability_rows=[
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        )
        strategy = SpeculativeDecodeStrategy(
            drafter,
            verifier,
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(caches),
            gamma=2,
        )
        emitted = strategy.step(state)
        self.assertEqual(verifier.seen_draft_tokens, (2, 2))
        self.assertEqual(emitted, [2, 2, 1])
        self.assertEqual(state.output_tokens, [2, 2, 1])
        self.assertEqual(caches.target.committed_len, 3)
        self.assertEqual(caches.draft.committed_len, 3)
        self.assertEqual(caches.target.spec_len, 0)
        self.assertEqual(caches.draft.spec_len, 0)
        self.assertFalse(caches.target.can_read_position(3))
        self.assertFalse(caches.draft.can_read_position(3))

    def test_rejected_suffix_is_not_visible_after_integration_step(self) -> None:
        state = RuntimeState(prompt_tokens=[3])
        caches = KVCacheSet(
            target=LogicalKVCache("target", max_seq_len=16, max_speculative_tokens=3),
            draft=LogicalKVCache("draft", max_seq_len=16, max_speculative_tokens=3),
        )
        caches.target.append_committed(len(state.prompt_tokens))
        drafter = HeuristicDraftRunner(metadata(), caches.draft)
        drafter.prefill(state)
        verifier = TargetVerifierFixture(
            caches.target,
            probability_rows=[
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        )
        strategy = SpeculativeDecodeStrategy(
            drafter,
            verifier,
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(caches),
            gamma=3,
        )
        emitted = strategy.step(state)
        self.assertEqual(emitted, [0])
        self.assertEqual(caches.target.committed_len, 1)
        self.assertEqual(caches.draft.committed_len, 1)
        self.assertEqual(caches.target.spec_len, 0)
        self.assertEqual(caches.draft.spec_len, 0)
        self.assertFalse(caches.target.can_read_position(1))
        self.assertFalse(caches.draft.can_read_position(1))

    def test_speculative_disabled_boundary_uses_baseline_without_speculative_imports(self) -> None:
        from vinf.baseline_loop import BaselineEngineLoop
        from vinf.config import EngineConfig

        config = EngineConfig()
        self.assertFalse(config.speculative.enabled)
        self.assertNotIn("speculative", BaselineEngineLoop.__module__)


if __name__ == "__main__":
    unittest.main()
