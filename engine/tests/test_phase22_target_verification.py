from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vinf.config import EngineConfig, GenerationConfig
from vinf.executors.reference import ReferenceExecutor
from vinf.executors.base import DraftProposal
from vinf.models import FIRST_SUPPORTED_SHAPE, load_target_model
from vinf.runtime.kv_cache import KVCacheSet, LogicalKVCache
from vinf.runtime.state import RuntimeState
from vinf.speculative import (
    CPUSpeculativeSampler,
    FallbackTargetVerifier,
    LogicalKVCommitManager,
    MegakernelTargetVerifier,
    SpeculativeDecodeStrategy,
)
from vinf.speculative.draft import HeuristicDraftRunner


def transition_model_dict():
    vocab_size = FIRST_SUPPORTED_SHAPE["vocab_size"]
    table = []
    for row in range(vocab_size):
        logits = [-10.0 for _ in range(vocab_size)]
        logits[(row + 1) % vocab_size] = 10.0
        logits[(row + 2) % vocab_size] = 5.0
        table.append(logits)
    return {
        "metadata": {
            "architecture": "llama",
            **FIRST_SUPPORTED_SHAPE,
            "dtype": "fp16",
            "tokenizer_id": "tiny-tokenizer",
        },
        "weights": {"transition_logits": table},
    }


def load_executor(generation: GenerationConfig | None = None) -> ReferenceExecutor:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    with tmp:
        json.dump(transition_model_dict(), tmp)
    model = load_target_model(Path(tmp.name), EngineConfig())
    return ReferenceExecutor(model, generation_config=generation or GenerationConfig(top_k=1))


class Phase22TargetVerificationTests(unittest.TestCase):
    def test_fallback_verifier_returns_gamma_plus_one_probability_rows(self) -> None:
        executor = load_executor(GenerationConfig(top_k=1))
        state = RuntimeState(prompt_tokens=[1])
        executor.prefill(state)
        verifier = FallbackTargetVerifier(executor)
        result = verifier.verify_many(state, (2, 3))
        rows = result.probability_rows_ref
        self.assertEqual(len(rows), 3)
        self.assertEqual(max(range(8), key=lambda idx: rows[0][idx]), 2)
        self.assertEqual(max(range(8), key=lambda idx: rows[1][idx]), 3)
        self.assertEqual(max(range(8), key=lambda idx: rows[2][idx]), 4)

    def test_verification_kv_writes_stay_speculative_until_commit(self) -> None:
        executor = load_executor(GenerationConfig(top_k=1))
        state = RuntimeState(prompt_tokens=[1])
        executor.prefill(state)
        cache = LogicalKVCache("target", max_seq_len=8, max_speculative_tokens=2)
        cache.append_committed(1)
        verifier = FallbackTargetVerifier(executor, speculative_kv=cache)
        result = verifier.verify_many(state, (2, 3))
        self.assertIs(result.speculative_kv_ref, cache)
        self.assertEqual(cache.committed_len, 1)
        self.assertEqual(cache.spec_len, 2)
        self.assertFalse(cache.can_read_position(1))
        self.assertTrue(cache.can_read_position(1, include_speculative=True))

    def test_all_token_acceptance_commits_only_accepted_target_kv(self) -> None:
        executor = load_executor(GenerationConfig(top_k=1))
        state = RuntimeState(prompt_tokens=[1])
        executor.prefill(state)
        caches = KVCacheSet(
            target=LogicalKVCache("target", max_seq_len=8, max_speculative_tokens=2),
            draft=LogicalKVCache("draft", max_seq_len=8, max_speculative_tokens=2),
        )
        caches.target.append_committed(1)
        drafter = ScriptedDraftRunner((2, 3), caches.draft, vocab_size=8)
        drafter.prefill(state)
        strategy = SpeculativeDecodeStrategy(
            drafter,
            FallbackTargetVerifier(executor, speculative_kv=caches.target),
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(caches),
            gamma=2,
        )
        emitted = strategy.step(state)
        self.assertEqual(emitted, [2, 3, 4])
        self.assertEqual(caches.target.committed_len, 3)
        self.assertEqual(caches.target.spec_len, 0)
        self.assertFalse(caches.target.can_read_position(3))

    def test_first_token_rejection_discards_rejected_target_kv(self) -> None:
        executor = load_executor(GenerationConfig(top_k=1))
        state = RuntimeState(prompt_tokens=[1])
        executor.prefill(state)
        caches = KVCacheSet(
            target=LogicalKVCache("target", max_seq_len=8, max_speculative_tokens=2),
            draft=LogicalKVCache("draft", max_seq_len=8, max_speculative_tokens=2),
        )
        caches.target.append_committed(1)
        drafter = HeuristicDraftRunner(executor.model.metadata, caches.draft)
        drafter.prefill(state)
        strategy = SpeculativeDecodeStrategy(
            drafter,
            FallbackTargetVerifier(executor, speculative_kv=caches.target),
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(caches),
            gamma=2,
        )
        emitted = strategy.step(state)
        self.assertEqual(emitted, [2])
        self.assertEqual(caches.target.committed_len, 1)
        self.assertEqual(caches.target.spec_len, 0)
        self.assertFalse(caches.target.can_read_position(1))

    def test_mixed_acceptance_commits_prefix_and_discards_suffix(self) -> None:
        executor = load_executor(GenerationConfig(top_k=1))
        state = RuntimeState(prompt_tokens=[2])
        executor.prefill(state)
        caches = KVCacheSet(
            target=LogicalKVCache("target", max_seq_len=8, max_speculative_tokens=2),
            draft=LogicalKVCache("draft", max_seq_len=8, max_speculative_tokens=2),
        )
        caches.target.append_committed(1)
        drafter = ScriptedDraftRunner((3, 3), caches.draft, vocab_size=8)
        drafter.prefill(state)
        strategy = SpeculativeDecodeStrategy(
            drafter,
            FallbackTargetVerifier(executor, speculative_kv=caches.target),
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(caches),
            gamma=2,
        )
        emitted = strategy.step(state)
        self.assertEqual(emitted, [3, 4])
        self.assertEqual(caches.target.committed_len, 2)
        self.assertEqual(caches.target.spec_len, 0)
        self.assertFalse(caches.target.can_read_position(2))

    def test_verifier_rows_match_reference_executor_parity(self) -> None:
        executor = load_executor(GenerationConfig(temperature=1.0))
        state = RuntimeState(prompt_tokens=[3])
        executor.prefill(state)
        direct = executor.verify_many(state, (4, 5)).probability_rows_ref
        wrapped = FallbackTargetVerifier(executor).verify_many(state, (4, 5)).probability_rows_ref
        self.assertEqual(wrapped, direct)

    def test_megakernel_verifier_wrapper_matches_fallback_rows(self) -> None:
        executor = load_executor(GenerationConfig(temperature=1.0))
        state = RuntimeState(prompt_tokens=[3])
        executor.prefill(state)
        fallback = FallbackTargetVerifier(executor).verify_many(state, (4, 5)).probability_rows_ref
        wrapped = MegakernelTargetVerifier(executor, use_cuda=False).verify_many(
            state, (4, 5)
        ).probability_rows_ref
        for actual_row, expected_row in zip(wrapped, fallback):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()


class ScriptedDraftRunner:
    def __init__(
        self, tokens: tuple[int, ...], cache: LogicalKVCache, *, vocab_size: int
    ) -> None:
        self.tokens = tokens
        self.cache = cache
        self.vocab_size = vocab_size

    def prefill(self, state: RuntimeState) -> None:
        self.cache.committed_len = len(state.prompt_tokens)
        self.cache.spec_len = 0

    def propose(self, state: RuntimeState, gamma: int) -> DraftProposal:
        _ = state
        if gamma != len(self.tokens):
            raise ValueError("scripted draft length must equal gamma")
        self.cache.write_speculative(gamma)
        return DraftProposal(
            token_ids=self.tokens,
            probability_rows_ref=[self._one_hot(token) for token in self.tokens],
        )

    def _one_hot(self, token: int) -> list[float]:
        return [1.0 if idx == token else 0.0 for idx in range(self.vocab_size)]
