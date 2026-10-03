from __future__ import annotations

from pathlib import Path
import unittest

from vinf.config import EngineConfig, SpeculativeConfig
from vinf.executors.base import DraftProposal, VerificationResult
from vinf.runtime.kv_cache import KVCacheSet, LogicalKVCache
from vinf.runtime.state import RuntimeState
from vinf.speculative import (
    CPUSpeculativeSampler,
    LogicalKVCommitManager,
    SpeculativeDecodeStrategy,
)
from vinf.speculative.sampler import SpeculativeDecision


class Phase20SpeculativeInterfaceTests(unittest.TestCase):
    def test_speculative_config_defaults_disabled(self) -> None:
        config = EngineConfig()
        self.assertFalse(config.speculative.enabled)
        self.assertEqual(config.speculative.gamma, 2)
        self.assertIsNone(config.speculative.draft_model_path)
        self.assertFalse(SpeculativeConfig().enabled)

    def test_strategy_uses_decoupled_runner_verifier_sampler_and_commit(self) -> None:
        class Draft:
            def propose(self, state: RuntimeState, gamma: int) -> DraftProposal:
                self_state = state
                assert self_state is not None
                assert gamma == 2
                return DraftProposal(
                    token_ids=(0, 1),
                    probability_rows_ref=[[1.0, 0.0], [0.0, 1.0]],
                )

        class Verifier:
            def verify_many(
                self, state: RuntimeState, draft_tokens: tuple[int, ...]
            ) -> VerificationResult:
                self_state = state
                assert self_state is not None
                assert draft_tokens == (0, 1)
                return VerificationResult(
                    probability_rows_ref=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
                )

        state = RuntimeState()
        strategy = SpeculativeDecodeStrategy(
            Draft(),
            Verifier(),
            CPUSpeculativeSampler(seed=0),
            LogicalKVCommitManager(),
            gamma=2,
        )
        self.assertEqual(strategy.step(state), [0, 1, 0])
        self.assertEqual(state.output_tokens, [0, 1, 0])

    def test_logical_kv_commit_manager_commits_accepted_and_discards_suffix(self) -> None:
        caches = KVCacheSet(
            target=LogicalKVCache("target", max_seq_len=8, max_speculative_tokens=3),
            draft=LogicalKVCache("draft", max_seq_len=8, max_speculative_tokens=3),
        )
        caches.target.append_committed(2)
        caches.draft.append_committed(2)
        caches.target.write_speculative(3)
        caches.draft.write_speculative(3)
        manager = LogicalKVCommitManager(caches)
        state = RuntimeState()
        manager.apply(
            state,
            SpeculativeDecision(accepted_count=2, emitted_tokens=(4, 5, 9), rejected=True),
        )
        self.assertEqual(state.output_tokens, [4, 5, 9])
        self.assertEqual(caches.target.committed_len, 4)
        self.assertEqual(caches.draft.committed_len, 4)
        self.assertEqual(caches.target.spec_len, 0)
        self.assertEqual(caches.draft.spec_len, 0)
        self.assertFalse(caches.target.can_read_position(4))

    def test_baseline_modules_do_not_import_speculative_package(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in [
            "src/vinf/baseline_loop.py",
            "src/vinf/baseline_decode.py",
            "src/vinf/engine.py",
            "src/vinf/runtime/strategy.py",
        ]:
            source = (root / relative).read_text()
            self.assertNotIn("vinf.speculative", source, relative)


if __name__ == "__main__":
    unittest.main()
