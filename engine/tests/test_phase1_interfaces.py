from __future__ import annotations

import unittest

from vinf.config import EngineConfig, GenerationConfig, SpeculativeConfig
from vinf.errors import ConfigurationError
from vinf.executors.base import DraftProposal, VerificationResult
from vinf.metrics import EngineMetrics, Timer
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.runtime.state import DecodeMode, RuntimeState
from vinf.sampling import GreedySampler
from vinf.speculative.controller import SpeculativeDecodeStrategy
from vinf.speculative.sampler import SpeculativeDecision


class Phase1InterfaceTests(unittest.TestCase):
    def test_config_errors_are_engine_errors(self) -> None:
        with self.assertRaises(ConfigurationError):
            EngineConfig(max_seq_len=0)
        with self.assertRaises(ConfigurationError):
            GenerationConfig(temperature=0)
        with self.assertRaises(ConfigurationError):
            SpeculativeConfig(enabled=True, gamma=0)

    def test_model_metadata_validation(self) -> None:
        metadata = ModelMetadata(
            architecture=ModelArchitecture.LLAMA,
            num_hidden_layers=16,
            num_attention_heads=32,
            num_kv_heads=8,
            hidden_size=2048,
            intermediate_size=8192,
            head_dim=64,
            vocab_size=128256,
            max_position_embeddings=4096,
            dtype="fp16",
        )
        self.assertEqual(
            metadata.hidden_size, metadata.num_attention_heads * metadata.head_dim
        )

    def test_runtime_state_append_tokens_stops_at_limit(self) -> None:
        state = RuntimeState(
            generation=GenerationConfig(max_new_tokens=2),
            mode=DecodeMode.BASELINE,
            prompt_tokens=[10, 11],
        )
        state.append_tokens([12])
        self.assertFalse(state.stopped)
        state.append_tokens([13])
        self.assertTrue(state.stopped)
        self.assertEqual(state.stop_reason, "max_new_tokens")
        self.assertEqual(state.all_tokens, [10, 11, 12, 13])

    def test_greedy_sampler(self) -> None:
        result = GreedySampler().sample([0.0, 2.0, 1.0], GenerationConfig())
        self.assertEqual(result.token_id, 1)

    def test_metrics_timer(self) -> None:
        metrics = EngineMetrics(
            speculative_tokens_proposed=4, speculative_tokens_accepted=3
        )
        with Timer(metrics, "phase"):
            pass
        self.assertEqual(metrics.speculative_acceptance_rate, 0.75)
        self.assertGreaterEqual(metrics.timings["phase"], 0)

    def test_speculative_strategy_contract(self) -> None:
        class Draft:
            def propose(self, state: RuntimeState, gamma: int) -> DraftProposal:
                self_state = state
                assert self_state is not None
                assert gamma == 2
                return DraftProposal(token_ids=(1, 2), probability_rows_ref="q")

        class Verifier:
            def verify_many(
                self, state: RuntimeState, draft_tokens: tuple[int, ...]
            ) -> VerificationResult:
                self_state = state
                assert self_state is not None
                assert draft_tokens == (1, 2)
                return VerificationResult(probability_rows_ref="p")

        class Sampler:
            def accept_or_correct(
                self,
                draft_tokens,
                draft_probability_rows,
                target_probability_rows,
            ):
                assert draft_tokens == (1, 2)
                assert draft_probability_rows == "q"
                assert target_probability_rows == "p"
                return SpeculativeDecision(
                    accepted_count=1, emitted_tokens=(1, 9), rejected=True
                )

        class KVCommit:
            def apply(
                self, state: RuntimeState, decision: SpeculativeDecision
            ) -> None:
                state.append_tokens(list(decision.emitted_tokens))

        state = RuntimeState(generation=GenerationConfig(max_new_tokens=4))
        strategy = SpeculativeDecodeStrategy(
            Draft(), Verifier(), Sampler(), KVCommit(), 2
        )
        self.assertEqual(strategy.step(state), [1, 9])
        self.assertEqual(state.output_tokens, [1, 9])


if __name__ == "__main__":
    unittest.main()
