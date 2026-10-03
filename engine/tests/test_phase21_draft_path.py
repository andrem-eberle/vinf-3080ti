from __future__ import annotations

import unittest

from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.runtime.kv_cache import LogicalKVCache
from vinf.runtime.state import RuntimeState
from vinf.speculative.draft import (
    HeuristicDraftRunner,
    validate_draft_tokenizer_compatibility,
)


def metadata(*, tokenizer_id: str = "tiny-tokenizer", vocab_size: int = 5) -> ModelMetadata:
    return ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_kv_heads=1,
        hidden_size=4,
        intermediate_size=8,
        head_dim=4,
        vocab_size=vocab_size,
        max_position_embeddings=16,
        dtype="fp16",
        tokenizer_id=tokenizer_id,
    )


class Phase21DraftPathTests(unittest.TestCase):
    def test_repeat_last_heuristic_proposes_tokens_and_full_probability_rows(self) -> None:
        runner = HeuristicDraftRunner(
            metadata=metadata(),
            kv_cache=LogicalKVCache("draft", max_seq_len=16, max_speculative_tokens=4),
        )
        state = RuntimeState(prompt_tokens=[1, 3])
        runner.prefill(state)
        proposal = runner.propose(state, gamma=3)
        self.assertEqual(proposal.token_ids, (3, 3, 3))
        self.assertEqual(runner.last_token_ids, (3, 3, 3))
        rows = proposal.probability_rows_ref
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row, [0.0, 0.0, 0.0, 1.0, 0.0])
            self.assertAlmostEqual(sum(row), 1.0)
        self.assertEqual(runner.last_probability_rows[0], (0.0, 0.0, 0.0, 1.0, 0.0))

    def test_empty_context_drafts_token_zero(self) -> None:
        runner = HeuristicDraftRunner(
            metadata=metadata(),
            kv_cache=LogicalKVCache("draft", max_seq_len=16, max_speculative_tokens=2),
        )
        proposal = runner.propose(RuntimeState(), gamma=1)
        self.assertEqual(proposal.token_ids, (0,))
        self.assertEqual(proposal.probability_rows_ref, [[1.0, 0.0, 0.0, 0.0, 0.0]])

    def test_draft_kv_prefill_write_and_rollback_by_logical_length(self) -> None:
        cache = LogicalKVCache("draft", max_seq_len=16, max_speculative_tokens=4)
        runner = HeuristicDraftRunner(metadata=metadata(), kv_cache=cache)
        state = RuntimeState(prompt_tokens=[1, 2, 3])
        runner.prefill(state)
        self.assertEqual(cache.committed_len, 3)
        runner.propose(state, gamma=2)
        self.assertEqual(cache.spec_len, 2)
        self.assertTrue(cache.can_read_position(4, include_speculative=True))
        runner.discard_speculative()
        self.assertFalse(cache.can_read_position(3))
        runner.propose(state, gamma=2)
        cache.commit_speculative(1)
        runner.rollback_to(3)
        self.assertEqual(cache.committed_len, 3)
        self.assertEqual(cache.spec_len, 0)

    def test_tokenizer_compatibility_checks_id_and_vocab(self) -> None:
        validate_draft_tokenizer_compatibility(metadata(), metadata())
        with self.assertRaises(ValueError):
            validate_draft_tokenizer_compatibility(
                metadata(tokenizer_id="target"),
                metadata(tokenizer_id="draft"),
            )
        with self.assertRaises(ValueError):
            validate_draft_tokenizer_compatibility(metadata(vocab_size=5), metadata(vocab_size=6))

    def test_draft_cost_is_measured(self) -> None:
        runner = HeuristicDraftRunner(
            metadata=metadata(),
            kv_cache=LogicalKVCache("draft", max_seq_len=16, max_speculative_tokens=4),
        )
        runner.propose(RuntimeState(prompt_tokens=[2]), gamma=2)
        runner.propose(RuntimeState(prompt_tokens=[3]), gamma=1)
        self.assertEqual(runner.stats.calls, 2)
        self.assertEqual(runner.stats.tokens_proposed, 3)
        self.assertGreaterEqual(runner.stats.elapsed_seconds, 0.0)
        self.assertGreaterEqual(runner.stats.seconds_per_token, 0.0)

    def test_invalid_gamma_and_out_of_vocab_context_fail_loudly(self) -> None:
        runner = HeuristicDraftRunner(
            metadata=metadata(),
            kv_cache=LogicalKVCache("draft", max_seq_len=16, max_speculative_tokens=2),
        )
        with self.assertRaises(ValueError):
            runner.propose(RuntimeState(prompt_tokens=[1]), gamma=0)
        with self.assertRaises(ValueError):
            runner.propose(RuntimeState(prompt_tokens=[1]), gamma=3)
        with self.assertRaises(ValueError):
            runner.propose(RuntimeState(prompt_tokens=[99]), gamma=1)


if __name__ == "__main__":
    unittest.main()
