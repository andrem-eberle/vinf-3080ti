from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vinf.config import EngineConfig, GenerationConfig
from vinf.executors.reference import ReferenceExecutor
from vinf.models import FIRST_SUPPORTED_SHAPE, load_target_model
from vinf.runtime.state import RuntimeState
from vinf.sampling import CPUSampler, GreedySampler, logits_to_probabilities


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


class Phase4ReferenceExecutionTests(unittest.TestCase):
    def write_model(self) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        with tmp:
            json.dump(transition_model_dict(), tmp)
        return Path(tmp.name)

    def load_executor(
        self, generation: GenerationConfig | None = None, seed: int | None = 0
    ) -> ReferenceExecutor:
        model = load_target_model(self.write_model(), EngineConfig())
        return ReferenceExecutor(model, generation_config=generation, seed=seed)

    def test_reference_prefill_sets_model_and_position(self) -> None:
        executor = self.load_executor()
        state = RuntimeState(prompt_tokens=[2, 3, 4])
        executor.prefill(state)
        self.assertEqual(state.position, 3)
        self.assertIsNotNone(state.model)

    def test_reference_one_token_decode_returns_expected_greedy_token(self) -> None:
        executor = self.load_executor(GenerationConfig(temperature=1.0, top_k=1))
        state = RuntimeState(prompt_tokens=[3])
        executor.prefill(state)
        step = executor.decode_one(state)
        self.assertEqual(step.token_id, 4)
        self.assertEqual(len(step.logits_ref), FIRST_SUPPORTED_SHAPE["vocab_size"])
        self.assertAlmostEqual(sum(step.probabilities_ref), 1.0, places=6)

    def test_reference_decode_parity_against_known_transition_table(self) -> None:
        executor = self.load_executor(GenerationConfig(top_k=1))
        state = RuntimeState(
            prompt_tokens=[0], generation=GenerationConfig(max_new_tokens=4, top_k=1)
        )
        executor.prefill(state)
        for _ in range(4):
            step = executor.decode_one(state)
            state.append_tokens([step.token_id])
        self.assertEqual(state.output_tokens, [1, 2, 3, 4])

    def test_reference_verify_many_returns_gamma_plus_one_rows(self) -> None:
        executor = self.load_executor(GenerationConfig(top_k=1))
        state = RuntimeState(prompt_tokens=[1])
        executor.prefill(state)
        result = executor.verify_many(state, (2, 3))
        rows = result.probability_rows_ref
        self.assertEqual(len(rows), 3)
        self.assertEqual(max(range(8), key=lambda idx: rows[0][idx]), 2)
        self.assertEqual(max(range(8), key=lambda idx: rows[1][idx]), 3)
        self.assertEqual(max(range(8), key=lambda idx: rows[2][idx]), 4)

    def test_greedy_sampling(self) -> None:
        result = GreedySampler().sample([1.0, 3.0, 2.0], GenerationConfig())
        self.assertEqual(result.token_id, 1)

    def test_temperature_changes_distribution_sharpness(self) -> None:
        cold = logits_to_probabilities([0.0, 2.0], GenerationConfig(temperature=0.5))
        hot = logits_to_probabilities([0.0, 2.0], GenerationConfig(temperature=2.0))
        self.assertGreater(cold[1], hot[1])

    def test_top_k_sampling_masks_everything_but_k_tokens(self) -> None:
        probs = logits_to_probabilities([0.0, 1.0, 2.0], GenerationConfig(top_k=1))
        self.assertEqual(probs[0], 0.0)
        self.assertEqual(probs[1], 0.0)
        self.assertAlmostEqual(probs[2], 1.0)

    def test_top_p_sampling_keeps_minimal_probability_prefix(self) -> None:
        probs = logits_to_probabilities([5.0, 4.0, 0.0], GenerationConfig(top_p=0.8))
        self.assertGreater(probs[0], 0.0)
        self.assertGreater(probs[1], 0.0)
        self.assertEqual(probs[2], 0.0)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_cpu_sampler_is_deterministic_with_seed(self) -> None:
        config = GenerationConfig(temperature=1.0)
        sampler_a = CPUSampler(seed=123)
        sampler_b = CPUSampler(seed=123)
        seq_a = [sampler_a.sample([0.0, 1.0, 2.0], config).token_id for _ in range(8)]
        seq_b = [sampler_b.sample([0.0, 1.0, 2.0], config).token_id for _ in range(8)]
        self.assertEqual(seq_a, seq_b)


if __name__ == "__main__":
    unittest.main()

