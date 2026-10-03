from __future__ import annotations

import unittest

from vinf.config import GenerationConfig
from vinf.cuda.sampling import (
    CounterRNG,
    gpu_logits_process_cpu,
    gpu_probabilities_cpu,
    sample_from_logits_cpu,
    sample_token_id,
    speculative_accept_and_sample_cpu,
)
from vinf.sampling import logits_to_probabilities
from vinf.speculative.sampler import correction_distribution


class Phase25GPUSamplingAcceptanceTests(unittest.TestCase):
    def assert_close_list(self, actual, expected, tol=1e-9) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(a, e, delta=tol, msg=idx)

    def test_gpu_logits_processing_matches_sampling_policy_masks(self) -> None:
        processed = gpu_logits_process_cpu(
            [0.0, 1.0, 2.0, 3.0],
            GenerationConfig(temperature=2.0, top_k=2),
        )
        self.assertEqual(processed[0], float("-inf"))
        self.assertEqual(processed[1], float("-inf"))
        self.assertGreater(processed[3], processed[2])

    def test_gpu_softmax_probabilities_match_cpu_sampling_transform(self) -> None:
        config = GenerationConfig(temperature=1.5, top_k=3, top_p=0.9)
        logits = [0.0, 1.0, 2.0, 3.0]
        self.assert_close_list(gpu_probabilities_cpu(logits, config), logits_to_probabilities(logits, config))

    def test_counter_rng_is_deterministic(self) -> None:
        rng_a = CounterRNG(seed=123)
        rng_b = CounterRNG(seed=123)
        self.assertEqual([rng_a.uniform(i) for i in range(5)], [rng_b.uniform(i) for i in range(5)])

    def test_gpu_token_sampling_returns_only_token_id(self) -> None:
        token = sample_token_id([0.0, 1.0, 0.0], seed=7)
        self.assertEqual(token, 1)
        self.assertIsInstance(sample_from_logits_cpu([0.0, 5.0], GenerationConfig(), seed=7), int)

    def test_gpu_speculative_acceptance_returns_emitted_tokens_and_counters(self) -> None:
        emitted, accepted, rejected = speculative_accept_and_sample_cpu(
            (0, 1),
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            seed=1,
        )
        self.assertEqual(emitted, (0, 1, 0))
        self.assertEqual(accepted, 2)
        self.assertFalse(rejected)

    def test_gpu_correction_distribution_path_matches_formula(self) -> None:
        correction = correction_distribution([0.0, 1.0], [1.0, 0.0])
        emitted, accepted, rejected = speculative_accept_and_sample_cpu(
            (0,),
            [[1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            seed=1,
        )
        self.assertEqual(correction, [0.0, 1.0])
        self.assertEqual(emitted, (1,))
        self.assertEqual(accepted, 0)
        self.assertTrue(rejected)

    def test_cuda_probability_transform_matches_cpu_if_available(self) -> None:
        from vinf.cuda.sampling import gpu_probabilities_cuda

        logits = [0.0, 1.0, 2.0]
        expected = gpu_probabilities_cpu(logits, GenerationConfig())
        try:
            actual = gpu_probabilities_cuda(logits, GenerationConfig())
        except Exception as exc:
            self.skipTest(f"CUDA GPU sampling transform unavailable: {exc}")
        self.assert_close_list(actual, expected, tol=1e-5)


if __name__ == "__main__":
    unittest.main()
