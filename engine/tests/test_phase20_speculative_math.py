from __future__ import annotations

import math
import unittest

from vinf.speculative.sampler import (
    CPUSpeculativeSampler,
    acceptance_probability,
    correction_distribution,
    expected_speculative_speedup,
    expected_tokens_per_speculative_iteration,
)


class Phase20SpeculativeMathTests(unittest.TestCase):
    def assert_close_list(self, actual, expected, tol=1e-9) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_acceptance_probability_is_min_one_p_over_q(self) -> None:
        self.assertAlmostEqual(acceptance_probability(0.25, 0.5), 0.5)
        self.assertAlmostEqual(acceptance_probability(0.75, 0.5), 1.0)

    def test_zero_draft_probability_edge_case(self) -> None:
        self.assertEqual(acceptance_probability(0.2, 0.0), 1.0)
        self.assertEqual(acceptance_probability(0.0, 0.0), 0.0)

    def test_correction_distribution_is_normalized_positive_residual(self) -> None:
        correction = correction_distribution(
            [0.50, 0.30, 0.20],
            [0.20, 0.40, 0.40],
        )
        self.assert_close_list(correction, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(correction), 1.0)

    def test_correction_distribution_falls_back_to_target_when_residual_empty(self) -> None:
        self.assert_close_list(
            correction_distribution([0.2, 0.8], [0.2, 0.8]),
            [0.2, 0.8],
        )

    def test_all_accepted_samples_extra_token_from_p_gamma_plus_one(self) -> None:
        sampler = CPUSpeculativeSampler(seed=3)
        decision = sampler.accept_or_correct(
            (0, 1),
            [[0.9, 0.1], [0.1, 0.9]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        )
        self.assertEqual(decision.accepted_count, 2)
        self.assertFalse(decision.rejected)
        self.assertEqual(decision.emitted_tokens, (0, 1, 1))

    def test_first_rejection_samples_from_correction_distribution(self) -> None:
        sampler = CPUSpeculativeSampler(seed=0)
        decision = sampler.accept_or_correct(
            (0, 1),
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.1, 0.9], [0.0, 1.0], [0.5, 0.5]],
        )
        self.assertEqual(decision.accepted_count, 0)
        self.assertTrue(decision.rejected)
        self.assertEqual(decision.emitted_tokens, (1,))

    def test_mixed_acceptance_rejection_uses_matching_row_indices(self) -> None:
        sampler = CPUSpeculativeSampler(seed=2)
        decision = sampler.accept_or_correct(
            (0, 1),
            [[0.8, 0.2], [0.2, 0.8]],
            [[0.8, 0.2], [0.8, 0.2], [1.0, 0.0]],
        )
        self.assertEqual(decision.accepted_count, 1)
        self.assertTrue(decision.rejected)
        self.assertEqual(decision.emitted_tokens, (0, 0))

    def test_different_sampling_supports_are_handled_by_correction(self) -> None:
        correction = correction_distribution(
            [0.0, 0.7, 0.3],
            [1.0, 0.0, 0.0],
        )
        self.assert_close_list(correction, [0.0, 0.7, 0.3])

    def test_expected_tokens_formula(self) -> None:
        alpha = 0.75
        gamma = 3
        expected = (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)
        self.assertAlmostEqual(
            expected_tokens_per_speculative_iteration(alpha, gamma),
            expected,
        )
        self.assertEqual(expected_tokens_per_speculative_iteration(1.0, gamma), 4.0)

    def test_expected_speedup_formula(self) -> None:
        alpha = 0.75
        gamma = 3
        c = 0.1
        expected = (1.0 - alpha ** (gamma + 1)) / ((1.0 - alpha) * (gamma * c + 1.0))
        self.assertAlmostEqual(expected_speculative_speedup(alpha, gamma, c), expected)

    def test_invalid_shapes_fail_loudly(self) -> None:
        sampler = CPUSpeculativeSampler(seed=0)
        with self.assertRaises(ValueError):
            sampler.accept_or_correct((0, 1), [[1.0, 0.0]], [[1.0, 0.0]])
        with self.assertRaises(ValueError):
            correction_distribution([0.5, 0.5], [1.0])


if __name__ == "__main__":
    unittest.main()
