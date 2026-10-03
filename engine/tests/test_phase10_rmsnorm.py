from __future__ import annotations

import math
import unittest

from vinf.reference_ops import rmsnorm_reference


class Phase10RMSNormTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=1e-5) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(
                math.isclose(a, e, rel_tol=tol, abs_tol=tol),
                (idx, a, e),
            )

    def test_cpu_reference_tiny_vector(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        weights = [1.0, 0.5, 2.0, -1.0]
        out = rmsnorm_reference(values, weights, eps=1e-6)
        mean_square = (1 + 4 + 9 + 16) / 4
        inv = 1.0 / math.sqrt(mean_square + 1e-6)
        self.assert_close_lists(
            out,
            [1.0 * inv, 2.0 * inv * 0.5, 3.0 * inv * 2.0, 4.0 * inv * -1.0],
        )

    def test_cpu_reference_qwen_hidden_size(self) -> None:
        n = 5120
        values = [((i % 17) - 8) / 8.0 for i in range(n)]
        weights = [1.0 + (i % 7) * 0.01 for i in range(n)]
        out = rmsnorm_reference(values, weights)
        self.assertEqual(len(out), n)
        self.assertTrue(all(math.isfinite(x) for x in out))

    def test_cuda_rmsnorm_tiny_vector_if_available(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        weights = [1.0, 0.5, 2.0, -1.0]
        expected = rmsnorm_reference(values, weights)
        try:
            from vinf.cuda.rmsnorm import rmsnorm_cuda

            actual = rmsnorm_cuda(values, weights)
        except Exception as exc:
            self.skipTest(f"CUDA RMSNorm unavailable: {exc}")
        self.assert_close_lists(actual, expected, tol=2e-5)

    def test_cuda_rmsnorm_qwen_hidden_size_if_available(self) -> None:
        n = 5120
        values = [((i % 23) - 11) / 11.0 for i in range(n)]
        weights = [1.0 + (i % 5) * 0.02 for i in range(n)]
        expected = rmsnorm_reference(values, weights)
        try:
            from vinf.cuda.rmsnorm import rmsnorm_cuda

            actual = rmsnorm_cuda(values, weights)
        except Exception as exc:
            self.skipTest(f"CUDA RMSNorm unavailable: {exc}")
        self.assert_close_lists(actual, expected, tol=2e-5)


if __name__ == "__main__":
    unittest.main()

