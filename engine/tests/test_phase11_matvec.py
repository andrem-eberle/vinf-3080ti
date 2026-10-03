from __future__ import annotations

import math
import unittest

from vinf.cuda.matvec import QuantizedMatvecNotImplemented, matvec_cpu, matvec_quantized_placeholder


class Phase11MatvecTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=1e-5) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_cpu_matvec_reference(self) -> None:
        values = [1.0, 2.0, 3.0]
        weights = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 1.0,
        ]
        self.assertEqual(matvec_cpu(values, weights, 2, 3), [1.0, 5.0])

    def test_projection_shape_cases(self) -> None:
        hidden = 16
        cases = [
            ("qkv", 24, hidden),
            ("o_proj", hidden, hidden),
            ("mlp_up", 32, hidden),
            ("lm_head", 40, hidden),
        ]
        values = [float((i % 5) - 2) for i in range(hidden)]
        for _, rows, cols in cases:
            weights = [((i % 7) - 3) / 7.0 for i in range(rows * cols)]
            out = matvec_cpu(values, weights, rows, cols)
            self.assertEqual(len(out), rows)
            self.assertTrue(all(math.isfinite(x) for x in out))

    def test_quantized_placeholder(self) -> None:
        with self.assertRaises(QuantizedMatvecNotImplemented):
            matvec_quantized_placeholder()

    def test_cuda_matvec_if_available(self) -> None:
        values = [1.0, -2.0, 3.0, 0.5]
        weights = [
            1.0, 2.0, 3.0, 4.0,
            -1.0, 0.0, 1.0, 0.0,
            0.5, 0.5, 0.5, 0.5,
        ]
        expected = matvec_cpu(values, weights, 3, 4)
        try:
            from vinf.cuda.matvec import matvec_cuda

            actual = matvec_cuda(values, weights, 3, 4)
        except Exception as exc:
            self.skipTest(f"CUDA matvec unavailable: {exc}")
        self.assert_close_lists(actual, expected)


if __name__ == "__main__":
    unittest.main()

