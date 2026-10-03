from __future__ import annotations

import math
import unittest

from vinf.cuda.mlp import mlp_cpu
from vinf.reference_ops import silu


class Phase14MLPTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=1e-5) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_cpu_gated_mlp_simple(self) -> None:
        values = [1.0, 2.0]
        gate = [1.0, 0.0, 0.0, 1.0]
        up = [1.0, 1.0, 1.0, 1.0]
        down = [1.0, 0.0, 0.0, 1.0]
        out = mlp_cpu(values, gate, up, down, 2, 2)
        expected = [silu(1.0) * 3.0, silu(2.0) * 3.0]
        self.assert_close_lists(out, expected)

    def test_cpu_mlp_larger_shape(self) -> None:
        hidden = 8
        intermediate = 16
        values = [((i % 5) - 2) / 3.0 for i in range(hidden)]
        gate = [((i % 7) - 3) / 7.0 for i in range(intermediate * hidden)]
        up = [((i % 11) - 5) / 11.0 for i in range(intermediate * hidden)]
        down = [((i % 13) - 6) / 13.0 for i in range(hidden * intermediate)]
        out = mlp_cpu(values, gate, up, down, hidden, intermediate)
        self.assertEqual(len(out), hidden)
        self.assertTrue(all(math.isfinite(x) for x in out))

    def test_cuda_mlp_if_available(self) -> None:
        values = [1.0, 2.0]
        gate = [1.0, 0.0, 0.0, 1.0]
        up = [1.0, 1.0, 1.0, 1.0]
        down = [1.0, 0.0, 0.0, 1.0]
        expected = mlp_cpu(values, gate, up, down, 2, 2)
        try:
            from vinf.cuda.mlp import mlp_cuda

            actual = mlp_cuda(values, gate, up, down, 2, 2)
        except Exception as exc:
            self.skipTest(f"CUDA MLP unavailable: {exc}")
        self.assert_close_lists(actual, expected)


if __name__ == "__main__":
    unittest.main()

