from __future__ import annotations

import math
import unittest

from vinf.cuda.math import math_smoke_reference


class Phase9CUDAMathTests(unittest.TestCase):
    def test_math_smoke_reference(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        out = math_smoke_reference(values)
        self.assertEqual(out[:4], [2.0, 4.0, 6.0, 8.0])
        self.assertEqual(out[4], 10.0)
        self.assertEqual(out[8:12], [2.0, 3.0, 4.0, 5.0])

    def test_live_cuda_math_smoke_if_available(self) -> None:
        values = [1.0, -2.0, 3.5, 4.25, 8.0, -1.0]
        expected = math_smoke_reference(values)
        try:
            from vinf.cuda.math import run_math_smoke

            actual = run_math_smoke(values)
        except Exception as exc:
            self.skipTest(f"CUDA math smoke unavailable: {exc}")
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual[: len(values) + 1], expected[: len(values) + 1])):
            self.assertTrue(math.isclose(a, e, rel_tol=1e-6, abs_tol=1e-6), (idx, a, e))
        self.assertTrue(math.isclose(actual[len(values) + 1], expected[len(values) + 1], rel_tol=1e-3, abs_tol=1e-3))
        self.assertEqual(actual[len(values) + 4 : len(values) + 8], expected[len(values) + 4 : len(values) + 8])


if __name__ == "__main__":
    unittest.main()

