from __future__ import annotations

import math
import unittest

from vinf.cuda.lm_head import lm_head_cpu


class Phase15LMHeadTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=1e-5) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_cpu_lm_head_simple(self) -> None:
        out = lm_head_cpu(
            [1.0, 2.0],
            [1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
            3,
        )
        inv = 1.0 / math.sqrt((1.0 + 4.0) / 2.0 + 1e-6)
        self.assert_close_lists(out, [inv, 2 * inv, 3 * inv])

    def test_cpu_lm_head_logits_shape(self) -> None:
        hidden = 8
        vocab = 16
        values = [((i % 5) - 2) / 3.0 for i in range(hidden)]
        norm = [1.0 + (i % 3) * 0.1 for i in range(hidden)]
        lm = [((i % 7) - 3) / 7.0 for i in range(vocab * hidden)]
        out = lm_head_cpu(values, norm, lm, hidden, vocab)
        self.assertEqual(len(out), vocab)
        self.assertTrue(all(math.isfinite(x) for x in out))

    def test_cuda_lm_head_if_available(self) -> None:
        values = [1.0, 2.0]
        norm = [1.0, 1.0]
        lm = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        expected = lm_head_cpu(values, norm, lm, 2, 3)
        try:
            from vinf.cuda.lm_head import lm_head_cuda

            actual = lm_head_cuda(values, norm, lm, 2, 3)
        except Exception as exc:
            self.skipTest(f"CUDA LM head unavailable: {exc}")
        self.assert_close_lists(actual, expected)


if __name__ == "__main__":
    unittest.main()

