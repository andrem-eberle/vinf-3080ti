from __future__ import annotations

import math
import unittest

from vinf.cuda.rope_kv import rope_kv_cpu
from vinf.reference_ops import kv_append_reference, rope_reference


class Phase12RopeKVTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=1e-6) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_rope_reference(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        cos = [1.0, 1.0, 0.0, 0.0]
        sin = [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(rope_reference(values, cos, sin), [1.0, 2.0, -4.0, 3.0])

    def test_kv_append_reference_position(self) -> None:
        cache = [0.0] * (2 * 3 * 4)
        out = kv_append_reference(
            cache,
            [9.0, 8.0, 7.0, 6.0],
            head_idx=1,
            position=2,
            num_heads=2,
            max_seq=3,
            head_dim=4,
        )
        base = (1 * 3 + 2) * 4
        self.assertEqual(out[base : base + 4], [9.0, 8.0, 7.0, 6.0])
        self.assertEqual(sum(out[:base] + out[base + 4 :]), 0.0)

    def test_gqa_layout_multiple_heads(self) -> None:
        rope, cache = rope_kv_cpu(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            num_heads=4,
            max_seq=5,
            head_idx=2,
            position=3,
        )
        base = (2 * 5 + 3) * 4
        self.assertEqual(cache[base : base + 4], rope)

    def test_invalid_cache_position_fails(self) -> None:
        with self.assertRaises(ValueError):
            kv_append_reference([0.0] * 8, [1.0, 2.0], head_idx=0, position=4, num_heads=1, max_seq=4, head_dim=2)

    def test_cuda_rope_kv_if_available(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        cos = [1.0, 1.0, 0.0, 0.0]
        sin = [0.0, 0.0, 1.0, 1.0]
        expected_rope, expected_cache = rope_kv_cpu(
            values, cos, sin, num_heads=2, max_seq=3, head_idx=1, position=2
        )
        try:
            from vinf.cuda.rope_kv import rope_kv_cuda

            rope, cache = rope_kv_cuda(
                values, cos, sin, num_heads=2, max_seq=3, head_idx=1, position=2
            )
        except Exception as exc:
            self.skipTest(f"CUDA RoPE/KV unavailable: {exc}")
        self.assert_close_lists(rope, expected_rope)
        self.assert_close_lists(cache, expected_cache)


if __name__ == "__main__":
    unittest.main()

