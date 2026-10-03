from __future__ import annotations

import math
import unittest

from vinf.cuda.attention import attention_cpu


class Phase13AttentionTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=1e-5) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_cpu_attention_short_context(self) -> None:
        out = attention_cpu(
            [1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0, 20.0],
            kv_head_idx=0,
            seq_len=2,
            num_kv_heads=1,
            max_seq=2,
            head_dim=2,
            scale=1.0,
        )
        p0 = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
        p1 = 1.0 - p0
        self.assert_close_lists(out, [10.0 * p0, 20.0 * p1])

    def test_cpu_attention_longer_context(self) -> None:
        head_dim = 4
        seq_len = 8
        key = []
        value = []
        for pos in range(seq_len):
            key.extend([(pos + dim) / 10.0 for dim in range(head_dim)])
            value.extend([float(pos), float(pos + 1), float(pos + 2), float(pos + 3)])
        out = attention_cpu(
            [0.1, 0.2, 0.3, 0.4],
            key,
            value,
            kv_head_idx=0,
            seq_len=seq_len,
            num_kv_heads=1,
            max_seq=seq_len,
            head_dim=head_dim,
        )
        self.assertEqual(len(out), head_dim)
        self.assertTrue(all(math.isfinite(x) for x in out))

    def test_gqa_kv_head_selection(self) -> None:
        # Two KV heads, one position each, head 1 should be selected.
        key = [1.0, 0.0, 0.0, 1.0]
        value = [5.0, 6.0, 7.0, 8.0]
        out = attention_cpu(
            [0.0, 1.0],
            key,
            value,
            kv_head_idx=1,
            seq_len=1,
            num_kv_heads=2,
            max_seq=1,
            head_dim=2,
            scale=1.0,
        )
        self.assertEqual(out, [7.0, 8.0])

    def test_cuda_attention_if_available(self) -> None:
        query = [1.0, 0.0]
        key = [1.0, 0.0, 0.0, 1.0]
        value = [10.0, 0.0, 0.0, 20.0]
        expected = attention_cpu(
            query,
            key,
            value,
            kv_head_idx=0,
            seq_len=2,
            num_kv_heads=1,
            max_seq=2,
            head_dim=2,
            scale=1.0,
        )
        try:
            from vinf.cuda.attention import attention_cuda

            actual = attention_cuda(
                query,
                key,
                value,
                kv_head_idx=0,
                seq_len=2,
                num_kv_heads=1,
                max_seq=2,
                head_dim=2,
                scale=1.0,
            )
        except Exception as exc:
            self.skipTest(f"CUDA attention unavailable: {exc}")
        self.assert_close_lists(actual, expected)


if __name__ == "__main__":
    unittest.main()

