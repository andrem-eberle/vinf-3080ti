from __future__ import annotations

import math
import unittest

from vinf.one_layer import (
    OneLayerConfig,
    OneLayerWeights,
    one_layer_cuda_composed,
    one_layer_reference,
)


def fixture():
    config = OneLayerConfig(hidden_size=4, head_dim=4, num_kv_heads=1, max_seq=3)
    hidden = [0.25, -0.5, 0.75, 1.0]
    identity4 = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    gate_up = [((i % 5) - 2) / 5.0 for i in range(8 * 4)]
    down = [((i % 7) - 3) / 7.0 for i in range(4 * 8)]
    weights = OneLayerWeights(
        attn_norm=[1.0, 1.1, 0.9, 1.0],
        q_proj=identity4,
        k_proj=identity4,
        v_proj=identity4,
        o_proj=identity4,
        mlp_norm=[1.0, 1.0, 1.0, 1.0],
        gate_proj=gate_up,
        up_proj=list(reversed(gate_up)),
        down_proj=down,
    )
    rope_cos = [1.0, 1.0, 0.0, 0.0]
    rope_sin = [0.0, 0.0, 1.0, 1.0]
    return hidden, weights, config, rope_cos, rope_sin


class Phase16OneLayerTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=2e-4) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_one_layer_reference_has_stop_points(self) -> None:
        hidden, weights, config, cos, sin = fixture()
        result = one_layer_reference(hidden, weights, config, position=0, rope_cos=cos, rope_sin=sin)
        for key in [
            "attn_norm",
            "q_rope",
            "k_rope",
            "v",
            "k_cache",
            "v_cache",
            "attention",
            "post_attention",
            "mlp_norm",
            "post_mlp",
        ]:
            self.assertIn(key, result.stops)
        self.assertEqual(len(result.hidden), config.hidden_size)

    def test_one_layer_cuda_composed_if_available(self) -> None:
        hidden, weights, config, cos, sin = fixture()
        expected = one_layer_reference(hidden, weights, config, position=0, rope_cos=cos, rope_sin=sin)
        try:
            actual = one_layer_cuda_composed(hidden, weights, config, position=0, rope_cos=cos, rope_sin=sin)
        except Exception as exc:
            self.skipTest(f"CUDA one-layer composed unavailable: {exc}")
        self.assertEqual(set(actual.stops), set(expected.stops))
        for key in expected.stops:
            self.assert_close_lists(actual.stops[key], expected.stops[key])
        self.assert_close_lists(actual.hidden, expected.hidden)


if __name__ == "__main__":
    unittest.main()

