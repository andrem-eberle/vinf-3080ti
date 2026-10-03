from __future__ import annotations

import unittest

from vinf.baseline_decode import BaselineDecodeConfig
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.one_layer import OneLayerWeights
from vinf.speculative.verify_ops import (
    verify_attention_reference,
    verify_qkv_rope_reference,
    verify_layer_reference,
    verify_logits_reference,
)


def metadata() -> ModelMetadata:
    return ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_kv_heads=1,
        hidden_size=4,
        intermediate_size=8,
        head_dim=4,
        vocab_size=5,
        max_position_embeddings=8,
        dtype="fp16",
    )


def weights() -> OneLayerWeights:
    identity4 = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    gate_up = [((i % 5) - 2) / 5.0 for i in range(8 * 4)]
    down = [((i % 7) - 3) / 7.0 for i in range(4 * 8)]
    return OneLayerWeights(
        attn_norm=[1.0, 1.0, 1.0, 1.0],
        q_proj=identity4,
        k_proj=identity4,
        v_proj=identity4,
        o_proj=identity4,
        mlp_norm=[1.0, 1.0, 1.0, 1.0],
        gate_proj=gate_up,
        up_proj=gate_up,
        down_proj=down,
    )


class Phase24BVerifyOpsTests(unittest.TestCase):
    def test_multi_position_qkv_rope_writes_speculative_kv_region(self) -> None:
        config = BaselineDecodeConfig(metadata=metadata())
        result = verify_qkv_rope_reference(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            weights(),
            config,
            start_position=2,
            rope_cos_rows=[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            rope_sin_rows=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )
        self.assertEqual(len(result.queries), 2)
        self.assertNotEqual(result.key_cache[2 * 4 : 3 * 4], [0.0] * 4)
        self.assertNotEqual(result.key_cache[3 * 4 : 4 * 4], [0.0] * 4)
        self.assertEqual(result.key_cache[4 * 4 :], [0.0] * 16)

    def test_verify_attention_uses_committed_plus_causal_speculative_positions(self) -> None:
        config = BaselineDecodeConfig(metadata=metadata())
        qkv = verify_qkv_rope_reference(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            weights(),
            config,
            start_position=0,
            rope_cos_rows=[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            rope_sin_rows=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )
        attention = verify_attention_reference(
            qkv.queries,
            qkv.key_cache,
            qkv.value_cache,
            config,
            start_position=0,
        )
        self.assertEqual(len(attention), 2)
        self.assertNotEqual(attention[0], attention[1])

    def test_verify_layer_runs_per_position_mlp_residual_and_logits(self) -> None:
        config = BaselineDecodeConfig(metadata=metadata())
        hidden = verify_layer_reference(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            weights(),
            config,
            start_position=0,
            rope_cos_rows=[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            rope_sin_rows=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )
        logits = verify_logits_reference(
            hidden,
            final_norm=[1.0, 1.0, 1.0, 1.0],
            lm_head=[((i % 5) - 2) / 5.0 for i in range(5 * 4)],
            config=config,
        )
        self.assertEqual(len(hidden), 2)
        self.assertEqual(len(logits), 2)
        self.assertEqual(len(logits[0]), 5)


if __name__ == "__main__":
    unittest.main()
