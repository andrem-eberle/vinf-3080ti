from __future__ import annotations

import math
import unittest

from vinf.baseline_decode import BaselineDecodeConfig, BaselineTargetWeights
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.one_layer import OneLayerWeights
from vinf.prefill import (
    PrefillBackend,
    decode_after_prefill_reference,
    estimate_prefill_memory_bytes,
    target_prefill_from_tokens_reference,
    target_prefill_reference,
)
from tests.test_phase5_gguf import write_tiny_gguf
from vinf.gguf.parser import load_gguf


def fixture(max_seq: int = 5):
    metadata = ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_kv_heads=1,
        hidden_size=4,
        intermediate_size=8,
        head_dim=4,
        vocab_size=5,
        max_position_embeddings=max_seq,
        dtype="fp16",
    )
    config = BaselineDecodeConfig(metadata=metadata, num_sms=4, block_size=2)
    identity4 = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    gate_up = [((i % 5) - 2) / 5.0 for i in range(8 * 4)]
    down = [((i % 7) - 3) / 7.0 for i in range(4 * 8)]
    layers = tuple(
        OneLayerWeights(
            attn_norm=[1.0 + layer_idx * 0.05, 1.1, 0.9, 1.0],
            q_proj=identity4,
            k_proj=identity4,
            v_proj=identity4,
            o_proj=identity4,
            mlp_norm=[1.0, 1.0 + layer_idx * 0.05, 1.0, 1.0],
            gate_proj=gate_up,
            up_proj=list(reversed(gate_up)),
            down_proj=down,
        )
        for layer_idx in range(metadata.num_hidden_layers)
    )
    weights = BaselineTargetWeights(
        layers=layers,
        final_norm=[1.0, 1.0, 0.95, 1.05],
        lm_head=[((i % 9) - 4) / 9.0 for i in range(metadata.vocab_size * metadata.hidden_size)],
    )
    rope_cos = [[1.0, 1.0, 0.0, 0.0] for _ in range(max_seq)]
    rope_sin = [[0.0, 0.0, 1.0, 1.0] for _ in range(max_seq)]
    return weights, config, rope_cos, rope_sin


def prompt_states(count: int) -> list[list[float]]:
    return [
        [0.25 + idx * 0.1, -0.5 + idx * 0.05, 0.75 - idx * 0.02, 1.0 + idx * 0.03]
        for idx in range(count)
    ]


def token_prefill_fixture(max_seq: int = 5):
    metadata = ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_kv_heads=1,
        hidden_size=8,
        intermediate_size=16,
        head_dim=8,
        vocab_size=8,
        max_position_embeddings=max_seq,
        dtype="fp16",
    )
    config = BaselineDecodeConfig(metadata=metadata, num_sms=4, block_size=2)
    identity = [
        1.0 if row == col else 0.0
        for row in range(metadata.hidden_size)
        for col in range(metadata.hidden_size)
    ]
    gate_up = [((i % 5) - 2) / 5.0 for i in range(metadata.intermediate_size * metadata.hidden_size)]
    down = [((i % 7) - 3) / 7.0 for i in range(metadata.hidden_size * metadata.intermediate_size)]
    weights = BaselineTargetWeights(
        layers=(
            OneLayerWeights(
                attn_norm=[1.0] * metadata.hidden_size,
                q_proj=identity,
                k_proj=identity,
                v_proj=identity,
                o_proj=identity,
                mlp_norm=[1.0] * metadata.hidden_size,
                gate_proj=gate_up,
                up_proj=list(reversed(gate_up)),
                down_proj=down,
            ),
        ),
        final_norm=[1.0] * metadata.hidden_size,
        lm_head=[0.0] * (metadata.vocab_size * metadata.hidden_size),
    )
    rope_cos = [[1.0] * metadata.hidden_size for _ in range(max_seq)]
    rope_sin = [[0.0] * metadata.hidden_size for _ in range(max_seq)]
    return weights, config, rope_cos, rope_sin


class Phase18PrefillTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=2e-4) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_prefill_uses_fallback_backend_and_fills_target_kv(self) -> None:
        weights, config, cos, sin = fixture()
        state = target_prefill_reference(
            prompt_states(3),
            weights,
            config,
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assertEqual(state.backend, PrefillBackend.FALLBACK_STANDALONE_OPS)
        self.assertEqual(state.prompt_len, 3)
        self.assertEqual(len(state.layers), config.metadata.num_hidden_layers)
        cache_width = config.metadata.max_position_embeddings * config.metadata.head_dim
        for layer in state.layers:
            self.assertEqual(len(layer.key), cache_width)
            self.assertEqual(len(layer.value), cache_width)
            self.assertNotEqual(layer.key[: 3 * config.metadata.head_dim], [0.0] * 12)
            self.assertEqual(layer.key[3 * config.metadata.head_dim :], [0.0] * 8)

    def test_prefill_decode_handoff_matches_full_prefill(self) -> None:
        weights, config, cos, sin = fixture()
        prompt = prompt_states(3)
        next_hidden = prompt_states(4)[-1]
        prefill = target_prefill_reference(prompt, weights, config, rope_cos=cos, rope_sin=sin)
        handoff = decode_after_prefill_reference(
            next_hidden,
            prefill,
            weights,
            config,
            rope_cos=cos,
            rope_sin=sin,
        )
        full = target_prefill_reference(
            [*prompt, next_hidden],
            weights,
            config,
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assert_close_lists(handoff.hidden, full.final_hidden)
        for actual_layer, expected_layer in zip(handoff.layers, full.layers):
            self.assert_close_lists(actual_layer.key, expected_layer.key)
            self.assert_close_lists(actual_layer.value, expected_layer.value)

    def test_long_prompt_at_max_context_prefills_but_cannot_decode_more(self) -> None:
        weights, config, cos, sin = fixture(max_seq=4)
        state = target_prefill_reference(
            prompt_states(4),
            weights,
            config,
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assertEqual(state.prompt_len, 4)
        with self.assertRaises(ValueError):
            decode_after_prefill_reference(
                prompt_states(1)[0],
                state,
                weights,
                config,
                rope_cos=cos,
                rope_sin=sin,
            )

    def test_prefill_rejects_prompt_beyond_context(self) -> None:
        weights, config, cos, sin = fixture(max_seq=3)
        with self.assertRaises(ValueError):
            target_prefill_reference(
                prompt_states(4),
                weights,
                config,
                rope_cos=cos,
                rope_sin=sin,
            )

    def test_prefill_memory_estimate_accounts_for_kv_and_prompt_hidden(self) -> None:
        _, config, _, _ = fixture(max_seq=5)
        expected_values = (2 * 2 * 1 * 5 * 4) + (3 * 4)
        self.assertEqual(estimate_prefill_memory_bytes(config, 3), expected_values * 2)

    def test_token_backed_prefill_uses_gguf_embeddings(self) -> None:
        weights, config, cos, sin = token_prefill_fixture(max_seq=5)
        gguf = load_gguf(write_tiny_gguf())
        state = target_prefill_from_tokens_reference(
            [0, 1],
            gguf,
            weights,
            config,
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assertEqual(state.prompt_len, 2)
        self.assertEqual(state.backend, PrefillBackend.FALLBACK_STANDALONE_OPS)
        self.assertEqual(len(state.final_hidden), config.metadata.hidden_size)


if __name__ == "__main__":
    unittest.main()
