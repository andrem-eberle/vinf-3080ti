from __future__ import annotations

import unittest

from vinf.baseline_decode import BaselineDecodeConfig, BaselineTargetWeights
from vinf.baseline_loop import BaselineEngineLoop, TokenCodec
from vinf.config import GenerationConfig
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.one_layer import OneLayerWeights
from vinf.sampling import GreedySampler


def fixture(max_seq: int = 6, max_new_tokens: int = 2):
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
    loop = BaselineEngineLoop(
        weights=weights,
        config=config,
        generation=GenerationConfig(max_new_tokens=max_new_tokens),
        sampler=GreedySampler(),
    )
    rope_cos = [[1.0, 1.0, 0.0, 0.0] for _ in range(max_seq)]
    rope_sin = [[0.0, 0.0, 1.0, 1.0] for _ in range(max_seq)]
    return loop, rope_cos, rope_sin


def prompt_states(count: int) -> list[list[float]]:
    return [
        [0.25 + idx * 0.1, -0.5 + idx * 0.05, 0.75 - idx * 0.02, 1.0 + idx * 0.03]
        for idx in range(count)
    ]


class Phase19BaselineLoopTests(unittest.TestCase):
    def test_main_decode_loop_returns_generated_tokens_and_text(self) -> None:
        loop, cos, sin = fixture(max_new_tokens=2)
        result = loop.generate_from_hidden_states(
            prompt_states(3),
            prompt_tokens=[10, 11, 12],
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assertEqual(len(result.decode_result.tokens), 2)
        self.assertEqual(result.decode_result.text, " ".join(str(x) for x in result.decode_result.tokens))
        self.assertEqual(result.state.prompt_tokens, [10, 11, 12])
        self.assertEqual(result.state.position, 5)
        self.assertEqual(result.decode_result.stop_reason, "max_new_tokens")

    def test_loop_connects_prefill_decode_sampler_and_metrics(self) -> None:
        loop, cos, sin = fixture(max_new_tokens=1)
        result = loop.generate_from_hidden_states(
            prompt_states(2),
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assertEqual(result.metrics.generated_tokens, 1)
        self.assertEqual(result.metrics.target_calls, 1)
        self.assertIn("prefill", result.metrics.timings)
        self.assertIn("decode", result.metrics.timings)
        self.assertEqual(result.prefill.prompt_len, 3)

    def test_eos_stop_condition_wins_when_sampled(self) -> None:
        loop, cos, sin = fixture(max_new_tokens=4)
        loop.codec = TokenCodec(eos_token_id=0)
        result = loop.generate_from_hidden_states(
            prompt_states(2),
            rope_cos=cos,
            rope_sin=sin,
        )
        if result.decode_result.tokens[0] == 0:
            self.assertEqual(result.decode_result.stop_reason, "eos_token")
        else:
            self.assertNotEqual(result.decode_result.stop_reason, "eos_token")

    def test_stream_returns_generated_token_ids(self) -> None:
        loop, cos, sin = fixture(max_new_tokens=2)
        tokens = list(
            loop.stream_from_hidden_states(
                prompt_states(3),
                rope_cos=cos,
                rope_sin=sin,
            )
        )
        self.assertEqual(len(tokens), 2)

    def test_stress_stops_at_max_supported_context(self) -> None:
        loop, cos, sin = fixture(max_seq=4, max_new_tokens=4)
        result = loop.generate_from_hidden_states(
            prompt_states(3),
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assertEqual(len(result.decode_result.tokens), 1)
        self.assertEqual(result.decode_result.stop_reason, "max_position_embeddings")

    def test_prompt_tokens_must_match_prompt_hidden_states(self) -> None:
        loop, cos, sin = fixture()
        with self.assertRaises(ValueError):
            loop.generate_from_hidden_states(
                prompt_states(2),
                prompt_tokens=[1],
                rope_cos=cos,
                rope_sin=sin,
            )


if __name__ == "__main__":
    unittest.main()
