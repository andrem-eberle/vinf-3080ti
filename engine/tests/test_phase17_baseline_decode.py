from __future__ import annotations

import math
import unittest

from vinf.baseline_decode import (
    MAJOR_TIMING_SLOTS,
    BaselineDecodeConfig,
    BaselineTargetWeights,
    baseline_decode_cuda_composed,
    baseline_decode_reference,
)
from vinf.cuda.megakernel import TargetMegakernelRuntime, TIMING_WIDTH
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.one_layer import OneLayerWeights
from vinf.runtime.instructions import LMHead, Opcode


def fixture():
    metadata = ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_kv_heads=1,
        hidden_size=4,
        intermediate_size=8,
        head_dim=4,
        vocab_size=5,
        max_position_embeddings=3,
        dtype="fp16",
    )
    config = BaselineDecodeConfig(metadata=metadata, num_sms=4, block_size=2)
    hidden = [0.25, -0.5, 0.75, 1.0]
    identity4 = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    gate_up = [((i % 5) - 2) / 5.0 for i in range(8 * 4)]
    down = [((i % 7) - 3) / 7.0 for i in range(4 * 8)]
    layers = []
    for layer_idx in range(metadata.num_hidden_layers):
        scale = 1.0 + layer_idx * 0.05
        layers.append(
            OneLayerWeights(
                attn_norm=[scale, 1.1, 0.9, 1.0],
                q_proj=identity4,
                k_proj=identity4,
                v_proj=identity4,
                o_proj=identity4,
                mlp_norm=[1.0, scale, 1.0, 1.0],
                gate_proj=gate_up,
                up_proj=list(reversed(gate_up)),
                down_proj=down,
            )
        )
    lm_head = [((i % 9) - 4) / 9.0 for i in range(metadata.vocab_size * metadata.hidden_size)]
    weights = BaselineTargetWeights(
        layers=tuple(layers),
        final_norm=[1.0, 1.0, 0.95, 1.05],
        lm_head=lm_head,
    )
    rope_cos = [1.0, 1.0, 0.0, 0.0]
    rope_sin = [0.0, 0.0, 1.0, 1.0]
    return hidden, weights, config, rope_cos, rope_sin


class Phase17BaselineDecodeTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=2e-4) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_reference_instantiates_all_layers_and_emits_logits(self) -> None:
        hidden, weights, config, cos, sin = fixture()
        result = baseline_decode_reference(
            hidden,
            weights,
            config,
            position=0,
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assertEqual(len(result.hidden), config.metadata.hidden_size)
        self.assertEqual(len(result.logits), config.metadata.vocab_size)
        self.assertIn("layer_0.post_mlp", result.stops)
        self.assertIn("layer_1.post_mlp", result.stops)
        self.assertIn("logits", result.stops)

    def test_full_decode_schedule_covers_all_layers_and_lm_head(self) -> None:
        hidden, weights, config, cos, sin = fixture()
        result = baseline_decode_reference(
            hidden,
            weights,
            config,
            position=1,
            rope_cos=cos,
            rope_sin=sin,
        )
        flattened = [ins for queue in result.schedule.queues for ins in queue]
        opcodes = [ins.opcode() for ins in flattened]
        self.assertEqual(opcodes.count(Opcode.RMS_QKV_ROPE), config.metadata.num_hidden_layers)
        self.assertEqual(opcodes.count(Opcode.ATTENTION), config.metadata.num_hidden_layers)
        self.assertEqual(opcodes.count(Opcode.O_PROJ), config.metadata.num_hidden_layers)
        self.assertEqual(opcodes.count(Opcode.MLP_UPGATE), config.metadata.num_hidden_layers)
        self.assertEqual(opcodes.count(Opcode.DOWN_PROJ), config.metadata.num_hidden_layers)
        self.assertEqual([ins for ins in flattened if isinstance(ins, LMHead)][0].end_vocab_block, 3)

    def test_timing_slots_exist_for_major_ops(self) -> None:
        hidden, weights, config, cos, sin = fixture()
        result = baseline_decode_reference(
            hidden,
            weights,
            config,
            position=0,
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assertEqual(set(result.timings), set(MAJOR_TIMING_SLOTS))
        for value in result.timings.values():
            self.assertGreaterEqual(value, 0.0)

    def test_cuda_composed_logits_match_reference_if_available(self) -> None:
        hidden, weights, config, cos, sin = fixture()
        expected = baseline_decode_reference(
            hidden,
            weights,
            config,
            position=0,
            rope_cos=cos,
            rope_sin=sin,
        )
        try:
            actual = baseline_decode_cuda_composed(
                hidden,
                weights,
                config,
                position=0,
                rope_cos=cos,
                rope_sin=sin,
            )
        except Exception as exc:
            self.skipTest(f"CUDA composed baseline decode unavailable: {exc}")
        self.assert_close_lists(actual.hidden, expected.hidden)
        self.assert_close_lists(actual.logits, expected.logits)

    def test_target_megakernel_runtime_completes_one_token_decode(self) -> None:
        hidden, weights, config, cos, sin = fixture()
        expected = baseline_decode_reference(
            hidden,
            weights,
            config,
            position=0,
            rope_cos=cos,
            rope_sin=sin,
        )
        runtime = TargetMegakernelRuntime(num_sms=4, use_cuda=False)
        actual = runtime.decode_one_token(
            hidden,
            weights,
            config,
            position=0,
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assert_close_lists(actual.hidden, expected.hidden)
        self.assert_close_lists(actual.logits, expected.logits)
        self.assertIsNotNone(runtime.buffers)
        assert runtime.buffers is not None
        self.assertEqual(runtime.buffers.instruction_shape[0], 4)
        self.assertEqual(runtime.buffers.instruction_shape[2], 32)
        self.assertEqual(runtime.buffers.timing_shape[2], TIMING_WIDTH)
        self.assertEqual(runtime.last_decode_result, actual)
        self.assertEqual(len(runtime.read_timings()), 4)

    def test_target_megakernel_runtime_cuda_if_available(self) -> None:
        hidden, weights, config, cos, sin = fixture()
        expected = baseline_decode_reference(
            hidden,
            weights,
            config,
            position=0,
            rope_cos=cos,
            rope_sin=sin,
        )
        runtime = TargetMegakernelRuntime(num_sms=4, use_cuda=True)
        try:
            actual = runtime.decode_one_token(
                hidden,
                weights,
                config,
                position=0,
                rope_cos=cos,
                rope_sin=sin,
            )
        except Exception as exc:
            self.skipTest(f"CUDA target megakernel runtime unavailable: {exc}")
        self.assert_close_lists(actual.hidden, expected.hidden)
        self.assert_close_lists(actual.logits, expected.logits)


if __name__ == "__main__":
    unittest.main()
