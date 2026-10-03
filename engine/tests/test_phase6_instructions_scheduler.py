from __future__ import annotations

import unittest

from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.runtime.instructions import (
    Attention,
    INTS_PER_INSTRUCTION,
    LMHead,
    NoOp,
    Opcode,
    RMSQKVRope,
    Verify,
    cuda_instruction_abi_header,
)
from vinf.runtime.scheduler import (
    ScheduleCache,
    build_baseline_decode_schedule,
    build_verification_schedule,
    round_robin_schedule,
)


def tiny_metadata() -> ModelMetadata:
    return ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        hidden_size=32,
        intermediate_size=64,
        head_dim=8,
        vocab_size=80,
        max_position_embeddings=128,
        dtype="fp16",
    )


class Phase6InstructionSchedulerTests(unittest.TestCase):
    def test_instruction_serialization_fixed_width_and_field_order(self) -> None:
        ins = RMSQKVRope(layer_idx=3, start_block=4, end_block=8, position=11)
        words = ins.serialize()
        self.assertEqual(len(words), INTS_PER_INSTRUCTION)
        self.assertEqual(words[:5], [Opcode.RMS_QKV_ROPE, 3, 4, 8, 11])
        self.assertTrue(all(word == 0 for word in words[5:]))

    def test_attention_and_lm_head_serialization(self) -> None:
        self.assertEqual(
            Attention(1, 2, 3, 4).serialize()[:5],
            [Opcode.ATTENTION, 1, 2, 3, 4],
        )
        self.assertEqual(LMHead(5, 9).serialize()[:3], [Opcode.LM_HEAD, 5, 9])

    def test_noop_padding_in_tensorized_schedule(self) -> None:
        schedule = round_robin_schedule(
            [RMSQKVRope(0, 0, 1, 0), Attention(0, 0, 0, 1), LMHead(0, 1)],
            num_sms=2,
        )
        tensor = schedule.tensorize()
        self.assertEqual(tensor.shape, (2, 2, INTS_PER_INSTRUCTION))
        self.assertEqual(tensor.rows[1][1][0], Opcode.NOOP)

    def test_baseline_decode_schedule_contains_expected_ops(self) -> None:
        schedule = build_baseline_decode_schedule(tiny_metadata(), num_sms=4, position=7)
        flattened = [ins for queue in schedule.queues for ins in queue]
        opcodes = [ins.opcode() for ins in flattened]
        self.assertIn(Opcode.RMS_QKV_ROPE, opcodes)
        self.assertIn(Opcode.ATTENTION, opcodes)
        self.assertIn(Opcode.O_PROJ, opcodes)
        self.assertIn(Opcode.MLP_UPGATE, opcodes)
        self.assertIn(Opcode.DOWN_PROJ, opcodes)
        self.assertEqual(opcodes.count(Opcode.LM_HEAD), 1)
        self.assertEqual(schedule.tensorize().shape[0], 4)

    def test_verification_schedule_starts_with_verify_instruction(self) -> None:
        schedule = build_verification_schedule(
            tiny_metadata(),
            num_sms=2,
            position=10,
            num_verify_tokens=3,
            draft_token_start=20,
        )
        flattened = [ins for queue in schedule.queues for ins in queue]
        verify = [ins for ins in flattened if isinstance(ins, Verify)]
        self.assertEqual(len(verify), 1)
        self.assertEqual(verify[0].serialize()[:4], [Opcode.VERIFY, 10, 3, 20])

    def test_schedule_cache_reuses_built_schedule(self) -> None:
        cache = ScheduleCache()
        calls = 0

        def build():
            nonlocal calls
            calls += 1
            return round_robin_schedule([NoOp()], 1)

        first = cache.get_or_build(("baseline", 1), build)
        second = cache.get_or_build(("baseline", 1), build)
        self.assertIs(first, second)
        self.assertEqual(calls, 1)

    def test_cuda_instruction_abi_header_contains_opcodes_and_layout(self) -> None:
        header = cuda_instruction_abi_header()
        self.assertIn("#define VINF_INTS_PER_INSTRUCTION 32", header)
        self.assertIn("#define VINF_OPCODE_VERIFY 7", header)
        self.assertIn("RMSQKVRope", header)


if __name__ == "__main__":
    unittest.main()

