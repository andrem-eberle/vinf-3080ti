from __future__ import annotations

import unittest

from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.runtime.instructions import Opcode, Verify
from vinf.speculative.megakernel import (
    VerificationGlobals,
    build_speculative_megakernel_schedule,
    causal_verify_mask,
    verification_buffer_plan,
    verification_inputs,
)
from vinf.speculative.abi import VerificationGlobalsABI, verification_globals_abi_header


def metadata() -> ModelMetadata:
    return ModelMetadata(
        architecture=ModelArchitecture.LLAMA,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_kv_heads=1,
        hidden_size=4,
        intermediate_size=8,
        head_dim=4,
        vocab_size=5,
        max_position_embeddings=16,
        dtype="fp16",
    )


class Phase24SpeculativeMegakernelTests(unittest.TestCase):
    def test_verification_globals_validate_num_verify_tokens_and_max_gamma(self) -> None:
        globals = VerificationGlobals(position=4, num_verify_tokens=3, max_gamma=2)
        self.assertEqual(globals.num_verify_tokens, 3)
        with self.assertRaises(ValueError):
            VerificationGlobals(position=0, num_verify_tokens=4, max_gamma=2)

    def test_verification_globals_abi_field_order_and_header(self) -> None:
        abi = VerificationGlobalsABI(
            position=4,
            num_verify_tokens=3,
            max_gamma=2,
            draft_token_start=9,
        )
        self.assertEqual(abi.serialize(), (4, 3, 2, 9))
        header = verification_globals_abi_header()
        self.assertIn("struct VinfVerificationGlobals", header)
        self.assertIn("int num_verify_tokens;", header)
        self.assertIn("#define VINF_VERIFY_MAX_GAMMA_WORD 2", header)

    def test_buffer_plan_extends_activation_logits_probabilities_and_spec_kv(self) -> None:
        plan = verification_buffer_plan(metadata(), max_gamma=2)
        self.assertEqual(plan.activation_shape, (3, 4))
        self.assertEqual(plan.logits_shape, (3, 5))
        self.assertEqual(plan.probabilities_shape, (3, 5))
        self.assertEqual(plan.speculative_kv_shape, (2, 1, 3, 4))
        self.assertEqual(plan.draft_token_shape, (2,))
        self.assertEqual(plan.position_id_shape, (3,))

    def test_verification_inputs_prepare_draft_tokens_and_position_ids(self) -> None:
        inputs = verification_inputs((4, 5), position=7, max_gamma=2)
        self.assertEqual(inputs.draft_token_ids, (4, 5))
        self.assertEqual(inputs.position_ids, (7, 8, 9))
        with self.assertRaises(ValueError):
            verification_inputs((1, 2, 3), position=0, max_gamma=2)

    def test_causal_mask_allows_previous_speculative_positions_only(self) -> None:
        self.assertEqual(
            causal_verify_mask(3),
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
        )

    def test_verification_schedule_starts_with_verify_instruction(self) -> None:
        globals = VerificationGlobals(
            position=7,
            num_verify_tokens=3,
            max_gamma=2,
            draft_token_start=11,
        )
        schedule = build_speculative_megakernel_schedule(
            metadata(),
            num_sms=2,
            globals=globals,
        )
        flattened = [ins for queue in schedule.queues for ins in queue]
        verify = [ins for ins in flattened if isinstance(ins, Verify)]
        self.assertEqual(len(verify), 1)
        self.assertEqual(verify[0].serialize()[:4], [Opcode.VERIFY, 7, 3, 11])


if __name__ == "__main__":
    unittest.main()
