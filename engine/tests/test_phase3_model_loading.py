from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vinf.config import EngineConfig
from vinf.errors import UnsupportedModelError
from vinf.models import (
    FIRST_SUPPORTED_SHAPE,
    expected_reference_weight_specs,
    load_draft_model,
    load_model_metadata,
    load_model_weights,
    load_target_model,
    validate_tokenizer_compatibility,
)
from vinf.models.loader import load_model
from vinf.models.metadata import ModelArchitecture, ModelMetadata


def model_dict(*, tokenizer_id: str = "tiny-tokenizer", dtype: str = "fp16"):
    vocab_size = FIRST_SUPPORTED_SHAPE["vocab_size"]
    return {
        "metadata": {
            "architecture": "llama",
            **FIRST_SUPPORTED_SHAPE,
            "dtype": dtype,
            "tokenizer_id": tokenizer_id,
        },
        "weights": {
            "transition_logits": [
                [float((row + col) % vocab_size) for col in range(vocab_size)]
                for row in range(vocab_size)
            ]
        },
    }


class Phase3ModelLoadingTests(unittest.TestCase):
    def write_model(self, data) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        with tmp:
            json.dump(data, tmp)
        return Path(tmp.name)

    def test_load_metadata_and_weights(self) -> None:
        path = self.write_model(model_dict())
        metadata = load_model_metadata(path)
        weights = load_model_weights(path)
        self.assertEqual(metadata.architecture, ModelArchitecture.LLAMA)
        self.assertEqual(metadata.dtype, "fp16")
        specs = expected_reference_weight_specs(metadata)
        weights.validate(specs)

    def test_load_target_model_checks_shape_dtype_weights_and_memory(self) -> None:
        loaded = load_target_model(self.write_model(model_dict()), EngineConfig())
        self.assertTrue(loaded.memory_fits)
        self.assertEqual(loaded.metadata.vocab_size, FIRST_SUPPORTED_SHAPE["vocab_size"])
        tensor = loaded.weights.get("transition_logits")
        self.assertEqual(tensor.shape, (8, 8))

    def test_load_model_rejects_unsupported_dtype(self) -> None:
        with self.assertRaises(UnsupportedModelError):
            load_model(self.write_model(model_dict(dtype="fp32")), EngineConfig())

    def test_load_model_rejects_unsupported_shape(self) -> None:
        data = model_dict()
        data["metadata"]["vocab_size"] = 9
        data["weights"]["transition_logits"] = [[0.0] * 9 for _ in range(9)]
        with self.assertRaises(UnsupportedModelError):
            load_model(self.write_model(data), EngineConfig())

    def test_load_model_rejects_missing_weight(self) -> None:
        data = model_dict()
        data["weights"] = {}
        with self.assertRaises(UnsupportedModelError):
            load_model(self.write_model(data), EngineConfig())

    def test_load_model_rejects_weight_shape_mismatch(self) -> None:
        data = model_dict()
        data["weights"]["transition_logits"] = [[0.0] * 8 for _ in range(7)]
        with self.assertRaises(UnsupportedModelError):
            load_model(self.write_model(data), EngineConfig())

    def test_load_model_rejects_memory_budget_failure(self) -> None:
        config = EngineConfig(max_seq_len=1_000_000_000)
        with self.assertRaises(UnsupportedModelError):
            load_model(self.write_model(model_dict()), config)

    def test_draft_tokenizer_compatibility(self) -> None:
        target = load_target_model(self.write_model(model_dict()), EngineConfig())
        draft = load_draft_model(
            self.write_model(model_dict(tokenizer_id="tiny-tokenizer")),
            target,
            EngineConfig(),
        )
        validate_tokenizer_compatibility(target.metadata, draft.metadata)

    def test_draft_tokenizer_mismatch_fails(self) -> None:
        target = load_target_model(self.write_model(model_dict()), EngineConfig())
        with self.assertRaises(UnsupportedModelError):
            load_draft_model(
                self.write_model(model_dict(tokenizer_id="other-tokenizer")),
                target,
                EngineConfig(),
            )

    def test_metadata_validation_still_catches_bad_kv_head_ratio(self) -> None:
        with self.assertRaises(ValueError):
            ModelMetadata(
                architecture=ModelArchitecture.LLAMA,
                num_hidden_layers=1,
                num_attention_heads=3,
                num_kv_heads=2,
                hidden_size=8,
                intermediate_size=16,
                head_dim=4,
                vocab_size=8,
                max_position_embeddings=32,
                dtype="fp16",
            )


if __name__ == "__main__":
    unittest.main()
