from __future__ import annotations

import os

import unittest
from pathlib import Path

from vinf.errors import UnsupportedModelError
from vinf.gguf.mapper import metadata_from_gguf
from vinf.gguf.parser import load_gguf
from vinf.gguf.qwen_tensors import (
    load_qwen_token_embeddings,
    load_qwen_lm_head_rows,
    map_qwen_tensor_name,
    qwen_lm_head_logits_for_tokens,
    qwen_tensor_coverage,
    validate_qwen_tensor_dimensions,
)
from vinf.gguf.tokenizer import load_qwen_tokenizer


QWEN_GGUF = Path(os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf"))


class Phase27QwenTensorTests(unittest.TestCase):
    def test_real_qwen_tensor_coverage_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        gguf = load_gguf(QWEN_GGUF)
        metadata = metadata_from_gguf(gguf)
        coverage = qwen_tensor_coverage(gguf, metadata)
        self.assertTrue(coverage.complete)
        self.assertEqual(len(coverage.hybrid_layers) + len(coverage.full_attention_layers), 65)
        self.assertGreater(len(coverage.hybrid_layers), 0)
        self.assertGreater(len(coverage.full_attention_layers), 0)

    def test_real_qwen_tensor_dimensions_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        gguf = load_gguf(QWEN_GGUF)
        validate_qwen_tensor_dimensions(gguf, metadata_from_gguf(gguf))

    def test_qwen_tensor_name_mapping(self) -> None:
        self.assertEqual(map_qwen_tensor_name("token_embd.weight"), "embed_tokens.weight")
        self.assertEqual(map_qwen_tensor_name("blk.7.attn_qkv.weight"), "layers.7.attn_qkv.weight")

    def test_real_qwen_token_embedding_lookup_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        gguf = load_gguf(QWEN_GGUF)
        tokenizer = load_qwen_tokenizer(gguf)
        token_ids = tokenizer.encode("hello")
        embeddings = load_qwen_token_embeddings(gguf, token_ids[:2])
        self.assertEqual(len(embeddings), len(token_ids[:2]))
        self.assertEqual(len(embeddings[0]), 5120)
        self.assertTrue(all(value == value for value in embeddings[0][:128]))
        self.assertNotEqual(embeddings[0][:16], [0.0] * 16)

    def test_embedding_lookup_rejects_out_of_range_token_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        gguf = load_gguf(QWEN_GGUF)
        with self.assertRaises(UnsupportedModelError):
            load_qwen_token_embeddings(gguf, [248320])

    def test_real_qwen_lm_head_selected_rows_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        gguf = load_gguf(QWEN_GGUF)
        rows = load_qwen_lm_head_rows(gguf, [0, 1])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 5120)
        self.assertTrue(all(value == value for value in rows[0][:128]))

    def test_real_qwen_selected_lm_logits_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        gguf = load_gguf(QWEN_GGUF)
        hidden = load_qwen_token_embeddings(gguf, [0])[0]
        logits = qwen_lm_head_logits_for_tokens(gguf, hidden, [0, 1, 2])
        self.assertEqual(len(logits), 3)
        self.assertTrue(all(value == value for value in logits))

    def test_dimension_validation_fails_cleanly_for_wrong_metadata(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        gguf = load_gguf(QWEN_GGUF)
        metadata = metadata_from_gguf(gguf)
        bad_metadata = type(metadata)(
            architecture=metadata.architecture,
            num_hidden_layers=metadata.num_hidden_layers,
            num_attention_heads=metadata.num_attention_heads,
            num_kv_heads=metadata.num_kv_heads,
            hidden_size=metadata.hidden_size + 1,
            intermediate_size=metadata.intermediate_size,
            head_dim=metadata.head_dim,
            vocab_size=metadata.vocab_size,
            max_position_embeddings=metadata.max_position_embeddings,
            dtype=metadata.dtype,
            tokenizer_id=metadata.tokenizer_id,
        )
        with self.assertRaises(UnsupportedModelError):
            validate_qwen_tensor_dimensions(gguf, bad_metadata)


if __name__ == "__main__":
    unittest.main()
