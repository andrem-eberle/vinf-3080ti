from __future__ import annotations

import os

import unittest
from pathlib import Path

from vinf.errors import UnsupportedModelError
from vinf.gguf.parser import GGUFFile, GGUFMetadataValue, GGUFValueType, load_gguf
from vinf.gguf.tokenizer import ChatMessage, load_qwen_tokenizer


QWEN_GGUF = Path(os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf"))


class Phase28QwenTokenizerTests(unittest.TestCase):
    def test_loads_real_qwen_tokenizer_metadata_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        tokenizer = load_qwen_tokenizer(load_gguf(QWEN_GGUF))
        self.assertEqual(tokenizer.model, "gpt2")
        self.assertEqual(tokenizer.pre, "qwen35")
        self.assertEqual(tokenizer.vocab_size, 248320)
        self.assertEqual(tokenizer.special_tokens.bos_token_id, 248044)
        self.assertEqual(tokenizer.special_tokens.eos_token_id, 248046)
        self.assertEqual(tokenizer.special_tokens.padding_token_id, 248055)
        self.assertEqual(tokenizer.special_tokens.im_start_id, 248045)
        self.assertEqual(tokenizer.special_tokens.im_end_id, 248046)

    def test_round_trip_ascii_and_utf8_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        tokenizer = load_qwen_tokenizer(load_gguf(QWEN_GGUF))
        for text in ("hello world", "cafe naive", "olá mundo 日本語"):
            ids = tokenizer.encode(text)
            self.assertTrue(ids)
            self.assertEqual(tokenizer.decode(ids), text)

    def test_special_tokens_and_chat_boundaries_if_file_exists(self) -> None:
        if not QWEN_GGUF.exists():
            self.skipTest("local Qwen GGUF is not present")
        tokenizer = load_qwen_tokenizer(load_gguf(QWEN_GGUF))
        prompt = tokenizer.apply_chat_template(
            [
                ChatMessage("system", "You are concise."),
                ChatMessage("user", "Hello"),
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.assertTrue(prompt.startswith("<|im_start|>system\n"))
        self.assertIn("<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        ids = tokenizer.encode(prompt)
        self.assertEqual(ids[0], tokenizer.special_tokens.im_start_id)
        self.assertIn(tokenizer.special_tokens.im_end_id, ids)
        self.assertEqual(tokenizer.decode(ids), prompt)
        # User-defined special tokens (GGUF type 4) are atomic, like control tokens.
        think, end_think = tokenizer.encode("<think>"), tokenizer.encode("</think>")
        self.assertEqual(len(think), 1)
        self.assertEqual(len(end_think), 1)
        self.assertIn(think[0], ids)
        self.assertIn(end_think[0], ids)
        # Unused PAD slots (type 5) are not matched as specials.
        self.assertGreater(len(tokenizer.encode("[PAD248077]")), 1)

    def test_missing_tokenizer_metadata_fails_cleanly(self) -> None:
        gguf = GGUFFile(
            path=Path("missing.gguf"),
            version=3,
            metadata={
                "tokenizer.ggml.tokens": GGUFMetadataValue(GGUFValueType.ARRAY, ["a"]),
            },
            tensors={},
            data_start=0,
            alignment=32,
        )
        with self.assertRaises(UnsupportedModelError):
            load_qwen_tokenizer(gguf)


if __name__ == "__main__":
    unittest.main()
