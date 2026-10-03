from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vinf.gguf.mapper import map_tensor_name, metadata_from_gguf
from vinf.gguf.parser import (
    DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUFTensorType,
    GGUFValueType,
    load_gguf,
)


def align(value: int, alignment: int = DEFAULT_ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def pack_kv(key: str, value_type: GGUFValueType, value) -> bytes:
    out = bytearray()
    out += pack_string(key)
    out += struct.pack("<I", int(value_type))
    if value_type is GGUFValueType.STRING:
        out += pack_string(value)
    elif value_type is GGUFValueType.UINT32:
        out += struct.pack("<I", value)
    elif value_type is GGUFValueType.ARRAY:
        item_type, values = value
        out += struct.pack("<I", int(item_type))
        out += struct.pack("<Q", len(values))
        for item in values:
            if item_type is GGUFValueType.STRING:
                out += pack_string(item)
            else:
                raise AssertionError("fixture only supports string arrays")
    else:
        raise AssertionError(f"fixture cannot encode {value_type}")
    return bytes(out)


def pack_tensor(name: str, dims: tuple[int, ...], tensor_type: GGUFTensorType, offset: int) -> bytes:
    out = bytearray()
    out += pack_string(name)
    out += struct.pack("<I", len(dims))
    for dim in dims:
        out += struct.pack("<Q", dim)
    out += struct.pack("<I", int(tensor_type))
    out += struct.pack("<Q", offset)
    return bytes(out)


def write_tiny_gguf(*, tensor_type: GGUFTensorType = GGUFTensorType.F16) -> Path:
    metadata = [
        pack_kv("general.architecture", GGUFValueType.STRING, "llama"),
        pack_kv("general.alignment", GGUFValueType.UINT32, DEFAULT_ALIGNMENT),
        pack_kv("llama.block_count", GGUFValueType.UINT32, 1),
        pack_kv("llama.attention.head_count", GGUFValueType.UINT32, 2),
        pack_kv("llama.attention.head_count_kv", GGUFValueType.UINT32, 1),
        pack_kv("llama.embedding_length", GGUFValueType.UINT32, 8),
        pack_kv("llama.feed_forward_length", GGUFValueType.UINT32, 16),
        pack_kv("llama.context_length", GGUFValueType.UINT32, 32),
        pack_kv(
            "tokenizer.ggml.tokens",
            GGUFValueType.ARRAY,
            (GGUFValueType.STRING, [str(i) for i in range(8)]),
        ),
        pack_kv("tokenizer.ggml.model", GGUFValueType.STRING, "tiny-tokenizer"),
    ]
    tensors = [
        pack_tensor("token_embd.weight", (8, 8), tensor_type, 0),
    ]
    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))
    for item in metadata:
        header += item
    for item in tensors:
        header += item
    data_start = align(len(header))
    header += b"\x00" * (data_start - len(header))
    if tensor_type is GGUFTensorType.F16:
        payload = b"\x00" * (8 * 8 * 2)
    elif tensor_type is GGUFTensorType.Q4_0:
        payload = b"\x00" * (2 * 18)
    else:
        raise AssertionError("fixture does not know this tensor payload")
    tmp = tempfile.NamedTemporaryFile("wb", suffix=".gguf", delete=False)
    with tmp:
        tmp.write(header + payload)
    return Path(tmp.name)


class Phase5GGUFTests(unittest.TestCase):
    def test_parse_tiny_fp16_gguf_metadata_and_tensor_directory(self) -> None:
        gguf = load_gguf(write_tiny_gguf())
        self.assertEqual(gguf.version, 3)
        self.assertEqual(gguf.metadata_value("general.architecture"), "llama")
        tensor = gguf.tensors["token_embd.weight"]
        self.assertEqual(tensor.dimensions, (8, 8))
        self.assertEqual(tensor.tensor_type, GGUFTensorType.F16)
        self.assertEqual(tensor.nbytes, 128)
        self.assertEqual(tensor.absolute_offset, gguf.data_start)

    def test_map_gguf_metadata_to_internal_model_metadata(self) -> None:
        metadata = metadata_from_gguf(load_gguf(write_tiny_gguf()))
        self.assertEqual(metadata.hidden_size, 8)
        self.assertEqual(metadata.num_attention_heads, 2)
        self.assertEqual(metadata.num_kv_heads, 1)
        self.assertEqual(metadata.vocab_size, 8)
        self.assertEqual(metadata.dtype, "fp16")
        self.assertEqual(metadata.tokenizer_id, "tiny-tokenizer")

    def test_quantized_tensor_metadata_and_mmap_are_supported(self) -> None:
        gguf = load_gguf(write_tiny_gguf(tensor_type=GGUFTensorType.Q4_0))
        tensor = gguf.tensors["token_embd.weight"]
        self.assertTrue(tensor.is_quantized)
        self.assertEqual(tensor.block_size, 32)
        self.assertEqual(tensor.type_size, 18)
        self.assertEqual(tensor.nbytes, 36)
        self.assertEqual(metadata_from_gguf(gguf).dtype, "quantized")
        file_obj, mm, view = gguf.mmap_tensor("token_embd.weight")
        try:
            self.assertEqual(len(view), 36)
        finally:
            view.release()
            mm.close()
            file_obj.close()

    def test_tensor_name_mapping(self) -> None:
        self.assertEqual(map_tensor_name("token_embd.weight"), "embed_tokens.weight")
        self.assertEqual(map_tensor_name("blk.3.attn_q.weight"), "layers.3.attn_q.weight")

    def test_bad_magic_fails(self) -> None:
        tmp = tempfile.NamedTemporaryFile("wb", suffix=".gguf", delete=False)
        with tmp:
            tmp.write(b"NOPE")
        with self.assertRaises(Exception):
            load_gguf(tmp.name)

    def test_parser_streams_header_without_reading_entire_file(self) -> None:
        path = write_tiny_gguf()
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("no full-file read")):
            gguf = load_gguf(path)
        self.assertEqual(gguf.version, 3)


if __name__ == "__main__":
    unittest.main()
