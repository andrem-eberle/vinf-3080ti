from __future__ import annotations

import struct
import unittest

from vinf.errors import UnsupportedModelError
from vinf.gguf.dequant import (
    dequantize_tensor,
    load_dequantized_tensor,
    supported_cpu_dequant_types,
    unsupported_tensor_types,
)
from vinf.gguf.parser import GGUFTensorInfo, GGUFTensorType
from tests.test_phase5_gguf import write_tiny_gguf


def tensor(name: str, dims: tuple[int, ...], typ: GGUFTensorType) -> GGUFTensorInfo:
    return GGUFTensorInfo(
        name=name,
        dimensions=dims,
        tensor_type=typ,
        relative_offset=0,
        absolute_offset=0,
    )


class GGUFDequantTests(unittest.TestCase):
    def test_dequant_f32(self) -> None:
        info = tensor("x", (3,), GGUFTensorType.F32)
        data = memoryview(struct.pack("<fff", 1.0, -2.0, 3.5))
        self.assertEqual(dequantize_tensor(info, data).values, [1.0, -2.0, 3.5])

    def test_dequant_f16(self) -> None:
        info = tensor("x", (2,), GGUFTensorType.F16)
        data = memoryview(struct.pack("<ee", 1.5, -2.0))
        self.assertEqual(dequantize_tensor(info, data).values, [1.5, -2.0])

    def test_dequant_q8_0(self) -> None:
        info = tensor("x", (32,), GGUFTensorType.Q8_0)
        data = memoryview(struct.pack("<e32b", 0.5, *range(-16, 16)))
        actual = dequantize_tensor(info, data).values
        self.assertEqual(len(actual), 32)
        self.assertEqual(actual[0], -8.0)
        self.assertEqual(actual[-1], 7.5)

    def test_dequant_q4_0(self) -> None:
        info = tensor("x", (32,), GGUFTensorType.Q4_0)
        qs = bytes(low | (high << 4) for low, high in zip(range(16), reversed(range(16))))
        actual = dequantize_tensor(info, memoryview(struct.pack("<e", 0.5) + qs)).values
        self.assertEqual(len(actual), 32)
        self.assertEqual(actual[:4], [-4.0, -3.5, -3.0, -2.5])
        self.assertEqual(actual[16:20], [3.5, 3.0, 2.5, 2.0])

    def test_dequant_q4_k(self) -> None:
        # ggml layout: groups 2p/2p+1 are low/high nibbles of qs[32p:32p+32].
        info = tensor("x", (256,), GGUFTensorType.Q4_K)
        scales = bytes([1, 2, 3, 4, 0, 1, 2, 3, 5, 6, 7, 8])
        low_nibbles = list(range(16))
        high_nibbles = list(reversed(range(16)))
        qs = bytes(low | (high << 4) for low, high in zip(low_nibbles, high_nibbles))
        block = struct.pack("<ee", 0.5, 0.25) + scales + qs * 8
        actual = dequantize_tensor(info, memoryview(block)).values
        self.assertEqual(len(actual), 256)
        self.assertEqual(actual[:4], [0.0, 0.5, 1.0, 1.5])
        # Elements 16..31 are low nibbles of bytes 16..31, still group 0.
        self.assertEqual(actual[16:20], [0.0, 0.5, 1.0, 1.5])
        # Group 1 (sc=2, m=1): high nibbles of bytes 0..31.
        self.assertEqual(actual[32:34], [14.75, 13.75])
        self.assertEqual(actual[64], -0.5)
        self.assertEqual(actual[96], 29.25)
        self.assertEqual(actual[128], 0.0)
        self.assertEqual(actual[160], 45.0)
        self.assertEqual(actual[192], 0.0)
        self.assertEqual(actual[224], 60.0)

    def test_dequant_q3_k(self) -> None:
        # ggml layout: element e uses qs[32*(e//128) + e%32] >> 2*((e%128)//32),
        # hmask[e%32] bit e//32, and scale k=e//16 with high bits from byte 8 + k%4.
        info = tensor("x", (256,), GGUFTensorType.Q3_K)
        hmask = bytearray([0xFF] * 32)
        hmask[3] &= ~(1 << 5) & 0xFF  # clears the high bit of element e=163 only
        qs = bytearray([0xFF] * 64)
        qs[32] = 0x00  # second 128-half, byte 0: elements 128, 160, 192, 224
        low = bytes(k | (k + 8) << 4 for k in range(8))
        scales = low + bytes([0xE4] * 4)  # scale[k] = k + 16 * (k // 4) - 32
        block = bytes(hmask) + bytes(qs) + scales + struct.pack("<e", 0.5)
        actual = dequantize_tensor(info, memoryview(block)).values
        self.assertEqual(len(actual), 256)
        self.assertEqual(actual[0], -48.0)
        self.assertEqual(actual[3], -48.0)
        self.assertEqual(actual[35], -45.0)
        self.assertEqual(actual[80], -16.5)
        self.assertEqual(actual[128], 0.0)
        self.assertEqual(actual[129], 12.0)
        self.assertEqual(actual[160], 0.0)
        self.assertEqual(actual[163], -5.0)
        self.assertEqual(actual[255], 46.5)

    def test_iq3_s_grid_unpacks_two_3bit_values_per_byte(self) -> None:
        from vinf.gguf.dequant import _iq3_s_grid

        grid = _iq3_s_grid()
        self.assertEqual(len(grid), 512)
        self.assertEqual(grid[0], (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(grid[3], (11.0, 1.0, 1.0, 1.0))  # bytes 05 00
        self.assertEqual(grid[5], (1.0, 3.0, 1.0, 1.0))  # bytes 10 00

    def test_dequant_q2_k(self) -> None:
        # Expected values follow ggml's dequantize_row_q2_K loop order (halves n, shifts j, sub-blocks s).
        info = tensor("x", (256,), GGUFTensorType.Q2_K)
        scales = bytes((k % 16) | (((7 * k) % 16) << 4) for k in range(16))
        qs = bytes((i * 37 + 11) % 256 for i in range(64))
        block = scales + qs + struct.pack("<ee", 0.5, 0.25)
        actual = dequantize_tensor(info, memoryview(block)).values
        expected = []
        idx = 0
        for n in range(2):
            q = qs[32 * n : 32 * n + 32]
            for j in range(4):
                for s in range(2):
                    sc = scales[idx]
                    idx += 1
                    for l in range(16):
                        expected.append(0.5 * (sc & 0xF) * ((q[16 * s + l] >> (2 * j)) & 3) - 0.25 * (sc >> 4))
        self.assertEqual(actual, expected)

    def test_dequant_q5_k(self) -> None:
        info = tensor("x", (256,), GGUFTensorType.Q5_K)
        scales = bytes([1, 2, 3, 4, 0, 1, 2, 3, 5, 6, 7, 8])
        qh = bytes([0b01010101] * 32)
        qs_pair = bytes(low | (high << 4) for low, high in zip(range(16), reversed(range(16))))
        block = struct.pack("<ee", 0.5, 0.25) + scales + qh + qs_pair * 8
        actual = dequantize_tensor(info, memoryview(block)).values
        self.assertEqual(len(actual), 256)
        self.assertEqual(actual[:4], [8.0, 8.5, 9.0, 9.5])
        self.assertEqual(actual[32:36], [14.75, 13.75, 12.75, 11.75])
        self.assertEqual(actual[64], 23.5)
        self.assertEqual(actual[96], 29.25)
        self.assertEqual(actual[128], 40.0)
        self.assertEqual(actual[160], 45.0)
        self.assertEqual(actual[192], 56.0)
        self.assertEqual(actual[224], 60.0)

    def test_dequant_q6_k(self) -> None:
        info = tensor("x", (256,), GGUFTensorType.Q6_K)
        ql = bytes([0x10] * 128)
        qh = bytes([0b11100100] * 64)
        scales = struct.pack("<16b", *range(1, 17))
        block = ql + qh + scales + struct.pack("<e", 0.5)
        actual = dequantize_tensor(info, memoryview(block)).values
        self.assertEqual(len(actual), 256)
        self.assertEqual(actual[0], -16.0)
        self.assertEqual(actual[16], -32.0)
        self.assertEqual(actual[32], -24.0)
        self.assertEqual(actual[48], -32.0)
        self.assertEqual(actual[64], 2.5)
        self.assertEqual(actual[80], 3.0)
        self.assertEqual(actual[96], 59.5)
        self.assertEqual(actual[112], 68.0)

    def test_dequant_iq4_nl(self) -> None:
        info = tensor("x", (32,), GGUFTensorType.IQ4_NL)
        qs = bytes(low | (high << 4) for low, high in zip(range(16), reversed(range(16))))
        actual = dequantize_tensor(info, memoryview(struct.pack("<e", 0.5) + qs)).values
        self.assertEqual(len(actual), 32)
        self.assertEqual(actual[:4], [-63.5, -52.0, -41.5, -32.5])
        self.assertEqual(actual[16:20], [56.5, 44.5, 34.5, 26.5])

    def test_dequant_iq4_xs(self) -> None:
        info = tensor("x", (256,), GGUFTensorType.IQ4_XS)
        scales_h = 0xAAAA
        scales_l = bytes([0x11] * 4)
        qs = bytes(low | (high << 4) for low, high in zip(range(16), reversed(range(16)))) * 8
        block = struct.pack("<eH", 0.5, scales_h) + scales_l + qs
        actual = dequantize_tensor(info, memoryview(block)).values
        self.assertEqual(len(actual), 256)
        self.assertEqual(actual[:4], [-63.5, -52.0, -41.5, -32.5])
        self.assertEqual(actual[32:36], [-63.5, -52.0, -41.5, -32.5])

    def test_dequant_iq3_s(self) -> None:
        info = tensor("x", (256,), GGUFTensorType.IQ3_S)
        block = (
            struct.pack("<e", 0.5)
            + bytes(64)
            + bytes(8)
            + bytes([0b00000001] + [0] * 31)
            + bytes(4)
        )
        actual = dequantize_tensor(info, memoryview(block)).values
        self.assertEqual(len(actual), 256)
        self.assertEqual(actual[0], -0.5)
        self.assertEqual(actual[1:4], [0.5, 0.5, 0.5])
        self.assertEqual(actual[32], 0.5)

    def test_unsupported_qwen_quant_types_are_explicit(self) -> None:
        self.assertIn(GGUFTensorType.Q4_0, supported_cpu_dequant_types())
        self.assertIn(GGUFTensorType.Q3_K, supported_cpu_dequant_types())
        self.assertIn(GGUFTensorType.Q4_K, supported_cpu_dequant_types())
        self.assertIn(GGUFTensorType.Q5_K, supported_cpu_dequant_types())
        self.assertIn(GGUFTensorType.Q6_K, supported_cpu_dequant_types())
        self.assertIn(GGUFTensorType.Q8_0, supported_cpu_dequant_types())
        self.assertIn(GGUFTensorType.IQ3_S, supported_cpu_dequant_types())
        self.assertIn(GGUFTensorType.IQ4_NL, supported_cpu_dequant_types())
        self.assertIn(GGUFTensorType.IQ4_XS, supported_cpu_dequant_types())
        unsupported = unsupported_tensor_types(
            [
                tensor("ok", (32,), GGUFTensorType.Q8_0),
                tensor("q3", (256,), GGUFTensorType.Q3_K),
                tensor("q4", (256,), GGUFTensorType.Q4_K),
                tensor("q5", (256,), GGUFTensorType.Q5_K),
                tensor("q6", (256,), GGUFTensorType.Q6_K),
                tensor("iq4nl", (32,), GGUFTensorType.IQ4_NL),
                tensor("iq4xs", (256,), GGUFTensorType.IQ4_XS),
                tensor("iq", (256,), GGUFTensorType.IQ4_XS),
                tensor("iq3", (256,), GGUFTensorType.IQ3_S),
                tensor("iq2", (256,), GGUFTensorType.IQ1_S),
            ]
        )
        self.assertEqual(unsupported, {GGUFTensorType.IQ1_S})
        with self.assertRaises(UnsupportedModelError):
            iq1 = tensor("iq1", (256,), GGUFTensorType.IQ1_S)
            dequantize_tensor(iq1, memoryview(bytes(iq1.nbytes)))

    def test_load_dequantized_tensor_streams_single_mmap_tensor(self) -> None:
        from vinf.gguf.parser import load_gguf

        gguf = load_gguf(write_tiny_gguf())
        loaded = load_dequantized_tensor(gguf, "token_embd.weight")
        self.assertEqual(loaded.shape, (8, 8))
        self.assertEqual(len(loaded.values), 64)


if __name__ == "__main__":
    unittest.main()
