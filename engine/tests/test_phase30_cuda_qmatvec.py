from __future__ import annotations

import os

import math
import random
import struct
import unittest
from pathlib import Path

from vinf.errors import ExecutorUnavailableError
from vinf.gguf.dequant import dequantize_tensor
from vinf.gguf.parser import GGUFTensorInfo, GGUFTensorType, load_gguf
from vinf.reference_ops import matvec_reference

REAL_MODEL = Path(os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf"))

# Byte offsets of f16 scale fields inside one block, per type (kept finite and small in synthetic data).
F16_FIELDS = {
    GGUFTensorType.Q8_0: (0,),
    GGUFTensorType.IQ4_NL: (0,),
    GGUFTensorType.Q4_K: (0, 2),
    GGUFTensorType.Q5_K: (0, 2),
    GGUFTensorType.Q6_K: (208,),
    GGUFTensorType.Q3_K: (108,),
    GGUFTensorType.Q2_K: (80, 82),
    GGUFTensorType.IQ2_XXS: (0,),
    GGUFTensorType.IQ2_XS: (0,),
    GGUFTensorType.IQ2_S: (0,),
    GGUFTensorType.IQ3_XXS: (0,),
    GGUFTensorType.IQ4_XS: (0,),
    GGUFTensorType.IQ3_S: (0,),
}


def runtime_or_skip(test: unittest.TestCase):
    try:
        from vinf.cuda.qwen_runtime import CudaWeightRuntime

        return CudaWeightRuntime()
    except (ExecutorUnavailableError, RuntimeError) as exc:
        test.skipTest(f"CUDA qwen runtime unavailable: {exc}")


def synthetic_rows(tensor_type: GGUFTensorType, rows: int, cols: int, rng: random.Random) -> bytes:
    probe = GGUFTensorInfo("p", (cols,), tensor_type, 0, 0)
    if tensor_type is GGUFTensorType.F32:
        return struct.pack(f"<{rows * cols}f", *(rng.uniform(-1, 1) for _ in range(rows * cols)))
    if tensor_type is GGUFTensorType.F16:
        return struct.pack(f"<{rows * cols}e", *(rng.uniform(-1, 1) for _ in range(rows * cols)))
    out = bytearray()
    for _ in range(rows * cols // probe.block_size):
        block = bytearray(rng.getrandbits(8) for _ in range(probe.type_size))
        for offset in F16_FIELDS[tensor_type]:
            block[offset : offset + 2] = struct.pack("<e", rng.uniform(0.001, 0.05))
        out.extend(block)
    return bytes(out)


def cpu_matvec(raw: bytes, tensor_type: GGUFTensorType, rows: int, cols: int, x: list[float]) -> list[float]:
    info = GGUFTensorInfo("w", (rows * cols,), tensor_type, 0, 0)
    weights = dequantize_tensor(info, memoryview(raw)).values
    return matvec_reference(x, weights, rows, cols)


class Phase30CudaQuantizedMatvecTests(unittest.TestCase):
    def assert_close(self, gpu: list[float], cpu: list[float], label: str) -> None:
        self.assertEqual(len(gpu), len(cpu), label)
        scale = max(1e-6, max(abs(v) for v in cpu))
        for idx, (g, c) in enumerate(zip(gpu, cpu)):
            self.assertTrue(math.isfinite(g), (label, idx, g))
            self.assertLessEqual(abs(g - c), 1e-4 * scale + 1e-5, (label, idx, g, c))

    def test_synthetic_blocks_match_cpu_dequant_for_every_type(self) -> None:
        rt = runtime_or_skip(self)
        from vinf.cuda.qwen_runtime import CUDA_MATVEC_TYPES

        rng = random.Random(30)
        rows, cols = 5, 512
        x = [rng.uniform(-1, 1) for _ in range(cols)]
        for tensor_type in sorted(CUDA_MATVEC_TYPES):
            raw = synthetic_rows(tensor_type, rows, cols, rng)
            rt.upload_raw("w", raw, tensor_type, cols, rows)
            self.assert_close(rt.matvec("w", x), cpu_matvec(raw, tensor_type, rows, cols, x), tensor_type.name)
            rt.free("w")
        self.assertEqual(rt.device_bytes(), 0)

    def test_real_model_rows_match_cpu_dequant_for_every_model_type(self) -> None:
        if not REAL_MODEL.exists():
            self.skipTest("local Qwen GGUF is not present")
        rt = runtime_or_skip(self)
        gguf = load_gguf(REAL_MODEL)
        examples: dict[GGUFTensorType, str] = {}
        for name, tensor in gguf.tensors.items():
            if len(tensor.dimensions) == 2 and tensor.dimensions[0] % 256 == 0:
                examples.setdefault(tensor.tensor_type, name)
        rng = random.Random(31)
        for tensor_type, name in sorted(examples.items()):
            tensor = gguf.tensors[name]
            cols = tensor.dimensions[0]
            rows = 8
            row_bytes = cols // tensor.block_size * tensor.type_size
            file_obj, mm, view = gguf.mmap_tensor(name)
            raw = bytes(view[: rows * row_bytes])
            view.release()
            mm.close()
            file_obj.close()
            x = [rng.uniform(-1, 1) for _ in range(cols)]
            rt.upload_raw("w", raw, tensor_type, cols, rows)
            self.assert_close(rt.matvec("w", x), cpu_matvec(raw, tensor_type, rows, cols, x), f"{tensor_type.name} {name}")
            rt.free("w")

    def test_upload_gguf_tensor_and_mem_info(self) -> None:
        if not REAL_MODEL.exists():
            self.skipTest("local Qwen GGUF is not present")
        rt = runtime_or_skip(self)
        gguf = load_gguf(REAL_MODEL)
        free_before, total = rt.mem_info()
        self.assertGreater(total, 11 * 1024**3)
        nbytes = rt.upload_gguf_tensor(gguf, "blk.0.ssm_alpha.weight")
        self.assertEqual(rt.device_bytes(), nbytes)
        out = rt.matvec("blk.0.ssm_alpha.weight", [0.01] * 5120)
        self.assertEqual(len(out), 48)
        rt.free("blk.0.ssm_alpha.weight")
        self.assertGreater(free_before, 0)

    def test_invalid_uploads_fail_cleanly(self) -> None:
        rt = runtime_or_skip(self)
        with self.assertRaises(ValueError):
            rt.upload_raw("w", b"\x00" * 34, GGUFTensorType.Q4_0, 32, 1)
        with self.assertRaises(ValueError):
            rt.upload_raw("w", b"\x00" * 33, GGUFTensorType.Q8_0, 32, 1)
        with self.assertRaises(KeyError):
            rt.matvec("missing", [0.0] * 32)


if __name__ == "__main__":
    unittest.main()
