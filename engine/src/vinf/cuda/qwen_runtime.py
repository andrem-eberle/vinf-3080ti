from __future__ import annotations

import struct
from array import array
from typing import Sequence

from vinf.errors import ExecutorUnavailableError, UnsupportedModelError
from vinf.gguf.parser import GGUFFile, GGUFTensorType

CUDA_MATVEC_TYPES = frozenset(
    {
        GGUFTensorType.F32,
        GGUFTensorType.F16,
        GGUFTensorType.Q8_0,
        GGUFTensorType.Q2_K,
        GGUFTensorType.Q3_K,
        GGUFTensorType.Q4_K,
        GGUFTensorType.Q5_K,
        GGUFTensorType.Q6_K,
        GGUFTensorType.IQ4_NL,
        GGUFTensorType.IQ4_XS,
        GGUFTensorType.IQ3_S,
        GGUFTensorType.IQ2_XXS,
        GGUFTensorType.IQ2_XS,
        GGUFTensorType.IQ2_S,
        GGUFTensorType.IQ3_XXS,
    }
)


def _load_module():
    try:
        from vinf import _cuda_qwen_runtime
    except ImportError as exc:
        raise ExecutorUnavailableError(
            "vinf._cuda_qwen_runtime is not built; run `make cuda-qwen-runtime`"
        ) from exc
    return _cuda_qwen_runtime


def _iq3_s_grid_bytes() -> bytes:
    from vinf.gguf.dequant import _iq3_s_grid

    return bytes(int(value) for entry in _iq3_s_grid() for value in entry)


class CudaWeightRuntime:
    """Persistent VRAM store of raw GGUF tensors with quantized matvec.

    Weights stay in their GGUF block format on the device; no CPU dequantization.
    """

    def __init__(self) -> None:
        self._rt = _load_module().Runtime(_iq3_s_grid_bytes())

    def mem_info(self) -> tuple[int, int]:
        return self._rt.mem_info()

    def device_bytes(self) -> int:
        return self._rt.device_bytes()

    def has(self, name: str) -> bool:
        return self._rt.has(name)

    def free(self, name: str) -> None:
        self._rt.free(name)

    def upload_gguf_tensor(
        self, gguf: GGUFFile, name: str, *, device_name: str | None = None, resident: bool = True
    ) -> int:
        """Upload raw blocks; resident=False keeps them in pinned host memory, streamed on use."""
        tensor = gguf.tensors[name]
        if tensor.tensor_type not in CUDA_MATVEC_TYPES:
            raise UnsupportedModelError(f"{name}: no CUDA matvec for {tensor.tensor_type.name}")
        cols = tensor.dimensions[0]
        rows = 1
        for dim in tensor.dimensions[1:]:
            rows *= dim
        file_obj, mm, view = gguf.mmap_tensor(name)
        try:
            self._rt.upload(device_name or name, view, int(tensor.tensor_type), cols, rows, resident)
        finally:
            view.release()
            mm.close()
            file_obj.close()
        return tensor.nbytes

    def upload_raw(
        self,
        name: str,
        data: bytes | memoryview,
        tensor_type: GGUFTensorType,
        cols: int,
        rows: int,
        *,
        resident: bool = True,
    ) -> None:
        self._rt.upload(name, data, int(tensor_type), cols, rows, resident)

    def matvec(self, name: str, values: Sequence[float]) -> list[float]:
        out = array("f")
        out.frombytes(self._rt.matvec(name, array("f", values).tobytes()))
        return out.tolist()

    def read_floats(self, name: str, n: int = -1, offset: int = 0) -> list[float]:
        """Buffer contents as floats (fp16 KV caches are widened)."""
        data = self._rt.read(name, n, offset)
        if self._rt.buffer_elem(name) == 2:
            return list(struct.unpack(f"<{len(data) // 2}e", data))
        out = array("f")
        out.frombytes(data)
        return out.tolist()

    def __getattr__(self, name: str):
        # Device-op passthrough (qmv, rmsnorm, attention, gated_delta, argmax, ...).
        return getattr(self._rt, name)
