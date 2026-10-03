from __future__ import annotations

import mmap
from array import array

from vinf.errors import ExecutorUnavailableError, UnsupportedModelError
from vinf.gguf.parser import GGUFFile, GGUFTensorType

CPU_TYPES = frozenset(
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


class CpuQwenRuntime:
    """CPU twin of CudaWeightRuntime: same op names/signatures, weights read in place from mmap."""

    def __init__(self, threads: int = 0) -> None:
        try:
            from vinf import _cpu_qwen
        except ImportError as exc:
            raise ExecutorUnavailableError("vinf._cpu_qwen is not built; run `make cpu-qwen`") from exc
        from vinf.cuda.qwen_runtime import _iq3_s_grid_bytes

        self._rt = _cpu_qwen.CpuRuntime(_iq3_s_grid_bytes(), threads)
        self._maps: dict[str, tuple[object, mmap.mmap, memoryview]] = {}

    def _file_view(self, gguf: GGUFFile) -> memoryview:
        key = str(gguf.path)
        if key not in self._maps:
            file_obj = gguf.path.open("rb")
            mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps[key] = (file_obj, mm, memoryview(mm))
        return self._maps[key][2]

    def add_gguf_tensor(self, gguf: GGUFFile, name: str) -> int:
        tensor = gguf.tensors[name]
        if tensor.tensor_type not in CPU_TYPES:
            raise UnsupportedModelError(f"{name}: no CPU kernel for {tensor.tensor_type.name}")
        rows = 1
        for dim in tensor.dimensions[1:]:
            rows *= dim
        view = self._file_view(gguf)[tensor.absolute_offset : tensor.absolute_offset + tensor.nbytes]
        self._rt.add_tensor(name, view, int(tensor.tensor_type), tensor.dimensions[0], rows)
        return tensor.nbytes

    def add_raw(self, name: str, data: bytes, tensor_type: GGUFTensorType, cols: int, rows: int) -> None:
        self._rt.add_tensor(name, data, int(tensor_type), cols, rows)

    def read_floats(self, name: str, n: int = -1, offset: int = 0) -> list[float]:
        out = array("f")
        out.frombytes(self._rt.read(name, n, offset))
        return out.tolist()

    def write_floats(self, name: str, values, offset: int = 0) -> None:
        self._rt.write(name, array("f", values).tobytes(), offset)

    def __getattr__(self, name: str):
        return getattr(self._rt, name)
