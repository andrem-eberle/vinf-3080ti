"""Minimal dependency-free safetensors reader (single file or sharded index), memory-mapped."""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

from vinf.errors import UnsupportedModelError

DTYPE_CODES = {"F32": 0, "F16": 1, "BF16": 2}
DTYPE_WIDTH = {"F32": 4, "F16": 2, "BF16": 2}


@dataclass(frozen=True, slots=True)
class SafeTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: memoryview

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    @property
    def dtype_code(self) -> int:
        return DTYPE_CODES[self.dtype]


class SafeTensorsCheckpoint:
    """Tensors from `model.safetensors` or `model.safetensors.index.json` in a directory (or a single file)."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            index = path / "model.safetensors.index.json"
            if index.exists():
                files = sorted({path / f for f in json.loads(index.read_text())["weight_map"].values()})
            else:
                files = sorted(path.glob("*.safetensors"))
        else:
            files = [path]
        if not files:
            raise UnsupportedModelError(f"no .safetensors files in {path}")
        self.directory = path if path.is_dir() else path.parent
        self._maps = []
        self.tensors: dict[str, SafeTensor] = {}
        for file in files:
            self._read_file(file)

    def _read_file(self, file: Path) -> None:
        with file.open("rb") as fh:  # the mapping stays valid after the descriptor is closed
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._maps.append(mm)
        (header_len,) = struct.unpack("<Q", mm[:8])
        header = json.loads(mm[8 : 8 + header_len])
        base = 8 + header_len
        view = memoryview(mm)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info["dtype"]
            if dtype not in DTYPE_CODES:
                raise UnsupportedModelError(f"{file.name}: {name} has unsupported dtype {dtype}")
            start, end = info["data_offsets"]
            tensor = SafeTensor(name, dtype, tuple(info["shape"]), view[base + start : base + end])
            if tensor.numel * DTYPE_WIDTH[dtype] != end - start:
                raise UnsupportedModelError(f"{file.name}: {name} byte size does not match its shape")
            self.tensors[name] = tensor

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def __getitem__(self, name: str) -> SafeTensor:
        return self.tensors[name]

    def nbytes(self, names) -> int:
        return sum(self.tensors[n].data.nbytes for n in names)
