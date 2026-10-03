from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import mmap
import struct
from typing import Any

from vinf.errors import UnsupportedModelError


GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFTensorType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


GGUF_TYPE_TRAITS: dict[GGUFTensorType, tuple[int, int, bool]] = {
    GGUFTensorType.F32: (1, 4, False),
    GGUFTensorType.F16: (1, 2, False),
    GGUFTensorType.Q4_0: (32, 18, True),
    GGUFTensorType.Q4_1: (32, 20, True),
    GGUFTensorType.Q5_0: (32, 22, True),
    GGUFTensorType.Q5_1: (32, 24, True),
    GGUFTensorType.Q8_0: (32, 34, True),
    GGUFTensorType.Q8_1: (32, 40, True),
    GGUFTensorType.Q2_K: (256, 84, True),
    GGUFTensorType.Q3_K: (256, 110, True),
    GGUFTensorType.Q4_K: (256, 144, True),
    GGUFTensorType.Q5_K: (256, 176, True),
    GGUFTensorType.Q6_K: (256, 210, True),
    GGUFTensorType.Q8_K: (256, 292, True),
    GGUFTensorType.IQ2_XXS: (256, 66, True),
    GGUFTensorType.IQ2_XS: (256, 74, True),
    GGUFTensorType.IQ3_XXS: (256, 98, True),
    GGUFTensorType.IQ1_S: (256, 50, True),
    GGUFTensorType.IQ4_NL: (32, 18, True),
    GGUFTensorType.IQ3_S: (256, 110, True),
    GGUFTensorType.IQ2_S: (256, 82, True),
    GGUFTensorType.IQ4_XS: (256, 136, True),
    GGUFTensorType.I8: (1, 1, False),
    GGUFTensorType.I16: (1, 2, False),
    GGUFTensorType.I32: (1, 4, False),
    GGUFTensorType.I64: (1, 8, False),
    GGUFTensorType.F64: (1, 8, False),
    GGUFTensorType.BF16: (1, 2, False),
}


@dataclass(frozen=True, slots=True)
class GGUFMetadataValue:
    type: GGUFValueType
    value: Any


@dataclass(frozen=True, slots=True)
class GGUFTensorInfo:
    name: str
    dimensions: tuple[int, ...]
    tensor_type: GGUFTensorType
    relative_offset: int
    absolute_offset: int

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.dimensions:
            total *= dim
        return total

    @property
    def is_quantized(self) -> bool:
        return GGUF_TYPE_TRAITS[self.tensor_type][2]

    @property
    def block_size(self) -> int:
        return GGUF_TYPE_TRAITS[self.tensor_type][0]

    @property
    def type_size(self) -> int:
        return GGUF_TYPE_TRAITS[self.tensor_type][1]

    @property
    def nbytes(self) -> int:
        blocks = (self.numel + self.block_size - 1) // self.block_size
        return blocks * self.type_size

@dataclass(slots=True)
class GGUFFile:
    path: Path
    version: int
    metadata: dict[str, GGUFMetadataValue]
    tensors: dict[str, GGUFTensorInfo]
    data_start: int
    alignment: int

    def metadata_value(self, key: str, default: Any = None) -> Any:
        item = self.metadata.get(key)
        if item is None:
            return default
        return item.value

    def quantized_tensors(self) -> list[GGUFTensorInfo]:
        return [tensor for tensor in self.tensors.values() if tensor.is_quantized]

    def mmap_tensor(self, name: str):
        tensor = self.tensors[name]
        file_obj = self.path.open("rb")
        mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        start = tensor.absolute_offset
        end = start + tensor.nbytes
        return file_obj, mm, memoryview(mm)[start:end]


def load_gguf(path: str | Path, *, require_supported_tensors: bool = False) -> GGUFFile:
    parser = _GGUFParser(Path(path))
    parsed = parser.parse()
    if require_supported_tensors:
        _assert_known_tensor_types(parsed)
    return parsed


class _GGUFParser:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = path.open("rb")
        self.pos = 0

    def parse(self) -> GGUFFile:
        try:
            magic = self._read(4)
            if magic != GGUF_MAGIC:
                raise UnsupportedModelError("not a GGUF file")
            version = self._u32()
            if version not in {2, 3}:
                raise UnsupportedModelError(f"unsupported GGUF version: {version}")
            tensor_count = self._u64()
            metadata_count = self._u64()

            metadata = {}
            for _ in range(metadata_count):
                key = self._string()
                metadata[key] = self._metadata_value()

            tensor_infos = []
            for _ in range(tensor_count):
                name = self._string()
                n_dimensions = self._u32()
                dimensions = tuple(self._u64() for _ in range(n_dimensions))
                raw_type = self._u32()
                try:
                    tensor_type = GGUFTensorType(raw_type)
                except ValueError as exc:
                    raise UnsupportedModelError(f"unknown GGUF tensor type: {raw_type}") from exc
                relative_offset = self._u64()
                tensor_infos.append((name, dimensions, tensor_type, relative_offset))

            alignment = int(metadata.get("general.alignment", GGUFMetadataValue(GGUFValueType.UINT32, DEFAULT_ALIGNMENT)).value)
            data_start = _align(self.pos, alignment)
            tensors = {}
            for name, dimensions, tensor_type, relative_offset in tensor_infos:
                tensors[name] = GGUFTensorInfo(
                    name=name,
                    dimensions=dimensions,
                    tensor_type=tensor_type,
                    relative_offset=relative_offset,
                    absolute_offset=data_start + relative_offset,
                )

            return GGUFFile(
                path=self.path,
                version=version,
                metadata=metadata,
                tensors=tensors,
                data_start=data_start,
                alignment=alignment,
            )
        finally:
            self.file.close()

    def _metadata_value(self) -> GGUFMetadataValue:
        raw_type = self._u32()
        try:
            value_type = GGUFValueType(raw_type)
        except ValueError as exc:
            raise UnsupportedModelError(f"unknown GGUF metadata type: {raw_type}") from exc
        return GGUFMetadataValue(value_type, self._value_for_type(value_type))

    def _value_for_type(self, value_type: GGUFValueType) -> Any:
        match value_type:
            case GGUFValueType.UINT8:
                return self._unpack("<B", 1)
            case GGUFValueType.INT8:
                return self._unpack("<b", 1)
            case GGUFValueType.UINT16:
                return self._unpack("<H", 2)
            case GGUFValueType.INT16:
                return self._unpack("<h", 2)
            case GGUFValueType.UINT32:
                return self._u32()
            case GGUFValueType.INT32:
                return self._unpack("<i", 4)
            case GGUFValueType.FLOAT32:
                return self._unpack("<f", 4)
            case GGUFValueType.BOOL:
                return bool(self._unpack("<?", 1))
            case GGUFValueType.STRING:
                return self._string()
            case GGUFValueType.ARRAY:
                item_type = GGUFValueType(self._u32())
                count = self._u64()
                return [self._value_for_type(item_type) for _ in range(count)]
            case GGUFValueType.UINT64:
                return self._u64()
            case GGUFValueType.INT64:
                return self._unpack("<q", 8)
            case GGUFValueType.FLOAT64:
                return self._unpack("<d", 8)
        raise UnsupportedModelError(f"unsupported metadata value type: {value_type}")

    def _string(self) -> str:
        length = self._u64()
        raw = self._read(length)
        return raw.decode("utf-8")

    def _u32(self) -> int:
        return self._unpack("<I", 4)

    def _u64(self) -> int:
        return self._unpack("<Q", 8)

    def _unpack(self, fmt: str, size: int) -> Any:
        return struct.unpack(fmt, self._read(size))[0]

    def _read(self, size: int) -> bytes:
        raw = self.file.read(size)
        if len(raw) != size:
            raise UnsupportedModelError("truncated GGUF file")
        self.pos += size
        return raw


def _align(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise UnsupportedModelError("GGUF alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def _assert_known_tensor_types(gguf: GGUFFile) -> None:
    for tensor in gguf.tensors.values():
        if tensor.tensor_type not in GGUF_TYPE_TRAITS:
            raise UnsupportedModelError(
                f"unsupported GGUF tensor type {tensor.tensor_type.name} for {tensor.name}"
            )
