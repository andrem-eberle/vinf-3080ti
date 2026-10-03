from __future__ import annotations

from array import array
from collections.abc import Sequence

from vinf import _native


def native_version() -> str:
    return _native.native_version()


def argmax_float32(values: Sequence[float]) -> int:
    data = array("f", values)
    if not data:
        raise ValueError("values must not be empty")
    return int(_native.argmax_float32(data.tobytes()))

