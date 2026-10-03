from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vinf.errors import UnsupportedModelError
from vinf.models.layouts import WeightSpec


@dataclass(frozen=True, slots=True)
class WeightTensor:
    name: str
    values: Any
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class ModelWeights:
    tensors: dict[str, WeightTensor]

    def get(self, name: str) -> WeightTensor:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise UnsupportedModelError(f"missing weight tensor: {name}") from exc

    def validate(self, specs: tuple[WeightSpec, ...]) -> None:
        for spec in specs:
            if not spec.required:
                continue
            tensor = self.get(spec.name)
            if tensor.shape != spec.shape:
                raise UnsupportedModelError(
                    f"{spec.name} shape mismatch: expected {spec.shape}, got {tensor.shape}"
                )
            if tensor.dtype != spec.dtype:
                raise UnsupportedModelError(
                    f"{spec.name} dtype mismatch: expected {spec.dtype}, got {tensor.dtype}"
                )


def infer_nested_shape(values: Any) -> tuple[int, ...]:
    if not isinstance(values, list):
        return ()
    if not values:
        return (0,)
    first_shape = infer_nested_shape(values[0])
    for item in values:
        if infer_nested_shape(item) != first_shape:
            raise UnsupportedModelError("ragged nested weight arrays are not supported")
    return (len(values), *first_shape)

