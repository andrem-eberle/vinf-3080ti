from __future__ import annotations

from vinf.reference_ops import mlp_reference


def mlp_cpu(
    values: list[float],
    gate_weight: list[float],
    up_weight: list[float],
    down_weight: list[float],
    hidden_size: int,
    intermediate_size: int,
) -> list[float]:
    return mlp_reference(values, gate_weight, up_weight, down_weight, hidden_size, intermediate_size)


def mlp_cuda(
    values: list[float],
    gate_weight: list[float],
    up_weight: list[float],
    down_weight: list[float],
    hidden_size: int,
    intermediate_size: int,
) -> list[float]:
    from vinf import _cuda_mlp

    return [
        float(x)
        for x in _cuda_mlp.mlp(
            values,
            gate_weight,
            up_weight,
            down_weight,
            int(hidden_size),
            int(intermediate_size),
        )
    ]

