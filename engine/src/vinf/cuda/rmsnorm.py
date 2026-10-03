from __future__ import annotations

from vinf.reference_ops import rmsnorm_reference


def rmsnorm_cuda(values: list[float], weights: list[float], eps: float = 1e-6) -> list[float]:
    from vinf import _cuda_rmsnorm

    return [float(x) for x in _cuda_rmsnorm.rmsnorm(values, weights, float(eps))]


def rmsnorm_cpu(values: list[float], weights: list[float], eps: float = 1e-6) -> list[float]:
    return rmsnorm_reference(values, weights, eps)

