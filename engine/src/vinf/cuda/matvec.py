from __future__ import annotations

from vinf.reference_ops import matvec_reference


def matvec_cpu(values: list[float], weights: list[float], rows: int, cols: int) -> list[float]:
    return matvec_reference(values, weights, rows, cols)


def matvec_cuda(values: list[float], weights: list[float], rows: int, cols: int) -> list[float]:
    from vinf import _cuda_matvec

    return [float(x) for x in _cuda_matvec.matvec(values, weights, int(rows), int(cols))]


class QuantizedMatvecNotImplemented(NotImplementedError):
    pass


def matvec_quantized_placeholder(*args, **kwargs):
    raise QuantizedMatvecNotImplemented(
        "quantized GGUF matvec is not implemented yet; dequantization comes before CUDA quantized matvec"
    )
