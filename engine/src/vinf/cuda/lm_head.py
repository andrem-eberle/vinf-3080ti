from __future__ import annotations

from vinf.reference_ops import lm_head_reference


def lm_head_cpu(
    values: list[float],
    norm_weight: list[float],
    lm_head_weight: list[float],
    hidden_size: int,
    vocab_size: int,
    eps: float = 1e-6,
) -> list[float]:
    return lm_head_reference(values, norm_weight, lm_head_weight, hidden_size, vocab_size, eps)


def lm_head_cuda(
    values: list[float],
    norm_weight: list[float],
    lm_head_weight: list[float],
    hidden_size: int,
    vocab_size: int,
    eps: float = 1e-6,
) -> list[float]:
    from vinf import _cuda_lm_head

    return [
        float(x)
        for x in _cuda_lm_head.lm_head(
            values,
            norm_weight,
            lm_head_weight,
            int(hidden_size),
            int(vocab_size),
            float(eps),
        )
    ]

