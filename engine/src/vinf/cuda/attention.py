from __future__ import annotations

import math

from vinf.reference_ops import attention_reference


def attention_cpu(
    query: list[float],
    key_cache: list[float],
    value_cache: list[float],
    *,
    kv_head_idx: int,
    seq_len: int,
    num_kv_heads: int,
    max_seq: int,
    head_dim: int,
    scale: float | None = None,
) -> list[float]:
    return attention_reference(
        query,
        key_cache,
        value_cache,
        kv_head_idx=kv_head_idx,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        max_seq=max_seq,
        head_dim=head_dim,
        scale=scale,
    )


def attention_cuda(
    query: list[float],
    key_cache: list[float],
    value_cache: list[float],
    *,
    kv_head_idx: int,
    seq_len: int,
    num_kv_heads: int,
    max_seq: int,
    head_dim: int,
    scale: float | None = None,
) -> list[float]:
    from vinf import _cuda_attention

    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    return [
        float(x)
        for x in _cuda_attention.attention(
            query,
            key_cache,
            value_cache,
            int(kv_head_idx),
            int(seq_len),
            int(num_kv_heads),
            int(max_seq),
            int(head_dim),
            float(scale),
        )
    ]

