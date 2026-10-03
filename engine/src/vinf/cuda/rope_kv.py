from __future__ import annotations

from vinf.reference_ops import kv_append_reference, rope_reference


def rope_kv_cpu(
    values: list[float],
    cos: list[float],
    sin: list[float],
    *,
    num_heads: int,
    max_seq: int,
    head_idx: int,
    position: int,
) -> tuple[list[float], list[float]]:
    rope = rope_reference(values, cos, sin)
    cache = [0.0] * (num_heads * max_seq * len(values))
    cache = kv_append_reference(
        cache,
        rope,
        head_idx=head_idx,
        position=position,
        num_heads=num_heads,
        max_seq=max_seq,
        head_dim=len(values),
    )
    return rope, cache


def rope_kv_cuda(
    values: list[float],
    cos: list[float],
    sin: list[float],
    *,
    num_heads: int,
    max_seq: int,
    head_idx: int,
    position: int,
) -> tuple[list[float], list[float]]:
    from vinf import _cuda_rope_kv

    rope, cache = _cuda_rope_kv.rope_kv(
        values, cos, sin, int(num_heads), int(max_seq), int(head_idx), int(position)
    )
    return [float(x) for x in rope], [float(x) for x in cache]

