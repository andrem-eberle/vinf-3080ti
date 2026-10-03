from __future__ import annotations

import math


def rmsnorm_reference(
    values: list[float], weights: list[float], eps: float = 1e-6
) -> list[float]:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    mean_square = sum(value * value for value in values) / len(values)
    scale = 1.0 / math.sqrt(mean_square + eps)
    return [value * scale * weight for value, weight in zip(values, weights)]


def matvec_reference(values: list[float], weights: list[float], rows: int, cols: int) -> list[float]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if len(values) != cols:
        raise ValueError("values length must equal cols")
    if len(weights) != rows * cols:
        raise ValueError("weights length must equal rows * cols")
    out = []
    for row in range(rows):
        base = row * cols
        total = 0.0
        for col in range(cols):
            total += weights[base + col] * values[col]
        out.append(total)
    return out


def rope_reference(values: list[float], cos: list[float], sin: list[float]) -> list[float]:
    if len(values) != len(cos) or len(values) != len(sin):
        raise ValueError("values/cos/sin lengths must match")
    if len(values) % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    out = [0.0 for _ in values]
    for i in range(0, len(values), 2):
        x0 = values[i]
        x1 = values[i + 1]
        c = cos[i]
        s = sin[i]
        out[i] = x0 * c - x1 * s
        out[i + 1] = x0 * s + x1 * c
    return out


def kv_append_reference(
    cache: list[float],
    values: list[float],
    *,
    head_idx: int,
    position: int,
    num_heads: int,
    max_seq: int,
    head_dim: int,
) -> list[float]:
    if len(cache) != num_heads * max_seq * head_dim:
        raise ValueError("cache size does not match layout")
    if len(values) != head_dim:
        raise ValueError("values length must equal head_dim")
    if not 0 <= head_idx < num_heads:
        raise ValueError("head_idx out of range")
    if not 0 <= position < max_seq:
        raise ValueError("position out of range")
    out = list(cache)
    base = (head_idx * max_seq + position) * head_dim
    out[base : base + head_dim] = values
    return out


def attention_reference(
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
    if len(query) != head_dim:
        raise ValueError("query length must equal head_dim")
    expected_cache = num_kv_heads * max_seq * head_dim
    if len(key_cache) != expected_cache or len(value_cache) != expected_cache:
        raise ValueError("cache size does not match layout")
    if not 0 <= kv_head_idx < num_kv_heads:
        raise ValueError("kv_head_idx out of range")
    if not 0 < seq_len <= max_seq:
        raise ValueError("seq_len out of range")
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    scores = []
    for pos in range(seq_len):
        base = (kv_head_idx * max_seq + pos) * head_dim
        dot = 0.0
        for dim in range(head_dim):
            dot += query[dim] * key_cache[base + dim]
        scores.append(dot * scale)
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    denom = sum(exps)
    probs = [value / denom for value in exps]
    out = [0.0 for _ in range(head_dim)]
    for pos, prob in enumerate(probs):
        base = (kv_head_idx * max_seq + pos) * head_dim
        for dim in range(head_dim):
            out[dim] += prob * value_cache[base + dim]
    return out


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def mlp_reference(
    values: list[float],
    gate_weight: list[float],
    up_weight: list[float],
    down_weight: list[float],
    hidden_size: int,
    intermediate_size: int,
) -> list[float]:
    gate = matvec_reference(values, gate_weight, intermediate_size, hidden_size)
    up = matvec_reference(values, up_weight, intermediate_size, hidden_size)
    activated = [silu(g) * u for g, u in zip(gate, up)]
    return matvec_reference(activated, down_weight, hidden_size, intermediate_size)


def lm_head_reference(
    values: list[float],
    norm_weight: list[float],
    lm_head_weight: list[float],
    hidden_size: int,
    vocab_size: int,
    eps: float = 1e-6,
) -> list[float]:
    normalized = rmsnorm_reference(values, norm_weight, eps)
    return matvec_reference(normalized, lm_head_weight, vocab_size, hidden_size)
