from __future__ import annotations

from dataclasses import dataclass

from vinf.reference_ops import (
    attention_reference,
    kv_append_reference,
    matvec_reference,
    mlp_reference,
    rmsnorm_reference,
    rope_reference,
)


@dataclass(frozen=True, slots=True)
class OneLayerWeights:
    attn_norm: list[float]
    q_proj: list[float]
    k_proj: list[float]
    v_proj: list[float]
    o_proj: list[float]
    mlp_norm: list[float]
    gate_proj: list[float]
    up_proj: list[float]
    down_proj: list[float]


@dataclass(frozen=True, slots=True)
class OneLayerConfig:
    hidden_size: int
    head_dim: int
    num_kv_heads: int
    max_seq: int
    eps: float = 1e-6


@dataclass(frozen=True, slots=True)
class OneLayerResult:
    hidden: list[float]
    stops: dict[str, list[float]]


def one_layer_reference(
    hidden: list[float],
    weights: OneLayerWeights,
    config: OneLayerConfig,
    *,
    position: int,
    rope_cos: list[float],
    rope_sin: list[float],
) -> OneLayerResult:
    stops: dict[str, list[float]] = {}
    h = list(hidden)
    normed = rmsnorm_reference(h, weights.attn_norm, config.eps)
    stops["attn_norm"] = normed

    q = matvec_reference(normed, weights.q_proj, config.head_dim, config.hidden_size)
    k = matvec_reference(normed, weights.k_proj, config.head_dim, config.hidden_size)
    v = matvec_reference(normed, weights.v_proj, config.head_dim, config.hidden_size)
    q_rope = rope_reference(q, rope_cos, rope_sin)
    k_rope = rope_reference(k, rope_cos, rope_sin)
    stops["q_rope"] = q_rope
    stops["k_rope"] = k_rope
    stops["v"] = v

    cache_size = config.num_kv_heads * config.max_seq * config.head_dim
    k_cache = kv_append_reference(
        [0.0] * cache_size,
        k_rope,
        head_idx=0,
        position=position,
        num_heads=config.num_kv_heads,
        max_seq=config.max_seq,
        head_dim=config.head_dim,
    )
    v_cache = kv_append_reference(
        [0.0] * cache_size,
        v,
        head_idx=0,
        position=position,
        num_heads=config.num_kv_heads,
        max_seq=config.max_seq,
        head_dim=config.head_dim,
    )
    stops["k_cache"] = k_cache
    stops["v_cache"] = v_cache

    attn = attention_reference(
        q_rope,
        k_cache,
        v_cache,
        kv_head_idx=0,
        seq_len=position + 1,
        num_kv_heads=config.num_kv_heads,
        max_seq=config.max_seq,
        head_dim=config.head_dim,
    )
    stops["attention"] = attn

    o = matvec_reference(attn, weights.o_proj, config.hidden_size, config.head_dim)
    h = [a + b for a, b in zip(h, o)]
    stops["post_attention"] = h

    mlp_norm = rmsnorm_reference(h, weights.mlp_norm, config.eps)
    stops["mlp_norm"] = mlp_norm
    mlp = mlp_reference(
        mlp_norm,
        weights.gate_proj,
        weights.up_proj,
        weights.down_proj,
        config.hidden_size,
        config.hidden_size * 2,
    )
    h = [a + b for a, b in zip(h, mlp)]
    stops["post_mlp"] = h
    return OneLayerResult(hidden=h, stops=stops)


def one_layer_cuda_composed(
    hidden: list[float],
    weights: OneLayerWeights,
    config: OneLayerConfig,
    *,
    position: int,
    rope_cos: list[float],
    rope_sin: list[float],
) -> OneLayerResult:
    from vinf.cuda.attention import attention_cuda
    from vinf.cuda.matvec import matvec_cuda
    from vinf.cuda.mlp import mlp_cuda
    from vinf.cuda.rmsnorm import rmsnorm_cuda
    from vinf.cuda.rope_kv import rope_kv_cuda

    stops: dict[str, list[float]] = {}
    h = list(hidden)
    normed = rmsnorm_cuda(h, weights.attn_norm, config.eps)
    stops["attn_norm"] = normed

    q = matvec_cuda(normed, weights.q_proj, config.head_dim, config.hidden_size)
    k = matvec_cuda(normed, weights.k_proj, config.head_dim, config.hidden_size)
    v = matvec_cuda(normed, weights.v_proj, config.head_dim, config.hidden_size)
    q_rope, _ = rope_kv_cuda(
        q,
        rope_cos,
        rope_sin,
        num_heads=config.num_kv_heads,
        max_seq=config.max_seq,
        head_idx=0,
        position=position,
    )
    k_rope, k_cache = rope_kv_cuda(
        k,
        rope_cos,
        rope_sin,
        num_heads=config.num_kv_heads,
        max_seq=config.max_seq,
        head_idx=0,
        position=position,
    )
    _, v_cache = rope_kv_cuda(
        v,
        [1.0] * config.head_dim,
        [0.0] * config.head_dim,
        num_heads=config.num_kv_heads,
        max_seq=config.max_seq,
        head_idx=0,
        position=position,
    )
    stops["q_rope"] = q_rope
    stops["k_rope"] = k_rope
    stops["v"] = v
    stops["k_cache"] = k_cache
    stops["v_cache"] = v_cache

    attn = attention_cuda(
        q_rope,
        k_cache,
        v_cache,
        kv_head_idx=0,
        seq_len=position + 1,
        num_kv_heads=config.num_kv_heads,
        max_seq=config.max_seq,
        head_dim=config.head_dim,
    )
    stops["attention"] = attn
    o = matvec_cuda(attn, weights.o_proj, config.hidden_size, config.head_dim)
    h = [a + b for a, b in zip(h, o)]
    stops["post_attention"] = h

    mlp_norm = rmsnorm_cuda(h, weights.mlp_norm, config.eps)
    stops["mlp_norm"] = mlp_norm
    mlp = mlp_cuda(
        mlp_norm,
        weights.gate_proj,
        weights.up_proj,
        weights.down_proj,
        config.hidden_size,
        config.hidden_size * 2,
    )
    h = [a + b for a, b in zip(h, mlp)]
    stops["post_mlp"] = h
    return OneLayerResult(hidden=h, stops=stops)

