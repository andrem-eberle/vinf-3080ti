from __future__ import annotations

from dataclasses import dataclass

from vinf.baseline_decode import BaselineDecodeConfig
from vinf.one_layer import OneLayerWeights
from vinf.reference_ops import (
    attention_reference,
    kv_append_reference,
    lm_head_reference,
    matvec_reference,
    mlp_reference,
    rmsnorm_reference,
    rope_reference,
)
from vinf.speculative.megakernel import causal_verify_mask


@dataclass(frozen=True, slots=True)
class VerifyQKVRopeResult:
    queries: list[list[float]]
    keys: list[list[float]]
    values: list[list[float]]
    key_cache: list[float]
    value_cache: list[float]


def verify_qkv_rope_reference(
    hidden_rows: list[list[float]],
    weights: OneLayerWeights,
    config: BaselineDecodeConfig,
    *,
    start_position: int,
    rope_cos_rows: list[list[float]],
    rope_sin_rows: list[list[float]],
    committed_key_cache: list[float] | None = None,
    committed_value_cache: list[float] | None = None,
) -> VerifyQKVRopeResult:
    metadata = config.metadata
    num_rows = len(hidden_rows)
    cache_size = metadata.num_kv_heads * metadata.max_position_embeddings * metadata.head_dim
    key_cache = list(committed_key_cache or [0.0] * cache_size)
    value_cache = list(committed_value_cache or [0.0] * cache_size)
    queries: list[list[float]] = []
    keys: list[list[float]] = []
    values: list[list[float]] = []
    for row_idx, hidden in enumerate(hidden_rows):
        normed = rmsnorm_reference(hidden, weights.attn_norm, config.eps)
        q = matvec_reference(normed, weights.q_proj, metadata.head_dim, metadata.hidden_size)
        k = matvec_reference(normed, weights.k_proj, metadata.head_dim, metadata.hidden_size)
        v = matvec_reference(normed, weights.v_proj, metadata.head_dim, metadata.hidden_size)
        q_rope = rope_reference(q, rope_cos_rows[row_idx], rope_sin_rows[row_idx])
        k_rope = rope_reference(k, rope_cos_rows[row_idx], rope_sin_rows[row_idx])
        position = start_position + row_idx
        key_cache = kv_append_reference(
            key_cache,
            k_rope,
            head_idx=0,
            position=position,
            num_heads=metadata.num_kv_heads,
            max_seq=metadata.max_position_embeddings,
            head_dim=metadata.head_dim,
        )
        value_cache = kv_append_reference(
            value_cache,
            v,
            head_idx=0,
            position=position,
            num_heads=metadata.num_kv_heads,
            max_seq=metadata.max_position_embeddings,
            head_dim=metadata.head_dim,
        )
        queries.append(q_rope)
        keys.append(k_rope)
        values.append(v)
    return VerifyQKVRopeResult(
        queries=queries,
        keys=keys,
        values=values,
        key_cache=key_cache,
        value_cache=value_cache,
    )


def verify_attention_reference(
    queries: list[list[float]],
    key_cache: list[float],
    value_cache: list[float],
    config: BaselineDecodeConfig,
    *,
    start_position: int,
) -> list[list[float]]:
    metadata = config.metadata
    mask = causal_verify_mask(len(queries))
    out = []
    for row_idx, query in enumerate(queries):
        allowed_positions = start_position + row_idx + 1
        if not mask[row_idx][row_idx]:
            raise AssertionError("causal mask must allow self")
        out.append(
            attention_reference(
                query,
                key_cache,
                value_cache,
                kv_head_idx=0,
                seq_len=allowed_positions,
                num_kv_heads=metadata.num_kv_heads,
                max_seq=metadata.max_position_embeddings,
                head_dim=metadata.head_dim,
            )
        )
    return out


def verify_layer_reference(
    hidden_rows: list[list[float]],
    weights: OneLayerWeights,
    config: BaselineDecodeConfig,
    *,
    start_position: int,
    rope_cos_rows: list[list[float]],
    rope_sin_rows: list[list[float]],
) -> list[list[float]]:
    metadata = config.metadata
    qkv = verify_qkv_rope_reference(
        hidden_rows,
        weights,
        config,
        start_position=start_position,
        rope_cos_rows=rope_cos_rows,
        rope_sin_rows=rope_sin_rows,
    )
    attn_rows = verify_attention_reference(
        qkv.queries,
        qkv.key_cache,
        qkv.value_cache,
        config,
        start_position=start_position,
    )
    out = []
    for hidden, attn in zip(hidden_rows, attn_rows):
        o = matvec_reference(attn, weights.o_proj, metadata.hidden_size, metadata.head_dim)
        h = [a + b for a, b in zip(hidden, o)]
        mlp_norm = rmsnorm_reference(h, weights.mlp_norm, config.eps)
        mlp = mlp_reference(
            mlp_norm,
            weights.gate_proj,
            weights.up_proj,
            weights.down_proj,
            metadata.hidden_size,
            metadata.intermediate_size,
        )
        out.append([a + b for a, b in zip(h, mlp)])
    return out


def verify_logits_reference(
    hidden_rows: list[list[float]],
    *,
    final_norm: list[float],
    lm_head: list[float],
    config: BaselineDecodeConfig,
) -> list[list[float]]:
    metadata = config.metadata
    return [
        lm_head_reference(
            hidden,
            final_norm,
            lm_head,
            metadata.hidden_size,
            metadata.vocab_size,
            config.eps,
        )
        for hidden in hidden_rows
    ]
