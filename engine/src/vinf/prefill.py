from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vinf.baseline_decode import BaselineDecodeConfig, BaselineTargetWeights
from vinf.gguf.parser import GGUFFile
from vinf.gguf.qwen_tensors import load_qwen_token_embeddings
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


class PrefillBackend(StrEnum):
    FALLBACK_STANDALONE_OPS = "fallback_standalone_ops"


@dataclass(frozen=True, slots=True)
class LayerKVCache:
    key: list[float]
    value: list[float]


@dataclass(frozen=True, slots=True)
class TargetPrefillState:
    backend: PrefillBackend
    prompt_len: int
    final_hidden: list[float]
    layers: tuple[LayerKVCache, ...]
    memory_bytes: int


@dataclass(frozen=True, slots=True)
class PrefillDecodeResult:
    hidden: list[float]
    logits: list[float]
    layers: tuple[LayerKVCache, ...]


def target_prefill_reference(
    prompt_hidden_states: list[list[float]],
    weights: BaselineTargetWeights,
    config: BaselineDecodeConfig,
    *,
    rope_cos: list[float] | list[list[float]],
    rope_sin: list[float] | list[list[float]],
) -> TargetPrefillState:
    # TODO: replace this hidden-state scaffold with token embedding/model-backed prefill.
    _validate_prompt(prompt_hidden_states, config)
    layers = _empty_layer_caches(weights, config)
    hidden_at_positions = [list(hidden) for hidden in prompt_hidden_states]

    for layer_idx, layer_weights in enumerate(weights.layers):
        next_hidden_at_positions: list[list[float]] = []
        layer_cache = layers[layer_idx]
        for position, hidden in enumerate(hidden_at_positions):
            result = _run_layer_with_cache(
                hidden,
                layer_weights,
                layer_cache,
                config,
                position=position,
                rope_cos=_rope_for_position(rope_cos, position),
                rope_sin=_rope_for_position(rope_sin, position),
            )
            layer_cache = result.cache
            next_hidden_at_positions.append(result.hidden)
        layers = _replace_layer_cache(layers, layer_idx, layer_cache)
        hidden_at_positions = next_hidden_at_positions

    final_hidden = hidden_at_positions[-1] if hidden_at_positions else []
    return TargetPrefillState(
        backend=PrefillBackend.FALLBACK_STANDALONE_OPS,
        prompt_len=len(prompt_hidden_states),
        final_hidden=final_hidden,
        layers=layers,
        memory_bytes=estimate_prefill_memory_bytes(config, len(prompt_hidden_states)),
    )


def target_prefill_from_tokens_reference(
    prompt_tokens: list[int],
    gguf: GGUFFile,
    weights: BaselineTargetWeights,
    config: BaselineDecodeConfig,
    *,
    rope_cos: list[float] | list[list[float]],
    rope_sin: list[float] | list[list[float]],
) -> TargetPrefillState:
    prompt_hidden_states = load_qwen_token_embeddings(gguf, prompt_tokens)
    return target_prefill_reference(
        prompt_hidden_states,
        weights,
        config,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )


def decode_after_prefill_reference(
    hidden: list[float],
    prefill: TargetPrefillState,
    weights: BaselineTargetWeights,
    config: BaselineDecodeConfig,
    *,
    rope_cos: list[float] | list[list[float]],
    rope_sin: list[float] | list[list[float]],
) -> PrefillDecodeResult:
    if prefill.prompt_len >= config.metadata.max_position_embeddings:
        raise ValueError("prefill prompt_len leaves no room for decode")
    h = list(hidden)
    layers = prefill.layers
    position = prefill.prompt_len
    for layer_idx, layer_weights in enumerate(weights.layers):
        result = _run_layer_with_cache(
            h,
            layer_weights,
            layers[layer_idx],
            config,
            position=position,
            rope_cos=_rope_for_position(rope_cos, position),
            rope_sin=_rope_for_position(rope_sin, position),
        )
        layers = _replace_layer_cache(layers, layer_idx, result.cache)
        h = result.hidden
    logits = lm_head_reference(
        h,
        weights.final_norm,
        weights.lm_head,
        config.metadata.hidden_size,
        config.metadata.vocab_size,
        config.eps,
    )
    return PrefillDecodeResult(hidden=h, logits=logits, layers=layers)


def estimate_prefill_memory_bytes(config: BaselineDecodeConfig, prompt_len: int) -> int:
    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    metadata = config.metadata
    kv_values = (
        metadata.num_hidden_layers
        * 2
        * metadata.num_kv_heads
        * metadata.max_position_embeddings
        * metadata.head_dim
    )
    hidden_values = max(1, prompt_len) * metadata.hidden_size
    dtype_bytes = 2 if metadata.dtype == "fp16" else 4
    return (kv_values + hidden_values) * dtype_bytes


@dataclass(frozen=True, slots=True)
class _LayerRunResult:
    hidden: list[float]
    cache: LayerKVCache


def _run_layer_with_cache(
    hidden: list[float],
    weights: OneLayerWeights,
    cache: LayerKVCache,
    config: BaselineDecodeConfig,
    *,
    position: int,
    rope_cos: list[float],
    rope_sin: list[float],
) -> _LayerRunResult:
    metadata = config.metadata
    normed = rmsnorm_reference(hidden, weights.attn_norm, config.eps)
    q = matvec_reference(normed, weights.q_proj, metadata.head_dim, metadata.hidden_size)
    k = matvec_reference(normed, weights.k_proj, metadata.head_dim, metadata.hidden_size)
    v = matvec_reference(normed, weights.v_proj, metadata.head_dim, metadata.hidden_size)
    q_rope = rope_reference(q, rope_cos, rope_sin)
    k_rope = rope_reference(k, rope_cos, rope_sin)
    key = kv_append_reference(
        cache.key,
        k_rope,
        head_idx=0,
        position=position,
        num_heads=metadata.num_kv_heads,
        max_seq=metadata.max_position_embeddings,
        head_dim=metadata.head_dim,
    )
    value = kv_append_reference(
        cache.value,
        v,
        head_idx=0,
        position=position,
        num_heads=metadata.num_kv_heads,
        max_seq=metadata.max_position_embeddings,
        head_dim=metadata.head_dim,
    )
    attn = attention_reference(
        q_rope,
        key,
        value,
        kv_head_idx=0,
        seq_len=position + 1,
        num_kv_heads=metadata.num_kv_heads,
        max_seq=metadata.max_position_embeddings,
        head_dim=metadata.head_dim,
    )
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
    return _LayerRunResult(
        hidden=[a + b for a, b in zip(h, mlp)],
        cache=LayerKVCache(key=key, value=value),
    )


def _empty_layer_caches(
    weights: BaselineTargetWeights, config: BaselineDecodeConfig
) -> tuple[LayerKVCache, ...]:
    metadata = config.metadata
    cache_size = metadata.num_kv_heads * metadata.max_position_embeddings * metadata.head_dim
    return tuple(
        LayerKVCache(key=[0.0] * cache_size, value=[0.0] * cache_size)
        for _ in weights.layers
    )


def _replace_layer_cache(
    layers: tuple[LayerKVCache, ...], layer_idx: int, cache: LayerKVCache
) -> tuple[LayerKVCache, ...]:
    updated = list(layers)
    updated[layer_idx] = cache
    return tuple(updated)


def _rope_for_position(values: list[float] | list[list[float]], position: int) -> list[float]:
    if values and isinstance(values[0], list):
        return list(values[position])  # type: ignore[index]
    return list(values)  # type: ignore[arg-type]


def _validate_prompt(
    prompt_hidden_states: list[list[float]], config: BaselineDecodeConfig
) -> None:
    if not prompt_hidden_states:
        raise ValueError("prompt_hidden_states must not be empty")
    metadata = config.metadata
    if len(prompt_hidden_states) > metadata.max_position_embeddings:
        raise ValueError("prompt length exceeds max_position_embeddings")
    for hidden in prompt_hidden_states:
        if len(hidden) != metadata.hidden_size:
            raise ValueError("each prompt hidden state must equal hidden_size")
