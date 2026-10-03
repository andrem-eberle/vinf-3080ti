from __future__ import annotations

from dataclasses import dataclass

from vinf.errors import UnsupportedModelError
from vinf.gguf.parser import GGUFFile
from vinf.gguf.qwen_tensors import (
    load_dequantized_qwen_tensor,
    load_qwen_full_attention_layer_weights,
    load_qwen_linear_attention_layer_weights,
    load_qwen_token_embeddings,
    qwen_layer_schedule,
    qwen_lm_head_logits_for_tokens,
    qwen_tensor_coverage,
)
from vinf.models.metadata import ModelMetadata
from vinf.qwen_ops import (
    QwenKVCache,
    QwenLayerKind,
    QwenLayerWeights,
    QwenLinearAttentionCache,
    QwenLinearAttentionConfig,
    QwenLinearAttentionWeights,
    QwenRopeConfig,
    qwen_empty_kv_cache,
    qwen_empty_linear_attention_cache,
    qwen_full_attention_layer,
    qwen_linear_attention_config_from_metadata,
    qwen_linear_attention_layer,
    qwen_rope_frequencies,
)

QwenBlockWeights = QwenLayerWeights | QwenLinearAttentionWeights
QwenBlockCache = QwenKVCache | QwenLinearAttentionCache


@dataclass(frozen=True, slots=True)
class QwenBaselineWeights:
    layers: tuple[QwenBlockWeights, ...]
    final_norm: list[float]
    linear_config: QwenLinearAttentionConfig | None = None

    @property
    def layer_kinds(self) -> tuple[QwenLayerKind, ...]:
        return tuple(
            QwenLayerKind.FULL_ATTENTION if isinstance(layer, QwenLayerWeights) else QwenLayerKind.LINEAR_ATTENTION
            for layer in self.layers
        )


@dataclass(frozen=True, slots=True)
class QwenPrefillState:
    prompt_len: int
    final_hidden: list[float]
    caches: tuple[QwenBlockCache, ...]


@dataclass(frozen=True, slots=True)
class QwenDecodeResult:
    hidden: list[float]
    logits: list[float]
    token_ids: tuple[int, ...]
    caches: tuple[QwenBlockCache, ...]


def assert_qwen_decode_supported(gguf: GGUFFile, metadata: ModelMetadata) -> tuple[QwenLayerKind, ...]:
    coverage = qwen_tensor_coverage(gguf, metadata)
    if not coverage.complete:
        raise UnsupportedModelError("missing Qwen tensors: " + ", ".join(coverage.missing_tensors))
    schedule = qwen_layer_schedule(gguf, metadata)
    if QwenLayerKind.LINEAR_ATTENTION in schedule:
        qwen_linear_attention_config_from_metadata(gguf, metadata)
    return schedule


def load_qwen_baseline_weights(gguf: GGUFFile, metadata: ModelMetadata) -> QwenBaselineWeights:
    schedule = assert_qwen_decode_supported(gguf, metadata)
    linear_config = (
        qwen_linear_attention_config_from_metadata(gguf, metadata)
        if QwenLayerKind.LINEAR_ATTENTION in schedule
        else None
    )
    layers: list[QwenBlockWeights] = []
    for layer_idx, kind in enumerate(schedule):
        if kind is QwenLayerKind.FULL_ATTENTION:
            layers.append(load_qwen_full_attention_layer_weights(gguf, metadata, layer_idx))
        else:
            layers.append(load_qwen_linear_attention_layer_weights(gguf, metadata, layer_idx, linear_config))
    return QwenBaselineWeights(
        layers=tuple(layers),
        final_norm=load_dequantized_qwen_tensor(gguf, "output_norm.weight"),
        linear_config=linear_config,
    )


def qwen_empty_caches(metadata: ModelMetadata, weights: QwenBaselineWeights) -> tuple[QwenBlockCache, ...]:
    caches: list[QwenBlockCache] = []
    for kind in weights.layer_kinds:
        if kind is QwenLayerKind.FULL_ATTENTION:
            caches.append(qwen_empty_kv_cache(metadata))
        else:
            if weights.linear_config is None:
                raise UnsupportedModelError("linear-attention layers require a linear attention config")
            caches.append(qwen_empty_linear_attention_cache(weights.linear_config))
    return tuple(caches)


def qwen_prefill_tokens_reference(
    gguf: GGUFFile,
    metadata: ModelMetadata,
    weights: QwenBaselineWeights,
    prompt_tokens: list[int],
    rope: QwenRopeConfig,
    *,
    eps: float = 1e-6,
) -> QwenPrefillState:
    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if len(prompt_tokens) > metadata.max_position_embeddings:
        raise ValueError("prompt length exceeds max_position_embeddings")
    embeddings = load_qwen_token_embeddings(gguf, prompt_tokens)
    caches = qwen_empty_caches(metadata, weights)
    final_hidden: list[float] = []
    for position, embedding in enumerate(embeddings):
        final_hidden, caches = qwen_decode_embedding_reference(
            metadata,
            weights,
            embedding,
            caches,
            position=position,
            rope=rope,
            eps=eps,
        )
    return QwenPrefillState(
        prompt_len=len(prompt_tokens),
        final_hidden=final_hidden,
        caches=caches,
    )


def qwen_decode_embedding_reference(
    metadata: ModelMetadata,
    weights: QwenBaselineWeights,
    hidden: list[float],
    caches: tuple[QwenBlockCache, ...],
    *,
    position: int,
    rope: QwenRopeConfig,
    eps: float = 1e-6,
) -> tuple[list[float], tuple[QwenBlockCache, ...]]:
    if len(weights.layers) != len(caches):
        raise ValueError("weights/caches layer count mismatch")
    cos, sin = qwen_rope_frequencies(rope, position)
    h = list(hidden)
    updated: list[QwenBlockCache] = []
    for layer_idx, (layer_weights, cache) in enumerate(zip(weights.layers, caches)):
        if isinstance(layer_weights, QwenLayerWeights) and isinstance(cache, QwenKVCache):
            result = qwen_full_attention_layer(
                h,
                layer_weights,
                metadata,
                cache,
                position=position,
                rope_cos=cos,
                rope_sin=sin,
                eps=eps,
            )
        elif isinstance(layer_weights, QwenLinearAttentionWeights) and isinstance(cache, QwenLinearAttentionCache):
            result = qwen_linear_attention_layer(h, layer_weights, metadata, cache, eps=eps)
        else:
            raise ValueError(f"layer {layer_idx} weights/cache block kinds do not match")
        h = result.hidden
        updated.append(result.cache)
    return h, tuple(updated)


def qwen_decode_next_token_reference(
    gguf: GGUFFile,
    metadata: ModelMetadata,
    weights: QwenBaselineWeights,
    prefill: QwenPrefillState,
    token_id: int,
    rope: QwenRopeConfig,
    *,
    output_token_ids: list[int] | None = None,
    eps: float = 1e-6,
) -> QwenDecodeResult:
    if prefill.prompt_len >= metadata.max_position_embeddings:
        raise ValueError("prefill prompt_len leaves no room for decode")
    hidden = load_qwen_token_embeddings(gguf, [token_id])[0]
    h, caches = qwen_decode_embedding_reference(
        metadata,
        weights,
        hidden,
        prefill.caches,
        position=prefill.prompt_len,
        rope=rope,
        eps=eps,
    )
    token_ids = tuple(output_token_ids if output_token_ids is not None else range(metadata.vocab_size))
    logits = qwen_lm_head_logits_for_tokens(
        gguf,
        h,
        list(token_ids),
        norm_weight=weights.final_norm,
        eps=eps,
    )
    return QwenDecodeResult(hidden=h, logits=logits, token_ids=token_ids, caches=caches)


def qwen_prompt_logits_reference(
    gguf: GGUFFile,
    metadata: ModelMetadata,
    weights: QwenBaselineWeights,
    prompt_tokens: list[int],
    rope: QwenRopeConfig,
    *,
    output_token_ids: list[int] | None = None,
    eps: float = 1e-6,
) -> QwenDecodeResult:
    prefill = qwen_prefill_tokens_reference(
        gguf,
        metadata,
        weights,
        prompt_tokens,
        rope,
        eps=eps,
    )
    token_ids = tuple(output_token_ids if output_token_ids is not None else range(metadata.vocab_size))
    logits = qwen_lm_head_logits_for_tokens(
        gguf,
        prefill.final_hidden,
        list(token_ids),
        norm_weight=weights.final_norm,
        eps=eps,
    )
    return QwenDecodeResult(
        hidden=prefill.final_hidden,
        logits=logits,
        token_ids=token_ids,
        caches=prefill.caches,
    )


def qwen_greedy_next_token_reference(
    gguf: GGUFFile,
    metadata: ModelMetadata,
    weights: QwenBaselineWeights,
    prompt_tokens: list[int],
    rope: QwenRopeConfig,
    *,
    eps: float = 1e-6,
) -> int:
    result = qwen_prompt_logits_reference(
        gguf,
        metadata,
        weights,
        prompt_tokens,
        rope,
        eps=eps,
    )
    best_idx = max(range(len(result.logits)), key=lambda idx: result.logits[idx])
    return result.token_ids[best_idx]
