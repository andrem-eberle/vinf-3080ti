from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from vinf.errors import UnsupportedModelError
from vinf.gguf.parser import GGUFFile
from vinf.models.metadata import ModelMetadata
from vinf.reference_ops import matvec_reference, rmsnorm_reference, silu


@dataclass(frozen=True, slots=True)
class QwenRopeConfig:
    head_dim: int
    freq_base: float
    dimension_sections: tuple[int, ...]
    rotary_dim: int | None = None

    @property
    def rotated_dims(self) -> int:
        return self.head_dim if self.rotary_dim is None else self.rotary_dim


class QwenLayerKind(Enum):
    FULL_ATTENTION = "full_attention"
    LINEAR_ATTENTION = "linear_attention"


@dataclass(frozen=True, slots=True)
class QwenKVCache:
    key: list[float]
    value: list[float]
    num_kv_heads: int
    max_seq: int
    head_dim: int


@dataclass(frozen=True, slots=True)
class QwenAttentionWeights:
    q_proj: list[float]
    k_proj: list[float]
    v_proj: list[float]
    o_proj: list[float]


@dataclass(frozen=True, slots=True)
class QwenMLPWeights:
    gate_proj: list[float]
    up_proj: list[float]
    down_proj: list[float]


@dataclass(frozen=True, slots=True)
class QwenLayerWeights:
    """Full-attention block weights.

    qwen35 full-attention blocks use a gated query projection (``q_proj`` emits
    ``[query, gate]`` per head, 2 * head_dim rows each) plus per-head q/k
    RMSNorm. Plain Qwen blocks use an ungated ``q_proj`` and no q/k norm.
    """

    attn_norm: list[float]
    attention: QwenAttentionWeights
    post_attention_norm: list[float]
    mlp: QwenMLPWeights
    q_norm: list[float] | None = None
    k_norm: list[float] | None = None


@dataclass(frozen=True, slots=True)
class QwenLinearAttentionWeights:
    attn_norm: list[float]
    qkv_proj: list[float]
    gate_proj: list[float]
    beta_proj: list[float]
    alpha_proj: list[float]
    # GGUF `ssm_a` stores -exp(A_log), so per-head decay is exp(ssm_a * softplus(alpha + dt_bias)).
    ssm_a: list[float]
    dt_bias: list[float]
    # GGUF `ssm_conv1d` dims are (conv_kernel, conv_dim): kernel taps are contiguous per channel.
    conv1d: list[float]
    ssm_norm: list[float]
    out_proj: list[float]
    post_attention_norm: list[float]
    mlp: QwenMLPWeights


@dataclass(frozen=True, slots=True)
class QwenLinearAttentionConfig:
    key_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int
    # GGUF (llama.cpp conversion) orders value heads tiled: value head h uses key head
    # h % key_heads. HF checkpoints are grouped: h // (value_heads // key_heads).
    tiled_value_heads: bool = True

    def key_head_for(self, value_head: int) -> int:
        if self.tiled_value_heads:
            return value_head % self.key_heads
        return value_head // (self.value_heads // self.key_heads)

    @property
    def key_dim(self) -> int:
        return self.key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim


@dataclass(frozen=True, slots=True)
class QwenLinearAttentionCache:
    conv_state: list[float]
    recurrent_state: list[float]
    config: QwenLinearAttentionConfig


@dataclass(frozen=True, slots=True)
class QwenLayerResult:
    hidden: list[float]
    cache: QwenKVCache | QwenLinearAttentionCache
    stops: dict[str, list[float]]


def qwen_rmsnorm(values: list[float], weights: list[float], eps: float = 1e-6) -> list[float]:
    return rmsnorm_reference(values, weights, eps)


def qwen_rope_config_from_gguf(gguf: GGUFFile, metadata: ModelMetadata) -> QwenRopeConfig:
    freq_base = gguf.metadata_value(f"{metadata.architecture.value}.rope.freq_base")
    sections = gguf.metadata_value(f"{metadata.architecture.value}.rope.dimension_sections")
    if not isinstance(freq_base, float):
        raise UnsupportedModelError("missing qwen35 rope freq_base")
    if not isinstance(sections, list) or not all(isinstance(item, int) for item in sections):
        raise UnsupportedModelError("missing qwen35 rope dimension sections")
    return QwenRopeConfig(
        head_dim=metadata.head_dim,
        freq_base=freq_base,
        dimension_sections=tuple(sections),
        rotary_dim=_optional_int(gguf.metadata_value(f"{metadata.architecture.value}.rope.dimension_count")),
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def qwen_rope_frequencies(config: QwenRopeConfig, position: int) -> tuple[list[float], list[float]]:
    if position < 0:
        raise ValueError("position must be non-negative")
    rotary_dim = config.rotated_dims
    if rotary_dim % 2 != 0 or rotary_dim <= 0 or rotary_dim > config.head_dim:
        raise UnsupportedModelError("Qwen RoPE rotary dimension must be even and within head_dim")
    if sum(config.dimension_sections) > rotary_dim // 2:
        raise UnsupportedModelError("Qwen RoPE dimension sections exceed rotary dimensions")
    # Text-only prompts share one position id across all M-RoPE sections, so the
    # sectioned rotation reduces to NeoX RoPE over the first `rotary_dim` dims.
    cos_half: list[float] = []
    sin_half: list[float] = []
    for idx in range(rotary_dim // 2):
        theta = position / (config.freq_base ** (2.0 * idx / rotary_dim))
        cos_half.append(math.cos(theta))
        sin_half.append(math.sin(theta))
    return cos_half + cos_half, sin_half + sin_half


def qwen_apply_rope(values: list[float], cos: list[float], sin: list[float]) -> list[float]:
    """NeoX-style rotation of the first len(cos) dims; remaining dims pass through."""
    if len(cos) != len(sin) or len(cos) > len(values) or len(cos) % 2 != 0:
        raise ValueError("cos/sin must have equal even length no larger than values")
    out = list(values)
    half = len(cos) // 2
    for idx in range(half):
        x0 = values[idx]
        x1 = values[idx + half]
        c = cos[idx]
        s = sin[idx]
        out[idx] = x0 * c - x1 * s
        out[idx + half] = x1 * c + x0 * s
    return out


def qwen_empty_kv_cache(metadata: ModelMetadata) -> QwenKVCache:
    size = metadata.num_kv_heads * metadata.max_position_embeddings * metadata.head_dim
    return QwenKVCache(
        key=[0.0] * size,
        value=[0.0] * size,
        num_kv_heads=metadata.num_kv_heads,
        max_seq=metadata.max_position_embeddings,
        head_dim=metadata.head_dim,
    )


def qwen_empty_linear_attention_cache(config: QwenLinearAttentionConfig) -> QwenLinearAttentionCache:
    return QwenLinearAttentionCache(
        conv_state=[0.0] * (config.conv_dim * config.conv_kernel),
        recurrent_state=[0.0] * (config.value_heads * config.key_head_dim * config.value_head_dim),
        config=config,
    )


def qwen_gqa_attention(
    queries: list[float],
    key_cache: list[float],
    value_cache: list[float],
    *,
    num_attention_heads: int,
    num_kv_heads: int,
    max_seq: int,
    head_dim: int,
    seq_len: int,
) -> list[float]:
    if num_attention_heads % num_kv_heads != 0:
        raise UnsupportedModelError("Qwen attention heads must be divisible by KV heads")
    if len(queries) != num_attention_heads * head_dim:
        raise ValueError("query length must equal num_attention_heads * head_dim")
    out: list[float] = []
    group_size = num_attention_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    for head_idx in range(num_attention_heads):
        query = queries[head_idx * head_dim : (head_idx + 1) * head_dim]
        kv_head_idx = head_idx // group_size
        scores = []
        for pos in range(seq_len):
            base = (kv_head_idx * max_seq + pos) * head_dim
            scores.append(sum(query[i] * key_cache[base + i] for i in range(head_dim)) * scale)
        max_score = max(scores)
        probs = [math.exp(score - max_score) for score in scores]
        denom = sum(probs)
        probs = [prob / denom for prob in probs]
        head_out = [0.0] * head_dim
        for pos, prob in enumerate(probs):
            base = (kv_head_idx * max_seq + pos) * head_dim
            for dim in range(head_dim):
                head_out[dim] += prob * value_cache[base + dim]
        out.extend(head_out)
    return out


def qwen_mlp(values: list[float], weights: QwenMLPWeights, hidden_size: int, intermediate_size: int) -> list[float]:
    gate = matvec_reference(values, weights.gate_proj, intermediate_size, hidden_size)
    up = matvec_reference(values, weights.up_proj, intermediate_size, hidden_size)
    activated = [silu(g) * u for g, u in zip(gate, up)]
    return matvec_reference(activated, weights.down_proj, hidden_size, intermediate_size)


def qwen_lm_head(
    values: list[float],
    norm_weight: list[float],
    output_weight: list[float],
    hidden_size: int,
    vocab_size: int,
    eps: float = 1e-6,
) -> list[float]:
    normed = qwen_rmsnorm(values, norm_weight, eps)
    return matvec_reference(normed, output_weight, vocab_size, hidden_size)


def qwen_full_attention_layer(
    hidden: list[float],
    weights: QwenLayerWeights,
    metadata: ModelMetadata,
    cache: QwenKVCache,
    *,
    position: int,
    rope_cos: list[float],
    rope_sin: list[float],
    eps: float = 1e-6,
) -> QwenLayerResult:
    if position < 0 or position >= metadata.max_position_embeddings:
        raise ValueError("position out of range")
    stops: dict[str, list[float]] = {}
    head_dim = metadata.head_dim
    q_width = metadata.num_attention_heads * head_dim
    gated = len(weights.attention.q_proj) == 2 * q_width * metadata.hidden_size
    if not gated and len(weights.attention.q_proj) != q_width * metadata.hidden_size:
        raise UnsupportedModelError("Qwen q_proj size matches neither plain nor gated query layout")
    normed = qwen_rmsnorm(hidden, weights.attn_norm, eps)
    q_raw = matvec_reference(
        normed,
        weights.attention.q_proj,
        q_width * (2 if gated else 1),
        metadata.hidden_size,
    )
    if gated:
        q, gate = _split_gated_query(q_raw, metadata.num_attention_heads, head_dim)
    else:
        q, gate = q_raw, None
    k = matvec_reference(
        normed,
        weights.attention.k_proj,
        metadata.num_kv_heads * metadata.head_dim,
        metadata.hidden_size,
    )
    v = matvec_reference(
        normed,
        weights.attention.v_proj,
        metadata.num_kv_heads * metadata.head_dim,
        metadata.hidden_size,
    )
    q_rope = []
    for head in _split_heads(q, metadata.num_attention_heads, head_dim):
        if weights.q_norm is not None:
            head = qwen_rmsnorm(head, weights.q_norm, eps)
        q_rope.extend(qwen_apply_rope(head, rope_cos, rope_sin))
    k_rope = []
    for head in _split_heads(k, metadata.num_kv_heads, head_dim):
        if weights.k_norm is not None:
            head = qwen_rmsnorm(head, weights.k_norm, eps)
        k_rope.extend(qwen_apply_rope(head, rope_cos, rope_sin))
    key = list(cache.key)
    value = list(cache.value)
    for head_idx in range(metadata.num_kv_heads):
        base_src = head_idx * metadata.head_dim
        base_dst = (head_idx * metadata.max_position_embeddings + position) * metadata.head_dim
        key[base_dst : base_dst + metadata.head_dim] = k_rope[base_src : base_src + metadata.head_dim]
        value[base_dst : base_dst + metadata.head_dim] = v[base_src : base_src + metadata.head_dim]
    attn = qwen_gqa_attention(
        q_rope,
        key,
        value,
        num_attention_heads=metadata.num_attention_heads,
        num_kv_heads=metadata.num_kv_heads,
        max_seq=metadata.max_position_embeddings,
        head_dim=metadata.head_dim,
        seq_len=position + 1,
    )
    if gate is not None:
        attn = [value * _sigmoid(g) for value, g in zip(attn, gate)]
    projected = matvec_reference(
        attn,
        weights.attention.o_proj,
        metadata.hidden_size,
        metadata.num_attention_heads * metadata.head_dim,
    )
    h = [a + b for a, b in zip(hidden, projected)]
    mlp_norm = qwen_rmsnorm(h, weights.post_attention_norm, eps)
    mlp = qwen_mlp(mlp_norm, weights.mlp, metadata.hidden_size, metadata.intermediate_size)
    h = [a + b for a, b in zip(h, mlp)]
    stops.update({"attn_norm": normed, "q_rope": q_rope, "k_rope": k_rope, "attention": attn, "post_mlp": h})
    return QwenLayerResult(
        hidden=h,
        cache=QwenKVCache(
            key=key,
            value=value,
            num_kv_heads=metadata.num_kv_heads,
            max_seq=metadata.max_position_embeddings,
            head_dim=metadata.head_dim,
        ),
        stops=stops,
    )


@dataclass(frozen=True, slots=True)
class QwenMtpWeights:
    """qwen35 nextn / multi-token-prediction block (GGUF blk.<decoder_layers>)."""

    enorm: list[float]
    hnorm: list[float]
    eh_proj: list[float]
    shared_head_norm: list[float]
    block: QwenLayerWeights


def qwen_mtp_layer(
    embedding: list[float],
    hidden: list[float],
    weights: QwenMtpWeights,
    metadata: ModelMetadata,
    cache: QwenKVCache,
    *,
    position: int,
    rope_cos: list[float],
    rope_sin: list[float],
    eps: float = 1e-6,
) -> tuple[list[float], QwenKVCache]:
    """One MTP step at `position`: embedding of the next token plus a post-norm hidden state
    (target model output, or the previous MTP output when drafting recursively).

    Returns the post-norm MTP output (fed to the shared LM head and to the next draft step)
    and the updated MTP KV cache.
    """
    joined = qwen_rmsnorm(embedding, weights.enorm, eps) + qwen_rmsnorm(hidden, weights.hnorm, eps)
    x = matvec_reference(joined, weights.eh_proj, metadata.hidden_size, 2 * metadata.hidden_size)
    result = qwen_full_attention_layer(
        x, weights.block, metadata, cache, position=position, rope_cos=rope_cos, rope_sin=rope_sin, eps=eps
    )
    return qwen_rmsnorm(result.hidden, weights.shared_head_norm, eps), result.cache


def qwen_linear_attention_layer(
    hidden: list[float],
    weights: QwenLinearAttentionWeights,
    metadata: ModelMetadata,
    cache: QwenLinearAttentionCache,
    *,
    eps: float = 1e-6,
) -> QwenLayerResult:
    config = cache.config
    stops: dict[str, list[float]] = {}
    normed = qwen_rmsnorm(hidden, weights.attn_norm, eps)
    qkv = matvec_reference(normed, weights.qkv_proj, config.conv_dim, metadata.hidden_size)
    z = matvec_reference(normed, weights.gate_proj, config.value_dim, metadata.hidden_size)
    beta = [_sigmoid(value) for value in matvec_reference(normed, weights.beta_proj, config.value_heads, metadata.hidden_size)]
    alpha = matvec_reference(normed, weights.alpha_proj, config.value_heads, metadata.hidden_size)
    conv_out, conv_state = _causal_depthwise_conv_update(qkv, cache.conv_state, weights.conv1d, config)
    conv_out = [silu(value) for value in conv_out]
    query = conv_out[: config.key_dim]
    key = conv_out[config.key_dim : config.key_dim * 2]
    value = conv_out[config.key_dim * 2 :]
    if config.value_heads % config.key_heads != 0:
        raise UnsupportedModelError("Qwen linear attention value heads must divide key heads")
    query_heads = _split_heads(query, config.key_heads, config.key_head_dim)
    key_heads = _split_heads(key, config.key_heads, config.key_head_dim)
    value_heads = _split_heads(value, config.value_heads, config.value_head_dim)
    z_heads = _split_heads(z, config.value_heads, config.value_head_dim)
    state = list(cache.recurrent_state)
    outputs: list[float] = []
    for value_head_idx in range(config.value_heads):
        key_head_idx = config.key_head_for(value_head_idx)
        q = _l2norm(query_heads[key_head_idx], eps)
        q = [item / math.sqrt(config.key_head_dim) for item in q]
        k = _l2norm(key_heads[key_head_idx], eps)
        v = value_heads[value_head_idx]
        decay = math.exp(weights.ssm_a[value_head_idx] * _softplus(alpha[value_head_idx] + weights.dt_bias[value_head_idx]))
        state_base = value_head_idx * config.key_head_dim * config.value_head_dim
        _decay_recurrent_state(state, state_base, config, decay)
        out, state = _delta_rule_update(
            state,
            state_base,
            q,
            k,
            v,
            beta[value_head_idx],
            config,
        )
        gated = _gated_rmsnorm(out, weights.ssm_norm, z_heads[value_head_idx], eps)
        outputs.extend(gated)
    projected = matvec_reference(outputs, weights.out_proj, metadata.hidden_size, config.value_dim)
    h = [a + b for a, b in zip(hidden, projected)]
    mlp_norm = qwen_rmsnorm(h, weights.post_attention_norm, eps)
    mlp = qwen_mlp(mlp_norm, weights.mlp, metadata.hidden_size, metadata.intermediate_size)
    h = [a + b for a, b in zip(h, mlp)]
    stops.update({"attn_norm": normed, "linear_qkv": qkv, "linear_conv": conv_out, "linear_core": outputs, "post_mlp": h})
    return QwenLayerResult(
        hidden=h,
        cache=QwenLinearAttentionCache(
            conv_state=conv_state,
            recurrent_state=state,
            config=config,
        ),
        stops=stops,
    )


def qwen_linear_attention_config_from_metadata(gguf: GGUFFile, metadata: ModelMetadata) -> QwenLinearAttentionConfig:
    prefix = metadata.architecture.value
    fields = {}
    for name in ("group_count", "time_step_rank", "state_size", "conv_kernel", "inner_size"):
        value = gguf.metadata_value(f"{prefix}.ssm.{name}")
        if value is None:
            raise UnsupportedModelError(f"missing {prefix}.ssm.{name} for linear-attention layers")
        fields[name] = int(value)
    config = QwenLinearAttentionConfig(
        key_heads=fields["group_count"],
        value_heads=fields["time_step_rank"],
        key_head_dim=fields["state_size"],
        value_head_dim=fields["state_size"],
        conv_kernel=fields["conv_kernel"],
    )
    if config.value_dim != fields["inner_size"]:
        raise UnsupportedModelError(
            f"{prefix}.ssm.inner_size {fields['inner_size']} != time_step_rank * state_size {config.value_dim}"
        )
    if config.value_heads % config.key_heads != 0:
        raise UnsupportedModelError("Qwen linear attention value heads must be a multiple of key heads")
    return config


def qwen_layer_kinds_from_interval(num_layers: int, full_attention_interval: int) -> tuple[QwenLayerKind, ...]:
    """qwen35 places a full-attention block at every `interval`-th layer (1-based), linear attention elsewhere."""
    if full_attention_interval <= 0:
        raise UnsupportedModelError("qwen35 full_attention_interval must be positive")
    return tuple(
        QwenLayerKind.FULL_ATTENTION
        if (layer_idx + 1) % full_attention_interval == 0
        else QwenLayerKind.LINEAR_ATTENTION
        for layer_idx in range(num_layers)
    )


def _causal_depthwise_conv_update(
    values: list[float],
    state: list[float],
    weights: list[float],
    config: QwenLinearAttentionConfig,
) -> tuple[list[float], list[float]]:
    if len(values) != config.conv_dim:
        raise ValueError("linear attention qkv length does not match conv_dim")
    if len(state) != config.conv_dim * config.conv_kernel:
        raise ValueError("linear attention conv state size mismatch")
    if len(weights) != config.conv_kernel * config.conv_dim:
        raise ValueError("linear attention conv weight size mismatch")
    next_state = list(state)
    out = [0.0] * config.conv_dim
    for channel in range(config.conv_dim):
        base = channel * config.conv_kernel
        window = next_state[base + 1 : base + config.conv_kernel] + [values[channel]]
        next_state[base : base + config.conv_kernel] = window
        for kernel_idx, item in enumerate(window):
            out[channel] += item * weights[base + kernel_idx]
    return out, next_state


def _split_gated_query(values: list[float], heads: int, head_dim: int) -> tuple[list[float], list[float]]:
    query: list[float] = []
    gate: list[float] = []
    for head_idx in range(heads):
        base = head_idx * head_dim * 2
        query.extend(values[base : base + head_dim])
        gate.extend(values[base + head_dim : base + head_dim * 2])
    return query, gate


def _split_heads(values: list[float], heads: int, head_dim: int) -> list[list[float]]:
    return [values[idx * head_dim : (idx + 1) * head_dim] for idx in range(heads)]


def _l2norm(values: list[float], eps: float) -> list[float]:
    denom = math.sqrt(sum(value * value for value in values) + eps)
    return [value / denom for value in values]


def _softplus(value: float) -> float:
    if value > 20.0:
        return value
    return math.log1p(math.exp(value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _decay_recurrent_state(
    state: list[float],
    state_base: int,
    config: QwenLinearAttentionConfig,
    decay: float,
) -> None:
    count = config.key_head_dim * config.value_head_dim
    for idx in range(count):
        state[state_base + idx] *= decay


def _delta_rule_update(
    state: list[float],
    state_base: int,
    query: list[float],
    key: list[float],
    value: list[float],
    beta: float,
    config: QwenLinearAttentionConfig,
) -> tuple[list[float], list[float]]:
    predicted = [0.0] * config.value_head_dim
    for k_idx in range(config.key_head_dim):
        row = state_base + k_idx * config.value_head_dim
        for v_idx in range(config.value_head_dim):
            predicted[v_idx] += state[row + v_idx] * key[k_idx]
    delta = [(value[idx] - predicted[idx]) * beta for idx in range(config.value_head_dim)]
    for k_idx in range(config.key_head_dim):
        row = state_base + k_idx * config.value_head_dim
        for v_idx in range(config.value_head_dim):
            state[row + v_idx] += key[k_idx] * delta[v_idx]
    out = [0.0] * config.value_head_dim
    for k_idx in range(config.key_head_dim):
        row = state_base + k_idx * config.value_head_dim
        for v_idx in range(config.value_head_dim):
            out[v_idx] += state[row + v_idx] * query[k_idx]
    return out, state


def _gated_rmsnorm(values: list[float], weight: list[float], gate: list[float], eps: float) -> list[float]:
    if len(values) != len(weight) or len(values) != len(gate):
        raise ValueError("gated RMSNorm values/weight/gate lengths must match")
    mean_square = sum(value * value for value in values) / len(values)
    scale = 1.0 / math.sqrt(mean_square + eps)
    return [value * scale * norm * silu(gate_value) for value, norm, gate_value in zip(values, weight, gate)]
