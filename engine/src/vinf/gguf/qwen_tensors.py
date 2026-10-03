from __future__ import annotations

from dataclasses import dataclass

from vinf.errors import UnsupportedModelError
from vinf.gguf.dequant import dequantize_tensor
from vinf.gguf.parser import GGUFFile, GGUFTensorInfo
from vinf.models.metadata import ModelMetadata
from vinf.qwen_ops import (
    QwenAttentionWeights,
    QwenLayerKind,
    QwenLayerWeights,
    QwenLinearAttentionConfig,
    QwenLinearAttentionWeights,
    QwenMLPWeights,
    QwenMtpWeights,
    qwen_layer_kinds_from_interval,
    qwen_linear_attention_config_from_metadata,
)


HYBRID_LAYER_TENSORS = (
    "attn_gate.weight",
    "attn_norm.weight",
    "attn_qkv.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
    "ssm_a",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_norm.weight",
    "ssm_out.weight",
)

FULL_ATTENTION_LAYER_TENSORS = (
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_v.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
)


@dataclass(frozen=True, slots=True)
class QwenTensorCoverage:
    hybrid_layers: tuple[int, ...]
    full_attention_layers: tuple[int, ...]
    missing_tensors: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_tensors


def qwen_tensor_coverage(gguf: GGUFFile, metadata: ModelMetadata) -> QwenTensorCoverage:
    missing: list[str] = []
    hybrid: list[int] = []
    full: list[int] = []
    for layer_idx in range(metadata.num_hidden_layers):
        prefix = f"blk.{layer_idx}."
        has_hybrid = f"{prefix}attn_qkv.weight" in gguf.tensors
        has_full = f"{prefix}attn_q.weight" in gguf.tensors
        if has_hybrid and has_full:
            raise UnsupportedModelError(f"layer {layer_idx} mixes qkv and split attention tensors")
        if has_hybrid:
            hybrid.append(layer_idx)
            required = HYBRID_LAYER_TENSORS
        elif has_full:
            full.append(layer_idx)
            required = FULL_ATTENTION_LAYER_TENSORS
        else:
            missing.append(f"{prefix}<attention tensors>")
            continue
        for suffix in required:
            name = prefix + suffix
            if name not in gguf.tensors:
                missing.append(name)
    for name in ("token_embd.weight", "output.weight", "output_norm.weight"):
        if name not in gguf.tensors:
            missing.append(name)
    return QwenTensorCoverage(
        hybrid_layers=tuple(hybrid),
        full_attention_layers=tuple(full),
        missing_tensors=tuple(missing),
    )


def validate_qwen_tensor_dimensions(gguf: GGUFFile, metadata: ModelMetadata) -> None:
    coverage = qwen_tensor_coverage(gguf, metadata)
    if not coverage.complete:
        raise UnsupportedModelError("missing Qwen tensors: " + ", ".join(coverage.missing_tensors))
    _require_dims(gguf.tensors["token_embd.weight"], (metadata.hidden_size, metadata.vocab_size))
    _require_dims(gguf.tensors["output.weight"], (metadata.hidden_size, metadata.vocab_size))
    _require_dims(gguf.tensors["output_norm.weight"], (metadata.hidden_size,))
    for layer_idx in coverage.hybrid_layers:
        prefix = f"blk.{layer_idx}."
        _require_dims(gguf.tensors[prefix + "attn_norm.weight"], (metadata.hidden_size,))
        _require_dims(gguf.tensors[prefix + "attn_qkv.weight"], (metadata.hidden_size, 10240))
        _require_dims(gguf.tensors[prefix + "attn_gate.weight"], (metadata.hidden_size, 6144))
        _require_dims(gguf.tensors[prefix + "ffn_down.weight"], (metadata.intermediate_size, metadata.hidden_size))
        _require_dims(gguf.tensors[prefix + "ffn_gate.weight"], (metadata.hidden_size, metadata.intermediate_size))
        _require_dims(gguf.tensors[prefix + "ffn_up.weight"], (metadata.hidden_size, metadata.intermediate_size))
    for layer_idx in coverage.full_attention_layers:
        prefix = f"blk.{layer_idx}."
        _require_dims(gguf.tensors[prefix + "attn_q.weight"], (metadata.hidden_size, 12288))
        _require_dims(gguf.tensors[prefix + "attn_k.weight"], (metadata.hidden_size, metadata.num_kv_heads * metadata.head_dim))
        _require_dims(gguf.tensors[prefix + "attn_v.weight"], (metadata.hidden_size, metadata.num_kv_heads * metadata.head_dim))
        _require_dims(gguf.tensors[prefix + "attn_output.weight"], (6144, metadata.hidden_size))


def qwen_decoder_layer_count(gguf: GGUFFile, metadata: ModelMetadata) -> int:
    """Main-stack layer count; trailing nextn (MTP) blocks are excluded from normal decode."""
    nextn = int(gguf.metadata_value(f"{metadata.architecture.value}.nextn_predict_layers", 0) or 0)
    count = metadata.num_hidden_layers - nextn
    if count <= 0:
        raise UnsupportedModelError("qwen35 model has no decoder layers outside nextn_predict_layers")
    return count


def qwen_layer_schedule(gguf: GGUFFile, metadata: ModelMetadata) -> tuple[QwenLayerKind, ...]:
    """Per-layer block kind for the decoder stack, cross-checked against the GGUF tensors."""
    num_layers = qwen_decoder_layer_count(gguf, metadata)
    coverage = qwen_tensor_coverage(gguf, metadata)
    present = {idx: QwenLayerKind.LINEAR_ATTENTION for idx in coverage.hybrid_layers}
    present.update({idx: QwenLayerKind.FULL_ATTENTION for idx in coverage.full_attention_layers})
    missing_layers = [idx for idx in range(num_layers) if idx not in present]
    if missing_layers:
        raise UnsupportedModelError(f"qwen35 layers without attention tensors: {missing_layers}")
    from_tensors = tuple(present[idx] for idx in range(num_layers))
    interval = gguf.metadata_value(f"{metadata.architecture.value}.full_attention_interval")
    if interval is None:
        return from_tensors
    expected = qwen_layer_kinds_from_interval(num_layers, int(interval))
    for layer_idx, (want, have) in enumerate(zip(expected, from_tensors)):
        if want is not have:
            raise UnsupportedModelError(
                f"qwen35 layer {layer_idx} tensors describe a {have.value} block but "
                f"full_attention_interval={interval} requires {want.value}"
            )
    return expected


def map_qwen_tensor_name(name: str) -> str:
    if name in {"token_embd.weight", "output.weight", "output_norm.weight"}:
        return {
            "token_embd.weight": "embed_tokens.weight",
            "output.weight": "lm_head.weight",
            "output_norm.weight": "norm.weight",
        }[name]
    if name.startswith("blk."):
        parts = name.split(".")
        if len(parts) >= 4:
            return f"layers.{parts[1]}.{'.'.join(parts[2:])}"
    return name


def load_qwen_token_embeddings(gguf: GGUFFile, token_ids: list[int]) -> list[list[float]]:
    tensor = gguf.tensors.get("token_embd.weight")
    if tensor is None:
        raise UnsupportedModelError("missing Qwen token embedding tensor")
    if len(tensor.dimensions) != 2:
        raise UnsupportedModelError("Qwen token embedding tensor must be 2-D")
    hidden_size, vocab_size = tensor.dimensions
    if not token_ids:
        return []
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size:
            raise UnsupportedModelError(f"token id out of range for embeddings: {token_id}")

    file_obj, mm, view = gguf.mmap_tensor(tensor.name)
    try:
        return [
            _dequant_embedding_row(tensor, view, token_id, hidden_size)
            for token_id in token_ids
        ]
    finally:
        view.release()
        mm.close()
        file_obj.close()


def load_qwen_lm_head_rows(gguf: GGUFFile, token_ids: list[int]) -> list[list[float]]:
    tensor = gguf.tensors.get("output.weight")
    if tensor is None:
        raise UnsupportedModelError("missing Qwen output weight tensor")
    if len(tensor.dimensions) != 2:
        raise UnsupportedModelError("Qwen output weight tensor must be 2-D")
    hidden_size, vocab_size = tensor.dimensions
    if not token_ids:
        return []
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size:
            raise UnsupportedModelError(f"token id out of range for LM head: {token_id}")

    file_obj, mm, view = gguf.mmap_tensor(tensor.name)
    try:
        return [
            _dequant_embedding_row(tensor, view, token_id, hidden_size)
            for token_id in token_ids
        ]
    finally:
        view.release()
        mm.close()
        file_obj.close()


def qwen_lm_head_logits_for_tokens(
    gguf: GGUFFile,
    hidden: list[float],
    token_ids: list[int],
    *,
    norm_weight: list[float] | None = None,
    eps: float = 1e-6,
) -> list[float]:
    from vinf.reference_ops import matvec_reference, rmsnorm_reference

    rows = load_qwen_lm_head_rows(gguf, token_ids)
    if not rows:
        return []
    if norm_weight is not None:
        hidden = rmsnorm_reference(hidden, norm_weight, eps)
    weights = [value for row in rows for value in row]
    return matvec_reference(hidden, weights, len(rows), len(hidden))


def load_qwen_full_attention_layer_weights(
    gguf: GGUFFile, metadata: ModelMetadata, layer_idx: int
) -> QwenLayerWeights:
    coverage = qwen_tensor_coverage(gguf, metadata)
    if layer_idx in coverage.hybrid_layers:
        raise UnsupportedModelError(
            f"qwen35 hybrid SSM layer {layer_idx} cannot be loaded as a full-attention layer"
        )
    if layer_idx not in coverage.full_attention_layers:
        raise UnsupportedModelError(f"qwen35 layer {layer_idx} is not a full-attention layer")
    prefix = f"blk.{layer_idx}."
    return QwenLayerWeights(
        q_norm=_load_optional(gguf, prefix + "attn_q_norm.weight"),
        k_norm=_load_optional(gguf, prefix + "attn_k_norm.weight"),
        attn_norm=load_dequantized_qwen_tensor(gguf, prefix + "attn_norm.weight"),
        attention=QwenAttentionWeights(
            q_proj=load_dequantized_qwen_tensor(gguf, prefix + "attn_q.weight"),
            k_proj=load_dequantized_qwen_tensor(gguf, prefix + "attn_k.weight"),
            v_proj=load_dequantized_qwen_tensor(gguf, prefix + "attn_v.weight"),
            o_proj=load_dequantized_qwen_tensor(gguf, prefix + "attn_output.weight"),
        ),
        post_attention_norm=load_dequantized_qwen_tensor(
            gguf, prefix + "post_attention_norm.weight"
        ),
        mlp=QwenMLPWeights(
            gate_proj=load_dequantized_qwen_tensor(gguf, prefix + "ffn_gate.weight"),
            up_proj=load_dequantized_qwen_tensor(gguf, prefix + "ffn_up.weight"),
            down_proj=load_dequantized_qwen_tensor(gguf, prefix + "ffn_down.weight"),
        ),
    )


MTP_TENSORS = ("nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight")


def qwen_mtp_layer_index(gguf: GGUFFile, metadata: ModelMetadata) -> int:
    """Index of the first nextn/MTP block (right after the decoder stack)."""
    idx = qwen_decoder_layer_count(gguf, metadata)
    if idx >= metadata.num_hidden_layers:
        raise UnsupportedModelError("model has no nextn/MTP block (nextn_predict_layers = 0)")
    missing = [f"blk.{idx}.{name}" for name in MTP_TENSORS if f"blk.{idx}.{name}" not in gguf.tensors]
    if missing:
        raise UnsupportedModelError("MTP block is missing tensors: " + ", ".join(missing))
    return idx


def load_qwen_mtp_weights(gguf: GGUFFile, metadata: ModelMetadata) -> QwenMtpWeights:
    idx = qwen_mtp_layer_index(gguf, metadata)
    prefix = f"blk.{idx}."
    return QwenMtpWeights(
        enorm=load_dequantized_qwen_tensor(gguf, prefix + "nextn.enorm.weight"),
        hnorm=load_dequantized_qwen_tensor(gguf, prefix + "nextn.hnorm.weight"),
        eh_proj=load_dequantized_qwen_tensor(gguf, prefix + "nextn.eh_proj.weight"),
        shared_head_norm=load_dequantized_qwen_tensor(gguf, prefix + "nextn.shared_head_norm.weight"),
        block=load_qwen_full_attention_layer_weights(gguf, metadata, idx),
    )


def load_qwen_linear_attention_layer_weights(
    gguf: GGUFFile,
    metadata: ModelMetadata,
    layer_idx: int,
    config: QwenLinearAttentionConfig | None = None,
) -> QwenLinearAttentionWeights:
    coverage = qwen_tensor_coverage(gguf, metadata)
    if layer_idx in coverage.full_attention_layers:
        raise UnsupportedModelError(
            f"qwen35 full-attention layer {layer_idx} cannot be loaded as a linear-attention layer"
        )
    if layer_idx not in coverage.hybrid_layers:
        raise UnsupportedModelError(f"qwen35 layer {layer_idx} is not a linear-attention layer")
    if config is None:
        config = qwen_linear_attention_config_from_metadata(gguf, metadata)
    prefix = f"blk.{layer_idx}."
    hidden = metadata.hidden_size
    expected = {
        "attn_qkv.weight": (hidden, config.conv_dim),
        "attn_gate.weight": (hidden, config.value_dim),
        "ssm_beta.weight": (hidden, config.value_heads),
        "ssm_alpha.weight": (hidden, config.value_heads),
        "ssm_a": (config.value_heads,),
        "ssm_dt.bias": (config.value_heads,),
        "ssm_conv1d.weight": (config.conv_kernel, config.conv_dim),
        "ssm_norm.weight": (config.value_head_dim,),
        "ssm_out.weight": (config.value_dim, hidden),
    }
    for suffix, dims in expected.items():
        _require_dims(gguf.tensors[prefix + suffix], dims)
    return QwenLinearAttentionWeights(
        attn_norm=load_dequantized_qwen_tensor(gguf, prefix + "attn_norm.weight"),
        qkv_proj=load_dequantized_qwen_tensor(gguf, prefix + "attn_qkv.weight"),
        gate_proj=load_dequantized_qwen_tensor(gguf, prefix + "attn_gate.weight"),
        beta_proj=load_dequantized_qwen_tensor(gguf, prefix + "ssm_beta.weight"),
        alpha_proj=load_dequantized_qwen_tensor(gguf, prefix + "ssm_alpha.weight"),
        ssm_a=load_dequantized_qwen_tensor(gguf, prefix + "ssm_a"),
        dt_bias=load_dequantized_qwen_tensor(gguf, prefix + "ssm_dt.bias"),
        conv1d=load_dequantized_qwen_tensor(gguf, prefix + "ssm_conv1d.weight"),
        ssm_norm=load_dequantized_qwen_tensor(gguf, prefix + "ssm_norm.weight"),
        out_proj=load_dequantized_qwen_tensor(gguf, prefix + "ssm_out.weight"),
        post_attention_norm=load_dequantized_qwen_tensor(gguf, prefix + "post_attention_norm.weight"),
        mlp=QwenMLPWeights(
            gate_proj=load_dequantized_qwen_tensor(gguf, prefix + "ffn_gate.weight"),
            up_proj=load_dequantized_qwen_tensor(gguf, prefix + "ffn_up.weight"),
            down_proj=load_dequantized_qwen_tensor(gguf, prefix + "ffn_down.weight"),
        ),
    )


def _load_optional(gguf: GGUFFile, name: str) -> list[float] | None:
    return load_dequantized_qwen_tensor(gguf, name) if name in gguf.tensors else None


def load_dequantized_qwen_tensor(gguf: GGUFFile, name: str) -> list[float]:
    tensor = gguf.tensors[name]
    file_obj, mm, view = gguf.mmap_tensor(name)
    try:
        return dequantize_tensor(tensor, view).values
    finally:
        view.release()
        mm.close()
        file_obj.close()


def _require_dims(tensor: GGUFTensorInfo, expected: tuple[int, ...]) -> None:
    if tensor.dimensions != expected:
        raise UnsupportedModelError(
            f"{tensor.name} expected dimensions {expected}, got {tensor.dimensions}"
        )


def _dequant_embedding_row(
    tensor: GGUFTensorInfo, tensor_data: memoryview, token_id: int, hidden_size: int
) -> list[float]:
    start = token_id * hidden_size
    end = start + hidden_size
    block_size = tensor.block_size
    type_size = tensor.type_size
    first_block = start // block_size
    last_block = (end + block_size - 1) // block_size
    block_numel = (last_block - first_block) * block_size
    byte_start = first_block * type_size
    byte_end = last_block * type_size
    row_info = GGUFTensorInfo(
        name=tensor.name,
        dimensions=(block_numel,),
        tensor_type=tensor.tensor_type,
        relative_offset=tensor.relative_offset + byte_start,
        absolute_offset=tensor.absolute_offset + byte_start,
    )
    raw = tensor_data[byte_start:byte_end]
    try:
        values = dequantize_tensor(row_info, raw).values
    finally:
        raw.release()
    offset = start - first_block * block_size
    return values[offset : offset + hidden_size]
