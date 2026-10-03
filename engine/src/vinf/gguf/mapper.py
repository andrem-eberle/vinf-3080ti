from __future__ import annotations

from vinf.errors import UnsupportedModelError
from vinf.gguf.parser import GGUFFile, GGUFTensorType
from vinf.models.metadata import ModelArchitecture, ModelMetadata


GGUF_TO_INTERNAL_NAMES = {
    "token_embd.weight": "embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "norm.weight",
}


def metadata_from_gguf(gguf: GGUFFile) -> ModelMetadata:
    architecture = str(gguf.metadata_value("general.architecture", ""))
    if architecture not in {"llama", "qwen2", "qwen3", "qwen35", "qwen"}:
        raise UnsupportedModelError(f"unsupported GGUF architecture: {architecture}")

    prefix = architecture
    if architecture.startswith("qwen"):
        prefix = architecture

    block_count = _required_int(gguf, f"{prefix}.block_count")
    head_count = _required_int(gguf, f"{prefix}.attention.head_count")
    kv_heads = int(
        gguf.metadata_value(f"{prefix}.attention.head_count_kv", head_count)
    )
    embedding_length = _required_int(gguf, f"{prefix}.embedding_length")
    feed_forward_length = _required_int(gguf, f"{prefix}.feed_forward_length")
    context_length = int(gguf.metadata_value(f"{prefix}.context_length", 0)) or int(
        gguf.metadata_value("general.context_length", 0)
    )
    vocab_size = int(gguf.metadata_value("tokenizer.ggml.tokens", []).__len__())
    if vocab_size == 0:
        token_embd = gguf.tensors.get("token_embd.weight")
        if token_embd is None or len(token_embd.dimensions) < 2:
            raise UnsupportedModelError("cannot infer GGUF vocab_size")
        vocab_size = token_embd.dimensions[1]

    head_dim = int(
        gguf.metadata_value(f"{prefix}.attention.key_length", 0)
    ) or embedding_length // head_count

    dtype = _dtype_from_tensors(gguf)
    internal_architecture = (
        ModelArchitecture.QWEN35 if architecture == "qwen35" else ModelArchitecture.LLAMA
    )
    return ModelMetadata(
        architecture=internal_architecture,
        num_hidden_layers=block_count,
        num_attention_heads=head_count,
        num_kv_heads=kv_heads,
        hidden_size=embedding_length,
        intermediate_size=feed_forward_length,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=context_length or 1,
        dtype=dtype,
        tokenizer_id=str(gguf.metadata_value("tokenizer.ggml.model", architecture)),
    )


def map_tensor_name(gguf_name: str) -> str:
    if gguf_name in GGUF_TO_INTERNAL_NAMES:
        return GGUF_TO_INTERNAL_NAMES[gguf_name]
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".")
        if len(parts) >= 4 and parts[0] == "blk":
            layer = parts[1]
            rest = ".".join(parts[2:])
            return f"layers.{layer}.{rest}"
    return gguf_name


def _required_int(gguf: GGUFFile, key: str) -> int:
    value = gguf.metadata_value(key)
    if value is None:
        raise UnsupportedModelError(f"GGUF metadata missing required key: {key}")
    return int(value)


def _dtype_from_tensors(gguf: GGUFFile) -> str:
    tensor_types = {tensor.tensor_type for tensor in gguf.tensors.values()}
    if tensor_types <= {GGUFTensorType.F16}:
        return "fp16"
    if tensor_types <= {GGUFTensorType.F32}:
        return "fp32"
    if tensor_types <= {GGUFTensorType.F16, GGUFTensorType.F32}:
        return "fp16"
    return "quantized"
