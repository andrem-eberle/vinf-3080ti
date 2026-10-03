from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelArchitecture(StrEnum):
    LLAMA = "llama"
    QWEN35 = "qwen35"


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    architecture: ModelArchitecture
    num_hidden_layers: int
    num_attention_heads: int
    num_kv_heads: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    dtype: str
    tokenizer_id: str | None = None

    def __post_init__(self) -> None:
        positive_fields = {
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.dtype not in {"fp16", "fp32", "quantized"}:
            raise ValueError("dtype must be fp16, fp32, or quantized")
