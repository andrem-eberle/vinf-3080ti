from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vinf.models.metadata import ModelMetadata


class WeightLayout(StrEnum):
    ROW_MAJOR = "row_major"
    VECTOR = "vector"


@dataclass(frozen=True, slots=True)
class WeightSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: WeightLayout
    required: bool = True


def expected_reference_weight_specs(model: ModelMetadata) -> tuple[WeightSpec, ...]:
    return (
        WeightSpec(
            name="transition_logits",
            shape=(model.vocab_size, model.vocab_size),
            dtype="fp32",
            layout=WeightLayout.ROW_MAJOR,
        ),
    )


def expected_llama_weight_names() -> tuple[str, ...]:
    return (
        "qkv_proj_weights",
        "attn_ln_weights",
        "o_proj_weights",
        "mlp_ln_weights",
        "up_proj_weights",
        "gate_proj_weights",
        "down_proj_weights",
        "lm_head_norm_weights",
        "lm_head_weights",
        "k_cache",
        "v_cache",
        "rope_cos",
        "rope_sin",
    )

