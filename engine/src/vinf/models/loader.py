from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vinf.config import EngineConfig
from vinf.errors import UnsupportedModelError
from vinf.memory import MemoryPlanner
from vinf.models.layouts import expected_reference_weight_specs
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.models.weights import ModelWeights, WeightTensor, infer_nested_shape


FIRST_SUPPORTED_TARGET = "reference_transition_lm"
FIRST_SUPPORTED_DTYPE = "fp16"
FIRST_SUPPORTED_SHAPE = {
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_kv_heads": 1,
    "hidden_size": 8,
    "intermediate_size": 16,
    "head_dim": 4,
    "vocab_size": 8,
    "max_position_embeddings": 32,
}


class LoadedModel:
    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        weights: ModelWeights,
        tokenizer_id: str | None,
        memory_fits: bool,
    ) -> None:
        self.metadata = metadata
        self.weights = weights
        self.tokenizer_id = tokenizer_id
        self.memory_fits = memory_fits


def load_model(path: str | Path, config: EngineConfig) -> LoadedModel:
    data = _read_json(path)
    metadata = load_model_metadata_from_dict(data)
    _assert_supported_shape(metadata)

    weights = load_model_weights_from_dict(data)
    weights.validate(expected_reference_weight_specs(metadata))

    plan = MemoryPlanner(config).plan(metadata)
    if not plan.fits_gpu_budget:
        raise UnsupportedModelError("model memory plan does not fit available VRAM budget")

    return LoadedModel(
        metadata=metadata,
        weights=weights,
        tokenizer_id=metadata.tokenizer_id,
        memory_fits=True,
    )


def load_target_model(path: str | Path, config: EngineConfig) -> LoadedModel:
    return load_model(path, config)


def load_draft_model(path: str | Path, target: LoadedModel, config: EngineConfig) -> LoadedModel:
    draft = load_model(path, config)
    validate_tokenizer_compatibility(target.metadata, draft.metadata)
    return draft


def validate_tokenizer_compatibility(
    target: ModelMetadata, draft: ModelMetadata
) -> None:
    if target.tokenizer_id != draft.tokenizer_id:
        raise UnsupportedModelError(
            f"target/draft tokenizer mismatch: {target.tokenizer_id} != {draft.tokenizer_id}"
        )
    if target.vocab_size != draft.vocab_size:
        raise UnsupportedModelError("target/draft vocab_size mismatch")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise UnsupportedModelError("model file must contain a JSON object")
    return data


def load_model_metadata(path: str | Path) -> ModelMetadata:
    return load_model_metadata_from_dict(_read_json(path))


def load_model_metadata_from_dict(data: dict[str, Any]) -> ModelMetadata:
    raw = data.get("metadata")
    if not isinstance(raw, dict):
        raise UnsupportedModelError("model file missing metadata object")
    try:
        architecture = ModelArchitecture(raw["architecture"])
        return ModelMetadata(
            architecture=architecture,
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_kv_heads=int(raw["num_kv_heads"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            head_dim=int(raw["head_dim"]),
            vocab_size=int(raw["vocab_size"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            dtype=str(raw["dtype"]),
            tokenizer_id=raw.get("tokenizer_id"),
        )
    except KeyError as exc:
        raise UnsupportedModelError(f"metadata missing field: {exc.args[0]}") from exc
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc


def load_model_weights(path: str | Path) -> ModelWeights:
    return load_model_weights_from_dict(_read_json(path))


def load_model_weights_from_dict(data: dict[str, Any]) -> ModelWeights:
    raw = data.get("weights")
    if not isinstance(raw, dict):
        raise UnsupportedModelError("model file missing weights object")
    tensors = {}
    for name, values in raw.items():
        shape = infer_nested_shape(values)
        tensors[name] = WeightTensor(
            name=name,
            values=values,
            shape=shape,
            dtype="fp32",
        )
    return ModelWeights(tensors)


def _assert_supported_shape(metadata: ModelMetadata) -> None:
    if metadata.architecture is not ModelArchitecture.LLAMA:
        raise UnsupportedModelError("only llama-style metadata is supported")
    if metadata.dtype != FIRST_SUPPORTED_DTYPE:
        raise UnsupportedModelError(
            f"first supported target dtype is {FIRST_SUPPORTED_DTYPE}"
        )
    for field, expected in FIRST_SUPPORTED_SHAPE.items():
        actual = getattr(metadata, field)
        if actual != expected:
            raise UnsupportedModelError(
                f"first supported target requires {field}={expected}, got {actual}"
            )
