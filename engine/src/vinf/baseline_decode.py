from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from vinf.models.metadata import ModelMetadata
from vinf.one_layer import OneLayerConfig, OneLayerResult, OneLayerWeights
from vinf.reference_ops import lm_head_reference
from vinf.runtime.scheduler import Schedule, build_baseline_decode_schedule


MAJOR_TIMING_SLOTS = (
    "schedule",
    "rms_qkv_rope",
    "attention",
    "o_proj",
    "mlp",
    "lm_head",
)


@dataclass(frozen=True, slots=True)
class BaselineTargetWeights:
    layers: tuple[OneLayerWeights, ...]
    final_norm: list[float]
    lm_head: list[float]


@dataclass(frozen=True, slots=True)
class BaselineDecodeConfig:
    metadata: ModelMetadata
    eps: float = 1e-6
    num_sms: int = 80
    block_size: int = 16


@dataclass(frozen=True, slots=True)
class BaselineDecodeResult:
    hidden: list[float]
    logits: list[float]
    schedule: Schedule
    stops: dict[str, list[float]]
    timings: dict[str, float]


def baseline_decode_reference(
    hidden: list[float],
    weights: BaselineTargetWeights,
    config: BaselineDecodeConfig,
    *,
    position: int,
    rope_cos: list[float],
    rope_sin: list[float],
) -> BaselineDecodeResult:
    return _baseline_decode(
        hidden,
        weights,
        config,
        position=position,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        layer_runner=_one_layer_reference,
        lm_head_runner=lm_head_reference,
    )


def baseline_decode_cuda_composed(
    hidden: list[float],
    weights: BaselineTargetWeights,
    config: BaselineDecodeConfig,
    *,
    position: int,
    rope_cos: list[float],
    rope_sin: list[float],
) -> BaselineDecodeResult:
    from vinf.cuda.lm_head import lm_head_cuda
    from vinf.one_layer import one_layer_cuda_composed

    return _baseline_decode(
        hidden,
        weights,
        config,
        position=position,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        layer_runner=one_layer_cuda_composed,
        lm_head_runner=lm_head_cuda,
    )


def _baseline_decode(
    hidden: list[float],
    weights: BaselineTargetWeights,
    config: BaselineDecodeConfig,
    *,
    position: int,
    rope_cos: list[float],
    rope_sin: list[float],
    layer_runner: Callable[..., OneLayerResult],
    lm_head_runner: Callable[..., list[float]],
) -> BaselineDecodeResult:
    _validate_decode_inputs(hidden, weights, config, rope_cos, rope_sin)
    timings = {slot: 0.0 for slot in MAJOR_TIMING_SLOTS}

    start = perf_counter()
    schedule = build_baseline_decode_schedule(
        config.metadata,
        num_sms=config.num_sms,
        position=position,
        block_size=config.block_size,
    )
    timings["schedule"] = perf_counter() - start

    h = list(hidden)
    stops: dict[str, list[float]] = {}
    layer_config = OneLayerConfig(
        hidden_size=config.metadata.hidden_size,
        head_dim=config.metadata.head_dim,
        num_kv_heads=config.metadata.num_kv_heads,
        max_seq=config.metadata.max_position_embeddings,
        eps=config.eps,
    )

    for layer_idx, layer_weights in enumerate(weights.layers):
        layer_start = perf_counter()
        layer = layer_runner(
            h,
            layer_weights,
            layer_config,
            position=position,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        elapsed = perf_counter() - layer_start
        timings["rms_qkv_rope"] += elapsed
        timings["attention"] += elapsed
        timings["o_proj"] += elapsed
        timings["mlp"] += elapsed
        h = layer.hidden
        for name, values in layer.stops.items():
            stops[f"layer_{layer_idx}.{name}"] = values

    start = perf_counter()
    logits = lm_head_runner(
        h,
        weights.final_norm,
        weights.lm_head,
        config.metadata.hidden_size,
        config.metadata.vocab_size,
        config.eps,
    )
    timings["lm_head"] = perf_counter() - start
    stops["final_hidden"] = h
    stops["logits"] = logits
    return BaselineDecodeResult(
        hidden=h,
        logits=logits,
        schedule=schedule,
        stops=stops,
        timings=timings,
    )


def _one_layer_reference(
    hidden: list[float],
    weights: OneLayerWeights,
    config: OneLayerConfig,
    *,
    position: int,
    rope_cos: list[float],
    rope_sin: list[float],
) -> OneLayerResult:
    from vinf.one_layer import one_layer_reference

    return one_layer_reference(
        hidden,
        weights,
        config,
        position=position,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )


def _validate_decode_inputs(
    hidden: list[float],
    weights: BaselineTargetWeights,
    config: BaselineDecodeConfig,
    rope_cos: list[float],
    rope_sin: list[float],
) -> None:
    metadata = config.metadata
    if len(hidden) != metadata.hidden_size:
        raise ValueError("hidden length must equal metadata.hidden_size")
    if len(weights.layers) != metadata.num_hidden_layers:
        raise ValueError("weights.layers length must equal metadata.num_hidden_layers")
    if len(weights.final_norm) != metadata.hidden_size:
        raise ValueError("final_norm length must equal metadata.hidden_size")
    if len(weights.lm_head) != metadata.vocab_size * metadata.hidden_size:
        raise ValueError("lm_head length must equal vocab_size * hidden_size")
    if len(rope_cos) != metadata.head_dim or len(rope_sin) != metadata.head_dim:
        raise ValueError("rope_cos and rope_sin lengths must equal metadata.head_dim")
