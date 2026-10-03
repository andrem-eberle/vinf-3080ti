from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Protocol

from vinf.config import GenerationConfig


@dataclass(frozen=True, slots=True)
class SampleResult:
    token_id: int
    probability: float | None = None


class Sampler(Protocol):
    def sample(self, logits_or_probabilities: object, config: GenerationConfig) -> SampleResult:
        ...


class GreedySampler:
    def sample(self, logits_or_probabilities: object, config: GenerationConfig) -> SampleResult:
        _ = config
        if isinstance(logits_or_probabilities, (list, tuple)):
            if not logits_or_probabilities:
                raise ValueError("cannot sample from an empty sequence")
            token_id = max(
                range(len(logits_or_probabilities)),
                key=lambda idx: float(logits_or_probabilities[idx]),
            )
            return SampleResult(token_id=token_id)
        raise NotImplementedError("non-sequence sampling is not implemented yet")


class CPUSampler:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def sample(self, logits_or_probabilities: object, config: GenerationConfig) -> SampleResult:
        if not isinstance(logits_or_probabilities, (list, tuple)):
            raise NotImplementedError("non-sequence sampling is not implemented yet")
        probabilities = logits_to_probabilities(
            [float(x) for x in logits_or_probabilities], config
        )
        draw = self.rng.random()
        total = 0.0
        for token_id, probability in enumerate(probabilities):
            total += probability
            if draw <= total:
                return SampleResult(token_id=token_id, probability=probability)
        token_id = len(probabilities) - 1
        return SampleResult(token_id=token_id, probability=probabilities[token_id])


def logits_to_probabilities(
    logits: list[float], config: GenerationConfig
) -> list[float]:
    if not logits:
        raise ValueError("cannot process empty logits")
    scaled = [value / config.temperature for value in logits]
    filtered = apply_top_k(scaled, config.top_k)
    filtered = apply_top_p(filtered, config.top_p)
    return softmax(filtered)


def apply_top_k(logits: list[float], top_k: int | None) -> list[float]:
    if top_k is None or top_k >= len(logits):
        return list(logits)
    indexed = sorted(enumerate(logits), key=lambda item: item[1], reverse=True)
    keep = {idx for idx, _ in indexed[:top_k]}
    return [value if idx in keep else -math.inf for idx, value in enumerate(logits)]


def apply_top_p(logits: list[float], top_p: float | None) -> list[float]:
    if top_p is None or top_p >= 1:
        return list(logits)
    base_probs = softmax(logits)
    ordered = sorted(enumerate(base_probs), key=lambda item: item[1], reverse=True)
    keep: set[int] = set()
    cumulative = 0.0
    for idx, probability in ordered:
        keep.add(idx)
        cumulative += probability
        if cumulative >= top_p:
            break
    return [value if idx in keep else -math.inf for idx, value in enumerate(logits)]


def softmax(logits: list[float]) -> list[float]:
    finite_values = [value for value in logits if math.isfinite(value)]
    if not finite_values:
        raise ValueError("at least one logit must be finite")
    max_value = max(finite_values)
    exps = [math.exp(value - max_value) if math.isfinite(value) else 0.0 for value in logits]
    denom = sum(exps)
    if denom == 0:
        raise ValueError("softmax denominator is zero")
    return [value / denom for value in exps]
