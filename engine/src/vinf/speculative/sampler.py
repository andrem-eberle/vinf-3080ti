from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SpeculativeDecision:
    accepted_count: int
    emitted_tokens: tuple[int, ...]
    rejected: bool


class SpeculativeSampler(Protocol):
    def accept_or_correct(
        self,
        draft_tokens: tuple[int, ...],
        draft_probability_rows: object,
        target_probability_rows: object,
    ) -> SpeculativeDecision:
        ...


def acceptance_probability(
    target_probability: float,
    draft_probability: float,
    *,
    denominator_floor: float = 1e-30,
) -> float:
    if target_probability < 0 or draft_probability < 0:
        raise ValueError("probabilities must be non-negative")
    if draft_probability == 0:
        return 1.0 if target_probability > 0 else 0.0
    return min(1.0, target_probability / max(draft_probability, denominator_floor))


def correction_distribution(
    target_probabilities: list[float],
    draft_probabilities: list[float],
) -> list[float]:
    _validate_probability_row(target_probabilities)
    _validate_probability_row(draft_probabilities)
    if len(target_probabilities) != len(draft_probabilities):
        raise ValueError("target and draft probability rows must have the same length")
    residual = [max(0.0, p - q) for p, q in zip(target_probabilities, draft_probabilities)]
    total = sum(residual)
    if total == 0:
        return list(target_probabilities)
    return [value / total for value in residual]


def expected_tokens_per_speculative_iteration(alpha: float, gamma: int) -> float:
    _validate_alpha_gamma(alpha, gamma)
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def expected_speculative_speedup(alpha: float, gamma: int, draft_cost_ratio: float) -> float:
    if draft_cost_ratio < 0:
        raise ValueError("draft_cost_ratio must be non-negative")
    emitted = expected_tokens_per_speculative_iteration(alpha, gamma)
    return emitted / (gamma * draft_cost_ratio + 1.0)


class CPUSpeculativeSampler:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def accept_or_correct(
        self,
        draft_tokens: tuple[int, ...],
        draft_probability_rows: object,
        target_probability_rows: object,
    ) -> SpeculativeDecision:
        q_rows = _coerce_probability_rows(draft_probability_rows, len(draft_tokens))
        p_rows = _coerce_probability_rows(target_probability_rows, len(draft_tokens) + 1)
        accepted = 0
        rejected = False
        for idx, token in enumerate(draft_tokens):
            _validate_token(token, len(q_rows[idx]))
            accept_prob = acceptance_probability(p_rows[idx][token], q_rows[idx][token])
            if self.rng.random() <= accept_prob:
                accepted += 1
            else:
                rejected = True
                break
        if accepted == len(draft_tokens):
            next_token = self._sample(p_rows[len(draft_tokens)])
        else:
            correction = correction_distribution(p_rows[accepted], q_rows[accepted])
            next_token = self._sample(correction)
        return SpeculativeDecision(
            accepted_count=accepted,
            emitted_tokens=(*draft_tokens[:accepted], next_token),
            rejected=rejected,
        )

    def _sample(self, probabilities: list[float]) -> int:
        _validate_probability_row(probabilities)
        draw = self.rng.random()
        total = 0.0
        for token_id, probability in enumerate(probabilities):
            total += probability
            if draw <= total:
                return token_id
        return len(probabilities) - 1


def _coerce_probability_rows(rows: object, expected_rows: int) -> list[list[float]]:
    if not isinstance(rows, (list, tuple)):
        raise TypeError("probability rows must be a sequence")
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} probability rows")
    out = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            raise TypeError("probability row must be a sequence")
        values = [float(value) for value in row]
        _validate_probability_row(values)
        out.append(values)
    return out


def _validate_probability_row(probabilities: list[float]) -> None:
    if not probabilities:
        raise ValueError("probability row must not be empty")
    if any(value < 0 for value in probabilities):
        raise ValueError("probabilities must be non-negative")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("probability row must have positive mass")


def _validate_token(token: int, vocab_size: int) -> None:
    if token < 0 or token >= vocab_size:
        raise ValueError("draft token outside probability row")


def _validate_alpha_gamma(alpha: float, gamma: int) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
