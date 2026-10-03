from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class EngineMetrics:
    generated_tokens: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    verification_calls: int = 0
    speculative_tokens_proposed: int = 0
    speculative_tokens_accepted: int = 0
    rejection_position_histogram: dict[int, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def speculative_acceptance_rate(self) -> float:
        if self.speculative_tokens_proposed == 0:
            return 0.0
        return self.speculative_tokens_accepted / self.speculative_tokens_proposed

    @property
    def tokens_per_target_call(self) -> float:
        if self.target_calls == 0:
            return 0.0
        return self.generated_tokens / self.target_calls

    def add_time(self, name: str, seconds: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def record_rejection_position(self, position: int) -> None:
        self.rejection_position_histogram[position] = (
            self.rejection_position_histogram.get(position, 0) + 1
        )


class Timer:
    def __init__(self, metrics: EngineMetrics, name: str) -> None:
        self.metrics = metrics
        self.name = name
        self.started_at = 0.0

    def __enter__(self) -> "Timer":
        self.started_at = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.metrics.add_time(self.name, perf_counter() - self.started_at)
