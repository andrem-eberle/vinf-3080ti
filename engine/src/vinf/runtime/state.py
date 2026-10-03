from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from vinf.config import GenerationConfig
from vinf.models.metadata import ModelMetadata


class DecodeMode(StrEnum):
    BASELINE = "baseline"
    SPECULATIVE = "speculative"
    REFERENCE = "reference"


@dataclass(slots=True)
class RuntimeState:
    model: ModelMetadata | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    mode: DecodeMode = DecodeMode.BASELINE
    prompt_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    position: int = 0
    stopped: bool = False
    stop_reason: str | None = None

    @property
    def all_tokens(self) -> list[int]:
        return [*self.prompt_tokens, *self.output_tokens]

    def append_tokens(self, tokens: list[int]) -> None:
        self.output_tokens.extend(tokens)
        self.position += len(tokens)
        if len(self.output_tokens) >= self.generation.max_new_tokens:
            self.stopped = True
            self.stop_reason = "max_new_tokens"


@dataclass(frozen=True, slots=True)
class DecodeResult:
    tokens: tuple[int, ...]
    text: str
    stop_reason: str | None

