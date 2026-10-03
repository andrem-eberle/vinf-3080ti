from __future__ import annotations

from vinf.config import EngineConfig, GenerationConfig
from vinf.runtime.state import DecodeResult


class InferenceEngine:
    """Phase 0 engine shell.

    The real decode loop arrives in later phases. For now this class gives the
    package a stable construction point and keeps config validation centralized.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        _ = config or GenerationConfig()
        raise NotImplementedError("generation is not implemented in phase 0")

    def stream_generate(
        self, prompt: str, config: GenerationConfig | None = None
    ):  # type: ignore[no-untyped-def]
        _ = self.generate(prompt, config)
        yield from ()

    def generate_result(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> DecodeResult:
        text = self.generate(prompt, config)
        return DecodeResult(tokens=(), text=text, stop_reason=None)
