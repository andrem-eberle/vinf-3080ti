from __future__ import annotations

from collections.abc import Iterator

from vinf.config import EngineConfig, GenerationConfig
from vinf.engine import InferenceEngine


def generate(
    prompt: str,
    *,
    engine_config: EngineConfig | None = None,
    generation_config: GenerationConfig | None = None,
) -> str:
    engine = InferenceEngine(engine_config)
    return engine.generate(prompt, generation_config)


def stream_generate(
    prompt: str,
    *,
    engine_config: EngineConfig | None = None,
    generation_config: GenerationConfig | None = None,
) -> Iterator[str]:
    engine = InferenceEngine(engine_config)
    yield from engine.stream_generate(prompt, generation_config)

