from __future__ import annotations

from dataclasses import dataclass, field

from vinf.errors import ConfigurationError


@dataclass(slots=True)
class SpeculativeConfig:
    enabled: bool = False
    gamma: int = 2
    draft_model_path: str | None = None

    def __post_init__(self) -> None:
        if self.gamma < 0:
            raise ConfigurationError("gamma must be non-negative")
        if self.enabled and self.gamma == 0:
            raise ConfigurationError("enabled speculative decoding requires gamma > 0")


@dataclass(slots=True)
class EngineConfig:
    device: str = "cuda:0"
    target_gpu: str = "rtx_3080_ti"
    dtype: str = "fp16"
    max_seq_len: int = 2048
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)

    def __post_init__(self) -> None:
        if self.max_seq_len <= 0:
            raise ConfigurationError("max_seq_len must be positive")
        if self.dtype not in {"fp16", "fp32", "quantized"}:
            raise ConfigurationError("dtype must be fp16, fp32, or quantized")


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ConfigurationError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ConfigurationError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ConfigurationError("top_k must be positive when set")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ConfigurationError("top_p must be in (0, 1] when set")
