from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from vinf.config import EngineConfig
from vinf.errors import ConfigurationError
from vinf.models.metadata import ModelMetadata


class Residency(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    OPTIONAL_CPU = "optional_cpu"


class BufferRole(StrEnum):
    TARGET_WEIGHTS = "target_weights"
    DRAFT_WEIGHTS = "draft_weights"
    TARGET_KV = "target_kv"
    DRAFT_KV = "draft_kv"
    ACTIVATION = "activation"
    LOGITS = "logits"
    PROBABILITIES = "probabilities"
    INSTRUCTIONS = "instructions"
    TIMINGS = "timings"
    BARRIER = "barrier"


DTYPE_SIZES = {
    "fp16": 2,
    "fp32": 4,
    "quantized": 1,
    "int32": 4,
}


@dataclass(frozen=True, slots=True)
class BufferSpec:
    name: str
    role: BufferRole
    shape: tuple[int, ...]
    dtype: str
    residency: Residency = Residency.GPU
    required: bool = True
    handle: object | None = None

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            if dim <= 0:
                raise ConfigurationError(f"buffer {self.name} has invalid shape")
            total *= dim
        return total

    @property
    def nbytes(self) -> int:
        try:
            itemsize = DTYPE_SIZES[self.dtype]
        except KeyError as exc:
            raise ConfigurationError(f"unsupported dtype for {self.name}: {self.dtype}") from exc
        return self.numel * itemsize


@dataclass(slots=True)
class BufferRegistry:
    specs: dict[str, BufferSpec] = field(default_factory=dict)

    def add(self, spec: BufferSpec) -> None:
        if spec.name in self.specs:
            raise ConfigurationError(f"duplicate buffer spec: {spec.name}")
        self.specs[spec.name] = spec

    def get(self, name: str) -> BufferSpec:
        return self.specs[name]

    def by_role(self, role: BufferRole) -> list[BufferSpec]:
        return [spec for spec in self.specs.values() if spec.role == role]

    def bytes_by_residency(self, residency: Residency) -> int:
        return sum(spec.nbytes for spec in self.specs.values() if spec.residency == residency)

    @property
    def gpu_bytes(self) -> int:
        return self.bytes_by_residency(Residency.GPU)

    @property
    def cpu_bytes(self) -> int:
        return self.bytes_by_residency(Residency.CPU)


@dataclass(frozen=True, slots=True)
class ResidencyRules:
    target_weights: Residency = Residency.GPU
    target_kv: Residency = Residency.GPU
    activations: Residency = Residency.GPU
    logits: Residency = Residency.GPU
    instructions: Residency = Residency.GPU
    draft_weights: Residency = Residency.GPU
    draft_kv: Residency = Residency.GPU


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    registry: BufferRegistry
    available_vram_bytes: int
    reserved_vram_bytes: int

    @property
    def planned_gpu_bytes(self) -> int:
        return self.registry.gpu_bytes

    @property
    def fits_gpu_budget(self) -> bool:
        return self.planned_gpu_bytes <= self.available_vram_bytes - self.reserved_vram_bytes

    @property
    def remaining_gpu_bytes(self) -> int:
        return self.available_vram_bytes - self.reserved_vram_bytes - self.planned_gpu_bytes


class MemoryPlanner:
    RTX_3080_TI_VRAM_BYTES = 12 * 1024**3
    DEFAULT_RESERVED_BYTES = 512 * 1024**2

    def __init__(
        self,
        config: EngineConfig,
        *,
        available_vram_bytes: int | None = None,
        reserved_vram_bytes: int = DEFAULT_RESERVED_BYTES,
    ) -> None:
        self.config = config
        self.available_vram_bytes = available_vram_bytes or self.RTX_3080_TI_VRAM_BYTES
        self.reserved_vram_bytes = reserved_vram_bytes

    def plan(
        self,
        target: ModelMetadata,
        *,
        draft: ModelMetadata | None = None,
        max_gamma: int | None = None,
    ) -> MemoryPlan:
        max_gamma = max_gamma if max_gamma is not None else self.config.speculative.gamma
        registry = BufferRegistry()
        rules = ResidencyRules()

        registry.add(
            BufferSpec(
                name="target_weights",
                role=BufferRole.TARGET_WEIGHTS,
                shape=(self._estimate_transformer_weight_params(target),),
                dtype=target.dtype,
                residency=rules.target_weights,
            )
        )
        registry.add(
            BufferSpec(
                name="target_kv",
                role=BufferRole.TARGET_KV,
                shape=self._kv_shape(target, self.config.max_seq_len, max_gamma),
                dtype=target.dtype,
                residency=rules.target_kv,
            )
        )
        for spec in self.activation_buffer_plan(target, max_gamma=max_gamma):
            registry.add(spec)
        for spec in self.logits_probability_buffer_plan(target, max_gamma=max_gamma):
            registry.add(spec)

        registry.add(
            BufferSpec(
                name="instructions",
                role=BufferRole.INSTRUCTIONS,
                shape=(target.num_hidden_layers, 128, 32),
                dtype="int32",
                residency=rules.instructions,
            )
        )
        registry.add(
            BufferSpec(
                name="timings",
                role=BufferRole.TIMINGS,
                shape=(target.num_hidden_layers, 128, 128),
                dtype="int32",
                residency=rules.instructions,
                required=False,
            )
        )

        if draft is not None:
            registry.add(
                BufferSpec(
                    name="draft_weights",
                    role=BufferRole.DRAFT_WEIGHTS,
                    shape=(self._estimate_transformer_weight_params(draft),),
                    dtype=draft.dtype,
                    residency=rules.draft_weights,
                    required=self.config.speculative.enabled,
                )
            )
            registry.add(
                BufferSpec(
                    name="draft_kv",
                    role=BufferRole.DRAFT_KV,
                    shape=self._kv_shape(draft, self.config.max_seq_len, max_gamma),
                    dtype=draft.dtype,
                    residency=rules.draft_kv,
                    required=self.config.speculative.enabled,
                )
            )

        return MemoryPlan(
            registry=registry,
            available_vram_bytes=self.available_vram_bytes,
            reserved_vram_bytes=self.reserved_vram_bytes,
        )

    def activation_buffer_plan(
        self, model: ModelMetadata, *, max_gamma: int
    ) -> list[BufferSpec]:
        positions = max(1, max_gamma + 1)
        return [
            BufferSpec(
                name="hidden_states",
                role=BufferRole.ACTIVATION,
                shape=(positions, model.hidden_size),
                dtype=model.dtype,
            ),
            BufferSpec(
                name="attention_out",
                role=BufferRole.ACTIVATION,
                shape=(positions, model.hidden_size),
                dtype=model.dtype,
            ),
            BufferSpec(
                name="mlp_scratch",
                role=BufferRole.ACTIVATION,
                shape=(positions, model.intermediate_size),
                dtype=model.dtype,
            ),
        ]

    def logits_probability_buffer_plan(
        self, model: ModelMetadata, *, max_gamma: int
    ) -> list[BufferSpec]:
        positions = max(1, max_gamma + 1)
        return [
            BufferSpec(
                name="logits",
                role=BufferRole.LOGITS,
                shape=(positions, model.vocab_size),
                dtype="fp32",
            ),
            BufferSpec(
                name="probabilities",
                role=BufferRole.PROBABILITIES,
                shape=(positions, model.vocab_size),
                dtype="fp32",
                required=False,
            ),
        ]

    def _kv_shape(
        self, model: ModelMetadata, max_seq_len: int, max_gamma: int
    ) -> tuple[int, ...]:
        return (
            2,
            model.num_hidden_layers,
            model.num_kv_heads,
            max_seq_len + max(0, max_gamma),
            model.head_dim,
        )

    def _estimate_transformer_weight_params(self, model: ModelMetadata) -> int:
        per_layer = (
            model.hidden_size * (model.num_attention_heads + 2 * model.num_kv_heads) * model.head_dim
            + model.hidden_size * model.hidden_size
            + 3 * model.hidden_size * model.intermediate_size
            + 2 * model.hidden_size
        )
        lm_head = model.hidden_size * model.vocab_size + model.hidden_size
        return model.num_hidden_layers * per_layer + lm_head
