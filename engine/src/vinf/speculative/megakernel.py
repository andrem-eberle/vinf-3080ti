from __future__ import annotations

from dataclasses import dataclass

from vinf.models.metadata import ModelMetadata
from vinf.runtime.scheduler import Schedule, build_verification_schedule


@dataclass(frozen=True, slots=True)
class VerificationGlobals:
    position: int
    num_verify_tokens: int
    max_gamma: int
    draft_token_start: int = 0

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("position must be non-negative")
        if self.max_gamma <= 0:
            raise ValueError("max_gamma must be positive")
        if not 1 <= self.num_verify_tokens <= self.max_gamma + 1:
            raise ValueError("num_verify_tokens must be in [1, max_gamma + 1]")
        if self.draft_token_start < 0:
            raise ValueError("draft_token_start must be non-negative")


@dataclass(frozen=True, slots=True)
class VerificationBufferPlan:
    activation_shape: tuple[int, int]
    logits_shape: tuple[int, int]
    probabilities_shape: tuple[int, int]
    speculative_kv_shape: tuple[int, int, int, int]
    draft_token_shape: tuple[int, ...]
    position_id_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VerificationInputs:
    draft_token_ids: tuple[int, ...]
    position_ids: tuple[int, ...]


def verification_buffer_plan(
    model: ModelMetadata, *, max_gamma: int
) -> VerificationBufferPlan:
    if max_gamma <= 0:
        raise ValueError("max_gamma must be positive")
    positions = max_gamma + 1
    return VerificationBufferPlan(
        activation_shape=(positions, model.hidden_size),
        logits_shape=(positions, model.vocab_size),
        probabilities_shape=(positions, model.vocab_size),
        speculative_kv_shape=(
            model.num_hidden_layers,
            model.num_kv_heads,
            positions,
            model.head_dim,
        ),
        draft_token_shape=(max_gamma,),
        position_id_shape=(positions,),
    )


def verification_inputs(
    draft_token_ids: tuple[int, ...],
    *,
    position: int,
    max_gamma: int,
) -> VerificationInputs:
    if max_gamma <= 0:
        raise ValueError("max_gamma must be positive")
    if len(draft_token_ids) > max_gamma:
        raise ValueError("draft_token_ids length exceeds max_gamma")
    if position < 0:
        raise ValueError("position must be non-negative")
    return VerificationInputs(
        draft_token_ids=draft_token_ids,
        position_ids=tuple(position + idx for idx in range(len(draft_token_ids) + 1)),
    )


def causal_verify_mask(num_verify_tokens: int) -> list[list[bool]]:
    if num_verify_tokens <= 0:
        raise ValueError("num_verify_tokens must be positive")
    return [
        [key_pos <= query_pos for key_pos in range(num_verify_tokens)]
        for query_pos in range(num_verify_tokens)
    ]


def build_speculative_megakernel_schedule(
    model: ModelMetadata,
    *,
    num_sms: int,
    globals: VerificationGlobals,
) -> Schedule:
    return build_verification_schedule(
        model,
        num_sms=num_sms,
        position=globals.position,
        num_verify_tokens=globals.num_verify_tokens,
        draft_token_start=globals.draft_token_start,
    )
