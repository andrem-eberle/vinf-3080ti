from __future__ import annotations

from dataclasses import dataclass

from vinf.models.metadata import ModelMetadata
from vinf.runtime.instructions import (
    Attention,
    DownProj,
    INTS_PER_INSTRUCTION,
    Instruction,
    LMHead,
    MLPUpGate,
    NoOp,
    OProj,
    RMSQKVRope,
    Verify,
)


@dataclass(frozen=True, slots=True)
class InstructionTensor:
    rows: tuple[tuple[tuple[int, ...], ...], ...]

    @property
    def num_sms(self) -> int:
        return len(self.rows)

    @property
    def queue_len(self) -> int:
        return len(self.rows[0]) if self.rows else 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_sms, self.queue_len, INTS_PER_INSTRUCTION)


@dataclass(frozen=True, slots=True)
class Schedule:
    queues: tuple[tuple[Instruction, ...], ...]

    @property
    def num_sms(self) -> int:
        return len(self.queues)

    @property
    def max_queue_len(self) -> int:
        return max((len(queue) for queue in self.queues), default=0)

    def tensorize(self) -> InstructionTensor:
        max_len = self.max_queue_len
        rows = []
        for queue in self.queues:
            padded = list(queue) + [NoOp()] * (max_len - len(queue))
            rows.append(tuple(tuple(ins.serialize()) for ins in padded))
        return InstructionTensor(tuple(rows))


class ScheduleCache:
    def __init__(self) -> None:
        self._cache: dict[tuple, Schedule] = {}

    def get_or_build(self, key: tuple, builder) -> Schedule:  # type: ignore[no-untyped-def]
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]


def round_robin_schedule(
    instructions: list[Instruction], num_sms: int
) -> Schedule:
    if num_sms <= 0:
        raise ValueError("num_sms must be positive")
    queues: list[list[Instruction]] = [[] for _ in range(num_sms)]
    for idx, instruction in enumerate(instructions):
        queues[idx % num_sms].append(instruction)
    return Schedule(tuple(tuple(queue) for queue in queues))


def build_baseline_decode_schedule(
    model: ModelMetadata,
    *,
    num_sms: int,
    position: int,
    block_size: int = 16,
) -> Schedule:
    instructions: list[Instruction] = []
    hidden_blocks = _ceil_div(model.hidden_size, block_size)
    intermediate_blocks = _ceil_div(model.intermediate_size, block_size)
    vocab_blocks = _ceil_div(model.vocab_size, block_size)

    for layer_idx in range(model.num_hidden_layers):
        instructions.append(RMSQKVRope(layer_idx, 0, hidden_blocks, position))
        for kv_head_idx in range(model.num_kv_heads):
            instructions.append(Attention(layer_idx, kv_head_idx, 0, position + 1))
        instructions.append(OProj(layer_idx, 0, hidden_blocks))
        instructions.append(MLPUpGate(layer_idx, 0, intermediate_blocks))
        instructions.append(DownProj(layer_idx, 0, hidden_blocks))
    instructions.append(LMHead(0, vocab_blocks))

    return round_robin_schedule(instructions, num_sms)


def build_verification_schedule(
    model: ModelMetadata,
    *,
    num_sms: int,
    position: int,
    num_verify_tokens: int,
    draft_token_start: int = 0,
) -> Schedule:
    if num_verify_tokens <= 0:
        raise ValueError("num_verify_tokens must be positive")
    instructions = [
        Verify(
            position=position,
            num_verify_tokens=num_verify_tokens,
            draft_token_start=draft_token_start,
        )
    ]
    baseline = build_baseline_decode_schedule(
        model, num_sms=num_sms, position=position
    )
    flattened = instructions + [ins for queue in baseline.queues for ins in queue]
    return round_robin_schedule(flattened, num_sms)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

