from __future__ import annotations

from dataclasses import dataclass, field

from vinf.baseline_decode import (
    BaselineDecodeConfig,
    BaselineDecodeResult,
    BaselineTargetWeights,
    baseline_decode_cuda_composed,
    baseline_decode_reference,
)
from vinf.runtime.instructions import INTS_PER_INSTRUCTION, NoOp
from vinf.runtime.scheduler import (
    InstructionTensor,
    Schedule,
    build_baseline_decode_schedule,
    round_robin_schedule,
)


RTX_3080_TI_EXPECTED_SMS = 80
RTX_3080_TI_NUM_THREADS = 384
RTX_3080_TI_DYNAMIC_SHARED_MEMORY = 0
TIMING_WIDTH = 128


@dataclass(frozen=True, slots=True)
class LaunchDimensions:
    grid_blocks: int
    block_threads: int
    dynamic_shared_memory: int


@dataclass(slots=True)
class MegakernelBuffers:
    instructions: list[list[list[int]]]
    timings: list[list[list[int]]]

    @property
    def instruction_shape(self) -> tuple[int, int, int]:
        if not self.instructions:
            return (0, 0, INTS_PER_INSTRUCTION)
        return (
            len(self.instructions),
            len(self.instructions[0]),
            len(self.instructions[0][0]) if self.instructions[0] else INTS_PER_INSTRUCTION,
        )

    @property
    def timing_shape(self) -> tuple[int, int, int]:
        if not self.timings:
            return (0, 0, TIMING_WIDTH)
        return (
            len(self.timings),
            len(self.timings[0]),
            len(self.timings[0][0]) if self.timings[0] else TIMING_WIDTH,
        )


@dataclass(slots=True)
class MinimalMegakernelRuntime:
    num_sms: int = RTX_3080_TI_EXPECTED_SMS
    block_threads: int = RTX_3080_TI_NUM_THREADS
    dynamic_shared_memory: int = RTX_3080_TI_DYNAMIC_SHARED_MEMORY
    buffers: MegakernelBuffers | None = None
    last_timing_snapshot: list[list[list[int]]] = field(default_factory=list)

    def launch_dimensions(self) -> LaunchDimensions:
        return LaunchDimensions(
            grid_blocks=self.num_sms,
            block_threads=self.block_threads,
            dynamic_shared_memory=self.dynamic_shared_memory,
        )

    def allocate_for_schedule(self, schedule: Schedule) -> MegakernelBuffers:
        tensor = schedule.tensorize()
        instructions = [
            [list(row) for row in sm_rows]
            for sm_rows in tensor.rows
        ]
        timings = [
            [[0 for _ in range(TIMING_WIDTH)] for _ in sm_rows]
            for sm_rows in tensor.rows
        ]
        self.buffers = MegakernelBuffers(instructions=instructions, timings=timings)
        return self.buffers

    def allocate_noop(self, queue_len: int = 1) -> MegakernelBuffers:
        schedule = noop_schedule(self.num_sms, queue_len)
        return self.allocate_for_schedule(schedule)

    def read_timings(self) -> list[list[list[int]]]:
        if self.buffers is None:
            return []
        self.last_timing_snapshot = [
            [list(row) for row in sm_rows] for sm_rows in self.buffers.timings
        ]
        return self.last_timing_snapshot


@dataclass(slots=True)
class TargetMegakernelRuntime(MinimalMegakernelRuntime):
    use_cuda: bool = True
    last_decode_result: BaselineDecodeResult | None = None

    def decode_one_token(
        self,
        hidden: list[float],
        weights: BaselineTargetWeights,
        config: BaselineDecodeConfig,
        *,
        position: int,
        rope_cos: list[float],
        rope_sin: list[float],
    ) -> BaselineDecodeResult:
        schedule = build_baseline_decode_schedule(
            config.metadata,
            num_sms=self.num_sms,
            position=position,
            block_size=config.block_size,
        )
        self.allocate_for_schedule(schedule)
        runner = baseline_decode_cuda_composed if self.use_cuda else baseline_decode_reference
        result = runner(
            hidden,
            weights,
            BaselineDecodeConfig(
                metadata=config.metadata,
                eps=config.eps,
                num_sms=self.num_sms,
                block_size=config.block_size,
            ),
            position=position,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        self.last_decode_result = result
        self._write_timing_summary(result)
        return result

    def _write_timing_summary(self, result: BaselineDecodeResult) -> None:
        if self.buffers is None or not self.buffers.timings:
            return
        micros = [max(0, int(seconds * 1_000_000)) for seconds in result.timings.values()]
        for sm_rows in self.buffers.timings:
            for row in sm_rows:
                for idx, value in enumerate(micros[:TIMING_WIDTH]):
                    row[idx] = value


def noop_schedule(num_sms: int, queue_len: int = 1) -> Schedule:
    if queue_len <= 0:
        raise ValueError("queue_len must be positive")
    instructions = [NoOp() for _ in range(num_sms * queue_len)]
    return round_robin_schedule(instructions, num_sms)


def validate_noop_result(result: tuple[int, int, int]) -> None:
    opcode, arch, block_threads = result
    if opcode != 0:
        raise RuntimeError(f"unexpected noop opcode: {opcode}")
    if arch != 86:
        raise RuntimeError(f"unexpected CUDA arch marker: {arch}")
    if block_threads != RTX_3080_TI_NUM_THREADS:
        raise RuntimeError(
            f"unexpected block thread count: {block_threads}, expected {RTX_3080_TI_NUM_THREADS}"
        )


def noop_instruction_tensor(num_sms: int = RTX_3080_TI_EXPECTED_SMS) -> InstructionTensor:
    return noop_schedule(num_sms).tensorize()
