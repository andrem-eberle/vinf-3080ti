from __future__ import annotations

from dataclasses import dataclass, field

from vinf.config import GenerationConfig
from vinf.metrics import EngineMetrics, Timer
from vinf.runtime.state import DecodeMode, RuntimeState
from vinf.speculative.controller import SpeculativeDecodeStrategy


@dataclass(slots=True)
class SpeculativeHybridResult:
    state: RuntimeState
    metrics: EngineMetrics
    disabled: bool


@dataclass(slots=True)
class SpeculativeHybridEngine:
    strategy: SpeculativeDecodeStrategy
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    disable_below_tokens_per_target: float = 1.2
    warmup_iterations: int = 1
    disabled: bool = False

    def generate(self, prompt_tokens: list[int]) -> SpeculativeHybridResult:
        state = RuntimeState(
            generation=self.generation,
            mode=DecodeMode.SPECULATIVE,
            prompt_tokens=list(prompt_tokens),
            position=len(prompt_tokens),
        )
        metrics = EngineMetrics()
        iterations = 0
        while not state.stopped and not self.disabled:
            iterations += 1
            before = len(state.output_tokens)
            with Timer(metrics, "speculative_step"):
                emitted = self.strategy.step(state)
            produced = min(len(emitted), self.generation.max_new_tokens - before)
            if len(state.output_tokens) > self.generation.max_new_tokens:
                del state.output_tokens[self.generation.max_new_tokens :]
                state.position = len(state.prompt_tokens) + len(state.output_tokens)
                state.stopped = True
                state.stop_reason = "max_new_tokens"
            metrics.target_calls += 1
            metrics.verification_calls += 1
            metrics.draft_calls += 1
            metrics.generated_tokens += produced
            metrics.speculative_tokens_proposed += self.strategy.gamma
            accepted = max(0, min(len(emitted) - 1, self.strategy.gamma))
            metrics.speculative_tokens_accepted += accepted
            if accepted < self.strategy.gamma:
                metrics.record_rejection_position(accepted)
            if len(state.output_tokens) >= self.generation.max_new_tokens:
                state.stopped = True
                state.stop_reason = "max_new_tokens"
            if (
                iterations >= self.warmup_iterations
                and metrics.tokens_per_target_call < self.disable_below_tokens_per_target
            ):
                self.disabled = True
                if not state.stopped:
                    state.stopped = True
                    state.stop_reason = "speculation_disabled"
        return SpeculativeHybridResult(state=state, metrics=metrics, disabled=self.disabled)
