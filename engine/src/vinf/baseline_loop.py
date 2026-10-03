from __future__ import annotations

from dataclasses import dataclass, field

from vinf.baseline_decode import BaselineDecodeConfig, BaselineTargetWeights
from vinf.config import GenerationConfig
from vinf.cuda.megakernel import TargetMegakernelRuntime
from vinf.metrics import EngineMetrics, Timer
from vinf.prefill import (
    TargetPrefillState,
    decode_after_prefill_reference,
    target_prefill_reference,
)
from vinf.runtime.state import DecodeMode, DecodeResult, RuntimeState
from vinf.sampling import CPUSampler, Sampler


@dataclass(frozen=True, slots=True)
class TokenCodec:
    eos_token_id: int | None = None

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


@dataclass(slots=True)
class BaselineLoopResult:
    state: RuntimeState
    decode_result: DecodeResult
    metrics: EngineMetrics
    prefill: TargetPrefillState


@dataclass(slots=True)
class BaselineEngineLoop:
    weights: BaselineTargetWeights
    config: BaselineDecodeConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sampler: Sampler = field(default_factory=CPUSampler)
    runtime: TargetMegakernelRuntime = field(default_factory=TargetMegakernelRuntime)
    codec: TokenCodec = field(default_factory=TokenCodec)

    def generate_from_hidden_states(
        self,
        prompt_hidden_states: list[list[float]],
        *,
        prompt_tokens: list[int] | None = None,
        rope_cos: list[float] | list[list[float]],
        rope_sin: list[float] | list[list[float]],
    ) -> BaselineLoopResult:
        # TODO: replace prompt_hidden_states/prompt_tokens split with real tokenizer+embedding flow.
        if prompt_tokens is not None and len(prompt_tokens) != len(prompt_hidden_states):
            raise ValueError("prompt_tokens length must match prompt_hidden_states")
        state = RuntimeState(
            model=self.config.metadata,
            generation=self.generation,
            mode=DecodeMode.BASELINE,
            prompt_tokens=list(prompt_tokens or range(len(prompt_hidden_states))),
            position=0,
        )
        metrics = EngineMetrics()
        with Timer(metrics, "prefill"):
            prefill = target_prefill_reference(
                prompt_hidden_states,
                self.weights,
                self.config,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
        state.position = prefill.prompt_len

        current_hidden = list(prefill.final_hidden)
        for _ in range(self.generation.max_new_tokens):
            with Timer(metrics, "decode"):
                step = decode_after_prefill_reference(
                    current_hidden,
                    prefill,
                    self.weights,
                    self.config,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                )
            metrics.target_calls += 1
            sample = self.sampler.sample(step.logits, self.generation)
            state.append_tokens([sample.token_id])
            metrics.generated_tokens += 1
            prefill = TargetPrefillState(
                backend=prefill.backend,
                prompt_len=prefill.prompt_len + 1,
                final_hidden=step.hidden,
                layers=step.layers,
                memory_bytes=prefill.memory_bytes,
            )
            current_hidden = step.hidden
            if self.codec.eos_token_id is not None and sample.token_id == self.codec.eos_token_id:
                state.stopped = True
                state.stop_reason = "eos_token"
                break
            if state.stopped:
                break
            if prefill.prompt_len >= self.config.metadata.max_position_embeddings:
                state.stopped = True
                state.stop_reason = "max_position_embeddings"
                break

        tokens = tuple(state.output_tokens)
        return BaselineLoopResult(
            state=state,
            decode_result=DecodeResult(
                tokens=tokens,
                text=self.codec.decode(list(tokens)),
                stop_reason=state.stop_reason,
            ),
            metrics=metrics,
            prefill=prefill,
        )

    def stream_from_hidden_states(
        self,
        prompt_hidden_states: list[list[float]],
        *,
        prompt_tokens: list[int] | None = None,
        rope_cos: list[float] | list[list[float]],
        rope_sin: list[float] | list[list[float]],
    ):
        result = self.generate_from_hidden_states(
            prompt_hidden_states,
            prompt_tokens=prompt_tokens,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        yield from result.decode_result.tokens
