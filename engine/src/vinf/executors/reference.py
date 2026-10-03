from __future__ import annotations

from vinf.config import GenerationConfig
from vinf.executors.base import DecodeStep, VerificationResult
from vinf.models.loader import LoadedModel
from vinf.runtime.state import RuntimeState
from vinf.sampling import CPUSampler, logits_to_probabilities


class ReferenceExecutor:
    """Pure-Python executor for the tiny transition-logits reference model."""

    def __init__(
        self,
        model: LoadedModel,
        *,
        generation_config: GenerationConfig | None = None,
        seed: int | None = 0,
    ) -> None:
        self.model = model
        self.generation_config = generation_config or GenerationConfig()
        self.sampler = CPUSampler(seed=seed)
        self.transition_logits = model.weights.get("transition_logits").values

    def prefill(self, state: RuntimeState) -> None:
        state.model = self.model.metadata
        state.position = len(state.prompt_tokens)

    def decode_one(self, state: RuntimeState) -> DecodeStep:
        logits = self.next_logits(state)
        probabilities = logits_to_probabilities(logits, self.generation_config)
        sample = self.sampler.sample(logits, self.generation_config)
        return DecodeStep(
            token_id=sample.token_id,
            logits_ref=logits,
            probabilities_ref=probabilities,
        )

    def verify_many(
        self, state: RuntimeState, draft_tokens: tuple[int, ...]
    ) -> VerificationResult:
        rows = []
        context = state.all_tokens
        for i in range(len(draft_tokens) + 1):
            token_context = context + list(draft_tokens[:i])
            rows.append(
                logits_to_probabilities(
                    self._logits_for_last_token(token_context), self.generation_config
                )
            )
        return VerificationResult(probability_rows_ref=rows)

    def next_logits(self, state: RuntimeState) -> list[float]:
        return self._logits_for_last_token(state.all_tokens)

    def _logits_for_last_token(self, tokens: list[int]) -> list[float]:
        if not tokens:
            last_token = 0
        else:
            last_token = tokens[-1]
        if last_token < 0 or last_token >= self.model.metadata.vocab_size:
            raise ValueError(f"token {last_token} outside vocab")
        return [float(x) for x in self.transition_logits[last_token]]
