from __future__ import annotations

from dataclasses import dataclass

from vinf.executors.base import TargetExecutor, VerificationResult
from vinf.runtime.kv_cache import LogicalKVCache
from vinf.runtime.state import RuntimeState
from vinf.cuda.spec_verify import spec_verify_cpu, spec_verify_cuda


@dataclass(frozen=True, slots=True)
class FallbackTargetVerifier:
    target_executor: TargetExecutor
    speculative_kv: LogicalKVCache | None = None

    def verify_many(
        self, state: RuntimeState, draft_tokens: tuple[int, ...]
    ) -> VerificationResult:
        result = self.target_executor.verify_many(state, draft_tokens)
        rows = result.probability_rows_ref
        if not isinstance(rows, (list, tuple)):
            raise TypeError("target verifier must return probability rows")
        if len(rows) != len(draft_tokens) + 1:
            raise ValueError("target verifier must return gamma + 1 probability rows")
        if self.speculative_kv is not None and draft_tokens:
            self.speculative_kv.write_speculative(len(draft_tokens))
            return VerificationResult(
                probability_rows_ref=rows,
                speculative_kv_ref=self.speculative_kv,
            )
        return result


@dataclass(frozen=True, slots=True)
class MegakernelTargetVerifier:
    target_executor: TargetExecutor
    speculative_kv: LogicalKVCache | None = None
    use_cuda: bool = True

    def verify_many(
        self, state: RuntimeState, draft_tokens: tuple[int, ...]
    ) -> VerificationResult:
        fallback = self.target_executor.verify_many(state, draft_tokens)
        rows = fallback.probability_rows_ref
        if not isinstance(rows, list):
            rows = [list(row) for row in rows]  # type: ignore[union-attr]
        num_rows = len(draft_tokens) + 1
        vocab_size = len(rows[0]) if rows else 0
        if self.use_cuda:
            try:
                logits_like_rows = _probabilities_to_logits_like(rows)
                rows = spec_verify_cuda(
                    logits_like_rows,
                    num_verify_tokens=num_rows,
                    vocab_size=vocab_size,
                )
            except Exception:
                rows = spec_verify_cpu(
                    _probabilities_to_logits_like(rows),
                    num_verify_tokens=num_rows,
                    vocab_size=vocab_size,
                )
        if self.speculative_kv is not None and draft_tokens:
            self.speculative_kv.write_speculative(len(draft_tokens))
            return VerificationResult(probability_rows_ref=rows, speculative_kv_ref=self.speculative_kv)
        return VerificationResult(probability_rows_ref=rows)


def _probabilities_to_logits_like(rows: list[list[float]]) -> list[list[float]]:
    # The current CUDA verifier slice owns the per-position probability transform.
    # Convert probabilities to log-space so its softmax returns the same rows.
    import math

    return [[math.log(max(value, 1e-30)) for value in row] for row in rows]
