from __future__ import annotations

import math


def spec_verify_cpu(
    logits_rows: list[list[float]],
    *,
    num_verify_tokens: int,
    vocab_size: int,
) -> list[list[float]]:
    _validate_rows(logits_rows, num_verify_tokens, vocab_size)
    out = []
    for row in logits_rows:
        max_value = max(row)
        exps = [math.exp(value - max_value) for value in row]
        denom = sum(exps)
        out.append([value / denom for value in exps])
    return out


def spec_verify_cuda(
    logits_rows: list[list[float]],
    *,
    num_verify_tokens: int,
    vocab_size: int,
) -> list[list[float]]:
    _validate_rows(logits_rows, num_verify_tokens, vocab_size)
    from vinf import _cuda_spec_verify

    rows = _cuda_spec_verify.spec_verify(logits_rows, int(num_verify_tokens), int(vocab_size))
    return [[float(value) for value in row] for row in rows]


def _validate_rows(
    logits_rows: list[list[float]], num_verify_tokens: int, vocab_size: int
) -> None:
    if num_verify_tokens <= 0:
        raise ValueError("num_verify_tokens must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if len(logits_rows) != num_verify_tokens:
        raise ValueError("logits_rows must contain num_verify_tokens rows")
    for row in logits_rows:
        if len(row) != vocab_size:
            raise ValueError("each logits row must have vocab_size entries")
