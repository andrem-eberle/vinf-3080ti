from __future__ import annotations

import math

from vinf.config import GenerationConfig
from vinf.sampling import apply_top_k, apply_top_p, logits_to_probabilities
from vinf.speculative.sampler import acceptance_probability, correction_distribution


def gpu_logits_process_cpu(logits: list[float], config: GenerationConfig) -> list[float]:
    scaled = [value / config.temperature for value in logits]
    return apply_top_p(apply_top_k(scaled, config.top_k), config.top_p)


def gpu_softmax_cpu(logits: list[float]) -> list[float]:
    finite = [value for value in logits if math.isfinite(value)]
    if not finite:
        raise ValueError("at least one logit must be finite")
    max_value = max(finite)
    exps = [math.exp(value - max_value) if math.isfinite(value) else 0.0 for value in logits]
    denom = sum(exps)
    return [value / denom for value in exps]


def gpu_probabilities_cpu(logits: list[float], config: GenerationConfig) -> list[float]:
    return gpu_softmax_cpu(gpu_logits_process_cpu(logits, config))


def gpu_probabilities_cuda(logits: list[float], config: GenerationConfig) -> list[float]:
    from vinf.cuda.spec_verify import spec_verify_cuda

    processed = gpu_logits_process_cpu(logits, config)
    return spec_verify_cuda([processed], num_verify_tokens=1, vocab_size=len(processed))[0]


class CounterRNG:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed & 0xFFFFFFFFFFFFFFFF

    def uniform(self, counter: int) -> float:
        value = (self.seed + counter * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 31
        return ((value >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def sample_token_id(probabilities: list[float], *, seed: int, counter: int = 0) -> int:
    draw = CounterRNG(seed).uniform(counter)
    total = 0.0
    for token_id, probability in enumerate(probabilities):
        total += probability
        if draw <= total:
            return token_id
    return len(probabilities) - 1


def sample_from_logits_cpu(
    logits: list[float], config: GenerationConfig, *, seed: int, counter: int = 0
) -> int:
    return sample_token_id(gpu_probabilities_cpu(logits, config), seed=seed, counter=counter)


def speculative_accept_and_sample_cpu(
    draft_tokens: tuple[int, ...],
    draft_probability_rows: list[list[float]],
    target_probability_rows: list[list[float]],
    *,
    seed: int,
) -> tuple[tuple[int, ...], int, bool]:
    accepted = 0
    rng = CounterRNG(seed)
    for idx, token in enumerate(draft_tokens):
        accept_prob = acceptance_probability(
            target_probability_rows[idx][token],
            draft_probability_rows[idx][token],
        )
        if rng.uniform(idx) <= accept_prob:
            accepted += 1
        else:
            break
    if accepted == len(draft_tokens):
        next_token = sample_token_id(
            target_probability_rows[len(draft_tokens)],
            seed=seed,
            counter=len(draft_tokens),
        )
        rejected = False
    else:
        next_token = sample_token_id(
            correction_distribution(
                target_probability_rows[accepted],
                draft_probability_rows[accepted],
            ),
            seed=seed,
            counter=accepted + 1,
        )
        rejected = True
    return ((*draft_tokens[:accepted], next_token), accepted, rejected)
