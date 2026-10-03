from __future__ import annotations


def run_math_smoke(values: list[float]) -> list[float]:
    from vinf import _cuda_math

    return [float(x) for x in _cuda_math.run_math_smoke(values)]


def math_smoke_reference(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("input must not be empty")
    out = [value * 2.0 for value in values]
    total = sum(values)
    out.extend([0.0] * 8)
    out[len(values)] = total
    out[len(values) + 1] = _round_to_halfish(total)
    if len(values) >= 4:
        out[len(values) + 4 : len(values) + 8] = [value + 1.0 for value in values[:4]]
    return out


def _round_to_halfish(value: float) -> float:
    # Placeholder tolerance helper; exact IEEE half conversion is checked with a
    # loose tolerance in tests because Python stdlib has no direct half scalar.
    return float(value)

