from __future__ import annotations

def run_noop_smoke() -> tuple[int, int, int]:
    from vinf import _cuda_noop

    return tuple(int(x) for x in _cuda_noop.run_noop_smoke())


def cuda_device_count() -> int:
    from vinf import _cuda_noop

    return int(_cuda_noop.cuda_device_count())


def run_and_validate_noop_smoke() -> tuple[int, int, int]:
    from vinf.cuda.megakernel import validate_noop_result

    result = run_noop_smoke()
    validate_noop_result(result)
    return result
