from __future__ import annotations

from typing import Protocol

from vinf.runtime.state import RuntimeState


class DecodeStrategy(Protocol):
    def step(self, state: RuntimeState) -> list[int]:
        ...


class BaselineDecodeStrategy:
    def __init__(self, target_executor) -> None:  # type: ignore[no-untyped-def]
        self.target_executor = target_executor

    def step(self, state: RuntimeState) -> list[int]:
        result = self.target_executor.decode_one(state)
        return [result.token_id]

