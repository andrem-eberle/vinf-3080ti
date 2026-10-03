from __future__ import annotations

from dataclasses import dataclass

from vinf.errors import DecodeError


@dataclass(slots=True)
class LogicalKVCache:
    name: str
    max_seq_len: int
    max_speculative_tokens: int = 0
    committed_len: int = 0
    spec_len: int = 0

    @property
    def physical_capacity(self) -> int:
        return self.max_seq_len + self.max_speculative_tokens

    @property
    def visible_len(self) -> int:
        return self.committed_len

    @property
    def written_len(self) -> int:
        return self.committed_len + self.spec_len

    def can_read_position(self, position: int, *, include_speculative: bool = False) -> bool:
        limit = self.written_len if include_speculative else self.visible_len
        return 0 <= position < limit

    def require_read_position(
        self, position: int, *, include_speculative: bool = False
    ) -> None:
        if not self.can_read_position(position, include_speculative=include_speculative):
            raise DecodeError(
                f"{self.name} cannot read position {position}; "
                f"visible={self.visible_len}, written={self.written_len}"
            )

    def append_committed(self, count: int) -> None:
        self._check_count(count)
        if self.committed_len + count > self.max_seq_len:
            raise DecodeError(f"{self.name} committed length exceeds max_seq_len")
        self.committed_len += count
        self.spec_len = 0

    def write_speculative(self, count: int) -> None:
        self._check_count(count)
        if self.committed_len + count > self.physical_capacity:
            raise DecodeError(f"{self.name} speculative write exceeds physical capacity")
        self.spec_len = count

    def commit_speculative(self, count: int) -> None:
        self._check_count(count)
        if count > self.spec_len:
            raise DecodeError(f"{self.name} cannot commit more speculative tokens than written")
        if self.committed_len + count > self.max_seq_len:
            raise DecodeError(f"{self.name} committed length exceeds max_seq_len")
        self.committed_len += count
        self.spec_len = 0

    def discard_speculative(self) -> None:
        self.spec_len = 0

    def rollback_to(self, committed_len: int) -> None:
        if committed_len < 0 or committed_len > self.committed_len:
            raise DecodeError(f"{self.name} invalid rollback target")
        self.committed_len = committed_len
        self.spec_len = 0

    def _check_count(self, count: int) -> None:
        if count < 0:
            raise DecodeError(f"{self.name} count must be non-negative")


@dataclass(slots=True)
class KVCacheSet:
    target: LogicalKVCache
    draft: LogicalKVCache | None = None

    def discard_speculative(self) -> None:
        self.target.discard_speculative()
        if self.draft is not None:
            self.draft.discard_speculative()

