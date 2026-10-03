from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class VerificationGlobalsABI:
    position: int
    num_verify_tokens: int
    max_gamma: int
    draft_token_start: int

    def serialize(self) -> tuple[int, ...]:
        return tuple(int(getattr(self, field.name)) for field in fields(self))


def verification_globals_abi_header() -> str:
    lines = [
        "#pragma once",
        "",
        "// Generated ABI mirror for speculative verification globals.",
        "struct VinfVerificationGlobals {",
    ]
    for field in fields(VerificationGlobalsABI):
        lines.append(f"    int {field.name};")
    lines.extend(
        [
            "};",
            "",
            "#define VINF_VERIFICATION_GLOBALS_WORDS 4",
            "#define VINF_VERIFY_POSITION_WORD 0",
            "#define VINF_VERIFY_NUM_VERIFY_TOKENS_WORD 1",
            "#define VINF_VERIFY_MAX_GAMMA_WORD 2",
            "#define VINF_VERIFY_DRAFT_TOKEN_START_WORD 3",
        ]
    )
    return "\n".join(lines) + "\n"
