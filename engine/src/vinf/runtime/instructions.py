from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum


INTS_PER_INSTRUCTION = 32


class Opcode(IntEnum):
    NOOP = 0
    RMS_QKV_ROPE = 1
    ATTENTION = 2
    O_PROJ = 3
    MLP_UPGATE = 4
    DOWN_PROJ = 5
    LM_HEAD = 6
    VERIFY = 7
    # qwen35 fused megakernel (Phase 31b)
    QWEN_QMV = 16
    QWEN_LOAD = 17
    QWEN_ATTN_HEAD = 18
    QWEN_SSM_GROUP = 19
    QWEN_ARGMAX = 20


@dataclass(frozen=True, slots=True)
class Instruction:
    def opcode(self) -> Opcode:
        raise NotImplementedError

    def serialize_fields(self) -> list[int]:
        out: list[int] = []
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool):
                out.append(int(value))
            elif isinstance(value, int):
                out.append(value)
            elif isinstance(value, tuple):
                out.append(len(value))
                out.extend(int(item) for item in value)
            else:
                raise TypeError(f"unsupported instruction field {field.name}: {value!r}")
        return out

    def serialize(self) -> list[int]:
        words = [int(self.opcode()), *self.serialize_fields()]
        if len(words) > INTS_PER_INSTRUCTION:
            raise ValueError(
                f"{type(self).__name__} serializes to {len(words)} words; "
                f"limit is {INTS_PER_INSTRUCTION}"
            )
        return words + [0] * (INTS_PER_INSTRUCTION - len(words))


@dataclass(frozen=True, slots=True)
class NoOp(Instruction):
    def opcode(self) -> Opcode:
        return Opcode.NOOP


@dataclass(frozen=True, slots=True)
class RMSQKVRope(Instruction):
    layer_idx: int
    start_block: int
    end_block: int
    position: int

    def opcode(self) -> Opcode:
        return Opcode.RMS_QKV_ROPE


@dataclass(frozen=True, slots=True)
class Attention(Instruction):
    layer_idx: int
    kv_head_idx: int
    start_position: int
    end_position: int

    def opcode(self) -> Opcode:
        return Opcode.ATTENTION


@dataclass(frozen=True, slots=True)
class OProj(Instruction):
    layer_idx: int
    start_block: int
    end_block: int

    def opcode(self) -> Opcode:
        return Opcode.O_PROJ


@dataclass(frozen=True, slots=True)
class MLPUpGate(Instruction):
    layer_idx: int
    start_block: int
    end_block: int

    def opcode(self) -> Opcode:
        return Opcode.MLP_UPGATE


@dataclass(frozen=True, slots=True)
class DownProj(Instruction):
    layer_idx: int
    start_block: int
    end_block: int

    def opcode(self) -> Opcode:
        return Opcode.DOWN_PROJ


@dataclass(frozen=True, slots=True)
class LMHead(Instruction):
    start_vocab_block: int
    end_vocab_block: int

    def opcode(self) -> Opcode:
        return Opcode.LM_HEAD


@dataclass(frozen=True, slots=True)
class Verify(Instruction):
    position: int
    num_verify_tokens: int
    draft_token_start: int

    def opcode(self) -> Opcode:
        return Opcode.VERIFY


INSTRUCTION_CLASSES = (
    NoOp,
    RMSQKVRope,
    Attention,
    OProj,
    MLPUpGate,
    DownProj,
    LMHead,
    Verify,
)


def cuda_instruction_abi_header() -> str:
    lines = [
        "#pragma once",
        "",
        "// Generated ABI mirror for vinf runtime/instructions.py.",
        "#define VINF_INTS_PER_INSTRUCTION 32",
        "",
    ]
    for opcode in Opcode:
        lines.append(f"#define VINF_OPCODE_{opcode.name} {int(opcode)}")
    lines.append("")
    lines.extend(
        [
            "// Word 0 is always opcode.",
            "// RMSQKVRope: [1]=layer_idx [2]=start_block [3]=end_block [4]=position",
            "// Attention:  [1]=layer_idx [2]=kv_head_idx [3]=start_position [4]=end_position",
            "// OProj:      [1]=layer_idx [2]=start_block [3]=end_block",
            "// MLPUpGate:  [1]=layer_idx [2]=start_block [3]=end_block",
            "// DownProj:   [1]=layer_idx [2]=start_block [3]=end_block",
            "// LMHead:     [1]=start_vocab_block [2]=end_vocab_block",
            "// Verify:     [1]=position [2]=num_verify_tokens [3]=draft_token_start",
        ]
    )
    return "\n".join(lines) + "\n"



# ---- qwen35 fused megakernel instructions (Phase 31b) -----------------------------------
#
# Every row: [0]=opcode, [1]=wait0 counter, [2]=wait0 target, [3]=wait1 counter,
# [4]=wait1 target, [5]=signal counter, [6..]=op parameters. Counter -1 means none.
# Tensor fields index the megakernel tensor table; buffer fields index the buffer table.

QMV_FLAG_ADD = 1  # y[r] += W x (residual add fused into the projection)
QMV_FLAG_NORM = 2  # stage x as rmsnorm(x) * norm_weight
QMV_FLAG_ARGMAX = 4  # write this instruction's best (value, row) to partial slot
QMV_FLAG_SILU_MUL = 8  # stage x as silu(x) * x2


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenMkInstruction(Instruction):
    wait0_counter: int = -1
    wait0_target: int = 0
    wait1_counter: int = -1
    wait1_target: int = 0
    signal: int = -1


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenQmv(QwenMkInstruction):
    tensor: int
    slot: int = -1  # >= 0: weight bytes come from streaming slot
    x: int
    y: int
    row_start: int
    row_end: int
    flags: int = 0
    norm: int = -1
    partial: int = -1
    consumed_signal: int = -1  # streamed weights: signals slot consumption
    x2: int = -1

    def opcode(self) -> Opcode:
        return Opcode.QWEN_QMV


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenLoad(QwenMkInstruction):
    tensor: int
    slot: int
    byte_start: int
    byte_end: int

    def opcode(self) -> Opcode:
        return Opcode.QWEN_LOAD


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenAttnHead(QwenMkInstruction):
    kc: int
    vc: int
    head: int
    q_raw: int
    k: int
    v: int
    out: int
    q_norm: int
    k_norm: int

    def opcode(self) -> Opcode:
        return Opcode.QWEN_ATTN_HEAD


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenSsmGroup(QwenMkInstruction):
    key_head: int
    qkv: int
    z: int
    beta: int
    alpha: int
    conv_state: int
    ssm_state: int
    out: int
    conv_w: int
    ssm_a: int
    dt_bias: int
    ssm_norm: int

    def opcode(self) -> Opcode:
        return Opcode.QWEN_SSM_GROUP


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenArgmax(QwenMkInstruction):
    partials: int

    def opcode(self) -> Opcode:
        return Opcode.QWEN_ARGMAX


QWEN_MK_INSTRUCTION_CLASSES = (QwenQmv, QwenLoad, QwenAttnHead, QwenSsmGroup, QwenArgmax)


def qwen_mk_abi_header() -> str:
    """CUDA word offsets for the qwen35 megakernel rows, generated from the dataclasses."""
    lines = [
        "#pragma once",
        "",
        "// Generated by vinf.runtime.instructions.qwen_mk_abi_header(); do not edit.",
        f"#define VINF_MK_WORDS {INTS_PER_INSTRUCTION}",
        f"#define VINF_MK_OP_NOOP {int(Opcode.NOOP)}",
    ]
    for cls in QWEN_MK_INSTRUCTION_CLASSES:
        lines.append(f"#define VINF_MK_OP_{cls.__name__[4:].upper()} {int(cls.__new__(cls).opcode())}")
    lines.append("")
    for idx, field in enumerate(fields(QwenMkInstruction), start=1):
        lines.append(f"#define VINF_MK_{field.name.upper()} {idx}")
    base = len(fields(QwenMkInstruction))
    for cls in QWEN_MK_INSTRUCTION_CLASSES:
        prefix = cls.__name__[4:].upper()
        for idx, field in enumerate(fields(cls)[base:], start=base + 1):
            lines.append(f"#define VINF_MK_{prefix}_{field.name.upper()} {idx}")
    lines.extend([
        "",
        f"#define VINF_MK_FLAG_ADD {QMV_FLAG_ADD}",
        f"#define VINF_MK_FLAG_NORM {QMV_FLAG_NORM}",
        f"#define VINF_MK_FLAG_ARGMAX {QMV_FLAG_ARGMAX}",
        f"#define VINF_MK_FLAG_SILU_MUL {QMV_FLAG_SILU_MUL}",
    ])
    return "\n".join(lines) + "\n"
