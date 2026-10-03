from __future__ import annotations

import struct
from dataclasses import dataclass

from vinf.errors import UnsupportedModelError
from vinf.gguf.parser import GGUFFile, GGUFTensorInfo, GGUFTensorType


@dataclass(frozen=True, slots=True)
class DequantizedTensor:
    name: str
    values: list[float]
    shape: tuple[int, ...]


def dequantize_tensor(info: GGUFTensorInfo, data: memoryview) -> DequantizedTensor:
    if len(data) != info.nbytes:
        raise ValueError("tensor data length does not match GGUF tensor metadata")
    match info.tensor_type:
        case GGUFTensorType.F32:
            values = _dequant_f32(data, info.numel)
        case GGUFTensorType.F16:
            values = _dequant_f16(data, info.numel)
        case GGUFTensorType.Q4_0:
            values = _dequant_q4_0(data, info.numel)
        case GGUFTensorType.Q2_K:
            values = _dequant_q2_k(data, info.numel)
        case GGUFTensorType.Q3_K:
            values = _dequant_q3_k(data, info.numel)
        case GGUFTensorType.Q4_K:
            values = _dequant_q4_k(data, info.numel)
        case GGUFTensorType.Q5_K:
            values = _dequant_q5_k(data, info.numel)
        case GGUFTensorType.Q6_K:
            values = _dequant_q6_k(data, info.numel)
        case GGUFTensorType.Q8_0:
            values = _dequant_q8_0(data, info.numel)
        case GGUFTensorType.IQ4_NL:
            values = _dequant_iq4_nl(data, info.numel)
        case GGUFTensorType.IQ4_XS:
            values = _dequant_iq4_xs(data, info.numel)
        case GGUFTensorType.IQ3_S:
            values = _dequant_iq3_s(data, info.numel)
        case GGUFTensorType.IQ2_XXS:
            values = _dequant_iq2_xxs(data, info.numel)
        case GGUFTensorType.IQ2_XS:
            values = _dequant_iq2_xs(data, info.numel)
        case GGUFTensorType.IQ2_S:
            values = _dequant_iq2_s(data, info.numel)
        case GGUFTensorType.IQ3_XXS:
            values = _dequant_iq3_xxs(data, info.numel)
        case _:
            raise UnsupportedModelError(
                f"CPU dequantization for {info.tensor_type.name} is not implemented yet"
            )
    return DequantizedTensor(name=info.name, values=values, shape=info.dimensions)


def load_dequantized_tensor(gguf: GGUFFile, name: str) -> DequantizedTensor:
    file_obj, mm, view = gguf.mmap_tensor(name)
    try:
        return dequantize_tensor(gguf.tensors[name], view)
    finally:
        view.release()
        mm.close()
        file_obj.close()


def supported_cpu_dequant_types() -> set[GGUFTensorType]:
    return {
        GGUFTensorType.F32,
        GGUFTensorType.F16,
        GGUFTensorType.Q4_0,
        GGUFTensorType.Q2_K,
        GGUFTensorType.Q3_K,
        GGUFTensorType.Q4_K,
        GGUFTensorType.Q5_K,
        GGUFTensorType.Q6_K,
        GGUFTensorType.Q8_0,
        GGUFTensorType.IQ4_NL,
        GGUFTensorType.IQ4_XS,
        GGUFTensorType.IQ3_S,
        GGUFTensorType.IQ2_XXS,
        GGUFTensorType.IQ2_XS,
        GGUFTensorType.IQ2_S,
        GGUFTensorType.IQ3_XXS,
    }


def unsupported_tensor_types(tensors: list[GGUFTensorInfo]) -> set[GGUFTensorType]:
    supported = supported_cpu_dequant_types()
    return {tensor.tensor_type for tensor in tensors if tensor.tensor_type not in supported}


def _dequant_f32(data: memoryview, numel: int) -> list[float]:
    return [float(value[0]) for value in struct.iter_unpack("<f", data[: numel * 4])]


def _dequant_f16(data: memoryview, numel: int) -> list[float]:
    return [float(value[0]) for value in struct.iter_unpack("<e", data[: numel * 2])]


def _dequant_q4_0(data: memoryview, numel: int) -> list[float]:
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        d = float(struct.unpack_from("<e", data, offset)[0])
        offset += 2
        count = min(32, remaining)
        for idx in range(count):
            packed = data[offset + (idx % 16)]
            q = packed & 0x0F if idx < 16 else packed >> 4
            out.append(d * float(q - 8))
        offset += 16
        remaining -= count
    return out


def _dequant_q2_k(data: memoryview, numel: int) -> list[float]:
    # ggml block_q2_K: scales[16] (low nibble scale, high nibble min), qs[64], d, dmin (f16).
    # Element e uses qs[32*(e//128) + e%32] >> 2*((e%128)//32) & 3 and scales[e//16].
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        scales = data[offset : offset + 16]
        qs = data[offset + 16 : offset + 80]
        d = float(struct.unpack_from("<e", data, offset + 80)[0])
        dmin = float(struct.unpack_from("<e", data, offset + 82)[0])
        block_values = min(256, remaining)
        for e in range(block_values):
            sc = scales[e // 16]
            q = (qs[32 * (e // 128) + e % 32] >> (2 * ((e % 128) // 32))) & 0x03
            out.append(d * float(sc & 0x0F) * float(q) - dmin * float(sc >> 4))
        remaining -= block_values
        offset += 84
    return out


def _dequant_q3_k(data: memoryview, numel: int) -> list[float]:
    # ggml block_q3_K: hmask[32], qs[64], scales[12], d(f16). Element e of a block
    # reads 2 low bits from qs[32*(e//128) + e%32] at shift 2*((e%128)//32) and its
    # high bit from hmask[e%32] bit e//32 (bit clear => subtract 4).
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        hmask = data[offset : offset + 32]
        qs = data[offset + 32 : offset + 96]
        scales = _q3_k_scales(data[offset + 96 : offset + 108])
        d = float(struct.unpack_from("<e", data, offset + 108)[0])
        block_values = min(256, remaining)
        for e in range(block_values):
            low = (qs[32 * (e // 128) + e % 32] >> (2 * ((e % 128) // 32))) & 0x03
            high_bit = (hmask[e % 32] >> (e // 32)) & 0x01
            q = int(low) - (0 if high_bit else 4)
            out.append(d * float(scales[e // 16]) * float(q))
        remaining -= block_values
        offset += 110
    return out


def _q3_k_scales(scales: memoryview) -> list[int]:
    # 16 6-bit scales: low nibble from bytes[k % 8] (>> 4 for k >= 8),
    # high 2 bits from bytes[8 + k % 4] >> (2 * (k // 4)).
    out = []
    for k in range(16):
        low = (scales[k % 8] >> (4 * (k // 8))) & 0x0F
        high = (scales[8 + k % 4] >> (2 * (k // 4))) & 0x03
        out.append((low | (high << 4)) - 32)
    return out


def _dequant_q4_k(data: memoryview, numel: int) -> list[float]:
    # ggml block_q4_K: d, dmin (f16), scales[12], qs[128]. Groups 2p and 2p+1 are the
    # low and high nibbles of qs[32p : 32p + 32].
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        d = float(struct.unpack_from("<e", data, offset)[0])
        dmin = float(struct.unpack_from("<e", data, offset + 2)[0])
        scales = data[offset + 4 : offset + 16]
        qs = data[offset + 16 : offset + 144]
        block_values = min(256, remaining)
        for group in range(8):
            count = min(32, block_values - group * 32)
            if count <= 0:
                break
            scale, minimum = _q4_k_scale_min(scales, group)
            group_scale = d * float(scale)
            group_min = dmin * float(minimum)
            q_offset = (group // 2) * 32
            shift = 4 * (group % 2)
            for idx in range(count):
                q = (qs[q_offset + idx] >> shift) & 0x0F
                out.append(group_scale * float(q) - group_min)
        remaining -= block_values
        offset += 144
    return out


def _q4_k_scale_min(scales: memoryview, group: int) -> tuple[int, int]:
    if group < 4:
        return scales[group] & 0x3F, scales[group + 4] & 0x3F
    return (
        (scales[group + 4] & 0x0F) | ((scales[group - 4] >> 6) << 4),
        (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
    )


def _dequant_q5_k(data: memoryview, numel: int) -> list[float]:
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        d = float(struct.unpack_from("<e", data, offset)[0])
        dmin = float(struct.unpack_from("<e", data, offset + 2)[0])
        scales = data[offset + 4 : offset + 16]
        qh = data[offset + 16 : offset + 48]
        qs = data[offset + 48 : offset + 176]
        block_values = min(256, remaining)
        for pair in range(4):
            low_group = pair * 2
            high_group = low_group + 1
            low_scale, low_min = _q4_k_scale_min(scales, low_group)
            high_scale, high_min = _q4_k_scale_min(scales, high_group)
            low_d = d * float(low_scale)
            high_d = d * float(high_scale)
            low_m = dmin * float(low_min)
            high_m = dmin * float(high_min)
            high_bit_low = 1 << (pair * 2)
            high_bit_high = 1 << (pair * 2 + 1)
            q_offset = pair * 32
            for idx in range(32):
                if low_group * 32 + idx < block_values:
                    low_q = (qs[q_offset + idx] & 0x0F) + (
                        16 if qh[idx] & high_bit_low else 0
                    )
                    out.append(low_d * float(low_q) - low_m)
            for idx in range(32):
                if high_group * 32 + idx < block_values:
                    high_q = (qs[q_offset + idx] >> 4) + (
                        16 if qh[idx] & high_bit_high else 0
                    )
                    out.append(high_d * float(high_q) - high_m)
        remaining -= block_values
        offset += 176
    return out


def _dequant_q6_k(data: memoryview, numel: int) -> list[float]:
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        ql = data[offset : offset + 128]
        qh = data[offset + 128 : offset + 192]
        scales = struct.unpack_from("<16b", data, offset + 192)
        d = float(struct.unpack_from("<e", data, offset + 208)[0])
        block_values = min(256, remaining)
        flat_index = 0
        for chunk in range(2):
            ql_chunk = chunk * 64
            qh_chunk = chunk * 32
            for shift in (0, 4):
                for half in range(2):
                    ql_base = ql_chunk + half * 32
                    for idx in range(32):
                        if flat_index < block_values:
                            low = (ql[ql_base + idx] >> shift) & 0x0F
                            high = (qh[qh_chunk + idx] >> (shift + half * 2)) & 0x03
                            q = (low | (high << 4)) - 32
                            out.append(d * float(scales[flat_index // 16]) * float(q))
                        flat_index += 1
        remaining -= block_values
        offset += 210
    return out


def _dequant_q8_0(data: memoryview, numel: int) -> list[float]:
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        d = struct.unpack_from("<e", data, offset)[0]
        offset += 2
        count = min(32, remaining)
        qs = struct.unpack_from(f"<{count}b", data, offset)
        offset += 32
        out.extend(float(d) * float(q) for q in qs)
        remaining -= count
    return out


_IQ4_NL_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _dequant_iq4_nl(data: memoryview, numel: int) -> list[float]:
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        d = float(struct.unpack_from("<e", data, offset)[0])
        offset += 2
        count = min(32, remaining)
        for idx in range(count):
            packed = data[offset + (idx % 16)]
            q = packed & 0x0F if idx < 16 else packed >> 4
            out.append(d * float(_IQ4_NL_VALUES[q]))
        offset += 16
        remaining -= count
    return out


def _dequant_iq4_xs(data: memoryview, numel: int) -> list[float]:
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        d = float(struct.unpack_from("<e", data, offset)[0])
        scales_h = struct.unpack_from("<H", data, offset + 2)[0]
        scales_l = data[offset + 4 : offset + 8]
        qs = data[offset + 8 : offset + 136]
        block_values = min(256, remaining)
        for group in range(8):
            count = min(32, block_values - group * 32)
            if count <= 0:
                break
            low_byte = scales_l[group // 2]
            low = (low_byte >> (4 * (group % 2))) & 0x0F
            high = (scales_h >> (2 * group)) & 0x03
            group_scale = d * float(_i8(low | (high << 4)) - 32)
            q_offset = group * 16
            for idx in range(count):
                packed = qs[q_offset + (idx % 16)]
                q = packed & 0x0F if idx < 16 else packed >> 4
                out.append(group_scale * float(_IQ4_NL_VALUES[q]))
        remaining -= block_values
        offset += 136
    return out


def _dequant_iq3_s(data: memoryview, numel: int) -> list[float]:
    grid = _iq3_s_grid()
    out: list[float] = []
    offset = 0
    remaining = numel
    while remaining > 0:
        d = float(struct.unpack_from("<e", data, offset)[0])
        qs = data[offset + 2 : offset + 66]
        qh = data[offset + 66 : offset + 74]
        signs = data[offset + 74 : offset + 106]
        scales = data[offset + 106 : offset + 110]
        block_values = min(256, remaining)
        grid_values: list[float] = []
        for idx in range(64):
            high = (qh[idx // 8] >> (idx % 8)) & 0x01
            grid_values.extend(grid[qs[idx] | (high << 8)])
        for idx in range(block_values):
            scale_nibble = (scales[idx // 64] >> (4 * ((idx // 32) % 2))) & 0x0F
            scale = d * float(1 + 2 * scale_nibble)
            sign = -1.0 if (signs[idx // 8] >> (idx % 8)) & 0x01 else 1.0
            out.append(scale * grid_values[idx] * sign)
        remaining -= block_values
        offset += 110
    return out



_IQ3_S_GRID_HEX = (
    "000001000200050007001000110012001400160020002100250033004000420045004700510053006000620071007400"
    "770000010101020104011001110115012001230127013101350144016101650172010002010205020702100213021602"
    "210225023002340242024502470251025302700273020303110315032003220331033303360344035003520367037103"
    "750300041304170421042404320440044304510470040205040520052205260533054105450547056605730506061106"
    "130631065206710600070207040720072207260733075007540700100110021004101010111013101510171020102210"
    "311034103610541056106110721000110111031106111011141121113011331141115011521170117611001212121512"
    "171220122412321240124312551260127212011304130713101313132113271330133413411362137013031405141214"
    "141431143314421446145014541401151015131521153015321551152016241627164416461601170317101712172117"
    "351741176217701700200120032005200720102012201420162021202320272030203220412043204520502052206720"
    "702073207520002102211021132117212221252131213421422151210122042207222122232230223722412253225722"
    "712274220023022305231123222324233123332342235023662301240724202423243224352441247224752404251125"
    "222537254025532570250026022607262126552661260527112726273027432750270230113013301530173022303130"
    "333035304230443047305130633071300131033105311431213123314031603172317631003212322032323234325032"
    "013310331433213323332733303341334333473355337333033411341634223431345234603464340135103512352535"
    "323544355635733516364136013703372037223735370040044012402040244027403240414050407040024107411141"
    "134122413041354143415141554101420342104215422142334240425742624270420443114313432043224331433543"
    "004402442444374440447144054507452145624513463446604610471547304743475147025010501450225040504450"
    "475052506650745001510351055112512151325172510052115223523052365253520253075310532753445351536553"
    "735301540454205432544654125526555155535542560257045722571160136015603160336060600061206127616461"
    "126234624262556262627062006314632163406325644364626400650365346560650566406611671367007004700770"
    "207022703670407054706270027111712471437145710172047210721672217230725172027332733573537301740574"
    "13742074507422754275027631760077"
)
_IQ3_S_GRID: tuple[tuple[float, float, float, float], ...] | None = None


def _iq3_s_grid() -> tuple[tuple[float, float, float, float], ...]:
    global _IQ3_S_GRID
    if _IQ3_S_GRID is not None:
        return _IQ3_S_GRID
    # 512 entries x 4 values; each byte packs two 3-bit grid indices at bit 0 and bit 4.
    grid_map = (1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0)
    raw = bytes.fromhex(_IQ3_S_GRID_HEX)
    entries: list[tuple[float, float, float, float]] = []
    for byte_idx in range(0, len(raw), 2):
        lo, hi = raw[byte_idx], raw[byte_idx + 1]
        entries.append(
            (grid_map[lo & 0x07], grid_map[(lo >> 4) & 0x07], grid_map[hi & 0x07], grid_map[(hi >> 4) & 0x07])
        )
    _IQ3_S_GRID = tuple(entries)
    return _IQ3_S_GRID


def _i8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value >= 128 else value


# ---- IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS (ports of ggml dequantize_row_iq*; codebooks in iq_tables) ----


def _iq_tables():
    from vinf.gguf.iq_tables import iq_table

    return iq_table


def _signed(grid: bytes, entry: int, width: int, signs: int, first_bit: int = 0) -> list[float]:
    base = entry * width
    return [
        float(grid[base + j]) * (-1.0 if (signs >> (first_bit + j)) & 1 else 1.0) for j in range(width)
    ]


def _dequant_iq2_xxs(data: memoryview, numel: int) -> list[float]:
    t = _iq_tables()
    grid, ksigns = t("iq2xxs_grid"), t("ksigns_iq2xs")
    out: list[float] = []
    for b in range(numel // 256):
        blk = data[b * 66 : (b + 1) * 66]
        d = float(struct.unpack_from("<e", blk, 0)[0])
        for ib32 in range(8):
            aux0, aux1 = struct.unpack_from("<II", blk, 2 + 8 * ib32)
            db = d * (0.5 + (aux1 >> 28)) * 0.25
            for l in range(4):
                signs = ksigns[(aux1 >> (7 * l)) & 127]
                out.extend(db * v for v in _signed(grid, (aux0 >> (8 * l)) & 0xFF, 8, signs))
    return out


def _dequant_iq2_xs(data: memoryview, numel: int) -> list[float]:
    t = _iq_tables()
    grid, ksigns = t("iq2xs_grid"), t("ksigns_iq2xs")
    out: list[float] = []
    for b in range(numel // 256):
        blk = data[b * 74 : (b + 1) * 74]
        d = float(struct.unpack_from("<e", blk, 0)[0])
        for ib32 in range(8):
            sc = blk[66 + ib32]
            db = (d * (0.5 + (sc & 0xF)) * 0.25, d * (0.5 + (sc >> 4)) * 0.25)
            for l in range(4):
                (q,) = struct.unpack_from("<H", blk, 2 + 2 * (4 * ib32 + l))
                out.extend(db[l // 2] * v for v in _signed(grid, q & 511, 8, ksigns[q >> 9]))
    return out


def _dequant_iq2_s(data: memoryview, numel: int) -> list[float]:
    grid = _iq_tables()("iq2s_grid")
    out: list[float] = []
    for b in range(numel // 256):
        blk = data[b * 82 : (b + 1) * 82]
        d = float(struct.unpack_from("<e", blk, 0)[0])
        for ib32 in range(8):
            sc = blk[74 + ib32]
            db = (d * (0.5 + (sc & 0xF)) * 0.25, d * (0.5 + (sc >> 4)) * 0.25)
            qh = blk[66 + ib32]
            for l in range(4):
                entry = blk[2 + 4 * ib32 + l] | ((qh << (8 - 2 * l)) & 0x300)
                out.extend(db[l // 2] * v for v in _signed(grid, entry, 8, blk[2 + 32 + 4 * ib32 + l]))
    return out


def _dequant_iq3_xxs(data: memoryview, numel: int) -> list[float]:
    t = _iq_tables()
    grid, ksigns = t("iq3xxs_grid"), t("ksigns_iq2xs")
    out: list[float] = []
    for b in range(numel // 256):
        blk = data[b * 98 : (b + 1) * 98]
        d = float(struct.unpack_from("<e", blk, 0)[0])
        for ib32 in range(8):
            (aux,) = struct.unpack_from("<I", blk, 2 + 64 + 4 * ib32)
            db = d * (0.5 + (aux >> 28)) * 0.5
            for l in range(4):
                signs = ksigns[(aux >> (7 * l)) & 127]
                out.extend(db * v for v in _signed(grid, blk[2 + 8 * ib32 + 2 * l], 4, signs, 0))
                out.extend(db * v for v in _signed(grid, blk[2 + 8 * ib32 + 2 * l + 1], 4, signs, 4))
    return out
