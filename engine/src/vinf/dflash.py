"""DFlash / DFlash 2 block-diffusion drafter for qwen35 speculative decoding.

The drafter is a small Qwen3-style transformer conditioned on target-model features:
  context:  for every verified target position p, the target's hidden rows after layers
            `target_layer_ids` are concatenated, fused by `fc` + `hidden_norm`, and projected into
            each draft layer's K/V cache at position p ("KV injection", k_norm + RoPE applied).
  drafting: a block [x_P, mask, ..., mask] at positions P..P+k (x_P = last emitted token) runs
            through the draft layers; block queries attend to the injected context and to each
            other bidirectionally (sliding window); the draft `norm` + the target's LM head give
            the drafts for block rows 1..k in one pass.
Embedding table and LM head are shared with the target model.

DFlash 2 (z-lab/dflash `DFlash2DraftModel`) adds, per layer, two grouped dynamic causal
convolutions over block positions (`attention_conv` around attention, `mlp_conv` around the MLP:
`prepare` mixes the normed input, `finish` mixes the sub-block output, both with per-channel base
taps plus per-group taps projected from the normed input), and a candidate selector that walks the
block greedily: score = top-k logit + <pred_codebook[prev] * hidden_projection(h), succ_codebook[cand]>.
The always-full block (block_size tokens) matters: block attention is bidirectional.
"""

from __future__ import annotations

import json
import math
from array import array
from dataclasses import dataclass
from pathlib import Path

from vinf.errors import ConfigurationError, UnsupportedModelError
from vinf.gguf.parser import GGUFTensorType
from vinf.gguf.qwen_tensors import load_qwen_lm_head_rows, load_qwen_token_embeddings
from vinf.safetensors import SafeTensorsCheckpoint

LINEAR = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
LAYER_NORMS = ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")
CONVS = ("attention_conv", "mlp_conv")  # DFlash 2
IGNORED_SHARED = ("embed_tokens.weight", "lm_head.weight")  # shared with the target model


@dataclass(frozen=True, slots=True)
class DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int  # 0 = full attention
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    vocab_size: int
    version: int  # 1 = DFlash, 2 = DFlash 2
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_top_k: int = 0
    selector_rank: int = 0
    input_embedding_scale: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict) -> "DFlashConfig":
        dcfg = cfg.get("dflash_config", {})
        rope = cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta", 10000.0))
        window = cfg.get("sliding_window") if cfg.get("use_sliding_window", False) else 0
        arch = " ".join(cfg.get("architectures", []))
        try:
            return cls(
                hidden_size=int(cfg["hidden_size"]),
                intermediate_size=int(cfg["intermediate_size"]),
                num_layers=int(cfg["num_hidden_layers"]),
                num_heads=int(cfg["num_attention_heads"]),
                num_kv_heads=int(cfg["num_key_value_heads"]),
                head_dim=int(cfg.get("head_dim", int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]))),
                rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
                rope_theta=float(rope),
                sliding_window=int(window or 0),
                block_size=int(dcfg.get("block_size", cfg.get("block_size", 16))),
                mask_token_id=int(dcfg["mask_token_id"]),
                target_layer_ids=tuple(int(i) for i in dcfg["target_layer_ids"]),
                vocab_size=int(cfg["vocab_size"]),
                version=2 if "DFlash2" in arch or "conv_kernel_size" in dcfg else 1,
                conv_kernel_size=int(dcfg.get("conv_kernel_size", 0)),
                conv_group_size=int(dcfg.get("conv_group_size", 0)),
                selector_top_k=int(dcfg.get("selector_top_k", 0)),
                selector_rank=int(dcfg.get("selector_rank", 0)),
                input_embedding_scale=float(dcfg.get("input_embedding_scale", cfg.get("input_embedding_scale", 1.0))),
            )
        except KeyError as exc:
            raise UnsupportedModelError(f"DFlash config is missing {exc}") from exc

    @classmethod
    def load(cls, directory: str | Path) -> "DFlashConfig":
        return cls.from_dict(json.loads((Path(directory) / "config.json").read_text()))


def dflash_tensor_names(config: DFlashConfig) -> tuple[list[str], list[str]]:
    """(GPU matrices, GPU F32 vectors) the drafter needs, without any checkpoint prefix."""
    mats = ["fc.weight"] + [f"layers.{i}.{n}.weight" for i in range(config.num_layers) for n in LINEAR]
    vecs = ["hidden_norm.weight", "norm.weight"] + [
        f"layers.{i}.{n}.weight" for i in range(config.num_layers) for n in LAYER_NORMS
    ]
    if config.version == 2:
        mats += [f"layers.{i}.{c}.kernel_projection.weight" for i in range(config.num_layers) for c in CONVS]
        mats += ["candidate_selector.hidden_projection.weight"]
        vecs += [f"layers.{i}.{c}.base_kernel" for i in range(config.num_layers) for c in CONVS]
    return mats, vecs


def dflash_cpu_tensor_names(config: DFlashConfig) -> list[str]:
    """Selector codebooks, read in place on the CPU (only ~17 rows per block position are used)."""
    if config.version != 2:
        return []
    return ["candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"]


def _resolve(ckpt: SafeTensorsCheckpoint, names: list[str]) -> str:
    """Common prefix ("" or "model.") under which the checkpoint stores the drafter tensors."""
    for prefix in ("", "model."):
        if all(prefix + n in ckpt for n in names):
            return prefix
    missing = [n for n in names if n not in ckpt and "model." + n not in ckpt]
    raise UnsupportedModelError("DFlash checkpoint is missing tensors: " + ", ".join(missing[:8]))


class DFlashCheckpoint:
    def __init__(self, directory: str | Path, *, allow_unimplemented: bool = False) -> None:
        self.config = DFlashConfig.load(directory)
        self.ckpt = SafeTensorsCheckpoint(directory)
        mats, vecs = dflash_tensor_names(self.config)
        cpu = dflash_cpu_tensor_names(self.config)
        self.prefix = _resolve(self.ckpt, mats + vecs + cpu)
        known = {self.prefix + n for n in mats + vecs + cpu} | {self.prefix + n for n in IGNORED_SHARED} | set(IGNORED_SHARED)
        self.unimplemented = sorted(n for n in self.ckpt.tensors if n not in known)
        if self.unimplemented and not allow_unimplemented:
            raise UnsupportedModelError(
                "DFlash checkpoint has tensors this engine does not implement yet (DFlash 2 dynamic "
                "convolution / candidate selector?): " + ", ".join(self.unimplemented[:8])
                + ("" if len(self.unimplemented) <= 8 else f" ... ({len(self.unimplemented)} total)")
                + "; pass allow_unimplemented=True to draft with the plain block backbone"
            )
        c = self.config
        expect = {"fc.weight": (c.hidden_size, len(c.target_layer_ids) * c.hidden_size)}
        for i in range(c.num_layers):
            p = f"layers.{i}."
            expect[p + "self_attn.q_proj.weight"] = (c.num_heads * c.head_dim, c.hidden_size)
            expect[p + "self_attn.k_proj.weight"] = (c.num_kv_heads * c.head_dim, c.hidden_size)
            expect[p + "self_attn.v_proj.weight"] = (c.num_kv_heads * c.head_dim, c.hidden_size)
            expect[p + "self_attn.o_proj.weight"] = (c.hidden_size, c.num_heads * c.head_dim)
            expect[p + "mlp.gate_proj.weight"] = (c.intermediate_size, c.hidden_size)
            expect[p + "mlp.up_proj.weight"] = (c.intermediate_size, c.hidden_size)
            expect[p + "mlp.down_proj.weight"] = (c.hidden_size, c.intermediate_size)
            if c.version == 2:
                groups = c.hidden_size // c.conv_group_size
                for conv in CONVS:
                    expect[p + conv + ".base_kernel"] = (2, c.conv_kernel_size, c.hidden_size)
                    expect[p + conv + ".kernel_projection.weight"] = (2 * c.conv_kernel_size * groups, c.hidden_size)
        if c.version == 2:
            expect["candidate_selector.hidden_projection.weight"] = (c.selector_rank, c.hidden_size)
            expect["candidate_selector.predecessor_codebook"] = (c.vocab_size, c.selector_rank)
            expect["candidate_selector.successor_codebook"] = (c.vocab_size, c.selector_rank)
        for name, shape in expect.items():
            got = self.ckpt[self.prefix + name].shape
            if tuple(got) != shape:
                raise UnsupportedModelError(f"DFlash {name}: shape {got}, expected {shape}")

    def tensor(self, name: str):
        return self.ckpt[self.prefix + name]

    def floats(self, name: str) -> list[float]:
        from vinf import _cpu_qwen

        t = self.tensor(name)
        out = array("f")
        out.frombytes(_cpu_qwen.to_f32(t.data, t.dtype_code))
        return out.tolist()

    def gpu_bytes(self, max_context: int, quantize: bool = True) -> int:
        """VRAM for weights, KV caches, and scratch buffers (selector codebooks stay on the CPU)."""
        c = self.config
        mats, vecs = dflash_tensor_names(c)
        weights = sum(
            (self.tensor(n).numel // 32 * 34 if quantize else self.tensor(n).numel * 4) + 256 for n in mats
        ) + sum(self.tensor(n).numel * 4 + 256 for n in vecs)
        kv = 2 * c.num_layers * c.num_kv_heads * max_context * c.head_dim * 4
        rows = max(c.block_size, 8)
        scratch = rows * 4 * (6 * c.hidden_size + 2 * c.num_heads * c.head_dim + 2 * c.num_kv_heads * c.head_dim
                              + 2 * c.intermediate_size)
        return weights + kv + scratch + 16 * 1024**2


class DFlashDrafter:
    """GPU DFlash drafter bound to a `QwenGpuExecutor` (created with capture_layers=target_layer_ids)."""

    def __init__(self, executor, checkpoint: DFlashCheckpoint, *, quantize: bool = True) -> None:
        c = checkpoint.config
        self.ex, self.ck, self.config = executor, checkpoint, c
        s = executor.shapes
        if c.hidden_size != s.hidden:
            raise UnsupportedModelError(f"DFlash hidden size {c.hidden_size} != target hidden size {s.hidden}")
        if c.vocab_size != s.vocab:
            raise UnsupportedModelError(f"DFlash vocab {c.vocab_size} != target vocab {s.vocab}")
        if tuple(executor.capture_layers) != c.target_layer_ids:
            raise ConfigurationError(f"executor must capture target layers {c.target_layer_ids}")
        if c.mask_token_id >= s.vocab:
            raise UnsupportedModelError("DFlash mask token id is outside the target vocabulary")
        self.max_draft = c.block_size - 1  # verification needs max_batch >= k + 1 (checked by the decoder)
        if c.block_size > executor.max_batch:
            raise ConfigurationError(f"DFlash block size {c.block_size} needs executor max_batch >= {c.block_size}")
        rt = executor.rt
        mats, vecs = dflash_tensor_names(c)
        from vinf import _cpu_qwen

        for name in mats:
            t = checkpoint.tensor(name)
            rows, cols = t.shape
            if quantize:
                rt.upload_raw("dfl." + name, _cpu_qwen.quantize_q8_0(t.data, t.dtype_code), GGUFTensorType.Q8_0, cols, rows)
            else:
                rt.upload_raw("dfl." + name, _cpu_qwen.to_f32(t.data, t.dtype_code), GGUFTensorType.F32, cols, rows)
        for name in vecs:
            t = checkpoint.tensor(name)
            rt.upload_raw("dfl." + name, _cpu_qwen.to_f32(t.data, t.dtype_code), GGUFTensorType.F32, t.numel, 1)
        rows = max(c.block_size, executor.max_batch)
        q, kv = c.num_heads * c.head_dim, c.num_kv_heads * c.head_dim
        bufs = {"dfl_h": c.hidden_size, "dfl_xn": c.hidden_size, "dfl_proj": c.hidden_size,
                "dfl_ctx": c.hidden_size, "dfl_out": c.hidden_size, "dfl_q": q, "dfl_attn": q,
                "dfl_k": kv, "dfl_v": kv, "dfl_gate": c.intermediate_size, "dfl_up": c.intermediate_size}
        if c.version == 2:
            bufs.update({"dfl_dyn": 2 * c.conv_kernel_size * (c.hidden_size // c.conv_group_size),
                         "dfl_conv": c.hidden_size, "dfl_sel": c.selector_rank})
        for name, n in bufs.items():
            rt.alloc(name, n * rows)
        self._codebooks = None
        if c.version == 2:
            pc = checkpoint.tensor("candidate_selector.predecessor_codebook")
            sc = checkpoint.tensor("candidate_selector.successor_codebook")
            if pc.dtype != sc.dtype:
                raise UnsupportedModelError("DFlash 2 selector codebooks must share a dtype")
            self._codebooks = (pc.data, sc.data, pc.dtype_code)
        for i in range(c.num_layers):
            rt.alloc(f"dfl_kc.{i}", c.num_kv_heads * executor.max_context * c.head_dim)
            rt.alloc(f"dfl_vc.{i}", c.num_kv_heads * executor.max_context * c.head_dim)
        scale = c.input_embedding_scale
        self._mask_row = [v * scale for v in load_qwen_token_embeddings(executor.gguf, [c.mask_token_id])[0]]

    # ---- drafter interface (see qwen_speculative) -------------------------------------------

    def reset(self) -> None:
        pass  # context K/V is position-indexed; stale entries are overwritten before they are read

    def add_context(self, position: int, n: int) -> None:
        """Inject the n captured target feature rows in "cap_feat" as context at [position, position + n)."""
        rt, c = self.ex.rt, self.config
        if position + n > self.ex.max_context:
            raise ConfigurationError("DFlash context exceeds max_context")
        rt.qmv("dfl.fc.weight", "cap_feat", "dfl_ctx", n)
        rt.rmsnorm("dfl_ctx", "dfl.hidden_norm.weight", "dfl_ctx", c.hidden_size, c.rms_norm_eps, n)
        for i in range(c.num_layers):
            p = f"dfl.layers.{i}.self_attn."
            rt.qmv(p + "k_proj.weight", "dfl_ctx", "dfl_k", n)
            rt.qmv(p + "v_proj.weight", "dfl_ctx", "dfl_v", n)
            rt.rmsnorm("dfl_k", p + "k_norm.weight", "dfl_k", c.head_dim, c.rms_norm_eps, c.num_kv_heads * n)
            rt.rope("dfl_k", c.num_kv_heads, c.head_dim, c.head_dim, position, c.rope_theta, n)
            rt.kv_append("dfl_k", "dfl_v", f"dfl_kc.{i}", f"dfl_vc.{i}", c.num_kv_heads, self.ex.max_context,
                         c.head_dim, position, n)

    def _conv_prepare(self, layer: int, conv: str, b: int) -> None:
        """DFlash 2: dynamic taps from the normed rows in "dfl_xn"; mix "dfl_xn" in place (via dfl_conv)."""
        c, rt = self.config, self.ex.rt
        p = f"dfl.layers.{layer}.{conv}."
        rt.qmv(p + "kernel_projection.weight", "dfl_xn", "dfl_dyn", b)
        rt.dyn_conv("dfl_xn", "dfl_dyn", p + "base_kernel", "dfl_conv", b, c.hidden_size, c.conv_kernel_size,
                    c.conv_group_size, 0)
        rt.copy("dfl_xn", 0, "dfl_conv", 0, b * c.hidden_size)

    def _conv_finish(self, layer: int, conv: str, b: int) -> None:
        """DFlash 2: mix the sub-block output in "dfl_proj" with the second kernel half."""
        c, rt = self.config, self.ex.rt
        rt.dyn_conv("dfl_proj", "dfl_dyn", f"dfl.layers.{layer}.{conv}.base_kernel", "dfl_conv", b, c.hidden_size,
                    c.conv_kernel_size, c.conv_group_size, 1)
        rt.copy("dfl_proj", 0, "dfl_conv", 0, b * c.hidden_size)

    def draft_block(self, last_token: int, position: int, k: int) -> list[int]:
        """Draft k tokens following last_token (at `position`). The block always spans block_size
        positions (bidirectional block attention, as trained); the first k drafts are returned."""
        if not 1 <= k <= self.max_draft:
            raise ConfigurationError(f"DFlash can draft 1..{self.max_draft} tokens per block")
        rt, c, ex = self.ex.rt, self.config, self.ex
        b = min(c.block_size, ex.max_context - position)
        if b < k + 1:
            raise ConfigurationError("DFlash block exceeds max_context")
        v2 = c.version == 2
        first = [v * c.input_embedding_scale for v in load_qwen_token_embeddings(ex.gguf, [last_token])[0]]
        rt.write("dfl_h", array("f", first + self._mask_row * (b - 1)).tobytes())
        eps = c.rms_norm_eps
        for i in range(c.num_layers):
            p = f"dfl.layers.{i}."
            a = p + "self_attn."
            rt.rmsnorm("dfl_h", p + "input_layernorm.weight", "dfl_xn", c.hidden_size, eps, b)
            if v2:
                self._conv_prepare(i, "attention_conv", b)
            rt.qmv(a + "q_proj.weight", "dfl_xn", "dfl_q", b)
            rt.qmv(a + "k_proj.weight", "dfl_xn", "dfl_k", b)
            rt.qmv(a + "v_proj.weight", "dfl_xn", "dfl_v", b)
            rt.rmsnorm("dfl_q", a + "q_norm.weight", "dfl_q", c.head_dim, eps, c.num_heads * b)
            rt.rmsnorm("dfl_k", a + "k_norm.weight", "dfl_k", c.head_dim, eps, c.num_kv_heads * b)
            rt.rope("dfl_q", c.num_heads, c.head_dim, c.head_dim, position, c.rope_theta, b)
            rt.rope("dfl_k", c.num_kv_heads, c.head_dim, c.head_dim, position, c.rope_theta, b)
            rt.kv_append("dfl_k", "dfl_v", f"dfl_kc.{i}", f"dfl_vc.{i}", c.num_kv_heads, ex.max_context,
                         c.head_dim, position, b)
            rt.attention("dfl_q", f"dfl_kc.{i}", f"dfl_vc.{i}", "dfl_attn", c.num_heads, c.num_kv_heads, c.head_dim,
                         ex.max_context, position + 1, b, 1, c.sliding_window)
            rt.qmv(a + "o_proj.weight", "dfl_attn", "dfl_proj", b)
            if v2:
                self._conv_finish(i, "attention_conv", b)
            rt.add("dfl_h", "dfl_proj", "dfl_h", c.hidden_size * b)
            rt.rmsnorm("dfl_h", p + "post_attention_layernorm.weight", "dfl_xn", c.hidden_size, eps, b)
            if v2:
                self._conv_prepare(i, "mlp_conv", b)
            rt.qmv(p + "mlp.gate_proj.weight", "dfl_xn", "dfl_gate", b)
            rt.qmv(p + "mlp.up_proj.weight", "dfl_xn", "dfl_up", b)
            rt.silu_mul("dfl_gate", "dfl_up", "dfl_gate", c.intermediate_size * b)
            rt.qmv(p + "mlp.down_proj.weight", "dfl_gate", "dfl_proj", b)
            if v2:
                self._conv_finish(i, "mlp_conv", b)
            rt.add("dfl_h", "dfl_proj", "dfl_h", c.hidden_size * b)
        rt.rmsnorm("dfl_h", "dfl.norm.weight", "dfl_xn", c.hidden_size, eps, b)
        n = b - 1  # draft rows 1..b-1
        rt.copy("dfl_out", 0, "dfl_xn", c.hidden_size, n * c.hidden_size)
        rt.qmv("output.weight", "dfl_out", "logits", n)
        if not v2:
            return rt.argmax_rows("logits", ex.shapes.vocab, n)[:k]
        return self._select(last_token, n)[:k]

    def _select(self, anchor: int, n: int) -> list[int]:
        """DFlash 2 candidate selector (greedy): walk the block choosing among the top-k logits by
        logit + <pred_codebook[prev] * hidden_projection(h_t), succ_codebook[cand]>."""
        from vinf import _cpu_qwen

        c, rt = self.config, self.ex.rt
        idx, val = rt.topk_rows("logits", self.ex.shapes.vocab, n, c.selector_top_k)
        rt.qmv("dfl.candidate_selector.hidden_projection.weight", "dfl_out", "dfl_sel", n)
        proj = rt.read_floats("dfl_sel", n * c.selector_rank)
        pc, sc, dtype = self._codebooks
        path, prev = [], anchor
        kk = c.selector_top_k
        for t in range(n):
            cands = idx[t * kk:(t + 1) * kk]
            h = array("f", proj[t * c.selector_rank:(t + 1) * c.selector_rank]).tobytes()
            pair = _cpu_qwen.bilinear_scores(pc, sc, dtype, c.selector_rank, prev, h, cands)
            scores = [u + v for u, v in zip(val[t * kk:(t + 1) * kk], pair)]
            best = max(range(kk), key=lambda j: (scores[j], -cands[j]))
            prev = cands[best]
            path.append(prev)
        return path

    # speculative-decoder hooks
    def observe_prefill(self, start: int, chunk: list[int], prompt: list[int]) -> None:
        self.add_context(start, len(chunk))

    def draft(self, last_token: int, position: int, k: int) -> list[int]:
        return self.draft_block(last_token, position, k)

    def observe_verify(self, position: int, keep: int, emitted: list[int]) -> None:
        self.add_context(position, keep)


def dflash_reference_draft(
    ck: DFlashCheckpoint,
    gguf,
    features: list[list[float]],
    last_token: int,
    position: int,
    k: int,
) -> tuple[list[int], list[list[float]]]:
    """Pure-Python reference of DFlash / DFlash 2 drafting (tiny shapes only), following
    z-lab/dflash: features[p] is the concatenated target feature row of context position p
    (len(features) == position). Returns the first k drafts and the post-norm draft rows."""
    from vinf.qwen_ops import matvec_reference, qwen_apply_rope, rmsnorm_reference, silu

    c = ck.config
    h, hd, eps = c.hidden_size, c.head_dim, c.rms_norm_eps
    v2 = c.version == 2
    b = min(c.block_size, len(features) + c.block_size)  # full block
    fc = ck.floats("fc.weight")
    hidden_norm = ck.floats("hidden_norm.weight")

    def rope(vec: list[float], pos: int) -> list[float]:
        freqs = [pos / (c.rope_theta ** (2.0 * j / hd)) for j in range(hd // 2)]
        cos = [math.cos(f) for f in freqs] * 2
        sin = [math.sin(f) for f in freqs] * 2
        return qwen_apply_rope(vec, cos, sin)

    def heads(vec: list[float], n: int) -> list[list[float]]:
        return [vec[i * hd:(i + 1) * hd] for i in range(n)]

    def conv(rows: list[list[float]], dyn: list[list[float]], base: list[float], sel: int) -> list[list[float]]:
        kk, grp = c.conv_kernel_size, c.conv_group_size
        groups = h // grp
        out = []
        for t in range(len(rows)):
            row = [0.0] * h
            for o in range(kk):
                if t - o < 0:
                    continue
                for ch in range(h):
                    tap = base[(sel * kk + o) * h + ch] + dyn[t][(sel * kk + o) * groups + ch // grp]
                    row[ch] += tap * rows[t - o][ch]
            out.append(row)
        return out

    ctx = [rmsnorm_reference(matvec_reference(f, fc, h, len(f)), hidden_norm, eps) for f in features]
    scale = c.input_embedding_scale
    block = [[v * scale for v in row] for row in load_qwen_token_embeddings(gguf, [last_token] + [c.mask_token_id] * (b - 1))]
    for i in range(c.num_layers):
        p = f"layers.{i}."
        w = {n: ck.floats(p + n + ".weight") for n in LINEAR + LAYER_NORMS}
        qn, kn = w["self_attn.q_norm"], w["self_attn.k_norm"]
        groups2k = 2 * c.conv_kernel_size * (h // c.conv_group_size) if v2 else 0
        cw = {cv: (ck.floats(p + cv + ".kernel_projection.weight"), ck.floats(p + cv + ".base_kernel"))
              for cv in CONVS} if v2 else {}

        def keys_values(x: list[float], pos: int):
            kk_ = [rope(rmsnorm_reference(hv, kn, eps), pos)
                   for hv in heads(matvec_reference(x, w["self_attn.k_proj"], c.num_kv_heads * hd, h), c.num_kv_heads)]
            vv = heads(matvec_reference(x, w["self_attn.v_proj"], c.num_kv_heads * hd, h), c.num_kv_heads)
            return kk_, vv

        keys, values = [], []
        for pos, x in enumerate(ctx):
            kk_, vv = keys_values(x, pos)
            keys.append(kk_)
            values.append(vv)
        xn_rows = [rmsnorm_reference(x, w["input_layernorm"], eps) for x in block]
        if v2:
            kp, base = cw["attention_conv"]
            dyn_a = [matvec_reference(x, kp, groups2k, h) for x in xn_rows]
            xn_rows = conv(xn_rows, dyn_a, base, 0)
        for t, xn in enumerate(xn_rows):
            kk_, vv = keys_values(xn, position + t)
            keys.append(kk_)
            values.append(vv)
        total = position + b
        attn_rows = []
        for t, xn in enumerate(xn_rows):
            qpos1 = position + t + 1
            start = qpos1 - c.sliding_window if c.sliding_window and qpos1 > c.sliding_window else 0
            qs = [rope(rmsnorm_reference(qv, qn, eps), position + t)
                  for qv in heads(matvec_reference(xn, w["self_attn.q_proj"], c.num_heads * hd, h), c.num_heads)]
            attn = []
            for head_idx, qv in enumerate(qs):
                g = head_idx // (c.num_heads // c.num_kv_heads)
                scores = [sum(a * bb for a, bb in zip(qv, keys[pos][g])) / math.sqrt(hd) for pos in range(start, total)]
                m = max(scores)
                probs = [math.exp(sc - m) for sc in scores]
                den = sum(probs)
                attn.extend(sum(pr * values[start + j][g][d] for j, pr in enumerate(probs)) / den for d in range(hd))
            attn_rows.append(matvec_reference(attn, w["self_attn.o_proj"], h, c.num_heads * hd))
        if v2:
            attn_rows = conv(attn_rows, dyn_a, cw["attention_conv"][1], 1)
        block = [[a + bb for a, bb in zip(x, o)] for x, o in zip(block, attn_rows)]
        mx_rows = [rmsnorm_reference(x, w["post_attention_layernorm"], eps) for x in block]
        if v2:
            kp, base = cw["mlp_conv"]
            dyn_m = [matvec_reference(x, kp, groups2k, h) for x in mx_rows]
            mx_rows = conv(mx_rows, dyn_m, base, 0)
        mlp_rows = []
        for mx in mx_rows:
            gate = matvec_reference(mx, w["mlp.gate_proj"], c.intermediate_size, h)
            up = matvec_reference(mx, w["mlp.up_proj"], c.intermediate_size, h)
            mlp_rows.append(matvec_reference([silu(a) * bb for a, bb in zip(gate, up)], w["mlp.down_proj"], h, c.intermediate_size))
        if v2:
            mlp_rows = conv(mlp_rows, dyn_m, cw["mlp_conv"][1], 1)
        block = [[a + bb for a, bb in zip(x, o)] for x, o in zip(block, mlp_rows)]
    norm = ck.floats("norm.weight")
    outs = [rmsnorm_reference(x, norm, eps) for x in block[1:]]
    lm = load_qwen_lm_head_rows(gguf, list(range(c.vocab_size)))
    logits = [[sum(a * bb for a, bb in zip(row, o)) for row in lm] for o in outs]
    if not v2:
        return [max(range(len(lg)), key=lambda j: lg[j]) for lg in logits][:k], outs
    hp = ck.floats("candidate_selector.hidden_projection.weight")
    pcb = ck.floats("candidate_selector.predecessor_codebook")
    scb = ck.floats("candidate_selector.successor_codebook")
    r = c.selector_rank
    path, prev = [], last_token
    for t, lg in enumerate(logits):
        cands = sorted(range(len(lg)), key=lambda j: (-lg[j], j))[: c.selector_top_k]
        proj = matvec_reference(outs[t], hp, r, h)
        def score(j):
            return lg[j] + sum(pcb[prev * r + q] * proj[q] * scb[j * r + q] for q in range(r))
        prev = max(cands, key=lambda j: (score(j), -j))
        path.append(prev)
    return path[:k], outs
