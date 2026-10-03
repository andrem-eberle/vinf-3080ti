from __future__ import annotations

import os

import math
import struct
import tempfile
import unittest
from pathlib import Path

from tests.test_phase5_gguf import write_tiny_gguf
from vinf.errors import UnsupportedModelError
from vinf.gguf.parser import (
    GGUFFile,
    GGUFMetadataValue,
    GGUFTensorInfo,
    GGUFTensorType,
    GGUFValueType,
    load_gguf,
)
from vinf.gguf.qwen_tensors import (
    load_qwen_full_attention_layer_weights,
    load_qwen_linear_attention_layer_weights,
    qwen_layer_schedule,
)
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.qwen_ops import (
    QwenAttentionWeights,
    QwenLayerKind,
    QwenLayerWeights,
    QwenLinearAttentionCache,
    QwenLinearAttentionConfig,
    QwenLinearAttentionWeights,
    QwenMLPWeights,
    QwenRopeConfig,
    qwen_apply_rope,
    qwen_empty_kv_cache,
    qwen_full_attention_layer,
    qwen_empty_linear_attention_cache,
    qwen_gqa_attention,
    qwen_layer_kinds_from_interval,
    qwen_linear_attention_config_from_metadata,
    qwen_linear_attention_layer,
    qwen_lm_head,
    qwen_mlp,
    qwen_rmsnorm,
    qwen_rope_config_from_gguf,
    qwen_rope_frequencies,
)
from vinf.qwen_runtime import (
    assert_qwen_decode_supported,
    load_qwen_baseline_weights,
    qwen_decode_next_token_reference,
    qwen_greedy_next_token_reference,
    qwen_prefill_tokens_reference,
    qwen_prompt_logits_reference,
)
from vinf.reference_ops import lm_head_reference, mlp_reference, rmsnorm_reference


def metadata() -> ModelMetadata:
    return ModelMetadata(
        architecture=ModelArchitecture.QWEN35,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        hidden_size=4,
        intermediate_size=8,
        head_dim=2,
        vocab_size=3,
        max_position_embeddings=4,
        dtype="fp16",
    )


def weights() -> QwenLayerWeights:
    identity4 = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    k_v = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ]
    gate = [((i % 5) - 2) / 5.0 for i in range(8 * 4)]
    up = list(reversed(gate))
    down = [((i % 7) - 3) / 7.0 for i in range(4 * 8)]
    return QwenLayerWeights(
        attn_norm=[1.0, 1.1, 0.9, 1.0],
        attention=QwenAttentionWeights(
            q_proj=identity4,
            k_proj=k_v,
            v_proj=k_v,
            o_proj=identity4,
        ),
        post_attention_norm=[1.0, 1.0, 1.0, 1.0],
        mlp=QwenMLPWeights(gate_proj=gate, up_proj=up, down_proj=down),
    )


class Phase29QwenOpsTests(unittest.TestCase):
    def assert_close_lists(self, actual, expected, tol=1e-6) -> None:
        self.assertEqual(len(actual), len(expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isclose(a, e, rel_tol=tol, abs_tol=tol), (idx, a, e))

    def test_qwen_rmsnorm_matches_reference(self) -> None:
        values = [1.0, -2.0, 3.0, -4.0]
        norm = [1.0, 1.1, 0.9, 1.0]
        self.assert_close_lists(qwen_rmsnorm(values, norm), rmsnorm_reference(values, norm))

    def test_qwen_rope_frequencies_and_rotation(self) -> None:
        config = QwenRopeConfig(head_dim=4, freq_base=10000.0, dimension_sections=(1, 1, 0, 0))
        cos, sin = qwen_rope_frequencies(config, position=1)
        self.assertEqual(len(cos), 4)
        rotated = qwen_apply_rope([1.0, 2.0, 3.0, 4.0], cos, sin)
        self.assert_close_lists(
            rotated,
            [
                1.0 * cos[0] - 3.0 * sin[0],
                2.0 * cos[1] - 4.0 * sin[1],
                3.0 * cos[0] + 1.0 * sin[0],
                4.0 * cos[1] + 2.0 * sin[1],
            ],
        )

    def test_qwen_rope_config_from_gguf_metadata(self) -> None:
        gguf = load_gguf(write_tiny_gguf())
        gguf.metadata["qwen35.rope.freq_base"] = type(next(iter(gguf.metadata.values())))(next(iter(gguf.metadata.values())).type, 10000.0)
        gguf.metadata["qwen35.rope.dimension_sections"] = type(next(iter(gguf.metadata.values())))(next(iter(gguf.metadata.values())).type, [1, 1, 0, 0])
        config = qwen_rope_config_from_gguf(gguf, metadata())
        self.assertEqual(config.freq_base, 10000.0)
        self.assertEqual(config.dimension_sections, (1, 1, 0, 0))

    def test_gqa_attention_maps_query_heads_to_kv_heads(self) -> None:
        queries = [1.0, 0.0, 0.0, 1.0]
        key = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        value = [10.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]
        out = qwen_gqa_attention(
            queries,
            key,
            value,
            num_attention_heads=2,
            num_kv_heads=1,
            max_seq=4,
            head_dim=2,
            seq_len=2,
        )
        self.assertEqual(len(out), 4)
        self.assertGreater(out[0], out[2])
        self.assertGreater(out[3], out[1])

    def test_qwen_mlp_and_lm_head_match_reference_shapes(self) -> None:
        w = weights()
        values = [0.1, -0.2, 0.3, -0.4]
        self.assert_close_lists(
            qwen_mlp(values, w.mlp, 4, 8),
            mlp_reference(values, w.mlp.gate_proj, w.mlp.up_proj, w.mlp.down_proj, 4, 8),
        )
        norm = [1.0, 1.0, 1.0, 1.0]
        lm = [((i % 5) - 2) / 5.0 for i in range(12)]
        self.assert_close_lists(qwen_lm_head(values, norm, lm, 4, 3), lm_head_reference(values, norm, lm, 4, 3))

    def test_kv_cache_layout_uses_qwen_hidden_dimensions(self) -> None:
        cache = qwen_empty_kv_cache(metadata())
        self.assertEqual(len(cache.key), 8)
        self.assertEqual(cache.num_kv_heads, 1)
        self.assertEqual(cache.head_dim, 2)

    def test_one_layer_qwen_reference_fixture_runs(self) -> None:
        meta = metadata()
        cos, sin = qwen_rope_frequencies(QwenRopeConfig(2, 10000.0, (1, 0, 0, 0)), position=0)
        result = qwen_full_attention_layer(
            [0.25, -0.5, 0.75, 1.0],
            weights(),
            meta,
            qwen_empty_kv_cache(meta),
            position=0,
            rope_cos=cos,
            rope_sin=sin,
        )
        self.assertEqual(len(result.hidden), meta.hidden_size)
        self.assertIn("attention", result.stops)
        self.assertNotEqual(result.cache.key[:2], [0.0, 0.0])

    def test_load_full_attention_layer_weights_from_tiny_gguf(self) -> None:
        meta = metadata()
        gguf = tiny_full_attention_gguf(meta)
        loaded = load_qwen_full_attention_layer_weights(gguf, meta, 0)
        self.assertEqual(len(loaded.attn_norm), meta.hidden_size)
        self.assertEqual(len(loaded.attention.q_proj), meta.num_attention_heads * meta.head_dim * meta.hidden_size)
        self.assertEqual(len(loaded.mlp.down_proj), meta.hidden_size * meta.intermediate_size)

    def test_real_qwen_hybrid_schedule_if_file_exists(self) -> None:
        path = Path(os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf"))
        if not path.exists():
            self.skipTest("local Qwen GGUF is not present")
        from vinf.gguf.mapper import metadata_from_gguf

        gguf = load_gguf(path)
        meta = metadata_from_gguf(gguf)
        schedule = assert_qwen_decode_supported(gguf, meta)
        # 65 blocks minus one nextn (MTP) block; full attention at every 4th layer.
        self.assertEqual(len(schedule), 64)
        full = [idx for idx, kind in enumerate(schedule) if kind is QwenLayerKind.FULL_ATTENTION]
        self.assertEqual(full, list(range(3, 64, 4)))
        config = qwen_linear_attention_config_from_metadata(gguf, meta)
        self.assertEqual((config.key_heads, config.value_heads, config.key_head_dim), (16, 48, 128))
        self.assertEqual((config.conv_dim, config.value_dim, config.conv_kernel), (10240, 6144, 4))
        rope = qwen_rope_config_from_gguf(gguf, meta)
        self.assertEqual(rope.rotated_dims, 64)

    def test_full_one_token_logits_on_tiny_gguf_fixture(self) -> None:
        meta = metadata()
        gguf = tiny_full_attention_gguf(meta)
        runtime_weights = load_qwen_baseline_weights(gguf, meta)
        rope = QwenRopeConfig(meta.head_dim, 10000.0, (1, 0, 0, 0))
        result = qwen_prompt_logits_reference(gguf, meta, runtime_weights, [0, 1], rope)
        self.assertEqual(result.token_ids, (0, 1, 2))
        self.assertEqual(len(result.logits), meta.vocab_size)
        self.assertTrue(all(value == value for value in result.logits))

    def test_prompt_prefill_decode_handoff_with_real_token_ids_on_tiny_gguf(self) -> None:
        meta = metadata()
        gguf = tiny_full_attention_gguf(meta)
        runtime_weights = load_qwen_baseline_weights(gguf, meta)
        rope = QwenRopeConfig(meta.head_dim, 10000.0, (1, 0, 0, 0))
        full = qwen_prompt_logits_reference(gguf, meta, runtime_weights, [0, 1, 2], rope)
        prefill = qwen_prefill_tokens_reference(gguf, meta, runtime_weights, [0, 1], rope)
        handoff = qwen_decode_next_token_reference(gguf, meta, runtime_weights, prefill, 2, rope)
        self.assert_close_lists(handoff.hidden, full.hidden, tol=1e-5)
        self.assert_close_lists(handoff.logits, full.logits, tol=1e-5)

    def test_greedy_one_token_generation_with_tiny_gguf(self) -> None:
        meta = metadata()
        gguf = tiny_full_attention_gguf(meta)
        runtime_weights = load_qwen_baseline_weights(gguf, meta)
        rope = QwenRopeConfig(meta.head_dim, 10000.0, (1, 0, 0, 0))
        token = qwen_greedy_next_token_reference(gguf, meta, runtime_weights, [0, 1], rope)
        logits = qwen_prompt_logits_reference(gguf, meta, runtime_weights, [0, 1], rope)
        self.assertEqual(token, logits.token_ids[max(range(len(logits.logits)), key=lambda idx: logits.logits[idx])])

    def test_partial_rope_passes_through_unrotated_dims(self) -> None:
        config = QwenRopeConfig(head_dim=6, freq_base=10000.0, dimension_sections=(1, 1, 0, 0), rotary_dim=4)
        cos, sin = qwen_rope_frequencies(config, position=3)
        self.assertEqual(len(cos), 4)
        self.assertEqual(cos[:2], cos[2:])
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        rotated = qwen_apply_rope(values, cos, sin)
        self.assertEqual(rotated[4:], [5.0, 6.0])
        self.assert_close_lists(rotated[:4], [
            1.0 * cos[0] - 3.0 * sin[0],
            2.0 * cos[1] - 4.0 * sin[1],
            3.0 * cos[0] + 1.0 * sin[0],
            4.0 * cos[1] + 2.0 * sin[1],
        ])

    def test_layer_kinds_from_full_attention_interval(self) -> None:
        kinds = qwen_layer_kinds_from_interval(8, 4)
        self.assertEqual(
            [kind is QwenLayerKind.FULL_ATTENTION for kind in kinds],
            [False, False, False, True, False, False, False, True],
        )
        with self.assertRaises(UnsupportedModelError):
            qwen_layer_kinds_from_interval(4, 0)

    def test_gated_full_attention_matches_manual_qwen35_block(self) -> None:
        meta = metadata()
        base = weights()
        # Gated q_proj rows per head: [query(head_dim), gate(head_dim)].
        q_proj = [((i % 9) - 4) / 9.0 for i in range(2 * 4 * 4)]
        q_norm = [1.2, 0.8]
        k_norm = [0.9, 1.1]
        gated_weights = QwenLayerWeights(
            attn_norm=base.attn_norm,
            attention=QwenAttentionWeights(q_proj=q_proj, k_proj=base.attention.k_proj, v_proj=base.attention.v_proj, o_proj=base.attention.o_proj),
            post_attention_norm=base.post_attention_norm,
            mlp=base.mlp,
            q_norm=q_norm,
            k_norm=k_norm,
        )
        hidden = [0.25, -0.5, 0.75, 1.0]
        cos, sin = [1.0, 1.0], [0.0, 0.0]
        result = qwen_full_attention_layer(
            hidden, gated_weights, meta, qwen_empty_kv_cache(meta), position=0, rope_cos=cos, rope_sin=sin
        )
        # Position 0 attends only to itself, so each head outputs v; gate scales it by sigmoid.
        normed = rmsnorm_reference(hidden, base.attn_norm)
        q_raw = [sum(q_proj[r * 4 + c] * normed[c] for c in range(4)) for r in range(8)]
        gates = q_raw[2:4] + q_raw[6:8]
        v = [sum(base.attention.v_proj[r * 4 + c] * normed[c] for c in range(4)) for r in range(2)]
        expected_attn = [v[d % 2] / (1.0 + math.exp(-gates[d])) for d in range(4)]
        self.assert_close_lists(result.stops["attention"], expected_attn)
        k = [sum(base.attention.k_proj[r * 4 + c] * normed[c] for c in range(4)) for r in range(2)]
        self.assert_close_lists(result.stops["k_rope"], rmsnorm_reference(k, k_norm))
        self.assert_close_lists(result.stops["q_rope"][:2], rmsnorm_reference(q_raw[0:2], q_norm))

    def test_linear_attention_layer_matches_independent_gated_delta_rule(self) -> None:
        meta = metadata()
        config = tiny_linear_config()
        layer = tiny_linear_weights(meta, config)
        cache = qwen_empty_linear_attention_cache(config)
        ref_state = ReferenceGatedDeltaNet(meta, config, layer)
        inputs = [[0.25, -0.5, 0.75, 1.0], [-0.1, 0.4, 0.2, -0.3], [0.6, 0.1, -0.7, 0.05], [0.3, 0.3, -0.2, 0.9]]
        for hidden in inputs:
            result = qwen_linear_attention_layer(hidden, layer, meta, cache)
            cache = result.cache
            self.assertIsInstance(cache, QwenLinearAttentionCache)
            self.assert_close_lists(result.hidden, ref_state.step(hidden), tol=1e-9)

    def test_linear_attention_conv_uses_channel_major_kernel_layout(self) -> None:
        meta = metadata()
        config = tiny_linear_config()
        layer = tiny_linear_weights(meta, config)
        result = qwen_linear_attention_layer([0.25, -0.5, 0.75, 1.0], layer, meta, qwen_empty_linear_attention_cache(config))
        qkv = result.stops["linear_qkv"]
        k = config.conv_kernel
        # Empty history: only the newest tap (last element of each channel's kernel row) contributes.
        expected = [silu(qkv[ch] * layer.conv1d[ch * k + k - 1]) for ch in range(config.conv_dim)]
        self.assert_close_lists(result.stops["linear_conv"], expected)

    def test_hybrid_gguf_schedule_skips_nextn_layer_and_loads_both_block_kinds(self) -> None:
        meta = hybrid_metadata()
        gguf = tiny_hybrid_gguf(meta)
        self.assertEqual(
            qwen_layer_schedule(gguf, meta),
            (QwenLayerKind.LINEAR_ATTENTION, QwenLayerKind.FULL_ATTENTION),
        )
        runtime_weights = load_qwen_baseline_weights(gguf, meta)
        self.assertIsInstance(runtime_weights.layers[0], QwenLinearAttentionWeights)
        self.assertIsInstance(runtime_weights.layers[1], QwenLayerWeights)
        self.assertIsNotNone(runtime_weights.layers[1].q_norm)
        with self.assertRaises(UnsupportedModelError):
            load_qwen_linear_attention_layer_weights(gguf, meta, 1)
        with self.assertRaises(UnsupportedModelError):
            load_qwen_full_attention_layer_weights(gguf, meta, 0)

    def test_hybrid_prefill_decode_handoff_on_tiny_gguf(self) -> None:
        meta = hybrid_metadata()
        gguf = tiny_hybrid_gguf(meta)
        runtime_weights = load_qwen_baseline_weights(gguf, meta)
        rope = qwen_rope_config_from_gguf(gguf, meta)
        full = qwen_prompt_logits_reference(gguf, meta, runtime_weights, [0, 1, 2], rope)
        prefill = qwen_prefill_tokens_reference(gguf, meta, runtime_weights, [0, 1], rope)
        self.assertIsInstance(prefill.caches[0], QwenLinearAttentionCache)
        handoff = qwen_decode_next_token_reference(gguf, meta, runtime_weights, prefill, 2, rope)
        self.assert_close_lists(handoff.hidden, full.hidden, tol=1e-9)
        self.assert_close_lists(handoff.logits, full.logits, tol=1e-9)
        token = qwen_greedy_next_token_reference(gguf, meta, runtime_weights, [0, 1, 2], rope)
        self.assertEqual(token, full.token_ids[max(range(len(full.logits)), key=lambda idx: full.logits[idx])])

    def test_hybrid_schedule_mismatch_with_interval_fails_cleanly(self) -> None:
        meta = hybrid_metadata()
        gguf = tiny_hybrid_gguf(meta)
        gguf.metadata["qwen35.full_attention_interval"] = GGUFMetadataValue(GGUFValueType.UINT32, 1)
        with self.assertRaises(UnsupportedModelError) as ctx:
            assert_qwen_decode_supported(gguf, meta)
        self.assertIn("full_attention_interval", str(ctx.exception))

    def test_hybrid_requires_ssm_metadata(self) -> None:
        meta = hybrid_metadata()
        gguf = tiny_hybrid_gguf(meta)
        del gguf.metadata["qwen35.ssm.state_size"]
        with self.assertRaises(UnsupportedModelError) as ctx:
            assert_qwen_decode_supported(gguf, meta)
        self.assertIn("ssm.state_size", str(ctx.exception))

    def test_hybrid_rejects_inconsistent_ssm_inner_size(self) -> None:
        meta = hybrid_metadata()
        gguf = tiny_hybrid_gguf(meta)
        gguf.metadata["qwen35.ssm.inner_size"] = GGUFMetadataValue(GGUFValueType.UINT32, 6)
        with self.assertRaises(UnsupportedModelError):
            qwen_linear_attention_config_from_metadata(gguf, meta)


def tiny_linear_config() -> QwenLinearAttentionConfig:
    return QwenLinearAttentionConfig(key_heads=1, value_heads=2, key_head_dim=2, value_head_dim=2, conv_kernel=3)


def _pattern(count: int, mod: int, shift: int, scale: float) -> list[float]:
    return [((i % mod) - shift) / scale for i in range(count)]


def tiny_linear_weights(meta: ModelMetadata, config: QwenLinearAttentionConfig) -> QwenLinearAttentionWeights:
    h = meta.hidden_size
    return QwenLinearAttentionWeights(
        attn_norm=[1.0, 1.1, 0.9, 1.0],
        qkv_proj=_pattern(config.conv_dim * h, 7, 3, 5.0),
        gate_proj=_pattern(config.value_dim * h, 5, 2, 4.0),
        beta_proj=_pattern(config.value_heads * h, 3, 1, 2.0),
        alpha_proj=_pattern(config.value_heads * h, 4, 2, 3.0),
        ssm_a=[-0.3, -0.05],
        dt_bias=[0.5, -1.0],
        conv1d=_pattern(config.conv_dim * config.conv_kernel, 5, 2, 3.0),
        ssm_norm=[0.9, 1.1],
        out_proj=_pattern(h * config.value_dim, 6, 3, 4.0),
        post_attention_norm=[1.0, 1.0, 1.0, 1.0],
        mlp=weights().mlp,
    )


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


class ReferenceGatedDeltaNet:
    """Independent HF-style recurrent gated delta rule (state[v_head][k][v], history rows per step)."""

    def __init__(self, meta: ModelMetadata, config: QwenLinearAttentionConfig, w: QwenLinearAttentionWeights) -> None:
        self.meta, self.c, self.w = meta, config, w
        self.history = [[0.0] * config.conv_dim for _ in range(config.conv_kernel - 1)]
        self.state = [[[0.0] * config.value_head_dim for _ in range(config.key_head_dim)] for _ in range(config.value_heads)]

    def _proj(self, x: list[float], weight: list[float], rows: int) -> list[float]:
        cols = len(x)
        return [sum(weight[r * cols + c] * x[c] for c in range(cols)) for r in range(rows)]

    def step(self, hidden: list[float]) -> list[float]:
        c, w = self.c, self.w
        x = rmsnorm_reference(hidden, w.attn_norm, 1e-6)
        mixed = self._proj(x, w.qkv_proj, c.conv_dim)
        z = self._proj(x, w.gate_proj, c.value_dim)
        b = [1.0 / (1.0 + math.exp(-v)) for v in self._proj(x, w.beta_proj, c.value_heads)]
        a = self._proj(x, w.alpha_proj, c.value_heads)
        window = self.history + [mixed]
        conv = []
        for ch in range(c.conv_dim):
            acc = sum(window[t][ch] * w.conv1d[ch * c.conv_kernel + t] for t in range(c.conv_kernel))
            conv.append(silu(acc))
        self.history = window[1:]
        kd, vd = c.key_head_dim, c.value_head_dim
        out_all: list[float] = []
        for vh in range(c.value_heads):
            kh = vh % c.key_heads  # GGUF tiled value-head order
            q = conv[kh * kd : (kh + 1) * kd]
            k = conv[c.key_dim + kh * kd : c.key_dim + (kh + 1) * kd]
            v = conv[2 * c.key_dim + vh * vd : 2 * c.key_dim + (vh + 1) * vd]
            qn = math.sqrt(sum(t * t for t in q) + 1e-6)
            kn = math.sqrt(sum(t * t for t in k) + 1e-6)
            q = [t / qn / math.sqrt(kd) for t in q]
            k = [t / kn for t in k]
            g = w.ssm_a[vh] * math.log1p(math.exp(a[vh] + w.dt_bias[vh]))
            S = [[s * math.exp(g) for s in row] for row in self.state[vh]]
            kv_mem = [sum(S[i][j] * k[i] for i in range(kd)) for j in range(vd)]
            delta = [(v[j] - kv_mem[j]) * b[vh] for j in range(vd)]
            S = [[S[i][j] + k[i] * delta[j] for j in range(vd)] for i in range(kd)]
            self.state[vh] = S
            o = [sum(S[i][j] * q[i] for i in range(kd)) for j in range(vd)]
            rms = math.sqrt(sum(t * t for t in o) / vd + 1e-6)
            zg = z[vh * vd : (vh + 1) * vd]
            out_all.extend(o[j] / rms * w.ssm_norm[j] * silu(zg[j]) for j in range(vd))
        h = [hi + oi for hi, oi in zip(hidden, self._proj(out_all, w.out_proj, self.meta.hidden_size))]
        m = rmsnorm_reference(h, w.post_attention_norm, 1e-6)
        mlp = mlp_reference(m, w.mlp.gate_proj, w.mlp.up_proj, w.mlp.down_proj, self.meta.hidden_size, self.meta.intermediate_size)
        return [hi + mi for hi, mi in zip(h, mlp)]


def hybrid_metadata() -> ModelMetadata:
    # Two decoder layers (linear, full) plus one trailing nextn (MTP) block.
    return ModelMetadata(
        architecture=ModelArchitecture.QWEN35,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_kv_heads=1,
        hidden_size=4,
        intermediate_size=8,
        head_dim=4,
        vocab_size=3,
        max_position_embeddings=8,
        dtype="fp16",
    )


def tiny_hybrid_gguf(meta: ModelMetadata) -> GGUFFile:
    config = tiny_linear_config()
    h, hd = meta.hidden_size, meta.head_dim
    specs: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (h, meta.vocab_size),
        "output.weight": (h, meta.vocab_size),
        "output_norm.weight": (h,),
        "blk.0.attn_norm.weight": (h,),
        "blk.0.attn_qkv.weight": (h, config.conv_dim),
        "blk.0.attn_gate.weight": (h, config.value_dim),
        "blk.0.ssm_beta.weight": (h, config.value_heads),
        "blk.0.ssm_alpha.weight": (h, config.value_heads),
        "blk.0.ssm_a": (config.value_heads,),
        "blk.0.ssm_dt.bias": (config.value_heads,),
        "blk.0.ssm_conv1d.weight": (config.conv_kernel, config.conv_dim),
        "blk.0.ssm_norm.weight": (config.value_head_dim,),
        "blk.0.ssm_out.weight": (config.value_dim, h),
    }
    for layer in (1, 2):
        specs.update({
            f"blk.{layer}.attn_norm.weight": (h,),
            f"blk.{layer}.attn_q.weight": (h, 2 * meta.num_attention_heads * hd),
            f"blk.{layer}.attn_q_norm.weight": (hd,),
            f"blk.{layer}.attn_k.weight": (h, meta.num_kv_heads * hd),
            f"blk.{layer}.attn_k_norm.weight": (hd,),
            f"blk.{layer}.attn_v.weight": (h, meta.num_kv_heads * hd),
            f"blk.{layer}.attn_output.weight": (meta.num_attention_heads * hd, h),
        })
    for layer in (0, 1, 2):
        specs.update({
            f"blk.{layer}.post_attention_norm.weight": (h,),
            f"blk.{layer}.ffn_gate.weight": (h, meta.intermediate_size),
            f"blk.{layer}.ffn_up.weight": (h, meta.intermediate_size),
            f"blk.{layer}.ffn_down.weight": (meta.intermediate_size, h),
        })
    specs["blk.2.nextn.eh_proj.weight"] = (2 * h, h)
    overrides = {"blk.0.ssm_a": [-0.3, -0.05]}
    u32 = lambda value: GGUFMetadataValue(GGUFValueType.UINT32, value)
    gguf_metadata = {
        "qwen35.full_attention_interval": u32(2),
        "qwen35.nextn_predict_layers": u32(1),
        "qwen35.ssm.group_count": u32(config.key_heads),
        "qwen35.ssm.time_step_rank": u32(config.value_heads),
        "qwen35.ssm.state_size": u32(config.key_head_dim),
        "qwen35.ssm.inner_size": u32(config.value_dim),
        "qwen35.ssm.conv_kernel": u32(config.conv_kernel),
        "qwen35.rope.freq_base": GGUFMetadataValue(GGUFValueType.FLOAT32, 10000.0),
        "qwen35.rope.dimension_count": u32(2),
        "qwen35.rope.dimension_sections": GGUFMetadataValue(GGUFValueType.ARRAY, [1, 0, 0, 0]),
    }
    return _f32_gguf(specs, overrides, gguf_metadata)


def _f32_gguf(
    specs: dict[str, tuple[int, ...]],
    overrides: dict[str, list[float]],
    gguf_metadata: dict[str, GGUFMetadataValue],
) -> GGUFFile:
    payload = bytearray()
    tensors = {}
    for name, dims in specs.items():
        offset = len(payload)
        numel = math.prod(dims)
        values = overrides.get(name) or [((idx % 7) - 3) / 7.0 for idx in range(numel)]
        payload.extend(struct.pack("<" + "f" * numel, *values))
        tensors[name] = GGUFTensorInfo(
            name=name,
            dimensions=dims,
            tensor_type=GGUFTensorType.F32,
            relative_offset=offset,
            absolute_offset=offset,
        )
    tmp = tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False)
    with tmp:
        tmp.write(payload)
    return GGUFFile(
        path=Path(tmp.name),
        version=3,
        metadata=gguf_metadata,
        tensors=tensors,
        data_start=0,
        alignment=32,
    )


def tiny_full_attention_gguf(meta: ModelMetadata) -> GGUFFile:
    specs = {
        "token_embd.weight": (meta.hidden_size, meta.vocab_size),
        "output.weight": (meta.hidden_size, meta.vocab_size),
        "output_norm.weight": (meta.hidden_size,),
        "blk.0.attn_norm.weight": (meta.hidden_size,),
        "blk.0.attn_q.weight": (meta.hidden_size, meta.num_attention_heads * meta.head_dim),
        "blk.0.attn_k.weight": (meta.hidden_size, meta.num_kv_heads * meta.head_dim),
        "blk.0.attn_v.weight": (meta.hidden_size, meta.num_kv_heads * meta.head_dim),
        "blk.0.attn_q_norm.weight": (meta.head_dim,),
        "blk.0.attn_k_norm.weight": (meta.head_dim,),
        "blk.0.attn_output.weight": (meta.num_attention_heads * meta.head_dim, meta.hidden_size),
        "blk.0.post_attention_norm.weight": (meta.hidden_size,),
        "blk.0.ffn_gate.weight": (meta.hidden_size, meta.intermediate_size),
        "blk.0.ffn_up.weight": (meta.hidden_size, meta.intermediate_size),
        "blk.0.ffn_down.weight": (meta.intermediate_size, meta.hidden_size),
    }
    payload = bytearray()
    tensors = {}
    for name, dims in specs.items():
        offset = len(payload)
        numel = math.prod(dims)
        payload.extend(struct.pack("<" + "f" * numel, *[((idx % 7) - 3) / 7.0 for idx in range(numel)]))
        tensors[name] = GGUFTensorInfo(
            name=name,
            dimensions=dims,
            tensor_type=GGUFTensorType.F32,
            relative_offset=offset,
            absolute_offset=offset,
        )
    tmp = tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False)
    with tmp:
        tmp.write(payload)
    return GGUFFile(
        path=Path(tmp.name),
        version=3,
        metadata={},
        tensors=tensors,
        data_start=0,
        alignment=32,
    )


if __name__ == "__main__":
    unittest.main()
