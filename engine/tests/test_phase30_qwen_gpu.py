from __future__ import annotations

import os

import math
import random
import unittest

from tests.test_phase29_qwen_ops import _f32_gguf
from vinf.errors import ExecutorUnavailableError, InsufficientMemoryError
from vinf.gguf.parser import GGUFFile, GGUFMetadataValue, GGUFValueType
from vinf.gguf.residency import TensorResidency, is_elementwise_weight
from vinf.models.metadata import ModelArchitecture, ModelMetadata
from vinf.qwen_ops import QwenRopeConfig
from vinf.qwen_runtime import load_qwen_baseline_weights, qwen_prompt_logits_reference

KEY_HEADS, VALUE_HEADS, STATE, CONV_K = 2, 4, 16, 4


def gpu_metadata() -> ModelMetadata:
    # 4 decoder layers (linear, full, linear, full) + 1 nextn block; dims are multiples of 32.
    return ModelMetadata(
        architecture=ModelArchitecture.QWEN35,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_kv_heads=1,
        hidden_size=64,
        intermediate_size=96,
        head_dim=32,
        vocab_size=40,
        max_position_embeddings=16,
        dtype="fp16",
    )


def gpu_fixture(meta: ModelMetadata, seed: int = 7) -> GGUFFile:
    rng = random.Random(seed)
    h, hd, inter = meta.hidden_size, meta.head_dim, meta.intermediate_size
    key_dim, value_dim = KEY_HEADS * STATE, VALUE_HEADS * STATE
    conv_dim = 2 * key_dim + value_dim
    specs: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (h, meta.vocab_size),
        "output.weight": (h, meta.vocab_size),
        "output_norm.weight": (h,),
    }
    for layer in range(5):
        p = f"blk.{layer}."
        if layer in (1, 3, 4):
            specs.update({
                p + "attn_q.weight": (h, 2 * meta.num_attention_heads * hd),
                p + "attn_q_norm.weight": (hd,),
                p + "attn_k.weight": (h, meta.num_kv_heads * hd),
                p + "attn_k_norm.weight": (hd,),
                p + "attn_v.weight": (h, meta.num_kv_heads * hd),
                p + "attn_output.weight": (meta.num_attention_heads * hd, h),
            })
        else:
            specs.update({
                p + "attn_qkv.weight": (h, conv_dim),
                p + "attn_gate.weight": (h, value_dim),
                p + "ssm_beta.weight": (h, VALUE_HEADS),
                p + "ssm_alpha.weight": (h, VALUE_HEADS),
                p + "ssm_a": (VALUE_HEADS,),
                p + "ssm_dt.bias": (VALUE_HEADS,),
                p + "ssm_conv1d.weight": (CONV_K, conv_dim),
                p + "ssm_norm.weight": (STATE,),
                p + "ssm_out.weight": (value_dim, h),
            })
        specs.update({
            p + "attn_norm.weight": (h,),
            p + "post_attention_norm.weight": (h,),
            p + "ffn_gate.weight": (h, inter),
            p + "ffn_up.weight": (h, inter),
            p + "ffn_down.weight": (inter, h),
        })
    specs["blk.4.nextn.eh_proj.weight"] = (2 * h, h)
    for norm in ("enorm", "hnorm", "shared_head_norm"):
        specs[f"blk.4.nextn.{norm}.weight"] = (h,)
    values: dict[str, list[float]] = {}
    for name, dims in specs.items():
        n = math.prod(dims)
        if name.endswith("norm.weight"):
            values[name] = [rng.uniform(0.6, 1.4) for _ in range(n)]
        elif name.endswith("ssm_a"):
            values[name] = [rng.uniform(-0.6, -0.02) for _ in range(n)]
        else:
            values[name] = [rng.uniform(-0.4, 0.4) for _ in range(n)]
    u32 = lambda value: GGUFMetadataValue(GGUFValueType.UINT32, value)  # noqa: E731
    metadata = {
        "qwen35.full_attention_interval": u32(2),
        "qwen35.nextn_predict_layers": u32(1),
        "qwen35.ssm.group_count": u32(KEY_HEADS),
        "qwen35.ssm.time_step_rank": u32(VALUE_HEADS),
        "qwen35.ssm.state_size": u32(STATE),
        "qwen35.ssm.inner_size": u32(value_dim),
        "qwen35.ssm.conv_kernel": u32(CONV_K),
        "qwen35.rope.freq_base": GGUFMetadataValue(GGUFValueType.FLOAT32, 10000.0),
        "qwen35.rope.dimension_count": u32(16),
        "qwen35.rope.dimension_sections": GGUFMetadataValue(GGUFValueType.ARRAY, [4, 4, 0, 0]),
    }
    return _f32_gguf(specs, values, metadata)


def executor_or_skip(test: unittest.TestCase, gguf, meta, **kwargs):
    try:
        from vinf.qwen_gpu import QwenGpuExecutor

        return QwenGpuExecutor(gguf, meta, **kwargs)
    except (ExecutorUnavailableError, RuntimeError) as exc:
        test.skipTest(f"CUDA qwen runtime unavailable: {exc}")


def reference_logits(gguf, meta, tokens: list[int]) -> list[float]:
    rope = QwenRopeConfig(meta.head_dim, 10000.0, (4, 4, 0, 0), rotary_dim=16)
    weights = load_qwen_baseline_weights(gguf, meta)
    return qwen_prompt_logits_reference(gguf, meta, weights, tokens, rope).logits


class Phase30QwenGpuExecutorTests(unittest.TestCase):
    PROMPT = [3, 17, 5, 29, 11]

    def assert_close(self, actual, expected, tol=2e-4) -> None:
        self.assertEqual(len(actual), len(expected))
        scale = max(abs(v) for v in expected)
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertLessEqual(abs(a - e), tol * max(1.0, scale), (idx, a, e))

    def run_prompt(self, ex, tokens):
        ex.reset()
        for token in tokens:
            ex.forward_token(token)
        return ex.logits()

    def test_resident_gpu_logits_match_cpu_reference(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, max_context=16)
        self.assertEqual(ex.plan.streamed_bytes, 0)
        self.assert_close(self.run_prompt(ex, self.PROMPT), reference_logits(gguf, meta, self.PROMPT))

    def test_streamed_weights_match_resident(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        probe = executor_or_skip(self, gguf, meta, max_context=16)
        r = probe.plan.reservation
        mandatory = sum(e.nbytes for e in probe.plan.entries if is_elementwise_weight(gguf.tensors[e.name]) and e.residency is TensorResidency.GPU)
        largest = max(gguf.tensors[e.name].nbytes for e in probe.plan.entries
                      if e.residency is TensorResidency.GPU and not is_elementwise_weight(gguf.tensors[e.name]) and e.name != "output.weight")
        tight = r.kv_cache_bytes + r.ssm_state_bytes + r.activation_bytes + 3 * largest + mandatory + 1
        from vinf.qwen_gpu import QwenGpuExecutor

        ex = QwenGpuExecutor(gguf, meta, max_context=16, free_vram_bytes=tight, safety_bytes=0, placement="stream")
        self.assertGreater(ex.plan.streamed_bytes, 0)
        self.assertIn("blk.0.ffn_up.weight", ex.plan.names(TensorResidency.PINNED_STREAM))
        before = ex.rt.streamed_bytes()
        logits = self.run_prompt(ex, self.PROMPT)
        self.assertGreater(ex.rt.streamed_bytes(), before)
        self.assert_close(logits, self.run_prompt(probe, self.PROMPT), tol=1e-5)
        # Out-of-order use of streamed tensors resynchronizes the prefetch ring.
        x = [0.01 * i for i in range(meta.hidden_size)]
        ex.rt.write("xn", __import__("array").array("f", x).tobytes())
        ex.rt.qmv("blk.2.ffn_up.weight", "xn", "mlp_up")
        ex.rt.qmv("blk.0.ffn_up.weight", "xn", "mlp_gate")
        a = ex.rt.read_floats("mlp_up", meta.intermediate_size)
        b = ex.rt.read_floats("mlp_gate", meta.intermediate_size)
        probe.rt.write("xn", __import__("array").array("f", x).tobytes())
        probe.rt.qmv("blk.2.ffn_up.weight", "xn", "mlp_up")
        probe.rt.qmv("blk.0.ffn_up.weight", "xn", "mlp_gate")
        self.assert_close(a, probe.rt.read_floats("mlp_up", meta.intermediate_size), tol=1e-6)
        self.assert_close(b, probe.rt.read_floats("mlp_gate", meta.intermediate_size), tol=1e-6)
        self.assert_close(self.run_prompt(ex, self.PROMPT), logits, tol=1e-6)

    def test_greedy_generation_matches_reference_argmax_chain(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, max_context=16)
        tokens, stats = ex.generate_greedy(self.PROMPT, 4)
        self.assertEqual(stats.generated_tokens, 4)
        context = list(self.PROMPT)
        expected = []
        for _ in range(4):
            logits = reference_logits(gguf, meta, context)
            token = max(range(len(logits)), key=lambda idx: logits[idx])
            expected.append(token)
            context.append(token)
        self.assertEqual(tokens, expected)

    def test_reset_clears_kv_and_ssm_state(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, max_context=16)
        first = self.run_prompt(ex, self.PROMPT)
        self.run_prompt(ex, [1, 2, 3])
        self.assert_close(self.run_prompt(ex, self.PROMPT), first, tol=1e-6)

    def test_grouped_head_order_changes_ssm_output(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        tiled = executor_or_skip(self, gguf, meta, max_context=16)
        from vinf.qwen_gpu import QwenGpuExecutor

        grouped = QwenGpuExecutor(gguf, meta, max_context=16, head_order="grouped")
        a, b = self.run_prompt(tiled, self.PROMPT), self.run_prompt(grouped, self.PROMPT)
        self.assertGreater(max(abs(x - y) for x, y in zip(a, b)), 1e-4)

    def test_insufficient_vram_fails_cleanly(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        executor_or_skip(self, gguf, meta, max_context=16)
        from vinf.qwen_gpu import QwenGpuExecutor

        with self.assertRaises(InsufficientMemoryError) as ctx:
            QwenGpuExecutor(gguf, meta, max_context=16, free_vram_bytes=1024)
        self.assertIn("insufficient VRAM", str(ctx.exception))

    def test_context_limit_is_enforced(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, max_context=4)
        from vinf.errors import ConfigurationError

        with self.assertRaises(ConfigurationError):
            ex.generate_greedy([1, 2, 3], 2)

    def test_greedy_generation_transfers_no_full_vocab_rows(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, max_context=16)
        inner = ex.rt._rt

        class NoReads:
            def __getattr__(self, name):
                if name == "read":
                    raise AssertionError("device->host buffer read during greedy decode")
                return getattr(inner, name)

        ex.rt._rt = NoReads()
        tokens, _ = ex.generate_greedy(self.PROMPT, 3)
        self.assertEqual(len(tokens), 3)

    def test_inference_dequantizes_only_embedding_rows_on_cpu(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, max_context=16)
        import vinf.gguf.qwen_tensors as qt

        original = qt.dequantize_tensor
        counted = []

        def counting(info, data):
            counted.append(info.numel)
            return original(info, data)

        qt.dequantize_tensor = counting
        try:
            ex.generate_greedy(self.PROMPT, 3)
        finally:
            qt.dequantize_tensor = original
        forwarded = len(self.PROMPT) + 2
        self.assertEqual(len(counted), forwarded)
        self.assertLessEqual(sum(counted), forwarded * (meta.hidden_size + 256))

    def test_unsupported_tensor_type_fails_before_upload(self) -> None:
        import dataclasses

        from vinf.errors import UnsupportedModelError
        from vinf.gguf.parser import GGUFTensorType

        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        executor_or_skip(self, gguf, meta, max_context=16)
        name = "blk.0.ffn_up.weight"
        gguf.tensors[name] = dataclasses.replace(gguf.tensors[name], tensor_type=GGUFTensorType.Q4_1)
        from vinf.qwen_gpu import QwenGpuExecutor

        with self.assertRaises(UnsupportedModelError) as ctx:
            QwenGpuExecutor(gguf, meta, max_context=16)
        self.assertIn("Q4_1", str(ctx.exception))

    def test_cli_reports_user_facing_errors(self) -> None:
        from pathlib import Path

        if not Path(os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf")).exists():
            self.skipTest("local Qwen GGUF is not present")
        import contextlib
        import io

        from vinf.qwen_cli import main

        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["--max-context", "0", "--prompt", "hi"])
        self.assertEqual(code, 2)
        self.assertIn("error: max_context", err.getvalue())

    def test_vram_report_accounts_runtime_allocations(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, max_context=16)
        self.assertEqual(ex.rt.device_bytes(), ex.plan.gpu_bytes)
        self.assertGreater(ex.rt.buffer_bytes(), 0)
        self.assertIn("VRAM used", ex.vram_report())


if __name__ == "__main__":
    unittest.main()
