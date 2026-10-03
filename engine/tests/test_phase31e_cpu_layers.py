from __future__ import annotations

import os

import random
import unittest
from pathlib import Path

from tests.test_phase30_cuda_qmatvec import cpu_matvec, synthetic_rows
from vinf.errors import ExecutorUnavailableError
from vinf.gguf.dequant import dequantize_tensor
from vinf.gguf.parser import GGUFTensorInfo, GGUFTensorType, load_gguf

REAL_MODEL = Path(os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf"))
QUANT_TYPES = (GGUFTensorType.Q8_0, GGUFTensorType.Q2_K, GGUFTensorType.Q3_K, GGUFTensorType.Q4_K, GGUFTensorType.Q5_K,
               GGUFTensorType.Q6_K, GGUFTensorType.IQ4_NL, GGUFTensorType.IQ4_XS, GGUFTensorType.IQ3_S,
               GGUFTensorType.IQ2_XXS, GGUFTensorType.IQ2_XS, GGUFTensorType.IQ2_S, GGUFTensorType.IQ3_XXS)


def cpu_or_skip(test):
    try:
        from vinf.cpu_qwen import CpuQwenRuntime

        return CpuQwenRuntime()
    except ExecutorUnavailableError as exc:
        test.skipTest(str(exc))


class Phase31eCpuMatmulTests(unittest.TestCase):
    """int8-activation CPU matmul vs CPU dequant (bound: sum|w| * amax(x group) / 254 per group)."""

    def check(self, rt, name, tensor_type, raw, rows, cols, xs):
        rt.add_raw(name, raw, tensor_type, cols, rows)
        ntok = len(xs)
        rt.alloc(name + ".x", cols * ntok)
        rt.alloc(name + ".y", rows * ntok)
        rt.write_floats(name + ".x", [v for x in xs for v in x])
        rt.qmv(name, name + ".x", name + ".y", ntok)
        got = rt.read_floats(name + ".y")
        w = dequantize_tensor(GGUFTensorInfo("w", (rows * cols,), tensor_type, 0, 0), memoryview(raw)).values
        float_path = tensor_type in (GGUFTensorType.F32, GGUFTensorType.F16)
        for t, x in enumerate(xs):
            ref = cpu_matvec(raw, tensor_type, rows, cols, x)
            for r in range(rows):
                bound = 0.0
                if not float_path:
                    for g in range(cols // 32):
                        amax = max(abs(v) for v in x[32 * g:32 * g + 32])
                        bound += sum(abs(v) for v in w[r * cols + 32 * g:r * cols + 32 * g + 32]) * amax / 254.0
                self.assertLessEqual(abs(got[t * rows + r] - ref[r]), 1.05 * bound + 1e-4 * (1 + abs(ref[r])),
                                     (tensor_type.name, t, r, got[t * rows + r], ref[r]))

    def test_all_types_synthetic_multi_token(self) -> None:
        rt = cpu_or_skip(self)
        rng = random.Random(3101)
        rows, cols = 12, 512
        for tensor_type in QUANT_TYPES + (GGUFTensorType.F32, GGUFTensorType.F16):
            raw = synthetic_rows(tensor_type, rows, cols, rng)
            xs = [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(3)]
            self.check(rt, f"w{int(tensor_type)}", tensor_type, raw, rows, cols, xs)

    def test_all_model_types_on_real_rows(self) -> None:
        if not REAL_MODEL.exists():
            self.skipTest("local Qwen GGUF is not present")
        rt = cpu_or_skip(self)
        gguf = load_gguf(REAL_MODEL)
        examples = {}
        for name, tensor in gguf.tensors.items():
            if len(tensor.dimensions) == 2 and tensor.dimensions[0] % 256 == 0:
                examples.setdefault(tensor.tensor_type, name)
        rng = random.Random(3102)
        for tensor_type, name in sorted(examples.items()):
            tensor = gguf.tensors[name]
            cols, rows = tensor.dimensions[0], 4
            row_bytes = cols // tensor.block_size * tensor.type_size
            file_obj, mm, view = gguf.mmap_tensor(name)
            raw = bytes(view[:rows * row_bytes])
            view.release()
            mm.close()
            file_obj.close()
            xs = [[rng.gauss(0, 1) for _ in range(cols)] for _ in range(2)]
            self.check(rt, name, tensor_type, raw, rows, cols, xs)


class Phase31eHybridExecutorTests(unittest.TestCase):
    PROMPT = [3, 17, 5, 29, 11, 2, 7, 19, 23, 1]

    def executors(self, **kwargs):
        import dataclasses

        kwargs.setdefault("kv_dtype", "f32")  # CPU layers keep fp32 KV; compare like with like

        from tests.test_phase30_qwen_gpu import gpu_fixture, gpu_metadata

        try:
            from vinf.qwen_gpu import QwenGpuExecutor

            meta = dataclasses.replace(gpu_metadata(), max_position_embeddings=32)
            gguf = gpu_fixture(meta)
            gpu = QwenGpuExecutor(gguf, meta, max_context=32, **kwargs)
            # Free VRAM just above the output head + activations: forces decoder layers onto the CPU.
            probe = QwenGpuExecutor(gguf, meta, max_context=32, **kwargs)
            fixed = sum(probe._buffer_specs().values()) * 4 + 32 * 1024**2
            tight = fixed + sum(gguf.tensors[n].nbytes for n in ("output.weight", "output_norm.weight") + probe._mtp_weight_names())
            tight += sum(v * 4 for n, v in probe._state_specs().items() if probe._layer_of(n) == probe.mtp_layer)
            tight += sum(gguf.tensors[n].nbytes for n in probe._layer_weight_names(0))
            tight += sum(v * 4 for n, v in probe._state_specs().items() if probe._layer_of(n) == 0) + 1
            hybrid = QwenGpuExecutor(gguf, meta, max_context=32, free_vram_bytes=tight, safety_bytes=0, **kwargs)
        except (ExecutorUnavailableError, RuntimeError) as exc:
            self.skipTest(str(exc))
        return meta, gguf, gpu, hybrid

    def assert_close(self, a, b, tol=1e-5):
        self.assertEqual(len(a), len(b))
        scale = max(1.0, max(abs(v) for v in b))
        for i, (x, y) in enumerate(zip(a, b)):
            self.assertLessEqual(abs(x - y), tol * scale, (i, x, y))

    def test_hybrid_places_tail_layers_on_cpu_and_matches_gpu(self) -> None:
        meta, gguf, gpu, hybrid = self.executors()
        self.assertEqual(gpu.cpu_layers, frozenset())
        self.assertTrue(hybrid.cpu_layers)
        self.assertEqual(hybrid.plan.streamed_bytes, 0)
        self.assertEqual(max(hybrid.cpu_layers), len(hybrid.schedule) - 1)
        self.assertIn("CPU-computed layers", hybrid.report())
        for ex in (gpu, hybrid):
            ex.reset()
            ex.forward_tokens(self.PROMPT[:6])
            ex.forward_tokens(self.PROMPT[6:])
        self.assert_close(hybrid.logits(), gpu.logits())
        self.assertEqual(hybrid.generate_greedy(self.PROMPT, 8)[0], gpu.generate_greedy(self.PROMPT, 8)[0])

    def test_hybrid_rollback_and_speculative_decoding(self) -> None:
        meta, gguf, gpu, hybrid = self.executors(mtp=True, snapshot_tokens=4)
        from vinf.qwen_speculative import QwenSpeculativeDecoder

        expected = gpu.generate_greedy(self.PROMPT, 12)[0]
        tokens, stats = QwenSpeculativeDecoder(hybrid, 3).generate(self.PROMPT, 12)
        self.assertEqual(tokens, expected)
        self.assertGreater(stats.steps, 0)


if __name__ == "__main__":
    unittest.main()
