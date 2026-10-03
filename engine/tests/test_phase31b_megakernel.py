from __future__ import annotations

import os

import unittest
from pathlib import Path

from tests.test_phase30_qwen_gpu import gpu_fixture, gpu_metadata
from vinf.errors import ExecutorUnavailableError
from vinf.gguf.residency import TensorResidency, is_elementwise_weight
from vinf.runtime.instructions import (
    QwenAttnHead,
    QwenQmv,
    QwenSsmGroup,
    qwen_mk_abi_header,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT = [3, 17, 5, 29, 11]


def executors_or_skip(test: unittest.TestCase, gguf, meta, **kwargs):
    try:
        from vinf.qwen_gpu import QwenGpuExecutor
        from vinf.qwen_megakernel import QwenMegakernelExecutor

        reference = QwenGpuExecutor(gguf, meta, max_context=16)
        mk = QwenMegakernelExecutor(QwenGpuExecutor(gguf, meta, max_context=16, **kwargs))
        return reference, mk
    except (ExecutorUnavailableError, RuntimeError) as exc:
        test.skipTest(f"CUDA megakernel unavailable: {exc}")


def per_op_logits(ex, tokens):
    ex.reset()
    for token in tokens:
        ex.forward_token(token)
    return ex.logits()


def mk_logits(mk, tokens):
    mk.reset()
    for token in tokens:
        mk.step(token)
    return mk.logits()


class Phase31bAbiTests(unittest.TestCase):
    def test_generated_cuda_abi_header_is_in_lockstep(self) -> None:
        self.assertEqual((ROOT / "csrc/common/qwen_mk_abi.h").read_text(), qwen_mk_abi_header())

    def test_serialized_field_offsets_match_header(self) -> None:
        defines = {}
        for line in qwen_mk_abi_header().splitlines():
            if line.startswith("#define "):
                _, key, value = line.split()
                defines[key] = int(value)
        row = QwenQmv(wait0_counter=7, wait0_target=9, signal=4, tensor=11, x=2, y=3, row_start=5, row_end=6, x2=8).serialize()
        self.assertEqual(row[0], defines["VINF_MK_OP_QMV"])
        self.assertEqual(row[defines["VINF_MK_WAIT0_COUNTER"]], 7)
        self.assertEqual(row[defines["VINF_MK_WAIT0_TARGET"]], 9)
        self.assertEqual(row[defines["VINF_MK_SIGNAL"]], 4)
        self.assertEqual(row[defines["VINF_MK_QMV_TENSOR"]], 11)
        self.assertEqual(row[defines["VINF_MK_QMV_ROW_END"]], 6)
        self.assertEqual(row[defines["VINF_MK_QMV_X2"]], 8)
        attn = QwenAttnHead(kc=1, vc=2, head=3, q_raw=4, k=5, v=6, out=7, q_norm=8, k_norm=9).serialize()
        self.assertEqual(attn[defines["VINF_MK_ATTNHEAD_K_NORM"]], 9)
        ssm = QwenSsmGroup(key_head=1, qkv=2, z=3, beta=4, alpha=5, conv_state=6, ssm_state=7, out=8,
                           conv_w=9, ssm_a=10, dt_bias=11, ssm_norm=12).serialize()
        self.assertEqual(ssm[defines["VINF_MK_SSMGROUP_SSM_NORM"]], 12)
        self.assertEqual(len(ssm), defines["VINF_MK_WORDS"])


class Phase31bMegakernelTests(unittest.TestCase):
    def assert_close(self, actual, expected, tol=2e-4) -> None:
        self.assertEqual(len(actual), len(expected))
        scale = max(1.0, max(abs(v) for v in expected))
        for idx, (a, e) in enumerate(zip(actual, expected)):
            self.assertLessEqual(abs(a - e), tol * scale, (idx, a, e))

    def test_resident_megakernel_matches_per_op_path(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        reference, mk = executors_or_skip(self, gguf, meta)
        self.assertEqual(mk.program.loader_sms, 0)
        expected = per_op_logits(reference, PROMPT)
        actual = mk_logits(mk, PROMPT)
        self.assert_close(actual, expected)
        self.assert_close(mk.hidden(), reference.hidden())

    def test_greedy_chain_matches_per_op_path(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        reference, mk = executors_or_skip(self, gguf, meta)
        expected, _ = reference.generate_greedy(PROMPT, 6)
        actual, stats = mk.generate_greedy(PROMPT, 6)
        self.assertEqual(actual, expected)
        self.assertEqual(stats.generated_tokens, 6)

    def test_streamed_megakernel_matches_per_op_path(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        reference, _ = executors_or_skip(self, gguf, meta)
        r = reference.plan.reservation
        entries = reference.plan.entries
        mandatory = sum(e.nbytes for e in entries if is_elementwise_weight(gguf.tensors[e.name]) and e.residency is TensorResidency.GPU)
        largest = max(gguf.tensors[e.name].nbytes for e in entries
                      if e.residency is TensorResidency.GPU and not is_elementwise_weight(gguf.tensors[e.name]) and e.name != "output.weight")
        tight = r.kv_cache_bytes + r.ssm_state_bytes + r.activation_bytes + 3 * largest + mandatory + 1
        from vinf.qwen_gpu import QwenGpuExecutor
        from vinf.qwen_megakernel import QwenMegakernelExecutor

        expected = per_op_logits(reference, PROMPT)
        for streaming in ("dma", "sm"):
            base = QwenGpuExecutor(gguf, meta, max_context=16, free_vram_bytes=tight, safety_bytes=0, placement="stream")
            mk = QwenMegakernelExecutor(base, streaming=streaming)
            self.assertGreater(len(mk.program.stream_order), 3)
            if streaming == "dma":
                self.assertEqual(mk.program.loader_sms, 0)
                self.assertEqual(len(mk.program.copy_plan), len(mk.program.stream_order))
            else:
                self.assertGreater(mk.program.loader_sms, 0)
            for _ in range(2):  # repeated runs reuse slots and counters
                self.assert_close(mk_logits(mk, PROMPT), expected)

    def test_dependency_timeout_aborts_cleanly(self) -> None:
        try:
            from vinf import _cuda_qwen_megakernel
            from vinf.cuda.qwen_runtime import _iq3_s_grid_bytes
        except ImportError as exc:
            self.skipTest(f"megakernel unavailable: {exc}")
        from array import array

        from vinf.runtime.instructions import QwenArgmax

        mk = _cuda_qwen_megakernel.Megakernel(_iq3_s_grid_bytes())
        info = mk.device_info()
        row = QwenArgmax(wait0_counter=0, wait0_target=1, partials=1).serialize()
        instr = array("i", row).tobytes()
        params = {"max_ctx": 4, "heads": 1, "kv_heads": 1, "hd": 32, "rot": 2, "freq_base": 10000.0, "eps": 1e-6,
                  "key_heads": 1, "value_heads": 1, "kd": 32, "vd": 32, "conv_k": 4,
                  "timeout_cycles": int(0.05 * info["clock_khz"] * 1000)}
        mk.configure(instr, 1, 1, [], [], 0, 0, 1, 1, params, 4096)
        token, error, block, entry = mk.run(0)
        self.assertEqual((error, block, entry), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()


class Phase31bQuantizedQmvTests(unittest.TestCase):
    """Single-instruction megakernel QMV on quantized blocks vs CPU dequantization.

    int8-path types quantize x per 32 elements, so the bound is the x-rounding error:
    |err| <= sum_i |w_i| * amax_group(x) / 254 (plus float slack).
    """

    def run_qmv(self, tensor_type, raw: bytes, rows: int, cols: int, x: list[float]) -> list[float]:
        from array import array

        from vinf import _cuda_qwen_megakernel
        from vinf.cuda.qwen_runtime import CudaWeightRuntime, _iq3_s_grid_bytes
        rt = CudaWeightRuntime()
        rt.upload_raw("w", raw, tensor_type, cols, rows)
        rt.alloc("x", cols)
        rt.alloc("y", rows)
        rt.write("x", array("f", x).tobytes())
        mk = _cuda_qwen_megakernel.Megakernel(_iq3_s_grid_bytes())
        info = mk.device_info()
        parts = 3
        queues = []
        for part in range(parts):
            queues.append([QwenQmv(tensor=0, x=0, y=1, row_start=rows * part // parts, row_end=rows * (part + 1) // parts, signal=0)])
        words = array("i")
        for q in queues:
            words.extend(q[0].serialize())
        params = {"max_ctx": 4, "heads": 1, "kv_heads": 1, "hd": 32, "rot": 2, "freq_base": 1e4, "eps": 1e-6,
                  "key_heads": 1, "value_heads": 1, "kd": 32, "vd": 32, "conv_k": 4,
                  "timeout_cycles": 5 * info["clock_khz"] * 1000}
        mk.configure(words.tobytes(), parts, 1, [rt.tensor_info("w")[:5]], [rt.buffer_ptr("x")[0], rt.buffer_ptr("y")[0]],
                     0, 0, 1, 1, params, info["warps"] * info["warp_buf_bytes"] + cols * 4 + 64)
        self.assertEqual(mk.run(0)[1], 0)
        return rt.read_floats("y")

    def check(self, tensor_type, raw, rows, cols, x, label):
        from tests.test_phase30_cuda_qmatvec import cpu_matvec
        from vinf.gguf.dequant import dequantize_tensor
        from vinf.gguf.parser import GGUFTensorInfo

        try:
            gpu = self.run_qmv(tensor_type, raw, rows, cols, x)
        except (ImportError, ExecutorUnavailableError, RuntimeError) as exc:
            self.skipTest(f"megakernel unavailable: {exc}")
        cpu = cpu_matvec(raw, tensor_type, rows, cols, x)
        w = dequantize_tensor(GGUFTensorInfo("w", (rows * cols,), tensor_type, 0, 0), memoryview(raw)).values
        for r in range(rows):
            bound = 0.0
            for g in range(cols // 32):
                amax = max(abs(v) for v in x[32 * g: 32 * g + 32])
                bound += sum(abs(v) for v in w[r * cols + 32 * g: r * cols + 32 * g + 32]) * amax / 254.0
            self.assertLessEqual(abs(gpu[r] - cpu[r]), 1.05 * bound + 1e-4 * (1 + abs(cpu[r])), (label, r, gpu[r], cpu[r]))

    def test_int8_path_types_on_synthetic_blocks(self) -> None:
        import random

        from tests.test_phase30_cuda_qmatvec import synthetic_rows
        from vinf.gguf.parser import GGUFTensorType

        rng = random.Random(311)
        rows, cols = 24, 512
        x = [rng.uniform(-1, 1) for _ in range(cols)]
        for tensor_type in (GGUFTensorType.Q8_0, GGUFTensorType.Q2_K, GGUFTensorType.Q3_K, GGUFTensorType.Q4_K, GGUFTensorType.Q5_K,
                            GGUFTensorType.Q6_K, GGUFTensorType.IQ4_NL, GGUFTensorType.IQ4_XS,
                            GGUFTensorType.IQ3_S, GGUFTensorType.F16, GGUFTensorType.IQ2_XXS,
                            GGUFTensorType.IQ2_XS, GGUFTensorType.IQ2_S, GGUFTensorType.IQ3_XXS):
            self.check(tensor_type, synthetic_rows(tensor_type, rows, cols, rng), rows, cols, x, tensor_type.name)

    def test_int8_path_types_on_real_model_rows(self) -> None:
        import random

        from vinf.gguf.parser import load_gguf

        path = Path(os.environ.get("VINF_QWEN_GGUF", "/home/z/mt2/Qwen3.8-27B-UD-Q3_K_XL.gguf"))
        if not path.exists():
            self.skipTest("local Qwen GGUF is not present")
        gguf = load_gguf(path)
        examples = {}
        for name, tensor in gguf.tensors.items():
            if len(tensor.dimensions) == 2 and tensor.dimensions[0] % 256 == 0:
                examples.setdefault(tensor.tensor_type, name)
        rng = random.Random(312)
        for tensor_type, name in sorted(examples.items()):
            tensor = gguf.tensors[name]
            cols, rows = tensor.dimensions[0], 6
            row_bytes = cols // tensor.block_size * tensor.type_size
            file_obj, mm, view = gguf.mmap_tensor(name)
            raw = bytes(view[: rows * row_bytes])
            view.release()
            mm.close()
            file_obj.close()
            x = [rng.gauss(0, 1) for _ in range(cols)]
            self.check(tensor_type, raw, rows, cols, x, f"{tensor_type.name} {name}")
