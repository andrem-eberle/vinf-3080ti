"""Phase 31i: wide prompt passes, tiled attention, and the prompt prefix cache."""

from __future__ import annotations

import dataclasses
import math
import random
import unittest
from array import array

from tests.test_phase30_qwen_gpu import gpu_fixture, gpu_metadata
from tests.test_phase31c_batched_forward import state_dump
from vinf.errors import ExecutorUnavailableError


def runtime_or_skip(test):
    try:
        from vinf.cuda.qwen_runtime import CudaWeightRuntime

        return CudaWeightRuntime()
    except (ExecutorUnavailableError, RuntimeError, ImportError) as exc:
        test.skipTest(f"CUDA qwen runtime unavailable: {exc}")


def executor_or_skip(test, gguf, meta, **kwargs):
    try:
        from vinf.qwen_gpu import QwenGpuExecutor

        return QwenGpuExecutor(gguf, meta, **kwargs)
    except (ExecutorUnavailableError, RuntimeError) as exc:
        test.skipTest(f"CUDA qwen runtime unavailable: {exc}")


def reference_attention(q, kc, vc, heads, kv_heads, hd, max_seq, seq_len0, ntok, bidirectional, window):
    out = []
    for t in range(ntok):
        end = seq_len0 + ntok - 1 if bidirectional else seq_len0 + t
        qpos1 = seq_len0 + t
        start = qpos1 - window if window > 0 and qpos1 > window else 0
        for h in range(heads):
            kvh = h // (heads // kv_heads)
            qh = q[(t * heads + h) * hd : (t * heads + h + 1) * hd]
            scores = []
            for p in range(start, end):
                base = (kvh * max_seq + p) * hd
                scores.append(sum(a * b for a, b in zip(qh, kc[base : base + hd])) / math.sqrt(hd))
            m = max(scores)
            e = [math.exp(x - m) for x in scores]
            z = sum(e)
            for d in range(hd):
                out.append(sum(w * vc[(kvh * max_seq + p) * hd + d] for w, p in zip(e, range(start, end))) / z)
    return out


class TiledAttentionTests(unittest.TestCase):
    def check(self, *, ntok, seq_len0, hd=32, heads=4, kv_heads=2, max_seq=96, bidirectional=0, window=0):
        rt = runtime_or_skip(self)
        rng = random.Random(ntok * 1000 + seq_len0)
        q = [rng.uniform(-1, 1) for _ in range(ntok * heads * hd)]
        kc = [rng.uniform(-1, 1) for _ in range(kv_heads * max_seq * hd)]
        vc = [rng.uniform(-1, 1) for _ in range(kv_heads * max_seq * hd)]
        for name, values in (("q", q), ("kc", kc), ("vc", vc)):
            rt.alloc(name, len(values))
            rt.write(name, array("f", values).tobytes())
        rt.alloc("o", len(q))
        rt.attention("q", "kc", "vc", "o", heads, kv_heads, hd, max_seq, seq_len0, ntok, bidirectional, window)
        got = rt.read_floats("o")
        want = reference_attention(q, kc, vc, heads, kv_heads, hd, max_seq, seq_len0, ntok, bidirectional, window)
        for idx, (a, b) in enumerate(zip(got, want)):
            self.assertAlmostEqual(a, b, delta=1e-4, msg=idx)

    def test_causal_wide_pass(self):
        self.check(ntok=37, seq_len0=20)  # several query tiles, partial tiles, keys before the pass

    def test_from_position_zero(self):
        self.check(ntok=17, seq_len0=1)

    def test_bidirectional_and_window(self):
        self.check(ntok=12, seq_len0=40, bidirectional=1)
        self.check(ntok=24, seq_len0=30, window=19)


class TensorCoreGemmTests(unittest.TestCase):
    def test_gemm_matches_matvec_for_every_type(self):
        from tests.test_phase30_cuda_qmatvec import synthetic_rows
        from vinf.cuda.qwen_runtime import CUDA_MATVEC_TYPES

        rt = runtime_or_skip(self)
        rng = random.Random(41)
        rows, cols, ntok = 70, 512, 37  # partial row and token tiles
        x = [rng.uniform(-1, 1) for _ in range(ntok * cols)]
        rt.alloc("x", ntok * cols)
        rt.write("x", array("f", x).tobytes())
        rt.alloc("y_ref", ntok * rows)
        rt.alloc("y_gemm", ntok * rows)
        for tensor_type in sorted(CUDA_MATVEC_TYPES):
            raw = synthetic_rows(tensor_type, rows, cols, rng)
            rt.upload_raw("w", raw, tensor_type, cols, rows)
            rt.set_gemm_min_rows(1000)  # fp32 matvec groups
            rt.qmv("w", "x", "y_ref", ntok)
            rt.set_gemm_min_rows(1)
            rt.qmv("w", "x", "y_gemm", ntok)
            rt.set_gemm_min_rows(0)
            ref, got = rt.read_floats("y_ref"), rt.read_floats("y_gemm")
            scale = max(abs(v) for v in ref)
            for idx, (a, b) in enumerate(zip(got, ref)):
                self.assertLessEqual(abs(a - b), 3e-3 * scale, (tensor_type.name, idx, a, b))
            rt.free("w")


class WidePrefillTests(unittest.TestCase):
    def setUp(self):
        self.meta = dataclasses.replace(gpu_metadata(), max_position_embeddings=512)
        self.gguf = gpu_fixture(self.meta)
        rng = random.Random(3)
        self.prompt = [rng.randrange(5, 40) for _ in range(150)]

    def test_wide_passes_match_narrow_passes(self):
        narrow = executor_or_skip(self, self.gguf, self.meta, max_context=256, prefill_batch=8)
        wide = executor_or_skip(self, self.gguf, self.meta, max_context=256, prefill_batch=40)
        for ex in (narrow, wide):
            ex.reset()
            ex.prefill(self.prompt)
        self.assertEqual(wide.position, 150)
        a, b = state_dump(narrow), state_dump(wide)
        for name in a:
            for idx, (x, y) in enumerate(zip(a[name], b[name])):
                self.assertAlmostEqual(x, y, delta=1e-4 * max(1.0, abs(x)), msg=(name, idx))
        self.assertEqual(narrow.greedy_next(), wide.greedy_next())
        for x, y in zip(narrow.logits(), wide.logits()):
            self.assertAlmostEqual(x, y, delta=1e-3)


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.meta = dataclasses.replace(gpu_metadata(), max_position_embeddings=512)
        self.gguf = gpu_fixture(self.meta)
        rng = random.Random(5)
        self.a = [rng.randrange(5, 40) for _ in range(140)]
        self.b = [rng.randrange(5, 40) for _ in range(90)]
        self.c = [rng.randrange(5, 40) for _ in range(120)]

    def backend(self, speculative: int, cache: bool):
        from vinf.gguf.tokenizer import QwenTokenizer, SpecialTokens
        from vinf.qwen_backend import QwenBackend

        ex = executor_or_skip(self, self.gguf, self.meta, max_context=512, prefill_batch=32, mtp=speculative > 0,
                              snapshot_tokens=speculative + 1 if speculative else 0)
        decoder = None
        if speculative:
            from vinf.qwen_speculative import QwenSpeculativeDecoder

            decoder = QwenSpeculativeDecoder(ex, speculative)
        tok = QwenTokenizer(tokens=tuple(f"t{i}" for i in range(40)), merges=(), token_types=(3,) * 3 + (1,) * 37,
                            special_tokens=SpecialTokens(0, 2, None, 1, 2), model="gpt2", pre="qwen35", chat_template=None)
        return QwenBackend(ex, tok, decoder=decoder, prefix_cache_bytes=(1 << 30) if cache else 0)

    def run_sequence(self, backend):
        prompts = [self.a, self.a + self.b, self.c, self.a + self.b + self.c, self.a + self.c]
        outs = []
        for prompt in prompts:
            tokens, stats = backend.generate(prompt, 12)
            outs.append((tokens, stats.cached_tokens))
        return outs

    def check(self, speculative: int):
        fresh = self.run_sequence(self.backend(speculative, cache=False))
        cached_backend = self.backend(speculative, cache=True)
        cached = self.run_sequence(cached_backend)
        self.assertEqual([t for t, _ in cached], [t for t, _ in fresh])
        reused = [n for _, n in cached]
        self.assertEqual(reused[0], 0)
        self.assertGreater(reused[1], 100)  # a + b resumes inside a
        self.assertEqual(reused[2], 0)  # unrelated prompt
        self.assertGreater(reused[3], 200)  # a + b + c resumes inside a + b (KV restored from host)
        self.assertGreater(reused[4], 100)  # a + c resumes inside a
        st = cached_backend.prefix_cache.stats
        self.assertGreater(st.kv_saves, 0)
        self.assertGreater(st.kv_loads, 0)

    def test_plain_greedy(self):
        self.check(0)

    def test_mtp_speculative(self):
        self.check(3)


if __name__ == "__main__":
    unittest.main()
