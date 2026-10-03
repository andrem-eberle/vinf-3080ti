from __future__ import annotations

import unittest

from tests.test_phase30_qwen_gpu import gpu_fixture, gpu_metadata
from vinf.errors import ConfigurationError, ExecutorUnavailableError
from vinf.qwen_ops import QwenLayerKind

PROMPT = [3, 17, 5, 29, 11, 2, 7]


def executor_or_skip(test, gguf, meta, **kwargs):
    try:
        from vinf.qwen_gpu import QwenGpuExecutor

        return QwenGpuExecutor(gguf, meta, max_context=16, **kwargs)
    except (ExecutorUnavailableError, RuntimeError) as exc:
        test.skipTest(f"CUDA qwen runtime unavailable: {exc}")


def state_dump(ex) -> dict[str, list[float]]:
    out = {}
    for layer_idx, kind in enumerate(ex.schedule):
        if kind is QwenLayerKind.FULL_ATTENTION:
            for name in (f"kc.{layer_idx}", f"vc.{layer_idx}"):
                cache = ex.rt.read_floats(name)
                # Only committed positions are meaningful.
                s = ex.shapes
                out[name] = [
                    cache[(h * ex.max_context + p) * s.head_dim + d]
                    for h in range(s.kv_heads)
                    for p in range(ex.position)
                    for d in range(s.head_dim)
                ]
        else:
            out[f"conv.{layer_idx}"] = ex.rt.read_floats(f"conv.{layer_idx}")
            out[f"ssm.{layer_idx}"] = ex.rt.read_floats(f"ssm.{layer_idx}")
    return out


class Phase31cBatchedForwardTests(unittest.TestCase):
    def assert_same(self, a, b, tol=1e-6) -> None:
        self.assertEqual(len(a), len(b))
        scale = max(1.0, max((abs(v) for v in b), default=0.0))
        for idx, (x, y) in enumerate(zip(a, b)):
            self.assertLessEqual(abs(x - y), tol * scale, (idx, x, y))

    def sequential(self, ex, tokens):
        ex.reset()
        for token in tokens:
            ex.forward_token(token)

    def test_batched_pass_matches_sequential_logits_and_state(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        seq = executor_or_skip(self, gguf, meta)
        bat = executor_or_skip(self, gguf, meta)
        self.sequential(seq, PROMPT)
        bat.reset()
        bat.forward_tokens(PROMPT[:4])
        bat.forward_tokens(PROMPT[4:])
        self.assertEqual(bat.position, seq.position)
        self.assert_same(bat.hidden(), seq.hidden())
        self.assert_same(bat.logits(), seq.logits())
        for name, values in state_dump(seq).items():
            self.assert_same(state_dump(bat)[name], values)

    def test_greedy_rows_match_sequential_argmax_per_position(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta)
        expected = []
        ex.reset()
        for token in PROMPT[:6]:
            ex.forward_token(token)
            expected.append(ex.greedy_next())
        ex.reset()
        ex.forward_tokens(PROMPT[:6])
        self.assertEqual(ex.greedy_rows(), expected)

    def test_rollback_restores_state_of_kept_prefix(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, snapshot_tokens=4)
        ref = executor_or_skip(self, gguf, meta)
        for keep in (1, 2, 3, 4):
            ex.reset()
            ex.forward_tokens(PROMPT[:3])
            ex.forward_tokens(PROMPT[3:7], snapshot=True)
            ex.rollback(keep)
            self.sequential(ref, PROMPT[: 3 + keep])
            self.assertEqual(ex.position, ref.position)
            for name, values in state_dump(ref).items():
                self.assert_same(state_dump(ex)[name], values)
            # Continuing after rollback matches continuing the sequential run.
            ex.forward_token(13)
            ref.forward_token(13)
            self.assert_same(ex.logits(), ref.logits())

    def test_batched_prefill_greedy_generation_matches_sequential(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta)
        batched, _ = ex.generate_greedy(PROMPT, 5)
        self.sequential(ex, PROMPT)
        expected = [ex.greedy_next()]
        while len(expected) < 5:
            ex.forward_token(expected[-1])
            expected.append(ex.greedy_next())
        self.assertEqual(batched, expected)

    def test_batch_limits_are_enforced(self) -> None:
        meta = gpu_metadata()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, max_batch=4)
        ex.reset()
        with self.assertRaises(ConfigurationError):
            ex.forward_tokens([1] * (ex.prefill_batch + 1))
        with self.assertRaises(ConfigurationError):
            ex.forward_tokens([1, 2], snapshot=True)  # snapshot_tokens == 0
        with self.assertRaises(ConfigurationError):
            ex.rollback(2)  # no pass yet


if __name__ == "__main__":
    unittest.main()
