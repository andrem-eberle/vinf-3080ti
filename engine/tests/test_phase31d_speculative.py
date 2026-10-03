from __future__ import annotations

import dataclasses
import unittest

from tests.test_phase30_qwen_gpu import gpu_fixture, gpu_metadata
from vinf.errors import ConfigurationError, ExecutorUnavailableError
from vinf.gguf.qwen_tensors import load_qwen_mtp_weights, load_qwen_token_embeddings, qwen_mtp_layer_index
from vinf.qwen_ops import QwenRopeConfig, qwen_empty_kv_cache, qwen_mtp_layer, qwen_rope_frequencies

PROMPT = [3, 17, 5, 29, 11, 2, 7, 19, 23, 1]  # longer than max_batch: chunked prefill + MTP fill


def executor_or_skip(test, gguf, meta, **kwargs):
    try:
        from vinf.qwen_gpu import QwenGpuExecutor

        kwargs.setdefault("kv_dtype", "f32")  # parity with the fp32 CPU reference
        return QwenGpuExecutor(gguf, meta, max_context=32, **kwargs)
    except (ExecutorUnavailableError, RuntimeError) as exc:
        test.skipTest(f"CUDA qwen runtime unavailable: {exc}")


def spec_meta():
    return dataclasses.replace(gpu_metadata(), max_position_embeddings=32)


class Phase31dMtpTests(unittest.TestCase):
    def test_mtp_layer_index_and_cpu_weights(self) -> None:
        meta = spec_meta()
        gguf = gpu_fixture(meta)
        self.assertEqual(qwen_mtp_layer_index(gguf, meta), 4)
        weights = load_qwen_mtp_weights(gguf, meta)
        self.assertEqual(len(weights.eh_proj), 2 * meta.hidden_size * meta.hidden_size)

    def test_gpu_mtp_matches_cpu_reference(self) -> None:
        meta = spec_meta()
        gguf = gpu_fixture(meta)
        ex = executor_or_skip(self, gguf, meta, mtp=True, snapshot_tokens=4)
        ex.reset()
        ex.forward_tokens(PROMPT[:3])
        ex.post_norm_rows()
        hidden_rows = ex.rt.read_floats("xn", 3 * meta.hidden_size)
        ex.mtp_load_hidden("xn", 0, 3)
        draft = ex.mtp_forward(PROMPT[1:4], 0)
        gpu_out = ex.rt.read_floats("mtp_out", 3 * meta.hidden_size)
        weights = load_qwen_mtp_weights(gguf, meta)
        rope = QwenRopeConfig(meta.head_dim, 10000.0, (4, 4, 0, 0), rotary_dim=16)
        cache = qwen_empty_kv_cache(meta)
        embeds = load_qwen_token_embeddings(gguf, PROMPT[1:4])
        h = meta.hidden_size
        for t in range(3):
            cos, sin = qwen_rope_frequencies(rope, t)
            out, cache = qwen_mtp_layer(embeds[t], hidden_rows[t * h:(t + 1) * h], weights, meta, cache,
                                        position=t, rope_cos=cos, rope_sin=sin)
            for a, b in zip(gpu_out[t * h:(t + 1) * h], out):
                self.assertLessEqual(abs(a - b), 2e-4 * max(1.0, abs(b)))
        from vinf.gguf.qwen_tensors import load_qwen_lm_head_rows

        lm = load_qwen_lm_head_rows(gguf, list(range(meta.vocab_size)))
        logits = [sum(w * x for w, x in zip(row, out)) for row in lm]
        self.assertEqual(draft, max(range(len(logits)), key=lambda i: logits[i]))


class Phase31dSpeculativeTests(unittest.TestCase):
    def greedy_reference(self, gguf, meta, n):
        ex = executor_or_skip(self, gguf, meta)
        tokens, _ = ex.generate_greedy(PROMPT, n)
        return tokens

    def decoder(self, gguf, meta, k):
        from vinf.qwen_speculative import QwenSpeculativeDecoder

        ex = executor_or_skip(self, gguf, meta, mtp=True, snapshot_tokens=k + 1)
        return QwenSpeculativeDecoder(ex, k)

    def test_mtp_speculative_output_equals_greedy(self) -> None:
        meta = spec_meta()
        gguf = gpu_fixture(meta)
        expected = self.greedy_reference(gguf, meta, 12)
        for k in (1, 2, 3):
            tokens, stats = self.decoder(gguf, meta, k).generate(PROMPT, 12)
            self.assertEqual(tokens, expected, f"k={k}")
            self.assertEqual(stats.generated_tokens, 12)
            self.assertGreater(stats.steps, 0)
            self.assertEqual(sum(stats.accepted_histogram.values()), stats.steps)

    def test_oracle_drafts_are_all_accepted(self) -> None:
        meta = spec_meta()
        gguf = gpu_fixture(meta)
        expected = self.greedy_reference(gguf, meta, 13)
        dec = self.decoder(gguf, meta, 3)
        emitted = []
        original = dec.drafter.draft

        def oracle(last_token, position, k):
            original(last_token, position, k)  # keep MTP state flowing
            start = len(emitted)
            return expected[start:start + k]

        dec.drafter.draft = oracle
        tokens, stats = dec.generate(PROMPT, 13, on_token=emitted.append)
        self.assertEqual(tokens, expected)
        self.assertEqual(stats.accepted, stats.drafted)
        self.assertEqual(stats.steps, 3)  # 1 + 3 * (3 + 1) = 13 tokens

    def test_rejected_drafts_still_produce_greedy_output(self) -> None:
        meta = spec_meta()
        gguf = gpu_fixture(meta)
        expected = self.greedy_reference(gguf, meta, 10)
        dec = self.decoder(gguf, meta, 3)
        emitted = []

        def wrong(last_token, position, k):
            start = len(emitted)
            return [(expected[start + i] + 1) % meta.vocab_size if start + i < len(expected) else 0 for i in range(k)]

        dec.drafter.draft = wrong
        tokens, stats = dec.generate(PROMPT, 10, on_token=emitted.append)
        self.assertEqual(tokens, expected)
        self.assertEqual(stats.accepted, 0)
        self.assertEqual(stats.steps, 9)

    def test_stop_token_and_limits(self) -> None:
        meta = spec_meta()
        gguf = gpu_fixture(meta)
        expected = self.greedy_reference(gguf, meta, 12)
        stop = expected[4]
        tokens, _ = self.decoder(gguf, meta, 3).generate(PROMPT, 12, stop_token_ids=frozenset({stop}))
        self.assertEqual(tokens, expected[: expected.index(stop) + 1])
        ex = executor_or_skip(self, gguf, meta, mtp=True, snapshot_tokens=2)
        from vinf.qwen_speculative import QwenSpeculativeDecoder

        with self.assertRaises(ConfigurationError):
            QwenSpeculativeDecoder(ex, 3)  # snapshot_tokens < k + 1


if __name__ == "__main__":
    unittest.main()
