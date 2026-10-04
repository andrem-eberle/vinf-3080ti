"""Phase 31g: concurrent sequences (paged KV pool, multi-sequence passes, continuous-batching scheduler)."""

from __future__ import annotations

import dataclasses
import json
import random
import threading
import unittest
import urllib.request

from tests.test_phase30_qwen_gpu import gpu_fixture, gpu_metadata
from vinf.errors import ExecutorUnavailableError


def fixture():
    meta = dataclasses.replace(gpu_metadata(), max_position_embeddings=512)
    return meta, gpu_fixture(meta)


def executor_or_skip(test, gguf, meta, **kwargs):
    try:
        from vinf.qwen_gpu import QwenGpuExecutor

        kwargs.setdefault("max_context", 512)
        kwargs.setdefault("prefill_batch", 32)
        kwargs.setdefault("page_size", 32)
        return QwenGpuExecutor(gguf, meta, **kwargs)
    except (ExecutorUnavailableError, RuntimeError) as exc:
        test.skipTest(f"CUDA qwen runtime unavailable: {exc}")


def tokenizer():
    from vinf.gguf.tokenizer import QwenTokenizer, SpecialTokens

    return QwenTokenizer(tokens=tuple(f"t{i}" for i in range(40)), merges=(), token_types=(3,) * 3 + (1,) * 37,
                         special_tokens=SpecialTokens(0, 2, None, 1, 2), model="gpt2", pre="qwen35", chat_template=None)


def backend(test, *, speculative=0, cache=True, **kwargs):
    from vinf.qwen_backend import QwenBackend

    meta, gguf = fixture()
    ex = executor_or_skip(test, gguf, meta, mtp=speculative > 0,
                          snapshot_tokens=speculative + 1 if speculative else 0, **kwargs)
    decoder = None
    if speculative:
        from vinf.qwen_speculative import QwenSpeculativeDecoder

        decoder = QwenSpeculativeDecoder(ex, speculative)
    return QwenBackend(ex, tokenizer(), decoder=decoder, prefix_cache_bytes=(1 << 30) if cache else 0)


def prompts(n, seed=11, lo=40, hi=150):
    rng = random.Random(seed)
    return [[rng.randrange(5, 40) for _ in range(rng.randrange(lo, hi))] for _ in range(n)]


def solo(test, prompt_list, max_new):
    """Reference: each prompt alone on a fresh single-sequence executor, plain greedy."""
    meta, gguf = fixture()
    ex = executor_or_skip(test, gguf, meta)
    return [ex.generate_greedy(p, max_new)[0] for p in prompt_list]


class MultiSequencePassTests(unittest.TestCase):
    def test_batched_decode_matches_each_sequence_alone(self):
        ps = prompts(3)
        want = solo(self, ps, 10)
        meta, gguf = fixture()
        ex = executor_or_skip(self, gguf, meta, max_seqs=3)
        seqs, outs = [], []
        for p in ps:
            seq = ex.seq if not seqs else ex.new_sequence()
            ex.activate(seq)
            ex.reset()
            ex.prefill(p)
            outs.append([ex.greedy_next()])
            seqs.append(seq)
        for _ in range(9):
            nexts = (ex.forward_multi(seqs, [o[-1] for o in outs]), ex.greedy_rows())[1]
            for o, t in zip(outs, nexts):
                o.append(t)
        self.assertEqual(outs, [w[:10] for w in want])
        self.assertEqual(sorted(p for s in seqs for p in s.pages), sorted(set(p for s in seqs for p in s.pages)))


class SchedulerTests(unittest.TestCase):
    def run_jobs(self, b, ps, max_news):
        sched = b.scheduler(log=lambda m: None)
        jobs = [sched.submit(p, n) for p, n in zip(ps, max_news)]
        outs = []
        for job in jobs:
            for _ in job:
                pass
            outs.append(job.out)
        return outs, jobs, sched

    def check(self, b, ps, max_news):
        want = solo(self, ps, max(max_news))
        outs, jobs, sched = self.run_jobs(b, ps, max_news)
        for out, w, n in zip(outs, want, max_news):
            stop = next((i for i, t in enumerate(w) if t in b.stop_token_ids), None)
            expected = w[: min(n, stop + 1 if stop is not None else n)]
            self.assertEqual(out, expected)
        return jobs, sched

    def test_concurrent_plain(self):
        b = backend(self, max_seqs=4, kv_pool_tokens=2048)
        jobs, sched = self.check(b, prompts(4), [12, 5, 9, 12])
        self.assertTrue(all(j.finish_reason in ("length", "stop") for j in jobs))

    def test_concurrent_mtp_speculative(self):
        b = backend(self, speculative=3, max_seqs=3, kv_pool_tokens=2048)
        jobs, sched = self.check(b, prompts(3, seed=4), [14, 6, 10])
        self.assertGreater(sched.spec_passes, 0)  # drafts verified for several sequences in one pass

    def test_prefix_resume_and_shared_prefix_copy(self):
        b = backend(self, speculative=3, max_seqs=3, kv_pool_tokens=2048)
        base = prompts(1, seed=8, lo=150, hi=151)[0]
        first = [base + [7, 8, 9]]
        self.check(b, first, [6])
        # Two concurrent continuations of the same conversation: one resumes in place, one copies.
        ps = [base + [7, 8, 9, 11, 12], base + [7, 8, 9, 13, 14, 15]]
        self.check(b, ps, [8, 8])
        st = b.prefix_cache.stats
        self.assertGreaterEqual(st.hits, 2)
        self.assertGreaterEqual(st.kv_copies + st.kv_loads, 1)

    def test_pool_exhaustion_swaps_out_and_resumes(self):
        # 6 pages of 32 tokens for 3 sequences of ~100-150 tokens + output: forces swaps.
        b = backend(self, max_seqs=3, kv_pool_tokens=6 * 32)
        ps = prompts(3, seed=21, lo=60, hi=90)
        jobs, sched = self.check(b, ps, [40, 40, 40])
        self.assertGreater(b.prefix_cache.stats.evictions, 0)

    def test_cancel(self):
        b = backend(self, max_seqs=2, kv_pool_tokens=1024)
        sched = b.scheduler(log=lambda m: None)
        job = sched.submit(prompts(1)[0], 400)
        for kind, _ in job:
            if kind == "token" and len(job.out) >= 2:
                job.cancel()
        self.assertEqual(job.finish_reason, "cancelled")
        self.assertEqual(sched.active(), 0)


class ConcurrentHttpTests(unittest.TestCase):
    def test_parallel_requests_match_sequential(self):
        from vinf.server import OpenAIService, make_server

        from tests.test_phase31h_server import tiny_tokenizer

        b = backend(self, max_seqs=3, kv_pool_tokens=2048)
        b.tokenizer = tiny_tokenizer()  # text <-> ids for the 40-token fixture vocabulary
        service = OpenAIService(b, log=lambda m: None)
        httpd = make_server(service, "127.0.0.1", 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/completions"
        texts = ["the quick brown fox jumps over the lazy dog", "hello world how are you", "abc def ghi jkl"]
        bodies = [{"prompt": t, "max_tokens": 10} for t in texts]

        def post(body):
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["choices"][0]["text"]

        try:
            sequential = [post(body) for body in bodies]
            results = [None] * len(bodies)

            def worker(i):
                results[i] = post(bodies[i])

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(bodies))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(results, sequential)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
