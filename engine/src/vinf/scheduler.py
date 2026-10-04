"""Continuous-batching scheduler: many concurrent requests on one model instance.

One thread owns the GPU. Each iteration it
  1. admits waiting requests while a sequence slot and enough KV pages are available (resuming from the
     prefix cache when a prompt extends an earlier one),
  2. runs one prompt pass (up to prefill_batch tokens) for the oldest request still processing its prompt,
  3. runs one decode step for every request that is generating: a single pass for all of them, so the
     weights are read (and streamed over PCIe) once. With MTP, every sequence contributes its next input
     plus its drafts and keeps its accepted prefix (speculative decoding inside the shared pass);
     without a drafter (or before its first draft) each sequence contributes one row.
Greedy decoding is deterministic per sequence: a request produces the same tokens whatever runs beside it.

When the KV page pool runs out mid-generation, the most recently admitted other request is swapped out
(its state checkpointed to host memory through the prefix cache) and re-queued; it resumes where it left
off once pages are free.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from vinf.errors import ConfigurationError
from vinf.qwen_gpu import KvPoolExhausted


@dataclass(eq=False)
class Job:
    prompt: list[int]
    max_new: int
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    out: list[int] = field(default_factory=list)
    state: str = "waiting"  # waiting | prefill | decode | done
    seq: object = None
    cursor: int = 0  # next prompt position to process
    cached_tokens: int = 0
    admitted: int = 0  # admission order (swap-out picks the newest)
    t_submit: float = field(default_factory=time.perf_counter)
    t_first: float | None = None
    t_done: float | None = None
    prompt_seconds: float = 0.0
    finish_reason: str | None = None
    error: BaseException | None = None

    @property
    def full(self) -> list[int]:
        """Tokens to have processed before the next prediction (prompt + generated, minus the last one,
        which is the next input); after a swap-out the generated part is part of the 'prompt'."""
        return self.prompt + self.out

    def cancel(self) -> None:
        self.cancelled.set()

    def __iter__(self):
        """Yields ("token", id) events; ends after ("done", finish_reason) or raises the job's error."""
        while True:
            kind, value = self.events.get()
            if kind == "error":
                raise value
            yield kind, value
            if kind == "done":
                return


class Scheduler:
    def __init__(self, backend, *, log=print) -> None:
        self.backend = backend
        self.ex = backend.base
        self.decoder = backend.decoder
        self.cache = backend.prefix_cache
        self.stop_ids = backend.stop_token_ids
        self.log = log
        self.lock = threading.Condition()
        self.waiting: list[Job] = []
        self.running: list[Job] = []
        self.admissions = 0
        self.passes = 0
        self.spec_passes = 0  # verification passes with more than one sequence
        self.accepted = 0  # accepted draft tokens
        self.thread = threading.Thread(target=self._loop, name="vinf-scheduler", daemon=True)
        self.thread.start()

    # ---- public -------------------------------------------------------------------------------

    def submit(self, prompt: list[int], max_new: int) -> Job:
        if not prompt:
            raise ConfigurationError("empty prompt")
        if len(prompt) >= self.ex.max_context:
            raise ConfigurationError(f"prompt is {len(prompt)} tokens; the context holds {self.ex.max_context}")
        job = Job(list(prompt), max(1, max_new))
        with self.lock:
            self.waiting.append(job)
            self.lock.notify()
        return job

    def active(self) -> int:
        with self.lock:
            return len(self.running) + len(self.waiting)

    # ---- loop ---------------------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            with self.lock:
                while not self.waiting and not self.running:
                    self.lock.wait()
            try:
                self._iteration()
            except Exception as exc:  # noqa: BLE001 - fail the affected requests, keep serving
                self.log(traceback.format_exc())
                with self.lock:
                    jobs = list(self.running)
                for job in jobs:
                    self._finish(job, error=exc)

    def _iteration(self) -> None:
        for job in [j for j in self.running if j.cancelled.is_set()]:
            self._finish(job, "cancelled")
        self._admit()
        prefilling = [j for j in self.running if j.state == "prefill"]
        if prefilling:
            self._prefill_pass(prefilling[0])
        decoding = [j for j in self.running if j.state == "decode" and not j.cancelled.is_set()]
        if decoding:
            if self.decoder is not None and all(self._spec_ok(j) for j in decoding):
                self._spec_batch_step(decoding)
            else:
                self._batch_step(decoding)

    # ---- admission ----------------------------------------------------------------------------

    def _admit(self) -> None:
        ex = self.ex
        while True:
            with self.lock:
                if not self.waiting:
                    return
                job = self.waiting[0]
                if job.cancelled.is_set():
                    self.waiting.pop(0)
                    job.state = "done"
                    job.finish_reason = "cancelled"
                    job.events.put(("done", "cancelled"))
                    continue
            if len(self.running) >= ex.max_seqs:
                return
            pages = -(-len(job.full) // ex.page_size)
            if self.running and pages > ex.free_page_count() + self._reclaimable_pages():
                return  # wait for pages
            try:
                seq, start = self._acquire(job.full)
            except KvPoolExhausted:
                if self.running:
                    return
                with self.lock:
                    self.waiting.pop(0)
                self._finish(job, error=KvPoolExhausted(
                    f"prompt of {len(job.full)} tokens does not fit the KV pool ({ex.kv_pages * ex.page_size} tokens)"))
                continue
            with self.lock:
                self.waiting.pop(0)
                self.running.append(job)
            self.admissions += 1
            job.seq, job.cursor, job.state, job.admitted = seq, start, "prefill", self.admissions
            if not job.out:
                job.cached_tokens = start
            if self.decoder is not None and hasattr(self.decoder.drafter, "bind"):
                self.decoder.drafter.forget(seq)

    def _reclaimable_pages(self) -> int:
        busy = {id(j.seq) for j in self.running}
        return sum(len(s.pages) for s in self.ex.sequences if id(s) not in busy and not s.busy)

    def _acquire(self, tokens: list[int]):
        if self.cache is not None:
            return self.cache.acquire(tokens)
        if not self.ex.free_slots:  # no prefix cache: idle sequences hold nothing worth keeping
            for seq in [s for s in self.ex.sequences if not s.busy]:
                self.ex.release_sequence(seq)
        seq = self.ex.new_sequence()
        seq.busy = True
        self.ex.activate(seq)
        return seq, 0

    def _release(self, seq) -> None:
        if self.decoder is not None and hasattr(self.decoder.drafter, "forget"):
            self.decoder.drafter.forget(seq)
        if self.cache is not None:
            self.cache.release(seq)
        else:
            self.ex.release_sequence(seq)

    # ---- prompt passes ------------------------------------------------------------------------

    def _prefill_pass(self, job: Job) -> None:
        ex = self.ex
        full = job.full
        ex.activate(job.seq)
        self._bind(job)
        start = job.cursor
        plen = len(job.prompt)
        # Prompt tokens run on the prompt kernels (tensor-core GEMM/attention); generated tokens being
        # replayed after a swap-out run on the decode kernels they were produced with, so a request's
        # output does not depend on whether it was swapped out.
        prompt_mode = start < plen
        chunk = full[start : min(start + ex.prefill_batch, plen)] if prompt_mode else full[start : start + ex.max_batch]
        last = start + len(chunk) == len(full)
        t0 = time.perf_counter()
        if last and self.cache is not None:
            self.cache.checkpoint(full)
        gemm = getattr(ex.rt, "set_gemm_min_rows", None) if prompt_mode else None
        if gemm is not None:
            gemm(1)
        try:
            self._with_pages(job, lambda: ex.forward_tokens(chunk))
            if self.decoder is not None:
                self.decoder.drafter.observe_prefill(start, chunk, full)
        except KvPoolExhausted as exc:
            self._finish(job, error=exc)
            return
        finally:
            if gemm is not None:
                gemm(0)
        self.passes += 1
        job.cursor = start + len(chunk)
        job.prompt_seconds += time.perf_counter() - t0
        if last:
            job.state = "decode"
            self._emit(job, [ex.greedy_next()])

    # ---- decode steps -------------------------------------------------------------------------

    def _bind(self, job: Job) -> None:
        if self.decoder is not None and hasattr(self.decoder.drafter, "bind"):
            self.decoder.drafter.bind(job.seq)

    def _spec_ok(self, job: Job) -> bool:
        drafter = self.decoder.drafter
        if not hasattr(drafter, "bind"):
            return False
        drafter.bind(job.seq)
        return drafter.ready

    def _spec_step(self, job: Job) -> None:
        ex = self.ex
        ex.activate(job.seq)
        self._bind(job)
        room = job.max_new - len(job.out)
        try:
            emitted = self._with_pages(job, lambda: self.decoder.step(job.out[-1], room))
        except KvPoolExhausted as exc:
            self._finish(job, error=exc)
            return
        self.passes += 1
        self._emit(job, emitted)

    def _spec_batch_step(self, jobs: list[Job]) -> None:
        """Draft for every sequence, verify all drafts in one pass, keep each sequence's accepted prefix."""
        ex, dec = self.ex, self.decoder
        drafter = dec.drafter
        rows = []
        for job in jobs:
            seq = job.seq
            ex.activate(seq)
            drafter.bind(seq)
            k = min(dec.k, ex.max_context - seq.position - 1, job.max_new - len(job.out) - 1)
            drafts = drafter.draft(job.out[-1], seq.position, k) if k >= 1 else []
            rows.append([job.out[-1]] + drafts)
        seqs = [j.seq for j in jobs]

        def run():
            ex.forward_verify(seqs, rows)
            return ex.greedy_rows()

        try:
            targets = self._with_pages(jobs[0], run, group=jobs)
        except KvPoolExhausted as exc:
            for job in jobs:
                self._finish(job, error=exc)
            return
        if targets is None:
            return
        self.passes += 1
        keeps, emitted, row0 = [], [], 0
        for job, r in zip(jobs, rows):
            t = targets[row0 : row0 + len(r)]
            drafts = r[1:]
            accepted = 0
            while accepted < len(drafts) and drafts[accepted] == t[accepted]:
                accepted += 1
            keep = accepted + 1
            drafter.bind(job.seq)
            drafter.observe_verify(job.seq.position - len(r), keep, t[:keep], src_row=row0)
            keeps.append(keep)
            emitted.append(t[:keep])
            row0 += len(r)
        ex.commit(keeps)
        self.spec_passes += len(jobs) > 1
        self.accepted += sum(keeps) - len(keeps)
        for job, toks in zip(jobs, emitted):
            self._emit(job, toks)

    def _batch_step(self, jobs: list[Job]) -> None:
        ex = self.ex
        seqs = [j.seq for j in jobs]

        def run():
            ex.forward_multi(seqs, [j.out[-1] for j in jobs])
            return ex.greedy_rows()

        try:
            nexts = self._with_pages(jobs[0], run, group=jobs)
        except KvPoolExhausted as exc:
            for job in jobs:
                self._finish(job, error=exc)
            return
        if nexts is None:
            return
        self.passes += 1
        drafter = self.decoder.drafter if self.decoder is not None else None
        for row, (job, token) in enumerate(zip(jobs, nexts)):
            if drafter is not None and hasattr(drafter, "bootstrap"):
                drafter.bind(job.seq)  # greedy_rows left post-norm rows in "xn": seed the MTP drafter
                drafter.bootstrap("xn", row, job.seq.position)
            self._emit(job, [token])

    def _with_pages(self, job: Job, fn, group: list[Job] | None = None):
        """Run fn; when the KV pool is exhausted, swap out the newest other running request and retry."""
        while True:
            try:
                return fn()
            except KvPoolExhausted:
                victims = [j for j in self.running if j is not job and j.state in ("decode", "prefill")
                           and (group is None or j not in group or len(group) > 1)]
                if not victims:
                    raise
                victim = max(victims, key=lambda j: j.admitted)
                self._swap_out(victim)
                if group is not None and victim in group:
                    return None  # the batch changed; the next iteration rebuilds it
                self.ex.activate(job.seq)  # the swap-out activated (and freed) the victim's sequence
                self._bind(job)

    def _swap_out(self, job: Job) -> None:
        ex = self.ex
        seq = job.seq
        ex.activate(seq)
        if self.cache is not None:
            self.cache.checkpoint(job.full)
        job.state = "waiting"
        job.seq = None
        if self.cache is not None:
            self.cache.release(seq)
            self.cache.evict_sequence(seq)
        else:
            self.ex.release_sequence(seq)
        if self.decoder is not None and hasattr(self.decoder.drafter, "forget"):
            self.decoder.drafter.forget(seq)
        with self.lock:
            self.running.remove(job)
            self.waiting.insert(0, job)
        self.log(f"swapped out a request at {len(job.full)} tokens (KV pool full); it resumes when pages free up")

    # ---- results ------------------------------------------------------------------------------

    def _emit(self, job: Job, tokens: list[int]) -> None:
        if job.t_first is None:
            job.t_first = time.perf_counter()
        for token in tokens:
            if len(job.out) >= job.max_new:
                break
            job.out.append(token)
            job.events.put(("token", token))
            if token in self.stop_ids:
                self._finish(job, "stop")
                return
        if len(job.out) >= job.max_new:
            self._finish(job, "length")
        elif job.seq is not None and job.seq.position + 1 >= self.ex.max_context:
            self._finish(job, "length")

    def _finish(self, job: Job, reason: str | None = None, *, error: BaseException | None = None) -> None:
        if job.state == "done":
            return
        job.state = "done"
        job.t_done = time.perf_counter()
        job.finish_reason = reason
        with self.lock:
            if job in self.running:
                self.running.remove(job)
            if job in self.waiting:
                self.waiting.remove(job)
        if job.seq is not None:
            self._release(job.seq)
            job.seq = None
        if error is not None:
            job.error = error
            job.events.put(("error", error))
        else:
            job.events.put(("done", reason))
