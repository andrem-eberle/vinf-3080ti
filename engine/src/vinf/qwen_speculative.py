"""Greedy speculative decoding for qwen35 with a pluggable drafter.

Drafters:
  MtpDrafter   — the model's own MTP (nextn) block, drafting k tokens recursively.
  DFlashDrafter (vinf.dflash) — block-diffusion drafter conditioned on captured target features,
                 drafting k tokens in one block pass.

Per step, with target state committed through position P-1 and the next input x_P known:
  1. drafts = drafter.draft(x_P, P, k)
  2. Verify: one multi-token target pass over [x_P, d_1..d_k] (weights read once), with per-token
     SSM/conv snapshots; greedy targets t_0..t_k from one LM-head pass over all rows.
  3. Accept the longest prefix d_1..d_a equal to t_0..t_{a-1}; emit t_0..t_a.
  4. drafter.observe_verify(P, a + 1, emitted); roll the target back to keep a + 1 inputs.
The emitted sequence is exactly the target model's greedy continuation, whatever the drafter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vinf.errors import ConfigurationError
from vinf.qwen_gpu import QwenGpuExecutor


@dataclass(slots=True)
class SpeculativeStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    steps: int = 0
    drafted: int = 0
    accepted: int = 0
    cached_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    draft_seconds: float = 0.0
    verify_seconds: float = 0.0
    accepted_histogram: dict[int, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_step(self) -> float:
        return (self.generated_tokens - 1) / self.steps if self.steps else 0.0

    @property
    def decode_tokens_per_second(self) -> float:
        return (self.generated_tokens - 1) / self.decode_seconds if self.decode_seconds > 0 else 0.0


class MtpDrafter:
    """Drafts with the model's MTP block. Rows whose post-norm target hidden is known but whose MTP KV
    entry is not yet computed from true hidden states stay "pending" in the executor's "spec_pend"
    buffer; each draft runs the MTP over them (the last row yields d_1) and then recursively."""

    def __init__(self, executor: QwenGpuExecutor, *, target_hidden: str = "post", draft_hidden: str = "post") -> None:
        if executor.mtp_layer is None:
            raise ConfigurationError("the MTP drafter needs an executor created with mtp=True")
        if target_hidden not in ("post", "pre") or draft_hidden not in ("post", "pre"):
            raise ConfigurationError("hidden conventions must be 'post' (after final norm) or 'pre'")
        self.ex = executor
        self.target_hidden = target_hidden  # target rows fed to the MTP: after or before output_norm
        self.draft_hidden = draft_hidden  # MTP rows fed back when drafting recursively
        self.max_draft = executor.max_batch - 1
        self.pending_tokens: list[int] = []
        self.pending_pos = 0
        # Per-sequence state: pending rows live at spec_pend rows [slot * max_batch, ...).
        self._states: dict[int, tuple[list[int], int]] = {}
        self._bound = None
        self.row_base = 0

    def reset(self) -> None:
        self.pending_tokens, self.pending_pos = [], 0

    def bind(self, seq) -> None:
        """Switch the drafter to sequence seq (its pending rows and state)."""
        if self._bound is not None:
            self._states[id(self._bound)] = (self.pending_tokens, self.pending_pos)
        self._bound = seq
        self.row_base = seq.slot * self.ex.max_batch
        self.pending_tokens, self.pending_pos = self._states.pop(id(seq), ([], 0))

    def forget(self, seq) -> None:
        self._states.pop(id(seq), None)
        if self._bound is seq:
            self._bound = None
            self.pending_tokens, self.pending_pos = [], 0

    def bootstrap(self, src: str, row: int, position: int) -> None:
        """Start drafting for the bound sequence from post-norm hidden row `row` of src (its last forwarded
        position, position - 1) after passes the drafter did not observe (e.g. multi-sequence decode)."""
        h = self.ex.shapes.hidden
        self.ex.rt.copy("spec_pend", self.row_base * h, src, row * h, h)
        self.pending_tokens, self.pending_pos = [-1], position - 1

    @property
    def ready(self) -> bool:
        return bool(self.pending_tokens)

    def _hidden_source(self) -> str:
        return self.ex.post_norm_rows() if self.target_hidden == "post" else "h"

    def observe_prefill(self, start: int, chunk: list[int], prompt: list[int]) -> None:
        ex, h = self.ex, self.ex.shapes.hidden
        src = self._hidden_source()
        last = start + len(chunk) == len(prompt)
        if last:  # keep the final prompt row; its MTP row needs the first generated token
            ex.rt.copy("spec_pend", self.row_base * h, src, (len(chunk) - 1) * h, h)
            self.pending_tokens, self.pending_pos = [-1], start + len(chunk) - 1
        known = len(chunk) - (1 if last else 0)
        if known:
            ex.mtp_load_hidden(src, 0, known)
            ex.mtp_forward(prompt[start + 1 : start + 1 + known], start)

    def draft(self, last_token: int, position: int, k: int) -> list[int]:
        ex = self.ex
        tokens = list(self.pending_tokens)
        tokens[-1] = last_token
        ex.mtp_load_hidden("spec_pend", self.row_base, len(tokens))
        drafts = [ex.mtp_forward(tokens, self.pending_pos)]
        last_row = len(tokens) - 1
        pos = self.pending_pos + len(tokens)
        recur = "mtp_out" if self.draft_hidden == "post" else "mtp_h"
        while len(drafts) < k:
            ex.mtp_load_hidden(recur, last_row, 1)
            drafts.append(ex.mtp_forward([drafts[-1]], pos))
            last_row = 0
            pos += 1
        return drafts

    def observe_verify(self, position: int, keep: int, emitted: list[int]) -> None:
        h = self.ex.shapes.hidden
        src = "xn" if self.target_hidden == "post" else "h"  # greedy_rows() left post-norm rows in "xn"
        self.ex.rt.copy("spec_pend", self.row_base * h, src, 0, keep * h)
        self.pending_tokens, self.pending_pos = list(emitted), position


class QwenSpeculativeDecoder:
    def __init__(
        self,
        executor: QwenGpuExecutor,
        k: int,
        *,
        drafter=None,
        target_hidden: str = "post",
        draft_hidden: str = "post",
    ) -> None:
        if k < 1:
            raise ConfigurationError("speculative k must be >= 1")
        if drafter is None:
            drafter = MtpDrafter(executor, target_hidden=target_hidden, draft_hidden=draft_hidden)
        if executor.snapshot_tokens < k + 1 or executor.max_batch < k + 1:
            raise ConfigurationError(f"k={k} needs max_batch and snapshot_tokens >= {k + 1}")
        if k > drafter.max_draft:
            raise ConfigurationError(f"this drafter drafts at most {drafter.max_draft} tokens per step")
        self.ex = executor
        self.k = k
        self.drafter = drafter

    def _prefill(self, prompt: list[int], prefix_cache=None) -> tuple[int, int]:
        """Batched target prefill (resuming from the prefix cache when possible), feeding the drafter;
        returns (first greedy token, reused prompt tokens)."""
        ex = self.ex
        start = prefix_cache.begin(prompt) if prefix_cache is not None else 0
        if prefix_cache is None:
            ex.reset()
        self._bind()
        ex.prefill(prompt, start=start,
                   before_last=(lambda: prefix_cache.checkpoint(prompt)) if prefix_cache is not None else None,
                   observe=lambda i, chunk: self.drafter.observe_prefill(i, chunk, prompt))
        return ex.greedy_next(), start

    def _bind(self) -> None:
        bind = getattr(self.drafter, "bind", None)
        if bind is not None:
            bind(self.ex.seq)

    def step(self, last_token: int, max_tokens: int) -> list[int]:
        """One draft + verify step for the active sequence (last_token is its next input, not yet
        forwarded); returns the emitted tokens (1..k+1, at most max_tokens)."""
        ex = self.ex
        position = ex.position
        k = min(self.k, ex.max_context - position - 1, max_tokens - 1)
        if k < 1:
            ex.forward_tokens([last_token])
            return [ex.greedy_next()]
        drafts = self.drafter.draft(last_token, position, k)
        ex.forward_tokens([last_token] + drafts, snapshot=True)
        targets = ex.greedy_rows()
        accepted = 0
        while accepted < k and drafts[accepted] == targets[accepted]:
            accepted += 1
        keep = accepted + 1
        emitted = targets[:keep]
        self.drafter.observe_verify(position, keep, emitted)
        ex.rollback(keep)
        self.last_accepted, self.last_drafted = accepted, k
        return emitted

    def generate(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        *,
        stop_token_ids: frozenset[int] = frozenset(),
        on_token=None,
        prefix_cache=None,
    ) -> tuple[list[int], SpeculativeStats]:
        ex = self.ex
        if not prompt_tokens:
            raise ConfigurationError("prompt_tokens must not be empty")
        if len(prompt_tokens) + max_new_tokens > ex.max_context:
            raise ConfigurationError(
                f"prompt ({len(prompt_tokens)}) + max_new_tokens ({max_new_tokens}) exceeds max_context {ex.max_context}"
            )
        self.drafter.reset()
        stats = SpeculativeStats(prompt_tokens=len(prompt_tokens))
        t0 = time.perf_counter()
        token, stats.cached_tokens = self._prefill(prompt_tokens, prefix_cache)
        stats.prefill_seconds = time.perf_counter() - t0
        out = [token]
        if on_token is not None:
            on_token(token)
        position = len(prompt_tokens)  # next input token out[-1] goes to this position
        t_decode = time.perf_counter()
        while len(out) < max_new_tokens and out[-1] not in stop_token_ids:
            k = min(self.k, ex.max_context - position - 1, max_new_tokens - len(out))
            if k < 1:
                break
            t1 = time.perf_counter()
            drafts = self.drafter.draft(out[-1], position, k)
            t2 = time.perf_counter()
            ex.forward_tokens([out[-1]] + drafts, snapshot=True)
            targets = ex.greedy_rows()
            accepted = 0
            while accepted < k and drafts[accepted] == targets[accepted]:
                accepted += 1
            keep = accepted + 1
            emitted = targets[:keep]
            self.drafter.observe_verify(position, keep, emitted)
            ex.rollback(keep)
            stats.verify_seconds += time.perf_counter() - t2
            stats.draft_seconds += t2 - t1
            stats.steps += 1
            stats.drafted += k
            stats.accepted += accepted
            stats.accepted_histogram[accepted] = stats.accepted_histogram.get(accepted, 0) + 1
            position += keep
            for tok in emitted:
                if len(out) >= max_new_tokens:
                    break
                out.append(tok)
                if on_token is not None:
                    on_token(tok)
                if tok in stop_token_ids:
                    break
        stats.decode_seconds = time.perf_counter() - t_decode
        stats.generated_tokens = len(out)
        return out, stats
