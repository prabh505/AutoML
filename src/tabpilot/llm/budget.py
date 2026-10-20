"""Per-run LLM budget enforcement (PRD §23.7, FR-LLM-02; A-004; DEC-017).

Every provider attempt (first try, retry, fallback, repair) consumes one call. Tokens come from
each response (estimated when the provider reports none). Wall time accumulates across calls,
including backoff sleeps. Hitting any ceiling raises `LLMBudgetExhausted`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tabpilot.llm.errors import LLMBudgetExhausted
from tabpilot.schemas import LLMBudget, LLMResponse
from tabpilot.schemas.run import BudgetLedger


@dataclass
class BudgetTracker:
    """Mutable usage against an `LLMBudget`, shared by every LLM step of a run."""

    budget: LLMBudget
    clock: Callable[[], float] = time.monotonic
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    _since: float | None = field(default=None, repr=False)

    def start(self) -> None:
        """Begin timing one `complete()` call."""
        self._since = self.clock()

    def stop(self) -> None:
        """Add the elapsed time since `start()` to the wall-time total."""
        if self._since is not None:
            self.wall_s += self.clock() - self._since
            self._since = None

    def wall_elapsed(self) -> float:
        running = self.clock() - self._since if self._since is not None else 0.0
        return self.wall_s + running

    def time_left(self) -> float:
        return max(0.0, self.budget.max_wall_s - self.wall_elapsed())

    def call_timeout(self, requested: float) -> float:
        """The timeout for the next attempt: request, per-call cap, and remaining wall time."""
        return max(0.0, min(requested, self.budget.per_call_timeout_s, self.time_left()))

    def check_call(self, purpose: str, prompt_tokens: int) -> None:
        """Raise before an attempt that would break a ceiling; otherwise count the attempt."""
        if self.calls >= self.budget.max_calls:
            raise LLMBudgetExhausted(purpose, "call budget exhausted")
        if self.time_left() <= 0:
            raise LLMBudgetExhausted(purpose, "LLM wall-time budget exhausted")
        if prompt_tokens > self.budget.max_input_tokens_per_call:
            raise LLMBudgetExhausted(purpose, "request exceeds the per-call input-token cap")
        if self.input_tokens + prompt_tokens > self.budget.max_input_tokens:
            raise LLMBudgetExhausted(purpose, "input-token budget exhausted")
        if self.output_tokens >= self.budget.max_output_tokens:
            raise LLMBudgetExhausted(purpose, "output-token budget exhausted")
        self.calls += 1

    def record(self, response: LLMResponse) -> None:
        """Add a successful response's token usage."""
        self.input_tokens += response.input_tokens or 0
        self.output_tokens += response.output_tokens or 0

    def add_to(self, ledger: BudgetLedger) -> BudgetLedger:
        """Copy LLM usage into the run's `budget_ledger/1.0` record."""
        return ledger.model_copy(update={
            "llm_calls_used": self.calls, "llm_input_tokens_used": self.input_tokens,
            "llm_output_tokens_used": self.output_tokens,
            "llm_seconds_used": round(self.wall_elapsed(), 3),
        })  # fmt: skip
