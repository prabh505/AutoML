"""`LLMRouter`: ordered providers with retries, backoff, fallback and budgets (PRD §23.5).

```
for provider in order where available() and consented:
    for attempt in 1..3:
        budget.check_call()
        try: return provider.complete(req)
        except RateLimited(retry_after): wait if ≤ 20 s and time is left, else next provider
        except Timeout / 5xx / ConnectionError: sleep min(2^attempt + jitter(0..1), 10)
        except AuthError / BadRequest: provider unavailable for the run; next provider
raise LLMUnavailable
```

Fallback reuses the identical request (T-LLM-06). Cache hits cost zero calls (§23.8). Logs and
events carry metadata only; prompt and response bodies are never logged (I-11), and on the
hosted profile nothing about them is persisted except their SHA-256 in `llm_calls.jsonl`.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from tabpilot.llm.budget import BudgetTracker
from tabpilot.llm.cache import LLMCache, cache_key, canonical, request_hash, sha256
from tabpilot.llm.consent import ConsentStore
from tabpilot.llm.errors import (
    AuthError,
    BadRequest,
    LLMBudgetExhausted,
    LLMConnectionError,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    RateLimited,
    ServerError,
)
from tabpilot.llm.normalize import estimate_tokens, extract_json
from tabpilot.llm.records import LLMCallRecorder
from tabpilot.logging import get_logger
from tabpilot.schemas import EventType, LLMRequest, LLMResponse

MAX_ATTEMPTS = 3
MAX_RATE_LIMIT_WAIT_S = 20.0
MAX_BACKOFF_S = 10.0
EventSink = Callable[[EventType, dict[str, Any]], None]

log = get_logger("llm")


class LLMProvider(Protocol):
    """§23.1 provider interface."""

    name: str

    def available(self) -> bool: ...
    def complete(self, req: LLMRequest) -> LLMResponse: ...
    def supports_json_mode(self) -> bool: ...
    def privacy_notice(self) -> str: ...


def _model_id(provider: LLMProvider) -> str:
    entry = getattr(provider, "entry", None)
    ids = entry.configured_model_ids if entry is not None else []
    return ids[0] if ids else provider.name


class LLMRouter:
    """Routes one request through the ordered, consented providers (FR-LLM-01)."""

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        consent: ConsentStore,
        cache: LLMCache | None = None,
        recorder: LLMCallRecorder | None = None,
        events: EventSink | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.providers = list(providers)
        self.consent = consent
        self.cache = cache or LLMCache(enabled=False)
        self.recorder = recorder
        self._events = events
        self._sleep = sleep
        self._jitter = jitter
        self.unavailable: set[str] = set()  # providers disabled for the rest of the run
        self._template: str | None = None

    def _emit(self, kind: EventType, **data: Any) -> None:
        if self._events is not None:
            self._events(kind, data)

    def _record(self, **fields: Any) -> None:
        if self.recorder is not None:
            if self._template is not None:
                fields["template"] = self._template
            self.recorder.write(**fields)

    def usable(self) -> list[LLMProvider]:
        """Providers in order that are configured, consented, and not disabled for the run."""
        return [p for p in self.providers if p.name not in self.unavailable
                and self.consent.allows(p.name) and p.available()]  # fmt: skip

    def complete(
        self, req: LLMRequest, budget: BudgetTracker, template: str | None = None
    ) -> LLMResponse:
        """§23.5. Raises `LLMBudgetExhausted` or `LLMUnavailable`; callers fall back (I-8).
        `template` (e.g. `planner_system@1.0`) is recorded in `llm_calls.jsonl`.
        """
        budget.start()
        self._template = template
        try:
            return self._complete(req, budget)
        finally:
            budget.stop()
            self._template = None

    def _complete(self, req: LLMRequest, budget: BudgetTracker) -> LLMResponse:
        req_hash = request_hash(req)
        prompt_tokens = estimate_tokens(req.system + canonical(req.user_json))
        previous: str | None = None
        for provider in self.usable():
            if previous is not None:
                self._emit(EventType.LLM_FALLBACK, **{"from": previous, "to": provider.name})
                log.info("llm_fallback", from_provider=previous, to_provider=provider.name)
            previous = provider.name
            key = cache_key(provider.name, _model_id(provider), req)
            hit = self.cache.get(key)
            if hit is not None:
                self._record(purpose=req.purpose, provider=provider.name, model_id=hit.model_id,
                             outcome="cache_hit", cached=True, request_sha256=req_hash,
                             response_sha256=sha256(hit.content))  # fmt: skip
                return hit.model_copy(update={"cached": True, "attempts": 1})
            response = self._try_provider(provider, req, budget, req_hash, prompt_tokens)
            if response is not None:
                self.cache.put(key, response)
                return response
        self._emit(EventType.LLM_UNAVAILABLE, purpose=req.purpose)
        log.warning("llm_unavailable", purpose=req.purpose)
        raise LLMUnavailable(req.purpose)

    def _try_provider(
        self,
        provider: LLMProvider,
        req: LLMRequest,
        budget: BudgetTracker,
        req_hash: str,
        prompt_tokens: int,
    ) -> LLMResponse | None:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            budget.check_call(req.purpose, prompt_tokens)  # raises LLMBudgetExhausted
            timeout = budget.call_timeout(req.timeout_s)
            self._emit(EventType.LLM_CALL_STARTED, provider=provider.name, purpose=req.purpose,
                       attempt=attempt)  # fmt: skip
            try:
                response = provider.complete(req.model_copy(update={"timeout_s": timeout}))
            except LLMError as exc:
                self._failed(provider, req, attempt, exc, req_hash)
                if isinstance(exc, RateLimited):
                    wait = exc.retry_after
                    if (
                        wait is not None
                        and wait <= MAX_RATE_LIMIT_WAIT_S
                        and wait < budget.time_left()
                    ):
                        self._sleep(wait)
                        continue
                    return None
                if isinstance(exc, AuthError | BadRequest):
                    self.unavailable.add(provider.name)
                    return None
                if isinstance(exc, LLMTimeout | ServerError | LLMConnectionError):
                    if attempt < MAX_ATTEMPTS:
                        pause = min(2**attempt + self._jitter(), MAX_BACKOFF_S)
                        if pause >= budget.time_left():
                            raise LLMBudgetExhausted(req.purpose, "no time left to retry") from None
                        self._sleep(pause)
                    continue
                return None
            budget.record(response)
            parsed = response.parsed if response.parsed is not None else extract_json(
                response.content)  # fmt: skip
            response = response.model_copy(update={"attempts": attempt, "parsed": parsed})
            self._record(purpose=req.purpose, provider=provider.name,
                         model_id=response.model_id, attempt=attempt, outcome="success",
                         latency_s=round(response.latency_s, 3),
                         input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                         tokens_estimated=response.tokens_estimated, cached=False,
                         request_sha256=req_hash,
                         response_sha256=sha256(response.content))  # fmt: skip
            self._emit(EventType.LLM_CALL_SUCCEEDED, provider=provider.name,
                       latency=round(response.latency_s, 3), tokens=(response.input_tokens or 0)
                       + (response.output_tokens or 0))  # fmt: skip
            return response
        return None

    def _failed(
        self, provider: LLMProvider, req: LLMRequest, attempt: int, exc: LLMError, req_hash: str
    ) -> None:
        error_class = type(exc).__name__
        self._record(purpose=req.purpose, provider=provider.name, attempt=attempt,
                     outcome="failed", error_class=error_class, cached=False,
                     request_sha256=req_hash)  # fmt: skip
        self._emit(EventType.LLM_CALL_FAILED, provider=provider.name, error_class=error_class)
        log.warning("llm_call_failed", provider=provider.name, error_class=error_class,
                    attempt=attempt)  # fmt: skip
