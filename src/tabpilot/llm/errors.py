"""Provider error classes and classification (PRD §23.5, §23.9).

Adapters convert every client exception into one of these classes, so the router's retry rules
depend only on the class. Messages never include response bodies (§23.9).
"""

from __future__ import annotations

from typing import Any

SERVER_ERROR_MIN = 500
RATE_LIMIT = 429
TIMEOUT_STATUS = 408
AUTH_STATUSES = frozenset({401, 403})


class LLMError(Exception):
    """A classified provider failure. `user_message` is generic and safe to display."""

    user_message = "The language model provider returned an error"


class RateLimited(LLMError):  # noqa: N818 - PRD names (§23.5)
    """HTTP 429. `retry_after` is in seconds when the provider says how long to wait."""

    user_message = "Provider rate limit reached"

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__(f"rate limited (retry_after={retry_after})")
        self.retry_after = retry_after


class LLMTimeout(LLMError):  # noqa: N818 - PRD names (§23.5)
    """The call exceeded its timeout."""

    user_message = "Provider did not respond in time"


class ServerError(LLMError):
    """HTTP 5xx."""

    user_message = "Provider is having problems"


class AuthError(LLMError):
    """HTTP 401/403: the key is missing, wrong, or not allowed. The provider is skipped."""

    user_message = "Provider rejected the API key"


class BadRequest(LLMError):  # noqa: N818 - PRD names (§23.5)
    """HTTP 4xx other than 401/403/408/429. `status` is kept for the record."""

    user_message = "Provider rejected the request"

    def __init__(self, status: int | None = None) -> None:
        super().__init__(f"bad request (status={status})")
        self.status = status


class LLMConnectionError(LLMError):
    """The provider could not be reached."""

    user_message = "Provider could not be reached"


class LLMUnavailable(Exception):  # noqa: N818 - PRD name (§23.1)
    """No provider produced a response. Callers switch to the deterministic fallback (I-8)."""

    def __init__(self, purpose: str, reason: str = "no provider available") -> None:
        super().__init__(f"{purpose}: {reason}")
        self.purpose = purpose
        self.reason = reason


class LLMBudgetExhausted(LLMUnavailable):
    """A per-run LLM ceiling was reached (§23.7, FR-LLM-02). Also a fallback trigger."""


def classify_status(status: int, retry_after: float | None = None) -> LLMError:
    """Map an HTTP status to its class (§23.5)."""
    if status == RATE_LIMIT:
        return RateLimited(retry_after)
    if status in AUTH_STATUSES:
        return AuthError(f"HTTP {status}")
    if status == TIMEOUT_STATUS:
        return LLMTimeout(f"HTTP {status}")
    if status >= SERVER_ERROR_MIN:
        return ServerError(f"HTTP {status}")
    return BadRequest(status)


def _retry_after(headers: Any) -> float | None:
    value = headers.get("retry-after") if headers is not None else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify(exc: BaseException) -> LLMError:  # noqa: PLR0911 - one exit per class
    """Classify an `openai` client exception (or any exception) without reading its body."""
    import openai  # noqa: PLC0415 - keeps import cost out of modules that only need the classes

    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeout("timeout")
    if isinstance(exc, openai.APIConnectionError):
        return LLMConnectionError("connection failed")
    if isinstance(exc, openai.APIStatusError):
        return classify_status(exc.status_code, _retry_after(exc.response.headers))
    if isinstance(exc, TimeoutError):
        return LLMTimeout("timeout")
    if isinstance(exc, ConnectionError | OSError):
        return LLMConnectionError("connection failed")
    return ServerError(type(exc).__name__)
