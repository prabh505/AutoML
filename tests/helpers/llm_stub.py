"""`ScriptedLLMProvider`: replays `fixtures/llm_responses/<provider>/<scenario>.json` (P03).

Implements the `LLMProvider` protocol of PRD §23.1 (`name`, `available`, `complete`,
`supports_json_mode`, `privacy_notice`). Each `complete` call consumes the next step:

- `response`: returns an `LLMResponse` built from an OpenAI-compatible body. `parsed` is set
  only when the content is plain JSON, because fence stripping is the router's job (§23.6).
- `http_error` / `timeout` / `connection_error`: raises the exception built by `raise_with`
  (default: the `Scripted*` classes below). P10 passes its own classified error types.

Every request is recorded in `requests` (T-LLM-06 payload identity, T-PRIV-02).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tabpilot.schemas import LLMRequest, LLMResponse

LLM_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "llm_responses"


class ScriptedProviderError(Exception):
    """Base class for scripted transport failures."""


class ScriptedRateLimited(ScriptedProviderError):
    def __init__(self, retry_after: float | None) -> None:
        super().__init__(f"429 retry_after={retry_after}")
        self.retry_after = retry_after


class ScriptedHTTPError(ScriptedProviderError):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


class ScriptedTimeout(ScriptedProviderError):
    """The scripted call timed out."""


class ScriptedConnectionError(ScriptedProviderError):
    """The scripted call could not connect."""


class ScriptExhausted(AssertionError):
    """The code under test made more calls than the script provides."""


def default_raise_with(step: dict[str, Any]) -> Exception:
    """Map a failure step to a scripted exception."""
    kind = step["kind"]
    if kind == "timeout":
        return ScriptedTimeout("timeout")
    if kind == "connection_error":
        return ScriptedConnectionError("connection refused")
    if step["status"] == 429:
        return ScriptedRateLimited(step.get("retry_after"))
    return ScriptedHTTPError(int(step["status"]))


def load_script(provider: str, scenario: str) -> list[dict[str, Any]]:
    """Load the steps of one scripted scenario."""
    data = json.loads((LLM_FIXTURES / provider / f"{scenario}.json").read_text(encoding="utf-8"))
    steps = data["steps"]
    assert isinstance(steps, list)
    return steps


@dataclass
class ScriptedLLMProvider:
    """A deterministic provider that replays scripted steps."""

    name: str
    steps: list[dict[str, Any]]
    json_mode: bool = True
    is_available: bool = True
    raise_with: Callable[[dict[str, Any]], Exception] = default_raise_with
    requests: list[LLMRequest] = field(default_factory=list)

    @classmethod
    def from_fixture(cls, provider: str, scenario: str, **kwargs: Any) -> ScriptedLLMProvider:
        return cls(name=provider, steps=load_script(provider, scenario), **kwargs)

    def available(self) -> bool:
        return self.is_available

    def supports_json_mode(self) -> bool:
        return self.json_mode

    def privacy_notice(self) -> str:
        return f"Scripted test provider '{self.name}': nothing leaves the test process."

    @property
    def calls(self) -> int:
        return len(self.requests)

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        if len(self.requests) > len(self.steps):
            raise ScriptExhausted(f"{self.name}: call {len(self.requests)} has no scripted step")
        step = self.steps[len(self.requests) - 1]
        if step["kind"] != "response":
            raise self.raise_with(step)
        body = step["body"]
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        return LLMResponse(
            provider=self.name, model_id=body["model"], content=content,
            parsed=parsed if isinstance(parsed, dict) else None,
            input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
            tokens_estimated=not usage, latency_s=0.0, cached=False, attempts=1,
        )  # fmt: skip
