"""Provider adapters (PRD §23.1–§23.4; FR-LLM-01).

Groq, OpenRouter and Ollama use the `openai` client with a `base_url`. Gemini lives in
`tabpilot.llm.adapters.gemini` so that switching it to the `google-genai` SDK (D-11) touches
one file.
Model ids, endpoints, JSON-mode support and key variable names all come from
`config/providers.yaml`; nothing provider-specific is hard-coded here. Keys are read from the
environment at call time and never stored on the adapter, logged, or written anywhere (I-10).

Must not: pandas; hardcoded model ids or quotas.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from tabpilot.config import ProviderEntry
from tabpilot.llm.cache import canonical
from tabpilot.llm.consent import privacy_notice
from tabpilot.llm.errors import BadRequest, LLMError, classify
from tabpilot.llm.normalize import normalize
from tabpilot.schemas import LLMRequest, LLMResponse, ProviderName

NOT_FOUND = 404
NO_KEY = "not-needed"  # Ollama's local server ignores the key, but the client requires one
ClientFactory = Callable[[str | None, str, float], Any]


def openai_client(base_url: str | None, api_key: str, timeout_s: float) -> Any:
    """A fresh `openai.OpenAI` client; the router owns retries, so the client never retries."""
    import openai  # noqa: PLC0415 - imported only when a real provider call is made

    return openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)


class OpenAICompatibleProvider:
    """`LLMProvider` (§23.1) over an OpenAI-compatible chat-completions endpoint."""

    needs_key = True

    def __init__(
        self,
        name: ProviderName,
        entry: ProviderEntry,
        env: Mapping[str, str],
        client_factory: ClientFactory = openai_client,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name.value
        self.provider = name
        self.entry = entry
        self._env = env
        self._client_factory = client_factory
        self._clock = clock

    @property
    def base_url(self) -> str | None:
        return self.entry.base_url

    def _api_key(self) -> str | None:
        if not self.needs_key:
            return NO_KEY
        return self._env.get(self.entry.api_key_env or "") or None

    def available(self) -> bool:
        """Configured model id and (where needed) a key are present. No network call."""
        return bool(self.entry.configured_model_ids) and self._api_key() is not None

    def supports_json_mode(self) -> bool:
        return self.entry.json_mode

    def privacy_notice(self) -> str:
        return privacy_notice(self.provider)

    def _messages(self, req: LLMRequest) -> list[dict[str, str]]:
        return [{"role": "system", "content": req.system},
                {"role": "user", "content": canonical(req.user_json)}]  # fmt: skip

    def _call(self, model_id: str, req: LLMRequest, timeout_s: float) -> LLMResponse:
        client = self._client_factory(self.base_url, self._api_key() or "", timeout_s)
        kwargs: dict[str, Any] = {
            "model": model_id, "messages": self._messages(req), "temperature": req.temperature,
            "max_tokens": req.max_output_tokens, "timeout": timeout_s,
        }  # fmt: skip
        if self.supports_json_mode():
            kwargs["response_format"] = {"type": "json_object"}
        start = self._clock()
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:  # every client failure is classified (§23.5)
            raise classify(exc) from None
        content = completion.choices[0].message.content or ""
        usage = completion.usage
        return normalize(
            provider=self.name, model_id=completion.model or model_id, content=content,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            prompt_text=req.system + "".join(m["content"] for m in self._messages(req)),
            latency_s=self._clock() - start,
        )  # fmt: skip

    def complete(self, req: LLMRequest) -> LLMResponse:
        """One attempt within `req.timeout_s`. A model id the provider no longer serves (404)
        falls through to the next configured id (free OpenRouter models rotate, §23.3); other
        errors propagate classified.
        """
        timeout = req.timeout_s
        ids = self.entry.configured_model_ids
        last: LLMError = BadRequest(NOT_FOUND)
        for model_id in ids:
            try:
                return self._call(model_id, req, timeout)
            except BadRequest as exc:
                if exc.status != NOT_FOUND:
                    raise
                last = exc
        raise last


class OllamaProvider(OpenAICompatibleProvider):
    """Local Ollama server: no key needed; available when a model id is configured."""

    needs_key = False
