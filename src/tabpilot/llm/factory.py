"""Build the run's providers, cache and router from settings (PRD §23.3, §6.4)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tabpilot.config import Settings
from tabpilot.llm.adapters import (
    ClientFactory,
    OllamaProvider,
    OpenAICompatibleProvider,
    openai_client,
)
from tabpilot.llm.adapters.gemini import GeminiProvider
from tabpilot.llm.cache import DEFAULT_DIR, LLMCache
from tabpilot.llm.consent import ConsentStore
from tabpilot.llm.records import LLMCallRecorder
from tabpilot.llm.router import EventSink, LLMRouter
from tabpilot.schemas import ExecutionProfileName, LLMBudget, ProviderName

ADAPTERS: dict[ProviderName, type[OpenAICompatibleProvider]] = {
    ProviderName.GROQ: OpenAICompatibleProvider,
    ProviderName.OPENROUTER: OpenAICompatibleProvider,
    ProviderName.GEMINI: GeminiProvider,
    ProviderName.OLLAMA: OllamaProvider,
}
WALL_SHARE = 0.20  # LLM wall time per run: 20% of the run budget (§23.7, A-004)


def build_providers(
    settings: Settings, env: Mapping[str, str], client_factory: ClientFactory = openai_client
) -> list[OpenAICompatibleProvider]:
    """Adapters in the profile's configured order (`order_hosted` / `order_local`)."""
    return [ADAPTERS[name](name, settings.providers[name], env, client_factory)
            for name in settings.provider_order if name in settings.providers]  # fmt: skip


def default_budget(time_budget_s: float, max_calls: int) -> LLMBudget:
    """§23.7 defaults with the A-004 wall-time share."""
    return LLMBudget(max_calls=max_calls, max_wall_s=WALL_SHARE * time_budget_s)


def build_router(
    settings: Settings,
    env: Mapping[str, str],
    *,
    consent: ConsentStore,
    records_dir: Path | None = None,
    events: EventSink | None = None,
    cache_enabled: bool = True,
    cache_dir: Path = DEFAULT_DIR,
    client_factory: ClientFactory = openai_client,
) -> LLMRouter:
    """Hosted: in-memory cache for the run only. Local: on-disk cache with a 7-day TTL."""
    hosted = settings.profile == ExecutionProfileName.HOSTED
    cache = LLMCache(enabled=cache_enabled, directory=None if hosted else cache_dir)
    recorder = LLMCallRecorder(records_dir) if records_dir is not None else None
    return LLMRouter(build_providers(settings, env, client_factory), consent=consent, cache=cache,
                     recorder=recorder, events=events)  # fmt: skip
