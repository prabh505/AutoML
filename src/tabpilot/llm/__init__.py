"""Provider access (PRD §23, FR-LLM-*).

Interface: LLMRouter.complete(req, budget). Depends on schemas, config.

Must not: DataFrames / pandas (payloads are dicts); logging prompt contents on hosted.
"""

from tabpilot.llm.adapters import OllamaProvider, OpenAICompatibleProvider
from tabpilot.llm.adapters.gemini import GeminiProvider
from tabpilot.llm.budget import BudgetTracker
from tabpilot.llm.cache import LLMCache, cache_key, request_hash
from tabpilot.llm.consent import ConsentStore, ProviderConsent, privacy_notice
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
    classify,
    classify_status,
)
from tabpilot.llm.factory import build_providers, build_router, default_budget
from tabpilot.llm.normalize import estimate_tokens, extract_json
from tabpilot.llm.records import LLMCallRecorder
from tabpilot.llm.router import LLMProvider, LLMRouter

__all__ = [
    "AuthError",
    "BadRequest",
    "BudgetTracker",
    "ConsentStore",
    "GeminiProvider",
    "LLMBudgetExhausted",
    "LLMCache",
    "LLMCallRecorder",
    "LLMConnectionError",
    "LLMError",
    "LLMProvider",
    "LLMRouter",
    "LLMTimeout",
    "LLMUnavailable",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "ProviderConsent",
    "RateLimited",
    "ServerError",
    "build_providers",
    "build_router",
    "cache_key",
    "classify",
    "classify_status",
    "default_budget",
    "estimate_tokens",
    "extract_json",
    "privacy_notice",
    "request_hash",
]
