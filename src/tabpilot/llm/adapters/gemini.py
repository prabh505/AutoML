"""Gemini adapter (PRD §23.2; D-11).

Uses Google's OpenAI-compatible endpoint by default. Switching to the `google-genai` SDK only
changes this file: subclass `OpenAICompatibleProvider` and override `_call`.
"""

from __future__ import annotations

from tabpilot.llm.adapters import OpenAICompatibleProvider

# Google's documented OpenAI-compatible base URL; `providers.yaml` may override it (verify before
# deployment).
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiProvider(OpenAICompatibleProvider):
    """Gemini through the OpenAI-compatible endpoint."""

    @property
    def base_url(self) -> str | None:
        return self.entry.base_url or GEMINI_OPENAI_BASE_URL
