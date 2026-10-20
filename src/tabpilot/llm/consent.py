"""Consent and privacy notices (PRD §24.4–§24.5, §23.2).

Consent is per session and per provider. The notice texts summarize §23.2 and must be checked
against each provider's current terms before deployment.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from tabpilot.schemas import ProviderName

VERIFY = "Summary only; verify the provider's current terms before deployment."
WHAT_IS_SENT = ("TabPilot sends only the compact dataset profile (column names, types and "
                "aggregate statistics), leaderboard summaries, and fixed prompt templates. It "
                "never sends raw rows, category labels, the holdout, file names, or your "
                "identity (§24.4).")  # fmt: skip
NOTICES: dict[ProviderName, str] = {
    ProviderName.GROQ: "Groq: metadata is sent to Groq and processed under its API terms.",
    ProviderName.GEMINI: ("Google Gemini: on the free tier, inputs may be used to improve "
                          "Google's products and may be reviewed by humans. Use only with "
                          "non-sensitive metadata."),
    ProviderName.OPENROUTER: ("OpenRouter: requests are routed to third-party model hosts; "
                              "free models rotate and each host has its own terms."),
    ProviderName.OLLAMA: "Ollama: the model runs on your machine; nothing leaves it.",
}  # fmt: skip


def privacy_notice(provider: ProviderName) -> str:
    """The notice shown before the first LLM-enabled run (§24.5)."""
    return f"{NOTICES[provider]} {WHAT_IS_SENT} {VERIFY}"


class ProviderConsent(BaseModel):
    """One session's consent decision for one provider."""

    provider: ProviderName
    granted: bool
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConsentStore:
    """Per-session consent. Providers without a granted decision are never called."""

    def __init__(self, decisions: list[ProviderConsent] | None = None) -> None:
        self._decisions: dict[ProviderName, ProviderConsent] = {}
        for decision in decisions or []:
            self.record(decision)

    @classmethod
    def granted_for(cls, providers: list[ProviderName]) -> ConsentStore:
        return cls([ProviderConsent(provider=p, granted=True) for p in providers])

    def record(self, decision: ProviderConsent) -> None:
        self._decisions[decision.provider] = decision

    def allows(self, provider: str) -> bool:
        try:
            name = ProviderName(provider)
        except ValueError:
            return False
        decision = self._decisions.get(name)
        return bool(decision and decision.granted)
