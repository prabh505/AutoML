"""Choices the Configure page offers: metrics per task, providers and their availability,
privacy notices, and the sample catalog (PRD §27.2, §24.5, §10.2, FR-ING-06).

Availability is decided from configuration only: a provider needs its API key variable to be set
and a model id configured; Ollama is offered on the `local` profile only (§24.5 "local only").
Key values are never read beyond a presence check and never returned (I-10).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tabpilot.config import Settings
from tabpilot.evaluation.metrics import default_primary
from tabpilot.ingestion.samples import Sample, samples
from tabpilot.llm.consent import WHAT_IS_SENT, privacy_notice
from tabpilot.schemas import Direction, ExecutionProfileName, MetricName, ProviderName, Task
from tabpilot.schemas._base import METRIC_DIRECTION, PRIMARY_METRICS

__all__ = [
    "WHAT_IS_SENT",
    "ProviderOption",
    "Sample",
    "metric_direction",
    "metric_options",
    "provider_options",
    "samples",
]


@dataclass(frozen=True)
class ProviderOption:
    """One provider row of the provider settings (§27.2): disabled ones carry the reason."""

    name: ProviderName
    available: bool
    reason: str | None
    notice: str


def metric_options(task: Task, positive_share: float | None = None
                   ) -> tuple[list[MetricName], MetricName]:  # fmt: skip
    """The §10.2 allowed primary metrics for `task` (default first) and the default."""
    default = default_primary(task, positive_share)
    rest = sorted((m for m in PRIMARY_METRICS[task] if m != default), key=lambda m: m.value)
    return [default, *rest], default


def metric_direction(metric: MetricName) -> str:
    """'higher is better' / 'lower is better' for axis labels (§27.2 score chart)."""
    up = METRIC_DIRECTION[metric] == Direction.MAXIMIZE
    return "higher is better" if up else "lower is better"


def provider_options(settings: Settings, env: Mapping[str, str]) -> list[ProviderOption]:
    """Providers in the profile's configured order, each available or disabled with a reason."""
    out = []
    for name in settings.provider_order:
        entry = settings.providers.get(name)
        reason: str | None = None
        if entry is None:
            reason = "not configured"
        elif name == ProviderName.OLLAMA and settings.profile != ExecutionProfileName.LOCAL:
            reason = "local profile only"
        elif entry.api_key_env and not env.get(entry.api_key_env):
            reason = f"no key (set {entry.api_key_env})"
        elif not entry.configured_model_ids:
            reason = "no model id configured"
        out.append(ProviderOption(name, reason is None, reason, privacy_notice(name)))
    return out
