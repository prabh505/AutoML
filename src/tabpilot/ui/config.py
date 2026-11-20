"""Build the `RunConfig` from the UI choices (PRD §28.2 `start_run(RunConfig)`, §3.2 stage 4).

Pure: it reads a mapping of session-state primitives and returns the config, or the list of
things the user still has to do. The Start button is enabled only when that list is empty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tabpilot.config import Settings
from tabpilot.runtime.catalog import metric_options
from tabpilot.schemas import (
    ColumnOverride,
    ColumnRole,
    MetricName,
    ProviderName,
    RunConfig,
    SemanticType,
    Task,
)
from tabpilot.ui import state as sk
from tabpilot.ui.components.target import NO_GROUP, confirmed_task

LLM_PROVIDERS = "providers"
LLM_OLLAMA = "ollama"
LLM_OFF = "off"
LLM_STUB = "stub"
LLM_LABELS = {
    LLM_PROVIDERS: "Use LLM providers",
    LLM_OLLAMA: "Use a local model (Ollama)",
    LLM_OFF: "No LLM (rule-based plan)",
    LLM_STUB: "Offline stub LLM (testing)",
}


@dataclass
class Draft:
    """A complete config, or what is missing."""

    config: RunConfig | None
    missing: list[str] = field(default_factory=list)


def _providers(state: Any, available: Sequence[str]) -> tuple[str, list[str]]:
    """(llm_mode, providers) for `RunConfig`."""
    mode = state.get(sk.LLM_MODE, LLM_PROVIDERS if available else LLM_OFF)
    if mode == LLM_OLLAMA:
        return LLM_PROVIDERS, [ProviderName.OLLAMA.value]
    if mode == LLM_PROVIDERS:
        chosen = [p for p in state.get(sk.PROVIDERS, list(available)) if p in available]
        return LLM_PROVIDERS, chosen
    return mode, []


def needs_consent(state: Any, available: Sequence[str]) -> list[str]:
    """Providers chosen in this session that have no consent yet (§24.5, per provider).
    Ollama runs on the user's machine; choosing it is the consent decision.
    """
    mode, providers = _providers(state, available)
    if mode != LLM_PROVIDERS or state.get(sk.LLM_MODE) == LLM_OLLAMA:
        return []
    granted = set(state.get(sk.CONSENTED, []))
    return [p for p in providers if p not in granted]


def overrides(state: Any) -> list[ColumnOverride]:
    out = []
    for column, o in sorted(state.get(sk.OVERRIDES, {}).items()):
        stype, role = o.get("semantic_type"), o.get("role")
        out.append(ColumnOverride(column=column,
                                  semantic_type=SemanticType(stype) if stype else None,
                                  role=ColumnRole(role) if role else None))  # fmt: skip
    return out


def draft(state: Any, settings: Settings, available: Sequence[str]) -> Draft:
    """The run's config from the current choices (FR-TGT-02: no start before the task click)."""
    missing: list[str] = []
    info = state.get(sk.TARGET_INFO) or {}
    upload_id = state.get(sk.UPLOAD_ID)
    if not upload_id:
        missing.append("upload a file or choose a sample")
    if not info.get("column") or info.get("error"):
        missing.append("choose a valid target column")
    task = confirmed_task(state)
    if info.get("column") and not info.get("error") and task is None:
        missing.append("confirm the task")
    if info.get("requires_confirmation") and not state.get(sk.CONFIRM_MISSING):
        missing.append("confirm the removal of rows without a target")
    mode, providers = _providers(state, available)
    if mode == LLM_PROVIDERS and not providers:
        missing.append("choose at least one LLM provider, or No LLM")
    if needs_consent(state, available):
        missing.append("answer the privacy notice for the chosen provider(s)")
    if missing or task is None:
        return Draft(None, missing)
    options, default = metric_options(task, info.get("positive_share"))
    metric = state.get(sk.METRIC)
    primary = MetricName(metric) if metric in {m.value for m in options} else default
    group = state.get(sk.GROUP)
    llm_mode = mode
    config = RunConfig(
        run_id=upload_id,
        target=info["column"],
        task=task,
        positive_label=info.get("positive_label") if task == Task.BINARY else None,
        primary_metric=primary,
        group_column=None if group in (None, NO_GROUP) else group,
        column_overrides=overrides(state),
        time_budget_s=int(state.get(sk.BUDGET, settings.limits.time_budget_default_s)),
        llm_mode=llm_mode,
        provider_order=[ProviderName(p) for p in providers],
        consented_providers=[ProviderName(p) for p in providers],
        profile_mode="minimal" if state.get(sk.MINIMAL) else "standard",
        execution_profile=settings.profile,
    )  # fmt: skip
    return Draft(config)
