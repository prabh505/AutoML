"""Report inputs and the facts table (PRD §21.1, §21.7 step 1; FR-RPT-01).

`ReportState` gathers everything a report is built from. `build_facts` turns it into
`{fact_id: (text_key, value, unit, rounding)}` for every number and name the narrative may
mention. Fact ids are assigned in a fixed order, so the same state always gives the same table.

PI-16: the facts table never holds category values (worst segments, class labels). It holds
column and model names only, so it is safe to send to an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tabpilot.evaluation.metrics import Problem
from tabpilot.explain import ErrorAnalysis, ExplanationResult
from tabpilot.schemas import (
    BudgetLedger,
    CandidateSpec,
    DatasetMetadata,
    EnsembleResult,
    ExperimentConfig,
    PlannerOutput,
    PlanValidation,
    ProfileSummary,
    RunEvent,
)
from tabpilot.schemas.experiment import HoldoutResult, LeaderboardEntry, SelectionRecord
from tabpilot.schemas.features import FeatureProvenance
from tabpilot.schemas.reflection import ReflectionRecord

ReportStatus = Literal["completed", "completed_partial", "failed", "cancelled"]
TOP_FEATURE_FACTS = 5
SCORE_DECIMALS = 4
CI_LEVEL_PERCENT = 95


@dataclass
class ReportState:
    """Everything one report is built from (§21.1 "Source of content" column)."""

    config: ExperimentConfig
    metadata: DatasetMetadata
    plan: PlannerOutput
    entries: list[LeaderboardEntry]
    selection: SelectionRecord
    holdout: HoldoutResult
    problem: Problem
    events: list[RunEvent] = field(default_factory=list)
    profile: ProfileSummary | None = None
    plan_source: Literal["llm", "rule_based"] = "rule_based"
    plan_validation: PlanValidation | None = None
    winner: CandidateSpec | None = None
    baseline_model: str | None = None
    provenance: list[FeatureProvenance] = field(default_factory=list)
    reflections: list[ReflectionRecord] = field(default_factory=list)
    ensemble: EnsembleResult | None = None
    thresholds: dict[str, Any] = field(default_factory=dict)
    ledger: BudgetLedger | None = None
    explanation: ExplanationResult | None = None
    errors: ErrorAnalysis | None = None
    winner_oof: NDArray[np.float32] | None = None
    y_dev: NDArray[Any] | None = None
    status: ReportStatus = "completed"
    warnings: list[str] = field(default_factory=list)

    def entry(self, entry_id: str) -> LeaderboardEntry | None:
        return next((e for e in self.entries if e.entry_id == entry_id), None)

    @property
    def winner_entry(self) -> LeaderboardEntry:
        found = self.entry(self.selection.winner)
        if found is None:
            raise ValueError("the selected entry is not on the leaderboard")
        return found


@dataclass(frozen=True)
class Fact:
    """One fact. Numbers carry their display rounding (decimals); names have `rounding=None`."""

    id: str
    key: str
    value: int | float | str
    unit: str = ""
    rounding: int | None = None

    @property
    def is_number(self) -> bool:
        return not isinstance(self.value, str)

    def display(self) -> str:
        """The value exactly as the report and the template narrative print it."""
        if isinstance(self.value, str):
            return self.value
        if isinstance(self.value, int):
            return str(self.value)
        return f"{self.value:.{self.rounding or 0}f}"

    def decimal(self) -> Decimal:
        return Decimal(self.display())


@dataclass
class FactsTable:
    """Facts in id order (`F1`, `F2`, …)."""

    facts: dict[str, Fact] = field(default_factory=dict)

    def add(self, key: str, value: int | float | str, unit: str = "",
            rounding: int | None = None) -> Fact:  # fmt: skip
        if isinstance(value, float) and rounding is None:
            raise ValueError(f"fact {key} needs a rounding")
        fact = Fact(f"F{len(self.facts) + 1}", key, value, unit,
                    None if isinstance(value, str) else (rounding or 0))  # fmt: skip
        self.facts[fact.id] = fact
        return fact

    def by_key(self, key: str) -> Fact | None:
        return next((f for f in self.facts.values() if f.key == key), None)

    def names(self) -> set[str]:
        return {f.value for f in self.facts.values() if isinstance(f.value, str)}

    def model_names(self) -> set[str]:
        return {str(f.value) for f in self.facts.values() if f.key.endswith("model")}

    def payload(self) -> dict[str, dict[str, Any]]:
        """JSON data for the narrator: no category values (PI-16)."""
        return {f.id: {"key": f.key, "value": f.display() if f.is_number else f.value,
                       "unit": f.unit} for f in self.facts.values()}  # fmt: skip

    def to_json(self) -> dict[str, Any]:
        return {f.id: {"key": f.key, "value": f.value, "unit": f.unit, "rounding": f.rounding,
                       "display": f.display()} for f in self.facts.values()}  # fmt: skip


def _score(scores: dict[str, Any] | None, metric: str) -> float | None:
    value = (scores or {}).get(metric)
    return float(value) if isinstance(value, int | float) else None


def _ok(entries: list[LeaderboardEntry]) -> list[LeaderboardEntry]:
    return [e for e in entries if e.status == "ok" and e.comparable and e.mean is not None]


def build_facts(state: ReportState) -> FactsTable:  # noqa: PLR0912, PLR0915 - one block per fact group
    """§21.7 step 1: every number and name the narrative may mention."""
    t = FactsTable()
    cfg = state.config
    metric = cfg.primary_metric.name.value
    win = state.winner_entry
    t.add("task", cfg.task.value)
    t.add("target", cfg.target)
    t.add("primary_metric", metric)
    t.add("final_model", str(win.model))
    t.add("cv_mean", float(win.mean or 0.0), rounding=SCORE_DECIMALS)
    t.add("cv_std", float(win.std or 0.0), rounding=SCORE_DECIMALS)
    t.add("n_folds", cfg.split.k * cfg.split.n_repeats, unit="folds")
    if cfg.split.n_repeats > 1:
        t.add("n_repeats", cfg.split.n_repeats, unit="repeats")
    t.add("n_models_compared", len(_ok(state.entries)), unit="models")
    failed = sum(1 for e in state.entries if e.status != "ok")
    if failed:
        t.add("n_models_failed", failed, unit="models")
    paired = state.selection.runner_up_paired
    runner = state.entry(paired.runner_up) if paired and paired.runner_up else None
    if paired and runner is not None and runner.mean is not None:
        t.add("runner_up_model", str(runner.model))
        t.add("runner_up_cv_mean", float(runner.mean), rounding=SCORE_DECIMALS)
        t.add("paired_mean_difference", float(paired.mean_diff), rounding=SCORE_DECIMALS)
        t.add("folds_won", paired.folds_won, unit="folds")
    hold = state.holdout
    final = _score(hold.final, metric)
    if hold.present and final is not None:
        t.add("holdout_score", final, rounding=SCORE_DECIMALS)
        base = _score(hold.baseline, metric)
        if base is not None and state.baseline_model:
            t.add("baseline_model", state.baseline_model)
            t.add("holdout_baseline_score", base, rounding=SCORE_DECIMALS)
        ci = (hold.final or {}).get("ci95")
        if isinstance(ci, list):
            t.add("ci_level", CI_LEVEL_PERCENT, unit="%")
            t.add("holdout_ci_low", float(ci[0]), rounding=SCORE_DECIMALS)
            t.add("holdout_ci_high", float(ci[1]), rounding=SCORE_DECIMALS)
    meta = state.metadata
    t.add("n_rows_valid", meta.n_rows_valid, unit="rows")
    removed = meta.n_rows_raw - meta.n_rows_valid
    if removed:
        t.add("n_rows_removed", removed, unit="rows")
    if state.profile is not None:
        t.add("n_rows_dev", state.profile.n_rows_dev, unit="rows")
        t.add("n_features", state.profile.n_features, unit="columns")
    excluded = sum(1 for c in meta.columns if c.exclusion_reason)
    if excluded:
        t.add("n_columns_excluded", excluded, unit="columns")
    if state.explanation is not None:
        for rank, name in enumerate(state.explanation.top_features()[:TOP_FEATURE_FACTS], 1):
            t.add(f"top_feature_{rank}", name)
    active = sum(1 for p in state.provenance if p.source != "baseline" and p.status == "active")
    if active:
        t.add("llm_features_active", active, unit="features")
    accepted = sum(1 for r in state.reflections if r.status == "accepted")
    if state.reflections:
        t.add("reflection_rounds", len(state.reflections), unit="rounds")
        t.add("reflection_rounds_accepted", accepted, unit="rounds")
    if state.ensemble is not None and state.ensemble.kept:
        t.add("ensemble_members", len(state.ensemble.members), unit="models")
    if state.ledger is not None:
        t.add("time_budget_s", state.ledger.time_budget_s, unit="s")
        t.add("trials_used", state.ledger.trials_used, unit="trials")
    return t
