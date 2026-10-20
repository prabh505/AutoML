"""Plan validation and repair (PRD §12.6–§12.7; FR-PLN-02).

`validate_plan` assumes the plan is wrong until proven right. It never calls the LLM and never
raises: a plan that cannot be used comes back with status `invalid` and the caller falls back to
the rule-based plan (§12.8). Issue codes match `fixtures/MANIFEST.yaml`: schema failures keep the
Pydantic error type (`float_type`, `too_long`, …); semantic rules use the codes below.

Status (§12.6): `valid` only with no issues; `repaired` with issues, none `severe`; `invalid` with
any `severe` issue. An advisory `cv_folds` that differs from the split is recorded only in
`cv_folds_ignored`; an allowed `metric_recommendation` is advisory and recorded nowhere else.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from tabpilot.evaluation.metrics import allowed_primary
from tabpilot.features import FeatureSchema, validate_specs
from tabpilot.logging import get_logger
from tabpilot.schemas import (
    MetricName,
    ModelName,
    PlannerOutput,
    PlanValidation,
    SemanticType,
    Simplification,
    TargetTransform,
    Task,
)
from tabpilot.schemas.plan import ColumnDecision, PlanIssue

ALLOCATION_TOLERANCE = 1e-6
NUMERIC_PARSEABLE = frozenset({SemanticType.NUMERIC, SemanticType.BOOLEAN})
UNSAFE_SHARE = 0.5
Severity = Literal["info", "repair", "severe"]
CostCheck = Callable[[PlannerOutput], tuple[PlannerOutput, list[Simplification]]]

log = get_logger("validate_plan")


@dataclass(frozen=True)
class PlanContext:
    """Run facts a plan is checked against. The planner can change none of them (§12.1)."""

    task: Task
    target: str
    primary_metric: MetricName
    feature_types: dict[str, SemanticType]
    available_models: list[ModelName]
    k: int
    excluded: frozenset[str] = frozenset()
    target_min: float | None = None
    minimums: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0
    cost_check: CostCheck | None = None

    def feature_schema(self) -> FeatureSchema:
        return FeatureSchema(columns=dict(self.feature_types), target=self.target,
                             excluded=self.excluded, minimums=dict(self.minimums),
                             n_rows=self.n_rows)  # fmt: skip


@dataclass
class _Issues:
    items: list[PlanIssue] = field(default_factory=list)

    def add(self, code: str, severity: Severity, message: str, path: str | None = None) -> None:
        self.items.append(PlanIssue(code=code[:60], severity=severity, message=message[:300],
                                    path=path[:120] if path else None))  # fmt: skip

    def status(self) -> Literal["valid", "repaired", "invalid"]:
        if any(i.severity == "severe" for i in self.items):
            return "invalid"
        return "repaired" if self.items else "valid"


def parse_plan(
    raw: str | dict[str, Any] | PlannerOutput,
) -> tuple[PlannerOutput | None, list[PlanIssue]]:
    """Structural parsing (Pydantic). Errors become `severe` issues with field paths."""
    issues = _Issues()
    if isinstance(raw, PlannerOutput):
        return raw, []
    data: Any = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.add("parse_error", "severe", f"not valid JSON: {exc.msg} at char {exc.pos}")
            return None, issues.items
    try:
        return PlannerOutput.model_validate(data), []
    except ValidationError as exc:
        for err in exc.errors():
            path = ".".join(str(p) for p in err["loc"])
            issues.add(str(err["type"]), "severe", str(err["msg"]), path or None)
        return None, issues.items


def _decisions(plan: PlannerOutput, ctx: PlanContext, issues: _Issues) -> list[ColumnDecision]:
    by_column: dict[str, list[ColumnDecision]] = defaultdict(list)
    for i, d in enumerate(plan.column_decisions):
        path = f"column_decisions.{i}.column"
        if d.column == ctx.target:
            issues.add("target_column", "repair", "decisions cannot target the target", path)
        elif d.column in ctx.excluded:
            issues.add("excluded_column", "repair", f"{d.column!r} is excluded", path)
        elif d.column not in ctx.feature_types:
            issues.add("unknown_column", "repair", f"{d.column!r} is not a listed column", path)
        elif (
            d.action == "treat_as_numeric" and ctx.feature_types[d.column] not in NUMERIC_PARSEABLE
        ):
            issues.add("not_numeric", "repair", f"{d.column!r} cannot be parsed as numeric", path)
        else:
            by_column[d.column].append(d)
    kept: list[ColumnDecision] = []
    for column, items in by_column.items():
        if len({d.action for d in items}) > 1:  # the conservative action wins: keep the default
            issues.add("contradictory_decision", "repair",
                       f"conflicting actions for {column!r}; kept with default type")  # fmt: skip
            continue
        kept.append(items[0])
    dropped = {d.column for d in kept if d.action == "drop"}
    if ctx.feature_types and dropped >= set(ctx.feature_types):
        issues.add("all_features_dropped", "severe", "the plan drops every feature column")
        return []
    return kept


def _models(plan: PlannerOutput, ctx: PlanContext, issues: _Issues) -> list[Any]:
    models = []
    for i, m in enumerate(plan.models):
        if m.name in ctx.available_models:
            models.append(m)
        else:
            issues.add("model_unavailable", "repair", f"{m.name} is not available",
                       f"models.{i}.name")  # fmt: skip
    if not models:
        issues.add("no_models", "severe", "no available model remains")
    return models


def _features(plan: PlannerOutput, ctx: PlanContext, issues: _Issues) -> list[Any]:
    accepted, found = validate_specs(plan.features, ctx.feature_schema())
    unsafe = sum(i.code == "unsafe_feature" for i in found)
    severe = bool(plan.features) and unsafe / len(plan.features) > UNSAFE_SHARE
    seen: set[tuple[str, str]] = set()
    for issue in found:
        if (issue.feature, issue.code) in seen:  # one issue per feature and code
            continue
        seen.add((issue.feature, issue.code))
        severity: Severity = "severe" if severe and issue.code == "unsafe_feature" else (
            "repair" if issue.fatal else "info")  # fmt: skip
        issues.add(issue.code, severity, issue.message, f"features[{issue.feature}]")
    return accepted


def _allocation(models: list[Any], plan: PlannerOutput, issues: _Issues) -> dict[ModelName, float]:
    names = [m.name for m in models]
    raw = {n: plan.tuning_allocation.get(n, 0.0) for n in names}
    total = sum(raw.values())
    normalized = ({n: v / total for n, v in raw.items()} if total > 0
                  else {n: 1.0 / len(names) for n in names} if names else {})  # fmt: skip
    removed = len(names) != len(plan.models)  # renormalizing is part of that repair
    if not removed and abs(total - 1.0) > ALLOCATION_TOLERANCE:
        issues.add("allocation_normalized", "repair", "tuning allocation normalized to sum 1.0")
    return normalized


def _semantic(
    plan: PlannerOutput, ctx: PlanContext, issues: _Issues
) -> tuple[PlannerOutput, bool, list[Simplification]]:
    decisions = _decisions(plan, ctx, issues)
    models = _models(plan, ctx, issues)
    features = _features(plan, ctx, issues)
    transform = plan.target_transform
    if transform == TargetTransform.LOG1P and (
        ctx.task != Task.REGRESSION or ctx.target_min is None or ctx.target_min < 0
    ):
        issues.add("target_transform_invalid", "repair",
                   "log1p needs a non-negative regression target",
                   "target_transform")  # fmt: skip
        transform = TargetTransform.NONE
    metric = plan.metric_recommendation
    if metric is not None and metric not in {m.value for m in allowed_primary(ctx.task)}:
        issues.add("metric_not_allowed", "repair", f"{metric!r} is not allowed for {ctx.task}",
                   "metric_recommendation")  # fmt: skip
        metric = None
    folds_ignored = plan.cv_folds != ctx.k  # advisory; the split is fixed (§12.6 note)
    repaired = plan.model_copy(update={
        "column_decisions": decisions, "models": models or plan.models, "features": features,
        "tuning_allocation": _allocation(models, plan, issues), "target_transform": transform,
        "metric_recommendation": metric,
    })  # fmt: skip
    simplifications: list[Simplification] = []
    if ctx.cost_check is not None and models:
        repaired, simplifications = ctx.cost_check(repaired)
        for step in simplifications:
            issues.add("cost_simplified", "repair", f"applied §29.5 step {step.value}")
    return repaired, folds_ignored, simplifications


def validate_plan(
    raw: str | dict[str, Any] | PlannerOutput, ctx: PlanContext
) -> tuple[PlannerOutput | None, PlanValidation]:
    """§12.6. Returns the repaired plan (None when it cannot be parsed) and its validation."""
    plan, parse_issues = parse_plan(raw)
    if plan is None:
        return None, PlanValidation(status="invalid", issues=parse_issues)
    issues = _Issues()
    try:
        repaired, folds_ignored, simplifications = _semantic(plan, ctx, issues)
    except Exception as exc:  # the validator never raises (FR-PLN-02 AC)
        log.error("plan_validation_error", error_class=type(exc).__name__)
        issues.add("internal_error", "severe", f"validator error: {type(exc).__name__}")
        return plan, PlanValidation(status="invalid", issues=issues.items)
    validation = PlanValidation(status=issues.status(), issues=issues.items,
                                cv_folds_ignored=folds_ignored,
                                simplifications=simplifications)  # fmt: skip
    return repaired, validation
