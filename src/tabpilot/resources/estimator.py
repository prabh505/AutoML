"""Memory estimator v0 and simplification planner (PRD §29.4–§29.5, FR-ING-05; PI-46).

Coefficients are initial guesses and are calibrated by P18 (T-PERF-03). Until then
`estimator.safety` is 1.5 (config). The random-forest term drops the duplicated `rows` factor of
the §29.4 formula (PI-46). The model is n_estimators × leaf nodes × 32 B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import psutil

from tabpilot.config import Settings
from tabpilot.schemas import (
    DatasetMetadata,
    ErrorCode,
    FeasibilityReport,
    ModelName,
    RunConfig,
    SemanticType,
    Simplification,
    TabPilotError,
    Task,
)
from tabpilot.schemas.dataset import ColumnInfo

BYTES_BY_DTYPE = {"float64": 8, "int64": 8, "datetime64[ns]": 8, "float32": 4, "int32": 4,
                  "category": 4, "int16": 2, "int8": 1, "boolean": 1}  # fmt: skip
DATETIME_EXPANSION = 7  # year, month, day, dow, hour, is_weekend, days_since_min (§14.2)
ONEHOT_CAP = 16
ONEHOT_CAP_SIMPLIFIED = 8
RF_TREES = 200
RF_NODE_BYTES = 32
CANDIDATES_KEPT = 8
CANDIDATES_KEPT_NO_ENSEMBLE = 3
SHAP_ROWS = 1000
SHAP_BASE_MB = 50.0
SUBSAMPLE_ROWS = 50_000


@dataclass(frozen=True)
class Plan:
    """The settings the estimate depends on; simplifications change them."""

    models: tuple[ModelName, ...]
    k: int = 5
    onehot_cap: int = ONEHOT_CAP
    linear_onehot: bool = True
    candidates_kept: int = CANDIDATES_KEPT
    train_rows_cap: int | None = None


def _features(meta: DatasetMetadata) -> list[ColumnInfo]:
    return [c for c in meta.columns if c.role.value == "feature"]


def _bytes_per_row(cols: list[ColumnInfo]) -> float:
    total = 0.0
    for c in cols:
        width = BYTES_BY_DTYPE.get(c.dtype or "float64", 8)
        total += width * (DATETIME_EXPANSION if c.semantic_type == SemanticType.DATETIME else 1)
    return total


def _onehot_columns(cols: list[ColumnInfo], cap: int) -> int:
    """Extra one-hot columns for linear models. Cardinality is unknown here, so cap it."""
    return sum(cap for c in cols if c.semantic_type == SemanticType.CATEGORICAL)


def _n_outputs(meta: DatasetMetadata) -> int:
    if meta.target is None or meta.target.task in (None, Task.REGRESSION):
        return 1
    return max(len(meta.target.class_counts or {}), 2)


def rf_model_mb(train_rows: float, n_estimators: int = RF_TREES) -> float:
    """Random-forest model size: trees × leaf nodes (≈ rows / 2) × 32 B (§29.4, PI-46)."""
    return n_estimators * max(1.0, train_rows / 2) * RF_NODE_BYTES / 1e6


def estimate_peak_mb(meta: DatasetMetadata, plan: Plan, settings: Settings) -> float:
    """Estimated peak worker RSS in MB (§29.4)."""
    rows = meta.n_rows_valid
    cols = _features(meta)
    base = _bytes_per_row(cols) * rows / 1e6
    encoded = base
    if ModelName.LINEAR in plan.models and plan.linear_onehot:
        encoded_linear = base + _onehot_columns(cols, plan.onehot_cap) * 4 * rows / 1e6
    else:
        encoded_linear = base
    fold = encoded * (plan.k - 1) / plan.k
    train_rows = min(rows, plan.train_rows_cap) if plan.train_rows_cap else rows
    per_model = {
        ModelName.LINEAR: 2.5 * encoded_linear,
        ModelName.LIGHTGBM: 1.5 * encoded + 50,
        ModelName.HIST_GB: 1.5 * encoded + 50,
        ModelName.RANDOM_FOREST: rf_model_mb(train_rows * (plan.k - 1) / plan.k),
        ModelName.XGBOOST: 2.0 * encoded + 80,
        ModelName.CATBOOST: 2.5 * encoded + 120,
    }
    model_peak = max((per_model[m] for m in plan.models), default=0.0)
    oof = rows * _n_outputs(meta) * 4 * plan.candidates_kept / 1e6
    n_encoded_features = len(cols) + (
        _onehot_columns(cols, plan.onehot_cap) if plan.linear_onehot else 0
    )
    shap = SHAP_BASE_MB + SHAP_ROWS * n_encoded_features * _n_outputs(meta) * 8 / 1e6
    peak = settings.estimator.process_base_mb + base + fold + model_peak + oof + shap
    return peak * settings.estimator.safety


def memory_limit_mb(settings: Settings, physical_ram_mb: float | None = None) -> float:
    """The soft RSS limit is the budget the estimate must fit (§29.1, §39 step 3)."""
    ram = physical_ram_mb if physical_ram_mb is not None else psutil.virtual_memory().total / 2**20
    return settings.limits.rss_limits_mb(ram)[0]


def _simplify(plan: Plan, step: Simplification) -> Plan:  # noqa: PLR0911 - one exit per §29.5 step
    """Apply one §29.5 step. Steps that cannot change memory return the plan unchanged."""
    if step == Simplification.WITHIN_FOLD_SUBSAMPLE:
        return replace(plan, train_rows_cap=SUBSAMPLE_ROWS)
    if step == Simplification.DISABLE_RANDOM_FOREST:
        return replace(plan, models=tuple(m for m in plan.models if m != ModelName.RANDOM_FOREST))
    if step == Simplification.REDUCE_FOLDS:
        return replace(plan, k=min(plan.k, 3))
    if step == Simplification.CAP_ONEHOT:
        if plan.onehot_cap > ONEHOT_CAP_SIMPLIFIED:
            return replace(plan, onehot_cap=ONEHOT_CAP_SIMPLIFIED)
        return replace(plan, linear_onehot=False)
    if step == Simplification.DISABLE_ENSEMBLE:
        return replace(plan, candidates_kept=CANDIDATES_KEPT_NO_ENSEMBLE)
    return plan  # REDUCE_TUNING_TRIALS changes time, not memory


def estimate_feasibility(
    meta: DatasetMetadata,
    config: RunConfig,
    settings: Settings,
    physical_ram_mb: float | None = None,
) -> FeasibilityReport:
    """Estimate peak memory; simplify in §29.5 order; reject if still over (FR-ING-05).

    Only steps that lower the estimate are listed. A rejected report must be raised with
    `raise_if_infeasible` before any training starts.
    """
    models = tuple(settings.enabled_models())
    plan = Plan(models=models, candidates_kept=CANDIDATES_KEPT if config.enable_ensemble
                else CANDIDATES_KEPT_NO_ENSEMBLE)  # fmt: skip
    limit = memory_limit_mb(settings, physical_ram_mb)
    estimate = estimate_peak_mb(meta, plan, settings)
    applied: list[Simplification] = []
    for step in Simplification:
        if estimate <= limit:
            break
        candidate = _simplify(plan, step)
        new_estimate = estimate_peak_mb(meta, candidate, settings)
        if new_estimate < estimate:
            plan, estimate = candidate, new_estimate
            applied.append(step)
    status = "ok" if not applied else "simplified"
    if estimate > limit:
        status = "rejected"
    return FeasibilityReport(status=status, estimated_peak_mb=round(estimate, 1),
                             limit_mb=round(limit, 1), simplifications=applied)  # fmt: skip


def raise_if_infeasible(report: FeasibilityReport) -> None:
    """Raise E-RESOURCE-INFEASIBLE for a rejected report (FR-ING-05, US-F4)."""
    if report.status == "rejected":
        raise TabPilotError(ErrorCode.RESOURCE_INFEASIBLE, est=math.ceil(report.estimated_peak_mb),
                            limit=math.floor(report.limit_mb))  # fmt: skip
