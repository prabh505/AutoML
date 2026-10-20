"""Evidence for the reflector (PRD §19.2; P12 deliverable 1).

Only computed summaries: the top of the leaderboard, normalized importances of **source**
columns (PI-16: one-hot columns are folded back into their source, so category labels never
appear), LLM feature provenance, the current plan without its rationale, and the remaining
budget. No raw rows, predictions, or holdout data.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from tabpilot.evaluation.metrics import selection_score
from tabpilot.evaluation.metrics import spec as metric_spec
from tabpilot.schemas import MetricName, PlannerOutput, Task
from tabpilot.schemas.experiment import LeaderboardEntry
from tabpilot.schemas.features import FeatureProvenance
from tabpilot.schemas.reflection import (
    FeatureImportance,
    ReflectionInput,
    ReflectionLeaderboardRow,
    RemainingBudget,
)
from tabpilot_runtime.transformers import LLM_PREFIX

TOP_ENTRIES = 5
TOP_FEATURES = 15


def source_column(output_name: str, sources: Iterable[str]) -> str | None:
    """Map a preprocessed column (`branch__col…`) back to its source column or LLM feature.

    The longest source name that the column (after its branch prefix) starts with wins, so
    `lowcat__contract_one_year` → `contract` and `num__when__year` → `when`.
    """
    _, _, rest = output_name.partition("__")
    rest = rest or output_name
    if rest.startswith(LLM_PREFIX):
        rest = rest.removeprefix(LLM_PREFIX)
    best = None
    for src in sources:
        if (rest == src or rest.startswith(src + "_")) and (best is None or len(src) > len(best)):
            best = src
    return best


def raw_importances(model_step: Any, columns: Sequence[str]) -> dict[str, float]:
    """Gain importance for boosting, impurity importance for forests, |coef| for linear models.
    Returns {preprocessed column: importance}; empty when the model exposes none.
    """
    est = getattr(model_step, "estimator_", model_step)
    if hasattr(est, "booster_"):
        values = np.asarray(est.booster_.feature_importance(importance_type="gain"), dtype=float)
    elif hasattr(est, "get_feature_importance"):
        values = np.asarray(est.get_feature_importance(), dtype=float)
    elif hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.atleast_2d(np.asarray(est.coef_, dtype=float))
        values = np.abs(coef).mean(axis=0)
    else:
        return {}
    if len(values) != len(columns):
        return {}
    return {c: float(v) for c, v in zip(columns, values, strict=True)}


def source_importances(raw: dict[str, float], sources: Iterable[str]) -> list[FeatureImportance]:
    """Fold preprocessed importances into source columns, normalize, keep the top 15."""
    names = list(sources)
    totals: dict[str, float] = defaultdict(float)
    for column, value in raw.items():
        src = source_column(column, names)
        if src is not None and value > 0:
            totals[src] += value
    total = sum(totals.values())
    if total <= 0:
        return []
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_FEATURES]
    return [FeatureImportance(name=n, importance=min(1.0, v / total)) for n, v in ranked]


def build_reflection_input(
    *,
    task: Task,
    metric: MetricName,
    entries: Sequence[LeaderboardEntry],
    importances: list[FeatureImportance],
    provenance: list[FeatureProvenance],
    plan: PlannerOutput,
    remaining: RemainingBudget,
    warnings: Sequence[str] = (),
) -> ReflectionInput:
    """§19.2. Leaderboard rows are the top 5 successful, comparable entries by mean."""
    usable = [e for e in entries if e.status == "ok" and e.comparable and e.mean is not None]
    top = sorted(usable, key=lambda e: (-selection_score(metric, e.mean or 0.0), e.entry_id))
    rows = [ReflectionLeaderboardRow(entry_id=e.entry_id, model=e.model, feature_set=e.feature_set,
                                     mean=e.mean or 0.0, std=e.std or 0.0,
                                     fold_scores=list(e.fold_scores), fit_seconds=e.fit_seconds)
            for e in top[:TOP_ENTRIES]]  # fmt: skip
    llm = [p for p in provenance if p.source != "baseline"]
    return ReflectionInput(task=task, primary_metric=metric_spec(metric), leaderboard_top=rows,
                           top_features=importances[:TOP_FEATURES], llm_features=llm,
                           current_plan=plan.model_copy(update={"rationale": ""}),
                           remaining_budget=remaining, warnings=list(warnings))  # fmt: skip
