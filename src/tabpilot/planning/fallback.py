"""Rule-based fallback plan (PRD §12.8, FR-PLN-03; DEC-004).

A deterministic function of the DEV profile, the task, and the available models. It returns the
same `PlannerOutput` contract an LLM plan must satisfy, so downstream code has one path.
"""

from __future__ import annotations

from tabpilot.models import default_class_weight
from tabpilot.schemas import (
    ClassWeight,
    ModelName,
    PlannerOutput,
    ProfileSummary,
    TargetTransform,
    Task,
)
from tabpilot.schemas.plan import Encoding, Imputation, PlannedModel

SKEW_FOR_LOG1P = 2.0
ALLOCATION = {ModelName.LIGHTGBM: 0.60, ModelName.HIST_GB: 0.60, ModelName.RANDOM_FOREST: 0.25,
              ModelName.LINEAR: 0.15}  # fmt: skip
PRIORITY = (ModelName.LIGHTGBM, ModelName.HIST_GB, ModelName.RANDOM_FOREST, ModelName.LINEAR,
            ModelName.XGBOOST, ModelName.CATBOOST)  # fmt: skip
MAX_PLAN_MODELS = 5
RATIONALE = (
    "Rule-based plan (§12.8): all available models, baseline features only, median / "
    "most-frequent imputation, one-hot ≤ 15 categories, frequency encoding > 50."
)


def _target_transform(summary: ProfileSummary, task: Task) -> TargetTransform:
    target = summary.target
    if (task == Task.REGRESSION and target.min is not None and target.min >= 0
            and target.skew is not None and target.skew > SKEW_FOR_LOG1P):  # fmt: skip
        return TargetTransform.LOG1P
    return TargetTransform.NONE


def rule_based_plan(
    summary: ProfileSummary, task: Task, available: list[ModelName]
) -> PlannerOutput:
    """§12.8: keep all non-excluded features, baseline features only, every available model.

    Tuning allocation 60% LightGBM (or its hist_gb fallback), 25% random forest, 15% linear,
    normalized over the available models; other models train with default parameters only.
    """
    weight = ClassWeight.NONE
    if task != Task.REGRESSION and summary.target.class_counts:
        weight = default_class_weight(summary.target.class_counts)
    names = [m for m in PRIORITY if m in available][:MAX_PLAN_MODELS]
    if not names:
        raise ValueError("no model is available")
    models = [PlannedModel(name=m, priority=min(i + 1, 5), class_weight=weight)
              for i, m in enumerate(names)]  # fmt: skip
    shares = {m: ALLOCATION[m] for m in names if m in ALLOCATION}
    total = sum(shares.values())
    allocation = {m: round(v / total, 6) for m, v in shares.items()} if total else {}
    return PlannerOutput(
        imputation=Imputation(numeric="median", categorical="most_frequent",
                              add_missing_indicators=True),
        encoding=Encoding(low_cardinality="onehot", high_cardinality="frequency"),
        models=models,
        tuning_allocation=allocation,
        target_transform=_target_transform(summary, task),
        cv_folds=5,
        rationale=RATIONALE,
    )  # fmt: skip
