"""P09 rule-based planner tests: T-PLN-09 and the §12.8 rules."""

from __future__ import annotations

import pytest

from tabpilot.planning import rule_based_plan
from tabpilot.schemas import (
    ClassWeight,
    ModelName,
    PlannerOutput,
    ProfileSummary,
    TargetTransform,
    Task,
)
from tabpilot.schemas.profile import TargetProfile
from tabpilot.tracking import dumps

ALL = [ModelName.LINEAR, ModelName.RANDOM_FOREST, ModelName.LIGHTGBM, ModelName.XGBOOST,
       ModelName.CATBOOST]  # fmt: skip


def summary(task: Task = Task.BINARY, **target: object) -> ProfileSummary:
    return ProfileSummary(task=task, n_rows_dev=1000, n_features=5, memory_mb=1.0,
                          duplicate_rows=0, conflicting_duplicate_rows=0, columns=[],
                          target=TargetProfile(**target))  # type: ignore[arg-type]  # fmt: skip


@pytest.mark.unit
def test_T_PLN_09_same_profile_gives_identical_plan_json() -> None:
    s = summary(class_counts={"no": 700, "yes": 300})
    first, second = rule_based_plan(s, Task.BINARY, ALL), rule_based_plan(s, Task.BINARY, ALL)
    assert dumps(first) == dumps(second)
    # the same contract an LLM plan must pass (§12.3)
    assert PlannerOutput.model_validate_json(first.model_dump_json()) == first


@pytest.mark.unit
def test_rules_models_allocation_encoding() -> None:
    plan = rule_based_plan(summary(class_counts={"no": 700, "yes": 300}), Task.BINARY, ALL)
    assert [m.name for m in plan.models] == [ModelName.LIGHTGBM, ModelName.RANDOM_FOREST,
                                             ModelName.LINEAR, ModelName.XGBOOST, ModelName.CATBOOST]  # fmt: skip
    assert [m.priority for m in plan.models] == [1, 2, 3, 4, 5]
    assert plan.tuning_allocation == {ModelName.LIGHTGBM: 0.6, ModelName.RANDOM_FOREST: 0.25,
                                      ModelName.LINEAR: 0.15}  # fmt: skip
    assert plan.features == []
    assert plan.column_decisions == []
    assert (plan.encoding.low_cardinality, plan.encoding.high_cardinality) == (
        "onehot",
        "frequency",
    )
    assert (plan.imputation.numeric, plan.imputation.categorical) == ("median", "most_frequent")
    assert plan.imputation.add_missing_indicators
    assert all(m.class_weight == ClassWeight.NONE for m in plan.models)


@pytest.mark.unit
def test_allocation_renormalized_over_available_models() -> None:
    plan = rule_based_plan(summary(), Task.BINARY, [ModelName.LINEAR, ModelName.HIST_GB])
    assert [m.name for m in plan.models] == [ModelName.HIST_GB, ModelName.LINEAR]
    assert plan.tuning_allocation == {ModelName.HIST_GB: pytest.approx(0.8),
                                      ModelName.LINEAR: pytest.approx(0.2)}  # fmt: skip
    only_xgb = rule_based_plan(summary(), Task.BINARY, [ModelName.XGBOOST])
    assert only_xgb.tuning_allocation == {}
    with pytest.raises(ValueError, match="no model"):
        rule_based_plan(summary(), Task.BINARY, [])


@pytest.mark.unit
def test_imbalance_turns_on_balanced_class_weights() -> None:
    plan = rule_based_plan(summary(class_counts={"a": 900, "b": 100}), Task.BINARY, ALL)
    assert {m.class_weight for m in plan.models} == {ClassWeight.BALANCED}


@pytest.mark.unit
@pytest.mark.parametrize(("target", "expected"), [
    ({"min": 0.0, "skew": 3.0}, TargetTransform.LOG1P),
    ({"min": 0.0, "skew": 1.5}, TargetTransform.NONE),
    ({"min": -1.0, "skew": 5.0}, TargetTransform.NONE),
    ({}, TargetTransform.NONE),
])  # fmt: skip
def test_log1p_only_for_nonnegative_skewed_regression(
    target: dict[str, float], expected: TargetTransform
) -> None:
    assert (
        rule_based_plan(summary(Task.REGRESSION, **target), Task.REGRESSION, ALL).target_transform
        == expected
    )
    assert (
        rule_based_plan(summary(Task.BINARY, **target), Task.BINARY, ALL).target_transform
        == TargetTransform.NONE
    )
