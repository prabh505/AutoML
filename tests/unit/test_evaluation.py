"""P09 evaluation tests: T-EVL-01…05, holdout (T-LEAK-03 guard level), refit (PRD §18, §11.10)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from tests.helpers.prepared import prepared

from tabpilot.evaluation.holdout import (
    NO_HOLDOUT_REASON,
    bootstrap_ci,
    evaluate_holdout,
    make_guard,
)
from tabpilot.evaluation.leaderboard import make_entry, not_comparable, secondary_metrics
from tabpilot.evaluation.metrics import Problem
from tabpilot.evaluation.selection import (
    NoValidModelError,
    SelectionFreeze,
    SelectionFrozenError,
    paired_differences,
    select,
)
from tabpilot.schemas import CandidateSpec, MetricName, ModelName, Task
from tabpilot.schemas.experiment import LeaderboardEntry, ModelConfig, TrainingResult
from tabpilot.splitting.guard import HoldoutAlreadyOpened, HoldoutSealed
from tabpilot.training import baseline_candidate, refit_final, refit_params, run_cv

RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"


def entry(
    eid: str,
    scores: list[float],
    *,
    model: ModelName = ModelName.LIGHTGBM,
    status: str = "ok",
    n_features: int = 10,
    predict: float = 0.1,
    metric: MetricName = MetricName.ROC_AUC,
    split_id: str = "a" * 64,
    run_metric: MetricName | None = None,
) -> LeaderboardEntry:
    split = prepared("binary_basic.csv").split.model_copy(update={"split_id": split_id})
    result = TrainingResult(entry_id=eid, status=status, fold_scores=scores if status == "ok" else [],  # type: ignore[arg-type]
                            predict_seconds_mean=predict, error=None if status == "ok" else "x")  # fmt: skip
    return make_entry(result, run_id=RID, model=model, iteration=0, feature_set="baseline",
                      metric=metric, split=split, run_split_id="a" * 64,
                      run_metric=run_metric or metric, n_features_out=n_features)  # fmt: skip


# ---------------------------------------------------------------- T-EVL-01 integrity
@pytest.mark.unit
def test_T_EVL_01_statistics_recompute_from_fold_scores() -> None:
    e = entry("0-lightgbm-00000001", [0.81, 0.79, 0.84, 0.80, 0.78])
    scores = np.array(e.fold_scores)
    assert e.mean == pytest.approx(scores.mean(), abs=1e-12)
    assert e.std == pytest.approx(scores.std(ddof=1), abs=1e-12)
    assert e.sem == pytest.approx(scores.std(ddof=1) / math.sqrt(5), abs=1e-12)
    failed = entry("0-linear-00000002", [], status="failed")
    assert (failed.mean, failed.fold_scores, failed.status) == (None, [], "failed")
    assert failed.warnings == ["x"]  # the failure reason is kept on the board (§18.4)


# ---------------------------------------------------------------- T-EVL-02 selection
@pytest.mark.unit
def test_T_EVL_02_best_mean_wins_and_direction_respected() -> None:
    a = entry("0-lightgbm-0000000a", [0.90, 0.91, 0.92, 0.90, 0.91])
    b = entry(
        "0-random_forest-0000000b", [0.80, 0.81, 0.79, 0.80, 0.82], model=ModelName.RANDOM_FOREST
    )
    record = select([b, a])
    assert record.winner == a.entry_id
    assert record.near_ties == []
    loss = [entry("0-lightgbm-0000000c", [0.30] * 4 + [0.31], metric=MetricName.LOG_LOSS),
            entry("0-xgboost-0000000d", [0.50] * 4 + [0.52], model=ModelName.XGBOOST,
                  metric=MetricName.LOG_LOSS)]  # fmt: skip
    assert select(loss).winner == "0-lightgbm-0000000c"  # lower log loss is better
    paired = record.runner_up_paired
    assert paired is not None
    assert paired.runner_up == b.entry_id
    assert paired.folds_won == 5
    assert paired.n_folds == 5
    assert paired.mean_diff == pytest.approx(np.mean(np.array(a.fold_scores) - b.fold_scores))


# ---------------------------------------------------------------- T-EVL-03 tie-breaks
@pytest.mark.unit
def test_T_EVL_03_near_ties_prefer_simpler_family_then_features_speed_id() -> None:
    noisy = [0.80, 0.90, 0.85, 0.75, 0.95]  # SEM ≈ 0.035 → near-tie margin ≈ 0.0088
    best = entry("0-lightgbm-00000001", noisy)
    simpler = entry("0-linear-00000002", [s - 0.004 for s in noisy], model=ModelName.LINEAR)
    far = entry(
        "0-random_forest-00000003", [s - 0.05 for s in noisy], model=ModelName.RANDOM_FOREST
    )
    record = select([best, simpler, far])
    assert record.winner == simpler.entry_id
    assert record.near_ties == [best.entry_id]
    assert record.tie_break == "simpler model family"
    same_family = [entry("0-lightgbm-00000004", noisy, n_features=30),
                   entry("0-lightgbm-00000005", [s - 0.001 for s in noisy], n_features=12)]  # fmt: skip
    assert select(same_family).winner == "0-lightgbm-00000005"
    assert select(same_family).tie_break == "fewer output features"
    faster = [entry("0-lightgbm-00000006", noisy, predict=0.5),
              entry("0-lightgbm-00000007", noisy, predict=0.2)]  # fmt: skip
    assert select(faster).winner == "0-lightgbm-00000007"
    ids = [entry("0-lightgbm-0000000b", noisy), entry("0-lightgbm-0000000a", noisy)]
    assert select(ids).winner == "0-lightgbm-0000000a"
    assert select(ids).tie_break == "lexicographic entry id"


# ---------------------------------------------------------------- T-EVL-04/05 exclusion
@pytest.mark.unit
def test_T_EVL_04_failed_and_skipped_entries_are_never_ranked() -> None:
    ok = entry("0-linear-00000001", [0.70] * 5, model=ModelName.LINEAR)
    failed = entry("0-lightgbm-00000002", [], status="failed")
    skipped = entry("0-random_forest-00000003", [], status="skipped_budget")
    assert select([failed, ok, skipped]).winner == ok.entry_id
    with pytest.raises(NoValidModelError):
        select([failed, skipped])


@pytest.mark.unit
def test_T_EVL_05_non_comparable_entries_are_excluded() -> None:
    other_split = entry("0-lightgbm-00000001", [0.99] * 5, split_id="b" * 64)
    other_metric = entry("0-xgboost-00000002", [0.99] * 5, model=ModelName.XGBOOST,
                         metric=MetricName.AVERAGE_PRECISION, run_metric=MetricName.ROC_AUC)  # fmt: skip
    fine = entry("0-linear-00000003", [0.60] * 5, model=ModelName.LINEAR)
    assert not other_split.comparable
    assert not other_metric.comparable
    assert fine.comparable
    assert select([other_split, other_metric, fine]).winner == fine.entry_id
    assert [e.entry_id for e in not_comparable([other_split, fine, other_metric])] == [
        other_split.entry_id, other_metric.entry_id]  # fmt: skip


@pytest.mark.unit
def test_selection_after_freeze_is_refused() -> None:
    freeze = SelectionFreeze()
    e = entry("0-linear-00000001", [0.7] * 5, model=ModelName.LINEAR)
    assert select([e], freeze).winner == e.entry_id
    freeze.freeze()
    with pytest.raises(SelectionFrozenError):
        select([e], freeze)


@pytest.mark.unit
def test_paired_differences_need_equal_fold_counts() -> None:
    a, b = entry("0-linear-00000001", [0.7] * 5), entry("0-linear-00000002", [0.6] * 4 + [0.8])
    paired = paired_differences(a, b)
    assert (paired.folds_won, paired.min_diff, paired.max_diff) == (
        4,
        pytest.approx(-0.1),
        pytest.approx(0.1),
    )
    with pytest.raises(ValueError, match="shorter"):
        paired_differences(a, entry("0-linear-00000003", [0.7] * 3))


# ---------------------------------------------------------------- holdout (FR-EVL-02)
def fitted(name: ModelName = ModelName.LINEAR) -> tuple[Any, Any]:
    prep = prepared("binary_basic.csv")
    cand = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=name, task=Task.BINARY, seed=1, n_jobs=1))  # fmt: skip
    model = refit_final(cand, [], prep.dev, prep.split, schema=prep.schema, registry=prep.registry)
    return prep, model


@pytest.mark.unit
def test_T_LEAK_03_holdout_opens_once_after_freeze() -> None:
    prep, model = fitted()
    freeze = SelectionFreeze()
    guard = make_guard(prep.split, freeze.is_frozen)
    with pytest.raises(HoldoutSealed):
        evaluate_holdout(guard, model, None, frame=prep.dev.frame, y=prep.dev.y,
                         problem=prep.problem, metric=MetricName.ROC_AUC, seed=1)  # fmt: skip
    freeze.freeze()
    result = evaluate_holdout(guard, model, model, frame=prep.dev.frame, y=prep.dev.y,
                              problem=prep.problem, metric=MetricName.ROC_AUC, seed=1)  # fmt: skip
    assert result.present
    assert len(guard.access_log) == 1
    assert result.final is not None
    assert result.baseline == result.final
    low, high = result.final["ci95"]  # type: ignore[misc]
    assert low <= result.final["roc_auc"] <= high  # type: ignore[operator]
    with pytest.raises(HoldoutAlreadyOpened):
        evaluate_holdout(guard, model, None, frame=prep.dev.frame, y=prep.dev.y,
                         problem=prep.problem, metric=MetricName.ROC_AUC, seed=1)  # fmt: skip
    assert len(guard.access_log) == 1
    assert len(guard.refused_log) == 2


@pytest.mark.unit
def test_no_holdout_path_states_reason() -> None:
    prep, model = fitted()
    result = evaluate_holdout(None, model, None, frame=prep.dev.frame, y=prep.dev.y,
                              problem=prep.problem, metric=MetricName.ROC_AUC, seed=1)  # fmt: skip
    assert not result.present
    assert result.reason == NO_HOLDOUT_REASON


@pytest.mark.unit
def test_bootstrap_ci_is_seeded_and_skips_undefined_resamples() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.column_stack([1 - (q := rng.random(200)), q])
    problem = Problem(Task.BINARY, ("0", "1"), 1)
    assert bootstrap_ci(y, p, problem, MetricName.ROC_AUC, seed=3) == bootstrap_ci(
        y, p, problem, MetricName.ROC_AUC, seed=3)  # fmt: skip
    tiny = np.array([0, 1])
    ci = bootstrap_ci(
        tiny, np.array([[0.9, 0.1], [0.2, 0.8]]), problem, MetricName.ROC_AUC, seed=1, resamples=50
    )
    assert ci == [1.0, 1.0]  # single-class resamples were skipped
    assert (
        bootstrap_ci(
            np.array([1]), np.array([[0.2, 0.8]]), problem, MetricName.ROC_AUC, seed=1, resamples=5
        )
        is None
    )


@pytest.mark.unit
def test_regression_holdout_has_point_estimates_only() -> None:
    prep = prepared("regression_basic.csv")
    cand = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=ModelName.LINEAR, task=Task.REGRESSION, seed=1, n_jobs=1))  # fmt: skip
    model = refit_final(cand, [], prep.dev, prep.split, schema=prep.schema, registry=prep.registry)
    result = evaluate_holdout(make_guard(prep.split, lambda: True), model, None,
                              frame=prep.dev.frame, y=prep.dev.y, problem=prep.problem,
                              metric=MetricName.RMSE, seed=1)  # fmt: skip
    assert result.final is not None
    assert set(result.final) == {"rmse"}
    assert result.baseline is None


# ---------------------------------------------------------------- refit (§11.10)
@pytest.mark.unit
def test_refit_rounds_are_mean_best_iteration_times_1_1() -> None:
    lgbm = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=ModelName.LIGHTGBM, task=Task.BINARY, seed=1,
                                           n_jobs=1, params={"learning_rate": 0.05}))  # fmt: skip
    assert refit_params(lgbm, [100, 120, 110]) == {"learning_rate": 0.05, "n_estimators": 121}
    cat = lgbm.model_copy(
        update={"model": lgbm.model.model_copy(update={"name": ModelName.CATBOOST})}
    )
    assert refit_params(cat, [10])["iterations"] == 11
    linear = lgbm.model_copy(
        update={"model": lgbm.model.model_copy(update={"name": ModelName.LINEAR})}
    )
    assert "n_estimators" not in refit_params(linear, [10])


@pytest.mark.unit
def test_refit_uses_all_dev_rows_without_early_stopping() -> None:
    prep = prepared("binary_basic.csv")
    cand = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=ModelName.LIGHTGBM, task=Task.BINARY, seed=1, n_jobs=1))  # fmt: skip
    cv = run_cv(cand, prep.dev, prep.split, schema=prep.schema, registry=prep.registry,
                metric=MetricName.ROC_AUC)  # fmt: skip
    model = refit_final(cv.candidate, cv.best_iterations, prep.dev, prep.split, schema=prep.schema,
                        registry=prep.registry)  # fmt: skip
    step = model.named_steps["model"]
    assert step.early_stopping is False
    assert step.best_iteration_ is None
    expected = math.floor(np.mean(cv.best_iterations) * 1.1 + 0.5)
    assert step.estimator_.n_estimators == expected
    assert len(model.named_steps["select"].columns_) > 0


@pytest.mark.unit
def test_baseline_is_default_lightgbm_or_best_fallback() -> None:
    template = ModelConfig(
        name=ModelName.LINEAR, task=Task.BINARY, seed=1, n_jobs=1, params={"C": 2.0}
    )
    assert (
        baseline_candidate([ModelName.LINEAR, ModelName.LIGHTGBM], template).model.name
        == ModelName.LIGHTGBM
    )
    fallback = baseline_candidate([ModelName.LINEAR, ModelName.HIST_GB], template)
    assert fallback.model.name == ModelName.HIST_GB
    assert fallback.model.params == {}
    assert baseline_candidate([ModelName.LINEAR], template).model.name == ModelName.LINEAR


@pytest.mark.unit
def test_secondary_metrics_from_oof_exclude_primary() -> None:
    prep = prepared("binary_basic.csv")
    cand = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=ModelName.LINEAR, task=Task.BINARY, seed=1, n_jobs=1))  # fmt: skip
    cv = run_cv(cand, prep.dev, prep.split, schema=prep.schema, registry=prep.registry,
                metric=MetricName.ROC_AUC)  # fmt: skip
    sec = secondary_metrics(cv.oof, prep.dev.y, prep.split, prep.problem, MetricName.ROC_AUC)
    assert MetricName.ROC_AUC not in sec
    assert {MetricName.LOG_LOSS, MetricName.ACCURACY, MetricName.BRIER} <= set(sec)
    assert secondary_metrics(None, prep.dev.y, prep.split, prep.problem, MetricName.ROC_AUC) == {}
