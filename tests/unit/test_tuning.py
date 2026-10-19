"""P09 tuning tests: T-TUN-01…04, T-LEAK-02, limits, defaults (PRD §17)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest
from tests.helpers.fit_spy import FIT_LOG, FitSpy
from tests.helpers.guarded_frame import guard
from tests.helpers.prepared import prepared
from tests.helpers.synthetic import synthetic

from tabpilot.config import load_settings
from tabpilot.evaluation.metrics import Problem
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import build_pipeline
from tabpilot.schemas import CandidateSpec, ExecutionProfileName, MetricName, ModelName, Task
from tabpilot.schemas.experiment import ModelConfig
from tabpilot.splitting import build_split, iter_folds
from tabpilot.training import DevData, run_cv
from tabpilot.tuning import SPACES, TuningLimits, limits_for, tune

REG = ModelRegistry(load_settings({}))
LOCAL, HOSTED = ExecutionProfileName.LOCAL, ExecutionProfileName.HOSTED


def cand(name: ModelName = ModelName.LIGHTGBM, task: Task = Task.BINARY) -> CandidateSpec:
    return CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=name, task=task, seed=5, n_jobs=1))  # fmt: skip


def dev_split(n: int = 500) -> tuple[DevData, Any, Any]:
    frame, y, schema = synthetic(n)
    split = build_split(pd.Series(y), Task.BINARY).spec
    split = split.model_copy(update={"n_repeats": 1, "fold_assignment": split.fold_assignment[:1]})
    return DevData(frame, y, Problem(Task.BINARY, ("0", "1"), 1)), split, schema


def run_tune(limits: TuningLimits, name: ModelName = ModelName.LIGHTGBM, **kw: Any) -> Any:
    dev, split, schema = kw.pop("data", None) or dev_split()
    return tune(cand(name), dev, split, schema=schema, registry=REG, metric=MetricName.ROC_AUC,
                allocation_s=kw.pop("allocation_s", 60.0), limits=limits, seed=7, profile=LOCAL,
                **kw)  # fmt: skip


# ---------------------------------------------------------------- T-TUN-01 budget
@pytest.mark.unit
def test_T_TUN_01_time_mode_stops_at_allocation() -> None:
    allocation = 6.0
    start = time.monotonic()
    result = run_tune(TuningLimits(max_trials=1000, per_trial_timeout_s=3.0, mode="time"),
                      allocation_s=allocation)  # fmt: skip
    elapsed = time.monotonic() - start
    assert elapsed <= allocation * 1.1 + 15
    assert result.stopped == "allocation"
    assert result.elapsed_s <= allocation * 1.1 + 15


@pytest.mark.unit
def test_per_trial_timeout_fails_slow_trials_with_reason() -> None:
    clock_values = iter(np.arange(0.0, 1e6, 5.0))  # every clock read advances 5 s

    result = run_tune(TuningLimits(max_trials=3, per_trial_timeout_s=1.0, mode="time"),
                      allocation_s=1e5, clock=lambda: float(next(clock_values)))  # fmt: skip
    assert [t.state for t in result.trials] == ["fail", "fail", "fail"]
    assert {t.error for t in result.trials} == {"trial timeout"}
    assert result.best.result.status == "ok"  # no completed trial → default-config CV (§17.5)
    assert result.best_params == {}


# ---------------------------------------------------------------- T-TUN-02 determinism
@pytest.mark.unit
def test_T_TUN_02_trials_mode_is_deterministic() -> None:
    limits = TuningLimits(max_trials=7, per_trial_timeout_s=1.0, mode="trials")
    first, second = run_tune(limits), run_tune(limits)
    assert first.best_params == second.best_params
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert [t.state for t in first.trials] == [t.state for t in second.trials]
    assert first.trials[0].params == {}  # trial 0 = library defaults
    assert first.trials[1].params["n_estimators"] == 2000
    assert first.completed + first.pruned + first.failed == 7


@pytest.mark.unit
def test_pruning_and_best_trial_tie_break() -> None:
    result = run_tune(TuningLimits(max_trials=12, per_trial_timeout_s=1.0, mode="trials"),
                      name=ModelName.LINEAR)  # fmt: skip
    assert result.pruned > 0  # median pruner on the running fold mean
    values = [t.value for t in result.trials if t.state == "complete"]
    best = next(t for t in result.trials if t.params == result.best_params)
    assert best.value == max(values)
    assert best.number == min(t.number for t in result.trials if t.value == max(values))


# ---------------------------------------------------------------- T-TUN-03 failures
@pytest.mark.unit
def test_T_TUN_03_failed_trials_recorded_study_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    space = SPACES[ModelName.LIGHTGBM]

    def flaky(trial: optuna.Trial, task: Task, profile: ExecutionProfileName) -> dict[str, Any]:
        params = space(trial, task, profile)
        return {**params, "num_leaves": 1} if trial.number % 2 else params

    monkeypatch.setitem(SPACES, ModelName.LIGHTGBM, flaky)
    result = run_tune(TuningLimits(max_trials=6, per_trial_timeout_s=1.0, mode="trials"))
    failed = [t for t in result.trials if t.state == "fail"]
    assert [t.number for t in failed] == [1, 3, 5]
    assert all((t.error or "").startswith("LightGBMError on fold 1") for t in failed)
    assert len(result.trials) == 6
    assert result.completed >= 2
    assert result.best_params.get("num_leaves") != 1


@pytest.mark.unit
def test_five_consecutive_failures_stop_the_study(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SPACES, ModelName.LIGHTGBM, lambda t, task, p: {"num_leaves": 1})
    result = run_tune(TuningLimits(max_trials=20, per_trial_timeout_s=1.0, mode="trials"))
    assert result.stopped == "consecutive_failures"
    assert [t.state for t in result.trials] == ["complete"] + ["fail"] * 5


# ---------------------------------------------------------------- T-TUN-04 subsampling
@pytest.mark.unit
def test_T_TUN_04_subsampling_touches_training_folds_only() -> None:
    dev, split, schema = dev_split(600)

    def recording(c: CandidateSpec, *a: Any, **kw: Any) -> Any:
        pipe = build_pipeline(c, *a, **kw)
        return pipe.set_params(baseline_features=FitSpy(pipe.named_steps["baseline_features"],
                                                        key="first_step"))  # fmt: skip

    run = run_cv(cand(ModelName.LINEAR), dev, split, schema=schema, registry=REG,
                 metric=MetricName.ROC_AUC, build=recording, train_row_cap=200)  # fmt: skip
    assert {len(rows) for rows in FIT_LOG["first_step"]} == {200}
    assert run.oof is not None
    assert np.all(np.isfinite(run.oof))
    valid_sizes = [len(v) for _, _, _, v in iter_folds(split)]
    assert sum(valid_sizes) == len(split.dev_indices)


@pytest.mark.unit
def test_subsampled_tuning_recomputes_best_on_full_folds() -> None:
    data = dev_split(600)
    limits = TuningLimits(max_trials=3, per_trial_timeout_s=1.0, mode="trials", train_row_cap=200)
    result = run_tune(limits, name=ModelName.LINEAR, data=data)
    dev, split, schema = data
    c = cand(ModelName.LINEAR)
    full = run_cv(c.model_copy(update={"model": c.model.model_copy(update={"params": result.best_params})}),
                  dev, split, schema=schema, registry=REG, metric=MetricName.ROC_AUC)  # fmt: skip
    assert result.best.result.fold_scores == full.result.fold_scores


@pytest.mark.unit
def test_limits_for_profiles_small_data_and_subsampling() -> None:
    hosted = limits_for(HOSTED, 200.0, 10_000, trials_left=60)
    assert (hosted.max_trials, hosted.per_trial_timeout_s, hosted.train_row_cap) == (30, 50.0, None)
    assert limits_for(HOSTED, 1000.0, 10_000, 60).per_trial_timeout_s == 120.0  # A-004
    assert limits_for(LOCAL, 1000.0, 10_000, 300).per_trial_timeout_s == 600.0
    assert limits_for(LOCAL, 100.0, 1_500, 300).max_trials == 20  # small data
    assert limits_for(LOCAL, 100.0, 10_000, 300).max_trials == 100
    assert limits_for(LOCAL, 100.0, 10_000, 4).max_trials == 4  # run total cap
    assert limits_for(HOSTED, 100.0, 60_000, 60).train_row_cap == 50_000
    assert limits_for(LOCAL, 100.0, 60_000, 60).train_row_cap is None
    assert limits_for(LOCAL, 1.0, 10_000, 300, mode="trials", max_trials=3).mode == "trials"


# ---------------------------------------------------------------- T-LEAK-02 HOLD untouched
@pytest.mark.unit
def test_T_LEAK_02_tuning_never_reads_holdout_rows() -> None:
    prep = prepared("binary_basic.csv")
    assert prep.split.holdout_present
    frame = prep.dev.frame
    guarded = guard(frame, frame.index[prep.split.holdout_indices])
    dev = DevData(guarded, prep.dev.y, prep.problem)
    result = tune(cand(ModelName.LIGHTGBM), dev, prep.split, schema=prep.schema,
                  registry=prep.registry, metric=MetricName.ROC_AUC, allocation_s=60.0,
                  limits=TuningLimits(4, 60.0, "trials"), seed=1, profile=LOCAL)  # fmt: skip
    assert result.completed + result.pruned == 4


@pytest.mark.unit
@pytest.mark.parametrize("name", list(ModelName))
def test_every_search_space_produces_trainable_params(name: ModelName) -> None:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    trial = study.ask()
    params = SPACES[name](trial, Task.REGRESSION, HOSTED)
    est = REG.make_estimator(ModelConfig(name=name, task=Task.REGRESSION, seed=0, n_jobs=1,
                                         params=params))  # fmt: skip
    assert est is not None
    if name == ModelName.RANDOM_FOREST:
        assert trial.distributions["max_depth"].choices == (6, 10, 16, 20)
