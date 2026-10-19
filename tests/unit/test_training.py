"""P08 CV runner tests: T-TRN-01…05, T-LEAK-07, T-LEAK-08, T-FE-08, OOF, budgets (PRD §11.4, §16)."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from tests.helpers.fit_spy import FIT_LOG, FitSpy
from tests.helpers.guarded_frame import guard
from tests.helpers.prepared import prepared
from tests.helpers.synthetic import synthetic

from tabpilot.config import load_settings
from tabpilot.evaluation.metrics import Problem, default_primary, score
from tabpilot.evaluation.predict import predict_aligned as _predict
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import PipelineSchema, build_pipeline
from tabpilot.schemas import (
    FEATURE_SPEC_ADAPTER,
    CandidateSpec,
    ExecutionProfileName,
    MetricName,
    ModelName,
    SemanticType,
    Task,
)
from tabpilot.schemas.experiment import ENTRY_ID_PATTERN, ModelConfig
from tabpilot.schemas.plan import Encoding
from tabpilot.splitting import build_split, iter_folds
from tabpilot.training import (
    BudgetExceeded,
    Cancelled,
    DevData,
    OOFStore,
    entry_id_for,
    rf_resource_guard,
    run_cv,
    sanitize_error,
)

REG = ModelRegistry(load_settings({}))
FAST: dict[ModelName, dict[str, Any]] = {
    ModelName.RANDOM_FOREST: {"n_estimators": 20}, ModelName.CATBOOST: {"iterations": 60},
    ModelName.HIST_GB: {"max_iter": 30}, ModelName.LIGHTGBM: {"n_estimators": 60},
    ModelName.XGBOOST: {"n_estimators": 60},
}  # fmt: skip
DATASETS = ["binary_basic.csv", "multiclass_basic.csv", "regression_basic.csv"]


def candidate(
    model: ModelName,
    task: Task,
    features: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
    **kw: Any,
) -> CandidateSpec:
    specs = [FEATURE_SPEC_ADAPTER.validate_python(f) for f in features or []]
    config = ModelConfig(name=model, task=task, seed=3, n_jobs=1,
                         params=FAST.get(model, {}) if params is None else params)  # fmt: skip
    return CandidateSpec(candidate_id="c", iteration=0, model=config, features=specs,
                         feature_set="baseline+llm" if specs else "baseline", **kw)  # fmt: skip


def synthetic_dev(n: int = 400, seed: int = 0) -> tuple[DevData, Any, PipelineSchema]:
    frame, y, schema = synthetic(n, seed)
    split = build_split(pd.Series(y), Task.BINARY).spec
    dev = DevData(frame, y, Problem(Task.BINARY, ("0", "1"), 1))
    return dev, split, schema


# ---------------------------------------------------------------- T-TRN-01 every model × task
@pytest.mark.unit
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model", list(ModelName))
def test_T_TRN_01_every_model_trains_every_task(dataset: str, model: ModelName) -> None:
    prep = prepared(dataset)
    metric = default_primary(prep.task)
    store = OOFStore()
    run = run_cv(candidate(model, prep.task), prep.dev, prep.split, schema=prep.schema,
                 registry=prep.registry, metric=metric, store=store)  # fmt: skip
    result = run.result
    assert result.status == "ok", result.error
    assert len(result.fold_scores) == prep.split.k * prep.split.n_repeats
    assert re.match(ENTRY_ID_PATTERN, result.entry_id)
    n_out = 1 if prep.task == Task.REGRESSION else prep.problem.n_classes
    assert run.oof is not None
    assert run.oof.dtype == np.float32
    assert run.oof.shape == (len(prep.split.dev_indices), n_out)
    assert np.all(np.isfinite(run.oof))
    if prep.task != Task.REGRESSION:
        np.testing.assert_allclose(run.oof.sum(axis=1), 1.0, atol=1e-4)
    # each fold score is reproducible from the stored OOF rows (§11.4)
    position = {row: i for i, row in enumerate(prep.split.dev_indices)}
    for i, (_, _, _, valid) in enumerate(iter_folds(prep.split)):
        rows = [position[r] for r in valid]
        pred = run.oof[rows].astype(np.float64)
        value = score(metric, prep.dev.y[valid], pred[:, 0] if n_out == 1 else pred, prep.problem)
        assert value == pytest.approx(result.fold_scores[i], rel=1e-4, abs=1e-5)
    assert result.oof_ref is not None
    np.testing.assert_array_equal(store.get(result.oof_ref), run.oof)
    if model in (ModelName.LIGHTGBM, ModelName.XGBOOST, ModelName.CATBOOST):
        assert len(run.best_iterations) == len(result.fold_scores)


@pytest.mark.unit
def test_repeated_cv_averages_oof_over_repeats() -> None:
    prep = prepared("binary_small.csv")
    assert prep.split.n_repeats == 3
    run = run_cv(candidate(ModelName.LINEAR, Task.BINARY), prep.dev, prep.split,
                 schema=prep.schema, registry=prep.registry, metric=MetricName.ROC_AUC)  # fmt: skip
    assert len(run.result.fold_scores) == 15
    assert run.oof is not None
    assert np.all(np.isfinite(run.oof))
    # A row's OOF is the mean of its 3 per-repeat predictions, so it differs from any one repeat.
    single = prep.split.model_copy(update={"n_repeats": 1,
                                           "fold_assignment": prep.split.fold_assignment[:1]})  # fmt: skip
    first = run_cv(candidate(ModelName.LINEAR, Task.BINARY), prep.dev, single, schema=prep.schema,
                   registry=prep.registry, metric=MetricName.ROC_AUC)  # fmt: skip
    assert first.oof is not None
    assert not np.allclose(first.oof, run.oof)


# ---------------------------------------------------------------- T-TRN-02 fallback params
@pytest.mark.unit
def test_T_TRN_02_fold_one_failure_retries_with_defaults() -> None:
    dev, split, schema = synthetic_dev()
    run = run_cv(candidate(ModelName.LIGHTGBM, Task.BINARY, params={"num_leaves": 1}), dev, split,
                 schema=schema, registry=REG, metric=MetricName.ROC_AUC)  # fmt: skip
    assert run.result.status == "ok"
    assert run.warnings[0] == "fallback_params"
    assert run.result.best_params == {}
    assert run.candidate.model.params == {}


@pytest.mark.unit
def test_T_TRN_02_failure_with_defaults_or_later_fold_is_final() -> None:
    dev, split, schema = synthetic_dev()
    calls: list[int] = []

    def flaky(cand: CandidateSpec, *a: Any, **kw: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError("worker RSS above hard limit")
        return build_pipeline(cand, *a, **kw)

    run = run_cv(candidate(ModelName.LINEAR, Task.BINARY, params={"C": 0.5}), dev, split,
                 schema=schema, registry=REG, metric=MetricName.ROC_AUC, build=flaky)  # fmt: skip
    assert run.result.status == "failed"
    assert len(calls) == 2
    assert run.result.error == "MemoryError on fold 2: worker RSS above hard limit"
    defaults = run_cv(candidate(ModelName.LIGHTGBM, Task.BINARY, params={}), dev, split,
                      schema=schema, registry=REG, metric=MetricName.ROC_AUC,
                      build=lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))  # fmt: skip
    assert defaults.result.status == "failed"
    assert "fallback_params" not in defaults.warnings


# ---------------------------------------------------------------- T-TRN-03 isolation
@pytest.mark.unit
def test_T_TRN_03_one_model_failure_does_not_affect_others() -> None:
    """US-F7: injected LightGBM failure → LightGBM `failed`, the others complete."""
    dev, split, schema = synthetic_dev()

    def broken_lightgbm(cand: CandidateSpec, *a: Any, **kw: Any) -> Any:
        pipe = build_pipeline(cand, *a, **kw)
        if cand.model.name == ModelName.LIGHTGBM:
            raise ValueError("could not convert 'secret_cell_value' to float")
        return pipe

    results = {m: run_cv(candidate(m, Task.BINARY, params={}), dev, split, schema=schema,
                         registry=REG, metric=MetricName.ROC_AUC, build=broken_lightgbm)
               for m in (ModelName.LINEAR, ModelName.LIGHTGBM, ModelName.RANDOM_FOREST)}  # fmt: skip
    assert results[ModelName.LIGHTGBM].result.status == "failed"
    error = results[ModelName.LIGHTGBM].result.error or ""
    assert error.startswith("ValueError on fold 1")
    assert "secret_cell_value" not in error
    assert results[ModelName.LINEAR].result.status == "ok"
    assert results[ModelName.RANDOM_FOREST].result.status == "ok"


class NaNModel(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Estimator whose probabilities are NaN (non-finite output, §16.3)."""

    best_iteration_ = None

    def fit(self, X: Any, y: Any) -> NaNModel:
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.full((len(X), len(self.classes_)), np.nan)


@pytest.mark.unit
def test_non_finite_predictions_fail_the_candidate() -> None:
    dev, split, schema = synthetic_dev()

    def nan_model(cand: CandidateSpec, *a: Any, **kw: Any) -> Any:
        pipe = build_pipeline(cand, *a, **kw)
        pipe.steps[-1] = ("model", NaNModel())
        return pipe

    run = run_cv(candidate(ModelName.LINEAR, Task.BINARY, params={}), dev, split, schema=schema,
                 registry=REG, metric=MetricName.ROC_AUC, build=nan_model)  # fmt: skip
    assert run.result.status == "failed"
    assert (
        run.result.error == "_NonFiniteOutputError on fold 1: non-finite or misshaped predictions"
    )


@pytest.mark.unit
def test_sanitize_error_removes_quoted_values_and_keys() -> None:
    exc = ValueError(
        "bad value \"Alice Smith\" in col 'x' key sk-abcdefghijklmnopqrstuvwxyz123456\nmore"
    )
    text = sanitize_error(exc, 3)
    assert text.startswith("ValueError on fold 3: bad value <…> in col <…>")
    assert "Alice" not in text
    assert "abcdefghijklmnop" not in text
    assert "more" not in text
    assert sanitize_error(KeyError(), 1) == "KeyError on fold 1"


# ---------------------------------------------------------------- T-TRN-04 early stopping
@pytest.mark.unit
@pytest.mark.parametrize("model", [ModelName.LIGHTGBM, ModelName.XGBOOST, ModelName.CATBOOST])
def test_T_TRN_04_early_stopping_uses_inner_split_of_training_fold(model: ModelName) -> None:
    dev, split, schema = synthetic_dev()
    _, _, train, valid = next(iter_folds(split))
    pipe = build_pipeline(candidate(model, Task.BINARY), schema, REG)
    pipe.fit(dev.frame.iloc[train], dev.y[train])
    step = pipe.named_steps["model"]
    es_rows = set(step.early_stopping_rows_)
    assert es_rows <= set(train)
    assert not es_rows & set(valid)
    assert len(es_rows) == pytest.approx(0.1 * len(train), abs=1)
    share = dev.y[sorted(es_rows)].mean()
    assert share == pytest.approx(dev.y[train].mean(), abs=0.05)  # stratified
    assert step.best_iteration_ is not None
    assert step.best_iteration_ >= 1
    off = build_pipeline(candidate(model, Task.BINARY), schema, REG, early_stopping=False)
    off.fit(dev.frame.iloc[train], dev.y[train])
    assert off.named_steps["model"].early_stopping_rows_ == []
    assert off.named_steps["model"].best_iteration_ is None


# ---------------------------------------------------------------- T-TRN-05 RF resource skip
@pytest.mark.unit
def test_T_TRN_05_random_forest_skipped_on_hosted_when_too_big() -> None:
    dev, split, schema = synthetic_dev()
    guard_small = rf_resource_guard(ExecutionProfileName.HOSTED, train_rows=1_000_000,
                                    rss_budget_mb=2_000)  # fmt: skip
    # 200 trees × 500,000 leaves × 32 B = 3,200 MB > 40% of 2,000 MB
    run = run_cv(candidate(ModelName.RANDOM_FOREST, Task.BINARY, params={}), dev, split, schema=schema,
                 registry=REG, metric=MetricName.ROC_AUC, resource_guard=guard_small)  # fmt: skip
    assert run.result.status == "skipped_resource"
    assert "exceeds 40% of the 2000 MB budget" in (run.result.error or "")
    assert guard_small(candidate(ModelName.LIGHTGBM, Task.BINARY)) is None
    local = rf_resource_guard(ExecutionProfileName.LOCAL, train_rows=1_000_000, rss_budget_mb=2_000)
    assert local(candidate(ModelName.RANDOM_FOREST, Task.BINARY)) is None
    fits = rf_resource_guard(ExecutionProfileName.HOSTED, train_rows=1_000, rss_budget_mb=2_000)
    assert fits(candidate(ModelName.RANDOM_FOREST, Task.BINARY)) is None


# ---------------------------------------------------------------- T-FE-08 restart on feature failure
@pytest.mark.unit
def test_T_FE_08_feature_failure_in_one_fold_restarts_from_fold_one() -> None:
    dev, split, schema = synthetic_dev()
    _, _, _, valid0 = next(iter_folds(split))
    frame = dev.frame.copy()
    frame["sparse"] = 1.0
    frame.loc[frame.index[valid0[:5]], "sparse"] = np.nan  # missing only in fold 1's valid rows
    dev = DevData(frame, dev.y, dev.problem)
    schema = PipelineSchema({**schema.columns, "sparse": SemanticType.NUMERIC}, schema.column_map,
                            schema.distinct, schema.missing_rate, schema.n_rows)  # fmt: skip
    feats = [{"id": "f1", "name": "miss", "op": "is_missing", "inputs": ["sparse"]},
             {"id": "f2", "name": "both", "op": "add", "inputs": ["z", "amount"]}]  # fmt: skip
    seen: list[list[str]] = []

    def recording(cand: CandidateSpec, *a: Any, **kw: Any) -> Any:
        seen.append([f.name for f in cand.features])
        return build_pipeline(cand, *a, **kw)

    run = run_cv(candidate(ModelName.LINEAR, Task.BINARY, features=feats), dev, split,
                 schema=schema, registry=REG, metric=MetricName.ROC_AUC, build=recording)  # fmt: skip
    assert run.result.status == "ok"
    assert run.dropped_features == {"miss": "degenerate"}
    assert [f.name for f in run.candidate.features] == ["both"]
    assert seen[0] == ["miss", "both"]
    assert all(s == ["both"] for s in seen[1:])  # every scored fold used the same feature set
    assert len(seen) == 1 + split.k * split.n_repeats


@pytest.mark.unit
def test_all_llm_features_failing_fails_the_candidate() -> None:
    dev, split, schema = synthetic_dev()
    frame = dev.frame.assign(const=2.0)
    schema = PipelineSchema({**schema.columns, "const": SemanticType.NUMERIC}, schema.column_map,
                            schema.distinct, schema.missing_rate, schema.n_rows)  # fmt: skip
    feats = [{"id": "f1", "name": "sq", "op": "square", "inputs": ["const"]}]
    run = run_cv(candidate(ModelName.LINEAR, Task.BINARY, features=feats),
                 DevData(frame, dev.y, dev.problem), split, schema=schema, registry=REG,
                 metric=MetricName.ROC_AUC)  # fmt: skip
    assert run.result.status == "failed"
    assert run.result.error == "all LLM features failed (§14.6)"


# ---------------------------------------------------------------- T-LEAK-07 fold-local fitting
def _spy(step: Any, key: str) -> Any:
    if step == "passthrough":
        return step
    if isinstance(step, Pipeline):
        return Pipeline([(name, _spy(s, f"{key}/{name}")) for name, s in step.steps])
    return FitSpy(step, key=key)


def spy_build(cand: CandidateSpec, *a: Any, **kw: Any) -> Any:
    pipe = build_pipeline(cand, *a, **kw)
    for name in ("baseline_features", "dsl_features", "select", "model"):
        pipe.set_params(**{name: FitSpy(pipe.named_steps[name], key=name)})
    pre = pipe.named_steps["preprocess"]
    pre.set_params(transformers=[(n, _spy(s, n), c) for n, s, c in pre.transformers])
    return pipe.set_output(transform="pandas")


@pytest.mark.unit
@pytest.mark.parametrize(("model", "high"), [(ModelName.LINEAR, "target"),
                                             (ModelName.RANDOM_FOREST, "frequency"),
                                             (ModelName.LIGHTGBM, "target")])  # fmt: skip
def test_T_LEAK_07_every_stateful_step_fits_on_training_rows_only(
    model: ModelName, high: str
) -> None:
    dev, split, schema = synthetic_dev()
    feats = [{"id": "f1", "name": "gm", "op": "group_mean", "inputs": ["low", "z"]},
             {"id": "f2", "name": "cl", "op": "clip", "inputs": ["z"],
              "params": {"lower_q": 0.05, "upper_q": 0.95}},
             {"id": "f3", "name": "bq", "op": "bin_quantile", "inputs": ["z"], "params": {"n_bins": 4}}]  # fmt: skip
    cand = candidate(model, Task.BINARY, features=feats, encoding=Encoding(high_cardinality=high))
    run = run_cv(cand, dev, split, schema=schema, registry=REG, metric=MetricName.ROC_AUC,
                 build=spy_build)  # fmt: skip
    assert run.result.status == "ok", run.result.error
    folds = [set(dev.frame.index[train]) for _, _, train, _ in iter_folds(split)]
    expected_keys = {"baseline_features", "dsl_features", "select", "model", "highcat", "lowcat",
                     "midcat", "bool"} | ({"num"} if model != ModelName.LIGHTGBM else set())  # fmt: skip
    present = {key.split("/")[0] for key in FIT_LOG}
    assert expected_keys <= present
    for key, fits in FIT_LOG.items():
        assert [set(rows) for rows in fits] == folds, key  # one fit per fold, on its training rows


# ---------------------------------------------------------------- T-LEAK-08 canary
CANARY_LEVELS = 90  # > 50 → target encoding; ≈ 22 rows/level survive rare grouping (DEC-032)
CANARY_ROWS = 2500  # 20% holdout → n_dev = 2,000 (PI-08)


def canary(seed: int) -> tuple[pd.DataFrame, np.ndarray, Any, PipelineSchema, CandidateSpec]:
    """Pure-noise binary target, one 90-level categorical, 3 noise numerics (PI-08, DEC-032)."""
    rng = np.random.default_rng(100 + seed)
    n = CANARY_ROWS
    frame = pd.DataFrame({
        "code": pd.Categorical(rng.choice([f"k{i}" for i in range(CANARY_LEVELS)], n)),
        "u": rng.normal(size=n), "v": rng.normal(size=n), "w": rng.normal(size=n),
    })  # fmt: skip
    y = rng.integers(0, 2, n).astype(np.int64)
    split = build_split(pd.Series(y), Task.BINARY, seed_root=seed).spec
    schema = PipelineSchema({"code": SemanticType.CATEGORICAL, "u": SemanticType.NUMERIC,
                             "v": SemanticType.NUMERIC, "w": SemanticType.NUMERIC},
                            distinct={"code": CANARY_LEVELS}, n_rows=len(split.dev_indices))  # fmt: skip
    cand = candidate(ModelName.LIGHTGBM, Task.BINARY, params={},
                     encoding=Encoding(high_cardinality="target"))  # fmt: skip
    return frame, y, split, schema, cand


def canary_auc(frame: pd.DataFrame, y: np.ndarray, split: Any, schema: PipelineSchema,
               cand: CandidateSpec) -> float:  # fmt: skip
    run = run_cv(cand, DevData(frame, y, Problem(Task.BINARY, ("0", "1"), 1)), split,
                 schema=schema, registry=REG, metric=MetricName.ROC_AUC)  # fmt: skip
    assert run.result.status == "ok", run.result.error
    return float(np.mean(run.result.fold_scores))


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(5))
def test_T_LEAK_08_canary_noise_with_target_encoding(seed: int) -> None:
    """Pure-noise target, high-cardinality categorical, forced target encoding: the pipeline's
    CV ROC-AUC stays ≤ 0.55 for each seed (PI-08).
    """
    frame, y, split, schema, cand = canary(seed)
    assert len(split.dev_indices) == 2000
    # the target encoder is genuinely exercised: > 50 real levels reach it on a training fold
    _, _, train, _ = next(iter_folds(split))
    pipe = build_pipeline(cand, schema, REG).fit(frame.iloc[train], y[train])
    assert len(pipe.named_steps["baseline_features"].rare_.keep_["code"]) > 50
    encoder = pipe.named_steps["preprocess"].named_transformers_["highcat"]
    assert type(encoder).__name__ == "TargetEncoder"
    assert canary_auc(frame, y, split, schema, cand) <= 0.55


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(5))
def test_T_LEAK_08_positive_control_detects_a_leaky_encoder(seed: int) -> None:
    """The same data with a target-mean encoding fitted on all of DEV (outside the folds, so
    validation rows see their own target) must score > 0.55: the canary can fail (PI-08).
    """
    frame, y, split, schema, cand = canary(seed)
    dev = np.asarray(split.dev_indices)
    means = pd.Series(y[dev]).groupby(frame["code"].iloc[dev].astype(str).to_numpy()).mean()
    leaky = frame.assign(leak=frame["code"].astype(str).map(means).astype(float))
    leaky_schema = PipelineSchema({**schema.columns, "leak": SemanticType.NUMERIC},
                                  distinct=schema.distinct, n_rows=schema.n_rows)  # fmt: skip
    assert canary_auc(leaky, y, split, leaky_schema, cand) > 0.55


# ---------------------------------------------------------------- budgets, cancel, holdout
@pytest.mark.unit
def test_budget_hook_stops_candidate_as_skipped_budget() -> None:
    dev, split, schema = synthetic_dev()
    folds: list[int] = []

    def hook() -> None:
        folds.append(1)
        if len(folds) == 2:
            raise BudgetExceeded

    run = run_cv(candidate(ModelName.LINEAR, Task.BINARY), dev, split, schema=schema,
                 registry=REG, metric=MetricName.ROC_AUC, budget_hook=hook)  # fmt: skip
    assert run.result.status == "skipped_budget"
    assert run.result.fold_scores == []


@pytest.mark.unit
def test_cancel_token_stops_before_next_fold() -> None:
    dev, split, schema = synthetic_dev()
    event = threading.Event()
    event.set()
    with pytest.raises(Cancelled):
        run_cv(candidate(ModelName.LINEAR, Task.BINARY), dev, split, schema=schema, registry=REG,
               metric=MetricName.ROC_AUC, cancel=event)  # fmt: skip


@pytest.mark.unit
def test_cv_never_touches_holdout_rows() -> None:
    prep = prepared("binary_basic.csv")
    assert prep.split.holdout_present
    frame = prep.dev.frame
    guarded = guard(frame, frame.index[prep.split.holdout_indices])
    run = run_cv(candidate(ModelName.LIGHTGBM, Task.BINARY), DevData(guarded, prep.dev.y,
                 prep.problem), prep.split, schema=prep.schema, registry=prep.registry,
                 metric=MetricName.ROC_AUC)  # fmt: skip
    assert run.result.status == "ok"


@pytest.mark.unit
def test_class_missing_in_training_fold_gets_zero_probability() -> None:
    prep = prepared("multiclass_basic.csv")
    rows = np.asarray(prep.split.dev_indices[:800])
    rows = rows[prep.dev.y[rows] != 2]
    pipe = build_pipeline(candidate(ModelName.LINEAR, Task.MULTICLASS), prep.schema, prep.registry)
    pipe.fit(prep.dev.frame.iloc[rows], prep.dev.y[rows])
    proba, missing = _predict(pipe, prep.dev.frame.iloc[rows[:50]], prep.problem)
    assert missing
    assert proba.shape == (50, prep.problem.n_classes)
    assert np.all(proba[:, 2] == 0)
    assert np.allclose(proba.sum(axis=1), 1)


# ---------------------------------------------------------------- data and store
@pytest.mark.unit
def test_dev_data_encodes_labels_in_problem_order() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "t": pd.Categorical(["no", "yes", "no"])})
    dev = DevData.from_frame(frame, "t", Problem(Task.BINARY, ("no", "yes"), 1))
    assert dev.y.tolist() == [0, 1, 0]
    assert list(dev.frame.columns) == ["x"]
    with pytest.raises(ValueError, match="outside"):
        DevData.from_frame(frame, "t", Problem(Task.BINARY, ("a", "b"), 1))
    reg = DevData.from_frame(pd.DataFrame({"x": [1], "t": [2.5]}), "t", Problem(Task.REGRESSION))
    assert reg.y.dtype == np.float64


@pytest.mark.unit
def test_oof_store_memory_and_disk(tmp_path: Path) -> None:
    store = OOFStore(tmp_path / "oof")
    oof = np.arange(6, dtype=np.float64).reshape(3, 2)
    ref = store.put("0-linear-0123abcd", oof)
    assert store.get(ref).dtype == np.float32
    path = store.persist("0-linear-0123abcd")
    np.testing.assert_array_equal(store.get(path), oof.astype(np.float32))
    with pytest.raises(ValueError, match="no directory"):
        OOFStore().persist("x")


@pytest.mark.unit
def test_entry_id_is_stable_and_candidate_specific() -> None:
    a = candidate(ModelName.LINEAR, Task.BINARY)
    assert entry_id_for(a) == entry_id_for(a.model_copy())
    assert entry_id_for(a) != entry_id_for(
        candidate(ModelName.LINEAR, Task.BINARY, params={"C": 2.0})
    )
