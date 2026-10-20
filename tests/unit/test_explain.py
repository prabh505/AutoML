"""P13 explainer tests: T-EXP-01…03, ensembles, sampling, error analysis (PRD §21.3–§21.5)."""

from __future__ import annotations

import itertools
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.helpers.synthetic import synthetic

from tabpilot.config import load_settings
from tabpilot.evaluation.metrics import Problem
from tabpilot.explain import (
    ExplainData,
    ExplainSettings,
    aggregate,
    error_analysis,
    explain,
    sample_rows,
)
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import build_pipeline
from tabpilot.reflection.input import source_column
from tabpilot.schemas import (
    CandidateSpec,
    ExecutionProfileName,
    MetricName,
    ModelName,
    TargetTransform,
    Task,
)
from tabpilot.schemas.experiment import ModelConfig
from tabpilot_runtime import VotingEnsemble

BIN = Problem(Task.BINARY, ("0", "1"), 1)
MULTI = Problem(Task.MULTICLASS, ("0", "1", "2"))
REG = Problem(Task.REGRESSION)
LOCAL = ExecutionProfileName.LOCAL


@cache
def data() -> tuple[pd.DataFrame, np.ndarray, Any]:
    return synthetic(400)


def fit(model: ModelName, task: Task, y: np.ndarray, **spec: Any) -> Any:
    frame, _, schema = data()
    cand = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                         model=ModelConfig(name=model, task=task, seed=1, n_jobs=1), **spec)  # fmt: skip
    return build_pipeline(cand, schema, ModelRegistry(load_settings({}))).fit(frame, y)


def settings(
    problem: Problem, metric: MetricName, cap: float = 120.0, **kw: Any
) -> ExplainSettings:
    _, _, schema = data()
    return ExplainSettings(problem=problem, metric=metric, sources=list(schema.columns), seed=3,
                           profile=kw.pop("profile", LOCAL), time_cap_s=cap, **kw)  # fmt: skip


def xdata(y: np.ndarray) -> ExplainData:
    return ExplainData(data()[0], y, np.arange(80))


def multiclass_y() -> np.ndarray:
    frame, y, _ = data()
    return np.asarray(y + (frame["z"] > 5).astype(int))


# ---------------------------------------------------------------- T-EXP-01
@pytest.mark.unit
@pytest.mark.parametrize(("model", "method"), [
    (ModelName.LINEAR, "shap_linear"), (ModelName.LIGHTGBM, "shap_tree"),
    (ModelName.XGBOOST, "shap_tree"), (ModelName.CATBOOST, "shap_tree"),
    (ModelName.RANDOM_FOREST, "shap_tree"),
])  # fmt: skip
def test_T_EXP_01_shap_per_model_family(model: ModelName, method: str) -> None:
    _, y, schema = data()
    result = explain(fit(model, Task.BINARY, y), xdata(y), settings(BIN, MetricName.ROC_AUC))
    assert result.method == method
    assert result.unavailable_reason is None
    assert len(result.shap) == 1
    assert result.shap[0].values.shape == (400, len(result.shap[0].columns))
    names = [i.name for i in result.global_importance]
    assert set(names) <= set(schema.columns)
    assert "amount" in names[:3]  # y depends on amount and low (synthetic)
    assert bool(result.gain) == (model != ModelName.LINEAR)  # native gain for tree models only
    assert result.section().top_features == names


@pytest.mark.unit
def test_T_EXP_01_tree_shap_is_additive_on_the_model_output() -> None:
    frame, y, _ = data()
    pipe = fit(ModelName.LIGHTGBM, Task.BINARY, y)
    result = explain(pipe, xdata(y), settings(BIN, MetricName.ROC_AUC))
    tab = pipe.steps[-1][1]
    X = tab.prepared(pipe[:-1].transform(frame))
    raw = tab.estimator_.predict(X, raw_score=True)
    import shap  # noqa: PLC0415

    base = float(np.ravel(shap.TreeExplainer(tab.estimator_).expected_value)[-1])
    np.testing.assert_allclose(result.shap[0].values.sum(axis=1) + base, raw, atol=1e-6)


@pytest.mark.unit
def test_one_hot_columns_are_summed_per_row_before_abs() -> None:
    """§21.3: one-hot columns are aggregated to their source column for the global chart."""
    _, y, schema = data()
    result = explain(
        fit(ModelName.LINEAR, Task.BINARY, y), xdata(y), settings(BIN, MetricName.ROC_AUC)
    )
    out = result.shap[0]
    low = [j for j, c in enumerate(out.columns) if source_column(c, list(schema.columns)) == "low"]
    assert len(low) > 1  # one-hot: several columns per source
    expected = float(np.abs(out.values[:, low].sum(axis=1)).mean())
    got = {i.name: i.value for i in result.global_importance}["low"]
    assert got == pytest.approx(expected)


@pytest.mark.unit
def test_positive_class_code_zero_flips_the_sign() -> None:
    _, y, _ = data()
    pipe = fit(ModelName.LIGHTGBM, Task.BINARY, y)
    pos1 = explain(pipe, xdata(y), settings(BIN, MetricName.ROC_AUC))
    pos0 = explain(pipe, xdata(y), settings(Problem(Task.BINARY, ("0", "1"), 0),
                                            MetricName.ROC_AUC))  # fmt: skip
    np.testing.assert_allclose(pos0.shap[0].values, -pos1.shap[0].values)


@pytest.mark.unit
@pytest.mark.parametrize("model", [ModelName.LIGHTGBM, ModelName.LINEAR, ModelName.RANDOM_FOREST])
def test_T_EXP_01_multiclass_keeps_three_classes_and_averages_all(model: ModelName) -> None:
    y = multiclass_y()
    result = explain(fit(model, Task.MULTICLASS, y), xdata(y), settings(MULTI, MetricName.LOG_LOSS))
    assert result.method in ("shap_tree", "shap_linear")
    counts = np.bincount(y)
    expected = sorted(range(3), key=lambda k: (-counts[k], k))
    assert [o.label for o in result.shap] == [MULTI.classes[k] for k in expected]
    assert result.global_importance[0].name == "z"  # the class split uses z


@pytest.mark.unit
def test_regression_and_log1p_note() -> None:
    frame, _, _ = data()
    y = np.asarray(frame["z"] * 2 + 20)
    plain = explain(
        fit(ModelName.LIGHTGBM, Task.REGRESSION, y), xdata(y), settings(REG, MetricName.RMSE)
    )
    assert plain.method == "shap_tree"
    assert plain.global_importance[0].name == "z"
    assert plain.notes == []
    logged = fit(ModelName.LIGHTGBM, Task.REGRESSION, y, target_transform=TargetTransform.LOG1P)
    result = explain(logged, xdata(y), settings(REG, MetricName.RMSE))
    assert result.method == "shap_tree"
    assert any("log1p" in n for n in result.notes)


# ---------------------------------------------------------------- T-EXP-02 fallback
@pytest.mark.unit
def test_T_EXP_02_unsupported_model_falls_back_to_permutation() -> None:
    _, y, schema = data()
    result = explain(
        fit(ModelName.HIST_GB, Task.BINARY, y), xdata(y), settings(BIN, MetricName.ROC_AUC)
    )
    assert result.method == "permutation"
    assert "no SHAP explainer for the hist_gb family" in (result.unavailable_reason or "")
    assert result.shap == []
    assert {i.name for i in result.permutation} <= set(schema.columns)
    assert result.permutation[0].name in {"amount", "low"}
    assert all(i.std >= 0 for i in result.permutation)
    assert result.top_features()[0] == result.permutation[0].name


@pytest.mark.unit
def test_T_EXP_02_failing_model_never_raises() -> None:
    class Broken:
        def predict_proba(self, X: Any) -> Any:
            raise RuntimeError("cell value 42 leaked?")

    _, y, _ = data()
    result = explain(Broken(), xdata(y), settings(BIN, MetricName.ROC_AUC))
    assert result.method == "none"
    assert result.unavailable_reason is not None
    assert "42" not in result.unavailable_reason  # exception messages are not reported (I-11)


# ---------------------------------------------------------------- T-EXP-03 timeout
def slow_clock(step: float) -> Any:
    ticks = itertools.count()
    return lambda: next(ticks) * step


@pytest.mark.unit
def test_T_EXP_03_timeout_path_uses_gain_for_trees() -> None:
    _, y, _ = data()
    result = explain(fit(ModelName.LIGHTGBM, Task.BINARY, y), xdata(y),
                     settings(BIN, MetricName.ROC_AUC, cap=60.0, clock=slow_clock(100.0)))  # fmt: skip
    assert result.method == "gain"
    reason = result.unavailable_reason or ""
    assert "SHAP pre-estimate" in reason
    assert "permutation pre-estimate" in reason
    assert result.gain[0].name in {"amount", "low", "when", "z"}


@pytest.mark.unit
def test_T_EXP_03_timeout_for_linear_reports_none() -> None:
    _, y, _ = data()
    result = explain(fit(ModelName.LINEAR, Task.BINARY, y), xdata(y),
                     settings(BIN, MetricName.ROC_AUC, cap=60.0, clock=slow_clock(100.0)))  # fmt: skip
    assert result.method == "none"
    assert "pre-estimate" in (result.unavailable_reason or "")
    assert result.section().unavailable_reason == result.unavailable_reason


@pytest.mark.unit
def test_zero_time_left_skips_shap_but_a_fast_estimate_runs() -> None:
    _, y, _ = data()
    pipe = fit(ModelName.LIGHTGBM, Task.BINARY, y)
    assert explain(pipe, xdata(y), settings(BIN, MetricName.ROC_AUC, cap=0.0)).method == "gain"
    assert (
        explain(pipe, xdata(y), settings(BIN, MetricName.ROC_AUC, cap=600.0)).method == "shap_tree"
    )


# ---------------------------------------------------------------- ensembles
@pytest.mark.unit
def test_ensemble_uses_top_member_shap_plus_permutation() -> None:
    _, y, _ = data()
    members = [fit(ModelName.LIGHTGBM, Task.BINARY, y), fit(ModelName.LINEAR, Task.BINARY, y)]
    ens = VotingEnsemble(members, [0.3, 0.7]).fit(None, None)
    result = explain(ens, xdata(y), settings(BIN, MetricName.ROC_AUC))
    assert result.method == "shap_member_plus_permutation"
    linear_cols = explain(members[1], xdata(y), settings(BIN, MetricName.ROC_AUC)).shap[0].columns
    assert result.shap[0].columns == linear_cols  # the 0.7-weight member
    assert result.permutation
    assert any("highest-weight" in n for n in result.notes)


# ---------------------------------------------------------------- sampling and aggregation
@pytest.mark.unit
def test_sample_sizes_and_determinism() -> None:
    assert len(sample_rows(5000, seed=1, family="lightgbm", profile=LOCAL)) == 1000
    hosted = ExecutionProfileName.HOSTED
    assert len(sample_rows(5000, seed=1, family="random_forest", profile=hosted)) == 300
    assert len(sample_rows(5000, seed=1, family="random_forest", profile=LOCAL)) == 1000
    assert len(sample_rows(200, seed=1, family="lightgbm", profile=LOCAL)) == 200
    a = sample_rows(5000, seed=7, family="lightgbm", profile=LOCAL)
    assert np.array_equal(a, sample_rows(5000, seed=7, family="lightgbm", profile=LOCAL))
    assert np.all(np.diff(a) > 0)


@pytest.mark.unit
def test_aggregate_folds_one_hot_into_sources() -> None:
    values = {"lowcat__contract_one_year": 1.0, "lowcat__contract_two_year": 2.0,
              "num__when__year": 0.5, "num__amount": 0.0, "unknown__zzz": 9.0}  # fmt: skip
    ranked = aggregate(values, ["contract", "when", "amount"])
    assert [(i.name, i.value) for i in ranked] == [("contract", 3.0), ("when", 0.5)]


# ---------------------------------------------------------------- error analysis
@pytest.mark.unit
def test_error_analysis_classification() -> None:
    y = np.array([0] * 60 + [1] * 60)
    proba = np.zeros((120, 2))
    pred = y.copy()
    pred[:10] = 1  # 10 false positives
    pred[60:65] = 0  # 5 false negatives
    proba[np.arange(120), pred] = 1.0
    frame = pd.DataFrame({"seg": ["a"] * 40 + ["b"] * 80, "tiny": ["t"] * 10 + ["u"] * 110})
    result = error_analysis(
        proba, y, frame, Problem(Task.BINARY, ("no", "yes"), 1), ["seg", "tiny"]
    )
    assert result.confusion is not None
    assert result.confusion.tolist() == [[50, 10], [5, 55]]
    assert result.per_class[1].precision == pytest.approx(55 / 65)
    assert result.per_class[1].recall == pytest.approx(55 / 60)
    assert result.overall_error == pytest.approx(15 / 120)
    seg = [s for s in result.segments if s.column == "seg"]
    assert [(s.rank, s.label) for s in seg] == [(1, "a"), (2, "b")]  # a holds all 10 FPs
    assert [s.label for s in result.segments if s.column == "tiny"] == ["u"]  # "t" < 30 rows
    norm = result.confusion_normalized
    assert norm is not None
    np.testing.assert_allclose(norm.sum(axis=1), 1.0)


@pytest.mark.unit
def test_error_analysis_regression_and_top_three_columns() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    pred = y + rng.normal(size=300)
    frame = pd.DataFrame({c: rng.choice(["x", None], 300) for c in "abcd"})
    result = error_analysis(pred.reshape(-1, 1), y, frame, REG, ["a", "b", "c", "d"])
    assert result.error_name == "mean absolute error"
    assert result.overall_error == pytest.approx(np.abs(y - pred).mean())
    assert sum(d.n for d in result.deciles) == 300
    assert len(result.deciles) == 10
    assert {s.column for s in result.segments} == {"a", "b", "c"}
    assert "(missing)" in {s.label for s in result.segments}
    residuals = result.residuals
    assert residuals is not None
    np.testing.assert_allclose(residuals, y - pred)
    with pytest.raises(ValueError, match="same rows"):
        error_analysis(pred[:10], y, frame, REG)
