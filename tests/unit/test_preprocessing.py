"""P08 pipeline tests: T-PRE-01…05, branches per family, encoders, pickling (PRD §15)."""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from tests.helpers.prepared import prepared
from tests.helpers.synthetic import synthetic

from tabpilot.config import load_settings
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import STEP_NAMES, build_pipeline, inner_pipeline
from tabpilot.schemas import (
    FEATURE_SPEC_ADAPTER,
    CandidateSpec,
    ExecutionProfileName,
    ModelName,
    TargetTransform,
    Task,
)
from tabpilot.schemas.experiment import ModelConfig
from tabpilot.schemas.plan import ColumnDecision, Encoding, Imputation
from tabpilot_runtime import PredictionSchemaError, TabularModel

SRC = Path(__file__).resolve().parents[2] / "src"
REG = ModelRegistry(load_settings({}))
FAST = {ModelName.RANDOM_FOREST: {"n_estimators": 20}, ModelName.CATBOOST: {"iterations": 60},
        ModelName.HIST_GB: {"max_iter": 30}, ModelName.LIGHTGBM: {"n_estimators": 60},
        ModelName.XGBOOST: {"n_estimators": 60}}  # fmt: skip


def candidate(
    model: ModelName,
    task: Task = Task.BINARY,
    features: list[dict[str, Any]] | None = None,
    **kw: Any,
) -> CandidateSpec:
    specs = [FEATURE_SPEC_ADAPTER.validate_python(f) for f in features or []]
    return CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline+llm" if specs else "baseline",
                         features=specs, model=ModelConfig(name=model, task=task, seed=1, n_jobs=1,
                                                            params=FAST.get(model, {})), **kw)  # fmt: skip


def fitted(model: ModelName, **kw: Any) -> tuple[Any, pd.DataFrame, np.ndarray]:
    frame, y, schema = synthetic()
    pipe = build_pipeline(candidate(model, **kw), schema, REG)
    return pipe.fit(frame, y), frame, y


def branches(pipe: Any) -> dict[str, Any]:
    return {
        name: step for name, step, _ in inner_pipeline(pipe).named_steps["preprocess"].transformers
    }


# ---------------------------------------------------------------- T-PRE-01 construction
@pytest.mark.unit
@pytest.mark.parametrize("model", list(ModelName))
def test_T_PRE_01_layout_per_family(model: ModelName) -> None:
    _, _, schema = synthetic()
    pipe = build_pipeline(candidate(model), schema, REG)
    assert isinstance(pipe, Pipeline)
    assert tuple(pipe.named_steps) == STEP_NAMES
    names = set(branches(pipe))
    if model == ModelName.CATBOOST:
        assert names == {"num", "bool", "cat"}
    else:
        assert names == {"num", "bool", "lowcat", "midcat", "highcat"}
    num = branches(pipe)["num"]
    if model == ModelName.LINEAR:
        assert [s for s, _ in num.steps] == ["impute", "scale"]
    elif model == ModelName.RANDOM_FOREST:
        assert type(num).__name__ == "SimpleImputer"
    else:
        assert num == "passthrough"  # native missing handling (§15.2)
    assert isinstance(pipe.named_steps["model"], TabularModel)


@pytest.mark.unit
def test_T_PRE_01_log1p_wraps_whole_pipeline() -> None:
    _, _, schema = synthetic()
    model = build_pipeline(candidate(ModelName.LIGHTGBM, Task.REGRESSION,
                                     target_transform=TargetTransform.LOG1P), schema, REG)  # fmt: skip
    assert isinstance(model, TransformedTargetRegressor)
    assert tuple(model.regressor.named_steps) == STEP_NAMES
    frame, _, _ = synthetic()
    y = np.exp(frame["z"].to_numpy())
    model.fit(frame, y)
    pred = model.predict(frame)
    assert np.all(pred > 0)  # predictions are back on the original scale (I-4)
    assert tuple(inner_pipeline(model).named_steps) == STEP_NAMES


@pytest.mark.unit
def test_column_decisions_drop_and_retype() -> None:
    _, _, schema = synthetic()
    decisions = [ColumnDecision(column="z", action="drop"),
                 ColumnDecision(column="amount", action="treat_as_categorical")]  # fmt: skip
    pipe = build_pipeline(candidate(ModelName.LIGHTGBM, column_decisions=decisions), schema, REG)
    frame, y, _ = synthetic()
    pipe.fit(frame, y)
    kinds = pipe.named_steps["type_cast"].kinds
    assert "z" not in kinds
    assert kinds["amount"] == "categorical"
    out = pipe[:-1].transform(frame)
    assert not any(c.endswith("__z") for c in out.columns)
    assert "highcat__amount" in out.columns  # 400 distinct → high-card branch


@pytest.mark.unit
def test_missing_indicators_A_006_and_datetime_parts() -> None:
    pipe, frame, _ = fitted(ModelName.LIGHTGBM)
    out = pipe[:-1].transform(frame)
    assert "bool__amount__missing" in out.columns  # 10% missing ≥ 5%
    assert not any(c.endswith("z__missing") for c in out.columns)
    assert {"num__when__year", "num__when__dayofweek", "num__when__days_since_min"} <= set(
        out.columns
    )
    assert "num__when__hour" not in out.columns  # no times in the data
    no_ind = build_pipeline(candidate(ModelName.LIGHTGBM,
                                      imputation=Imputation(add_missing_indicators=False)),
                            synthetic()[2], REG).fit(frame, synthetic()[1])  # fmt: skip
    assert not any(c.endswith("__missing") for c in no_ind[:-1].transform(frame).columns)


@pytest.mark.unit
def test_float32_matrix_reaches_the_model() -> None:
    pipe, frame, _ = fitted(ModelName.XGBOOST)
    matrix = pipe.named_steps["model"]._prepare(pipe[:-1].transform(frame))
    assert set(matrix.dtypes) == {np.dtype(np.float32)}


# ---------------------------------------------------------------- T-PRE-02 unknown categories
@pytest.mark.unit
@pytest.mark.parametrize("model", list(ModelName))
def test_T_PRE_02_unknown_categories_never_error(model: ModelName) -> None:
    pipe, frame, _ = fitted(model)
    new = frame.head(5).copy()
    for col, value in (("low", "zzz"), ("mid", "unseen"), ("high", "h999"), ("flag", "maybe")):
        new[col] = pd.Categorical([value] * 5)
    proba = pipe.predict_proba(new)
    assert proba.shape == (5, 2)
    assert np.all(np.isfinite(proba))


@pytest.mark.unit
def test_unknown_category_gets_reserved_ordinal_code() -> None:
    pipe, frame, _ = fitted(ModelName.LIGHTGBM)
    new = frame.head(2).copy()
    new["low"] = pd.Categorical(["zzz", None])
    out = pipe[:-1].transform(new)
    assert out["lowcat__low"].tolist() == [-1.0, -2.0]  # unknown, missing


@pytest.mark.unit
def test_lightgbm_receives_ordinal_columns_as_categorical() -> None:
    pipe, _, _ = fitted(ModelName.LIGHTGBM)
    model = pipe.named_steps["model"]
    assert model.raw_categorical_ == ["lowcat__low", "midcat__mid"]
    infos = model.estimator_.booster_.dump_model()["feature_infos"]
    assert {0, 1, 2} <= set(infos["lowcat__low"]["values"])  # categorical: category values
    assert infos["num__z"]["values"] == []  # numeric: a range only


@pytest.mark.unit
def test_catboost_receives_raw_categoricals() -> None:
    pipe, _, _ = fitted(ModelName.CATBOOST)
    model = pipe.named_steps["model"]
    assert model.raw_categorical_ == ["cat__low", "cat__mid", "cat__high"]
    names = list(model.feature_names_in_)
    assert [names[i] for i in model.estimator_.get_cat_feature_indices()] == model.raw_categorical_


@pytest.mark.unit
def test_linear_onehot_and_ordinal_options() -> None:
    pipe, frame, _ = fitted(ModelName.LINEAR)
    out = pipe[:-1].transform(frame)
    assert {"lowcat__low_a", "lowcat__low_b", "lowcat__low_c"} <= set(out.columns)
    mid = [c for c in out.columns if c.startswith("midcat__")]
    assert len(mid) <= 16  # max_categories=16
    ordinal = build_pipeline(candidate(ModelName.LINEAR, encoding=Encoding(low_cardinality="ordinal")),
                             synthetic()[2], REG).fit(frame, synthetic()[1])  # fmt: skip
    assert "lowcat__low" in ordinal[:-1].transform(frame).columns


# ---------------------------------------------------------------- target encoding (§11.8)
class RecordingTargetEncoder(TargetEncoder):
    calls: list[str] = []

    def fit(self, X: Any, y: Any) -> Any:
        RecordingTargetEncoder.calls.append("fit")
        return super().fit(X, y)

    def fit_transform(self, X: Any, y: Any) -> Any:
        RecordingTargetEncoder.calls.append("fit_transform")
        return super().fit_transform(X, y)


@pytest.mark.unit
def test_target_encoder_is_cross_fitted_inside_pipeline_fit() -> None:
    frame, y, schema = synthetic(n=600)
    pipe = build_pipeline(candidate(ModelName.LIGHTGBM, encoding=Encoding(high_cardinality="target")),
                          schema, REG)  # fmt: skip
    pre = pipe.named_steps["preprocess"]
    _name, te, _cols = next(t for t in pre.transformers if t[0] == "highcat")
    assert isinstance(te, TargetEncoder)
    assert te.cv.n_splits == 5
    recorder = RecordingTargetEncoder(**te.get_params())
    pre.set_params(
        transformers=[(n, recorder if n == "highcat" else s, c) for n, s, c in pre.transformers]
    )
    RecordingTargetEncoder.calls = []
    pipe.fit(frame, y)
    assert RecordingTargetEncoder.calls == ["fit_transform"]  # never a plain fit (§11.8)
    encoder = pipe.named_steps["preprocess"].named_transformers_["highcat"]
    train_rows = pipe[:-2].transform(frame)[["high"]]
    full_fold = encoder.transform(train_rows).ravel()
    cross_fitted = encoder.fit_transform(train_rows, y).ravel()
    assert not np.allclose(full_fold, cross_fitted)  # training rows were encoded out-of-fold


# ---------------------------------------------------------------- T-PRE-03 row vs batch
@pytest.mark.unit
@pytest.mark.parametrize("model", list(ModelName))
def test_T_PRE_03_row_by_row_equals_batch(model: ModelName) -> None:
    feats = [{"id": "f1", "name": "gm", "op": "group_mean", "inputs": ["low", "z"]},
             {"id": "f2", "name": "bins", "op": "bin_quantile", "inputs": ["z"], "params": {"n_bins": 4}}]  # fmt: skip
    frame, y, schema = synthetic()
    pipe = build_pipeline(candidate(model, features=feats), schema, REG).fit(frame, y)
    rows = frame.iloc[:15]
    batch = pipe[:-1].transform(rows)
    single = pd.concat([pipe[:-1].transform(rows.iloc[[i]]) for i in range(len(rows))])
    pd.testing.assert_frame_equal(single, batch)
    np.testing.assert_allclose(
        np.vstack([pipe.predict_proba(rows.iloc[[i]]) for i in range(len(rows))]),
        pipe.predict_proba(rows), rtol=1e-6, atol=1e-7)  # fmt: skip


# ---------------------------------------------------------------- T-PRE-04 ordering, names
@pytest.mark.unit
def test_T_PRE_04_column_order_extra_columns_and_original_names() -> None:
    pipe, frame, _ = fitted(ModelName.LIGHTGBM)
    expected = pipe.predict_proba(frame)
    shuffled = frame[list(reversed(frame.columns))].assign(extra_column=1.0)
    np.testing.assert_allclose(pipe.predict_proba(shuffled), expected)
    original = frame.rename(columns={"amount": "Amount ($)"})
    np.testing.assert_allclose(pipe.predict_proba(original), expected)
    assert pipe.named_steps["select"].columns_ == list(pipe[:4].transform(frame).columns)


# ---------------------------------------------------------------- T-PRE-05 missing columns
@pytest.mark.unit
def test_T_PRE_05_missing_input_columns_error_lists_them() -> None:
    pipe, frame, _ = fitted(ModelName.LINEAR)
    with pytest.raises(PredictionSchemaError) as err:
        pipe.predict_proba(frame.drop(columns=["amount", "low"]))
    assert err.value.missing == ["Amount ($)", "low"]
    assert "Amount ($), low" in str(err.value)


@pytest.mark.unit
def test_column_selector_rejects_missing_feature() -> None:
    pipe, frame, _ = fitted(ModelName.LINEAR)
    upstream = pipe[:4].transform(frame).drop(columns=["z"])
    with pytest.raises(PredictionSchemaError, match="z"):
        pipe.named_steps["select"].transform(upstream)


# ---------------------------------------------------------------- probability alignment
@pytest.mark.unit
@pytest.mark.parametrize("model", list(ModelName))
def test_probability_columns_follow_label_order(model: ModelName) -> None:
    prep = prepared("multiclass_basic.csv")
    rows = np.asarray(prep.split.dev_indices[:600])
    pipe = build_pipeline(candidate(model, Task.MULTICLASS), prep.schema, prep.registry)
    pipe.fit(prep.dev.frame.iloc[rows], prep.dev.y[rows])
    assert list(pipe.classes_) == list(range(prep.problem.n_classes))
    proba = pipe.predict_proba(prep.dev.frame.iloc[rows])
    hits = (proba.argmax(axis=1) == prep.dev.y[rows]).mean()
    assert hits > 1 / prep.problem.n_classes  # columns line up with codes, not shuffled


# ---------------------------------------------------------------- pickling (A-001)
@pytest.mark.unit
def test_fitted_pipelines_unpickle_without_tabpilot(tmp_path: Path) -> None:
    frame, _y, _ = synthetic()
    models = [fitted(m)[0] for m in (ModelName.LINEAR, ModelName.LIGHTGBM, ModelName.CATBOOST)]
    (tmp_path / "models.pkl").write_bytes(pickle.dumps(models))
    frame.to_pickle(tmp_path / "frame.pkl")
    pd.to_pickle([m.predict_proba(frame) for m in models], tmp_path / "expected.pkl")
    shutil.copytree(SRC / "tabpilot_runtime", tmp_path / "tabpilot_runtime")
    code = (
        "import importlib.util, pickle, numpy as np, pandas as pd\n"
        "assert importlib.util.find_spec('tabpilot') is None\n"
        "models = pickle.load(open('models.pkl', 'rb'))\n"
        "frame = pd.read_pickle('frame.pkl')\n"
        "for m, e in zip(models, pd.read_pickle('expected.pkl')):\n"
        "    np.testing.assert_allclose(m.predict_proba(frame), e)\n"
        "print('ok')\n"
    )
    env = {"PYTHONPATH": os.pathsep.join([str(tmp_path), sysconfig.get_paths()["purelib"]]),
           "PATH": "/usr/bin:/bin"}  # fmt: skip
    out = subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path, env=env,
                         capture_output=True, text=True, check=False)  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok"


@pytest.mark.unit
def test_hosted_profile_builds_capped_random_forest() -> None:
    _, _, schema = synthetic()
    pipe = build_pipeline(candidate(ModelName.RANDOM_FOREST), schema, REG,
                          profile=ExecutionProfileName.HOSTED)  # fmt: skip
    assert pipe.named_steps["model"].estimator.max_depth == 20
