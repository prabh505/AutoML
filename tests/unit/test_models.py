"""P08 model registry tests: availability, fallback, defaults, class weights (PRD §16.1–§16.2)."""

from __future__ import annotations

import logging

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from tabpilot.config import load_settings
from tabpilot.models import ModelRegistry, class_weight_params, default_class_weight
from tabpilot.schemas import ClassWeight, ExecutionProfileName, ModelName, Task
from tabpilot.schemas.experiment import ModelConfig

HOSTED, LOCAL = ExecutionProfileName.HOSTED, ExecutionProfileName.LOCAL


def config(name: ModelName, task: Task = Task.BINARY, **kw: object) -> ModelConfig:
    return ModelConfig(name=name, task=task, seed=7, n_jobs=1, **kw)  # type: ignore[arg-type]


@pytest.mark.unit
def test_available_per_profile() -> None:
    reg = ModelRegistry(load_settings({}))
    assert reg.available(HOSTED) == [ModelName.LINEAR, ModelName.RANDOM_FOREST, ModelName.LIGHTGBM]
    assert reg.available(LOCAL) == [ModelName.LINEAR, ModelName.RANDOM_FOREST, ModelName.LIGHTGBM,
                                    ModelName.XGBOOST, ModelName.CATBOOST]  # fmt: skip


@pytest.mark.unit
def test_hist_gb_replaces_unimportable_lightgbm(caplog: pytest.LogCaptureFixture) -> None:
    reg = ModelRegistry(load_settings({}), importable=lambda module: module not in ("lightgbm",
                                                                                  "catboost"))  # fmt: skip
    with caplog.at_level(logging.WARNING):
        names = reg.available(LOCAL)
    assert names == [ModelName.LINEAR, ModelName.RANDOM_FOREST, ModelName.HIST_GB,
                     ModelName.XGBOOST]  # fmt: skip
    assert "model_unavailable" in caplog.text


@pytest.mark.unit
def test_models_enabled_limit_is_respected() -> None:
    settings = load_settings({})
    limits = settings.profiles[LOCAL].model_copy(update={"models_enabled": [ModelName.LINEAR]})
    settings = settings.model_copy(update={"profiles": {**settings.profiles, LOCAL: limits}})
    assert ModelRegistry(settings).available(LOCAL) == [ModelName.LINEAR]


@pytest.mark.unit
def test_make_estimator_defaults_seed_and_overrides() -> None:
    reg = ModelRegistry(load_settings({}))
    est = reg.make_estimator(config(ModelName.LINEAR, params={"C": 0.5, "random_state": 99}))
    assert isinstance(est, LogisticRegression)
    assert (est.C, est.max_iter, est.solver, est.random_state) == (0.5, 2000, "lbfgs", 7)
    ridge = reg.make_estimator(config(ModelName.LINEAR, Task.REGRESSION))
    assert isinstance(ridge, Ridge)
    lgbm = reg.make_estimator(config(ModelName.LIGHTGBM))
    assert lgbm.get_params()["deterministic"]
    assert lgbm.get_params()["force_row_wise"]
    cat = reg.make_estimator(config(ModelName.CATBOOST, Task.MULTICLASS))
    assert cat.get_params()["allow_writing_files"] is False
    assert cat.get_params()["random_seed"] == 7
    assert reg.make_estimator(config(ModelName.XGBOOST)).get_params()["tree_method"] == "hist"
    assert isinstance(reg.make_estimator(config(ModelName.HIST_GB)), HistGradientBoostingClassifier)


@pytest.mark.unit
def test_random_forest_depth_capped_on_hosted_only() -> None:
    reg = ModelRegistry(load_settings({}))
    rf = config(ModelName.RANDOM_FOREST, Task.REGRESSION, params={"max_depth": 50})
    hosted = reg.make_estimator(rf, HOSTED)
    assert isinstance(hosted, RandomForestRegressor)
    assert hosted.max_depth == 20
    assert reg.make_estimator(config(ModelName.RANDOM_FOREST), HOSTED).max_depth == 20
    assert reg.make_estimator(rf, LOCAL).max_depth == 50


@pytest.mark.unit
@pytest.mark.parametrize(("name", "task", "expected"), [
    (ModelName.LINEAR, Task.BINARY, {"class_weight": "balanced"}),
    (ModelName.RANDOM_FOREST, Task.MULTICLASS, {"class_weight": "balanced"}),
    (ModelName.LIGHTGBM, Task.BINARY, {"is_unbalance": True}),
    (ModelName.LIGHTGBM, Task.MULTICLASS, {"class_weight": "balanced"}),
    (ModelName.CATBOOST, Task.BINARY, {"auto_class_weights": "Balanced"}),
    (ModelName.XGBOOST, Task.BINARY, {}),  # scale_pos_weight is set on the training fold
    (ModelName.LINEAR, Task.REGRESSION, {}),
])  # fmt: skip
def test_class_weight_mapping(name: ModelName, task: Task, expected: dict[str, object]) -> None:
    assert class_weight_params(name, task, ClassWeight.BALANCED) == expected
    assert class_weight_params(name, task, ClassWeight.NONE) == {}


@pytest.mark.unit
def test_balanced_weights_reach_the_estimator() -> None:
    reg = ModelRegistry(load_settings({}))
    lgbm = reg.make_estimator(config(ModelName.LIGHTGBM, class_weight=ClassWeight.BALANCED))
    assert lgbm.get_params()["is_unbalance"] is True
    cat = reg.make_estimator(config(ModelName.CATBOOST, class_weight=ClassWeight.BALANCED))
    assert cat.get_params()["auto_class_weights"] == "Balanced"


@pytest.mark.unit
def test_default_class_weight_threshold() -> None:
    assert default_class_weight({"a": 81, "b": 19}) == ClassWeight.BALANCED
    assert default_class_weight({"a": 80, "b": 20}) == ClassWeight.NONE
    assert default_class_weight({"a": 70, "b": 20, "c": 10}) == ClassWeight.BALANCED
