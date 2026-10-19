"""Model zoo registry: availability per profile and estimator factories (PRD §16.1–§16.2).

Estimator classes come from a fixed mapping of model name → factory function with literal
imports; nothing is looked up by a configured string. Library imports are inside the factories so
optional libraries (§16.1) load only when that model is used.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from typing import Any

from tabpilot.config import Settings
from tabpilot.logging import get_logger
from tabpilot.schemas import ClassWeight, ExecutionProfileName, ModelName, Task
from tabpilot.schemas.experiment import ModelConfig

LIBRARY_MODULES = {"scikit-learn": "sklearn", "lightgbm": "lightgbm", "xgboost": "xgboost",
                   "catboost": "catboost"}  # fmt: skip
IMBALANCE_MINORITY_SHARE = 0.20
RESERVED_PARAMS = frozenset({"random_state", "random_seed", "seed", "n_jobs", "thread_count"})

log = get_logger("models")


def _importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _is_clf(task: Task) -> bool:
    return task != Task.REGRESSION


def _linear(task: Task, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    from sklearn.linear_model import LogisticRegression, Ridge  # noqa: PLC0415

    if _is_clf(task):
        return LogisticRegression(random_state=seed, **params)
    params = {k: v for k, v in params.items() if k not in ("solver", "max_iter", "class_weight")}
    return Ridge(random_state=seed, **params)


def _random_forest(task: Task, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: PLC0415

    if _is_clf(task):
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    params = {k: v for k, v in params.items() if k != "class_weight"}
    return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **params)


def _lightgbm(task: Task, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: PLC0415

    params = {"verbose": -1, **params}
    if _is_clf(task):
        return LGBMClassifier(random_state=seed, n_jobs=n_jobs, **params)
    params = {k: v for k, v in params.items() if k not in ("class_weight", "is_unbalance")}
    return LGBMRegressor(random_state=seed, n_jobs=n_jobs, **params)


def _xgboost(task: Task, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    from xgboost import XGBClassifier, XGBRegressor  # noqa: PLC0415

    if _is_clf(task):
        return XGBClassifier(random_state=seed, n_jobs=n_jobs, **params)
    return XGBRegressor(random_state=seed, n_jobs=n_jobs, **params)


def _catboost(task: Task, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    from catboost import CatBoostClassifier, CatBoostRegressor  # noqa: PLC0415

    params = {"logging_level": "Silent", **params}
    if _is_clf(task):
        return CatBoostClassifier(random_seed=seed, thread_count=n_jobs, **params)
    params = {k: v for k, v in params.items() if k != "auto_class_weights"}
    return CatBoostRegressor(random_seed=seed, thread_count=n_jobs, **params)


def _hist_gb(task: Task, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    from sklearn.ensemble import (  # noqa: PLC0415
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    if _is_clf(task):
        return HistGradientBoostingClassifier(random_state=seed, **params)
    params = {k: v for k, v in params.items() if k != "class_weight"}
    return HistGradientBoostingRegressor(random_state=seed, **params)


Factory = Callable[[Task, dict[str, Any], int, int], Any]
FACTORIES: dict[ModelName, Factory] = {
    ModelName.LINEAR: _linear,
    ModelName.RANDOM_FOREST: _random_forest,
    ModelName.LIGHTGBM: _lightgbm,
    ModelName.XGBOOST: _xgboost,
    ModelName.CATBOOST: _catboost,
    ModelName.HIST_GB: _hist_gb,
}


def default_class_weight(class_counts: Mapping[str, int]) -> ClassWeight:
    """`balanced` when the minority class share is < 20% (§16.2)."""
    total = sum(class_counts.values())
    if total and min(class_counts.values()) / total < IMBALANCE_MINORITY_SHARE:
        return ClassWeight.BALANCED
    return ClassWeight.NONE


def class_weight_params(name: ModelName, task: Task, weight: ClassWeight) -> dict[str, Any]:
    """Library parameters for `class_weight="balanced"` (§16.2). XGBoost binary uses
    `scale_pos_weight`, computed on the training fold by `TabularModel`.
    """
    if weight != ClassWeight.BALANCED or not _is_clf(task):
        return {}
    if name == ModelName.LIGHTGBM:
        return {"is_unbalance": True} if task == Task.BINARY else {"class_weight": "balanced"}
    if name == ModelName.CATBOOST:
        return {"auto_class_weights": "Balanced"}
    if name == ModelName.XGBOOST:
        return {}
    return {"class_weight": "balanced"}


class ModelRegistry:
    """Models configured in `config/models.yaml`, filtered by importable libraries (§16.1)."""

    def __init__(self, settings: Settings, importable: Callable[[str], bool] = _importable) -> None:
        self.settings = settings
        self._importable = importable

    def library_available(self, name: ModelName) -> bool:
        library = self.settings.models[name].library
        return self._importable(LIBRARY_MODULES.get(library, library))

    def available(self, profile: ExecutionProfileName | None = None) -> list[ModelName]:
        """Models the planner may use on `profile`; `hist_gb` replaces an unimportable LightGBM."""
        profile = profile or self.settings.profile
        limits = self.settings.profiles.get(profile, self.settings.limits)
        allowed = limits.models_enabled
        names: list[ModelName] = []
        for name, entry in self.settings.models.items():
            if profile not in entry.profiles or entry.fallback_only:
                continue
            if allowed != "all_available" and name not in allowed:
                continue
            if self.library_available(name):
                names.append(name)
                continue
            log.warning("model_unavailable", model=name.value, library=entry.library)
            if name == ModelName.LIGHTGBM and self._fallback_ok(profile):
                names.append(ModelName.HIST_GB)
        return names

    def _fallback_ok(self, profile: ExecutionProfileName) -> bool:
        entry = self.settings.models.get(ModelName.HIST_GB)
        return bool(
            entry and profile in entry.profiles and self.library_available(ModelName.HIST_GB)
        )

    def params(
        self, config: ModelConfig, profile: ExecutionProfileName | None = None
    ) -> dict[str, Any]:
        """Configured defaults + class-weight mapping + candidate params (candidate wins)."""
        profile = profile or self.settings.profile
        entry = self.settings.models[config.name]
        params: dict[str, Any] = dict(entry.defaults)
        params.update(class_weight_params(config.name, config.task, config.class_weight))
        params.update({k: v for k, v in config.params.items() if k not in RESERVED_PARAMS})
        cap = entry.hosted_max_depth_cap
        if cap and profile == ExecutionProfileName.HOSTED:
            depth = params.get("max_depth")
            params["max_depth"] = cap if depth is None else min(int(depth), cap)
        return params

    def make_estimator(
        self, config: ModelConfig, profile: ExecutionProfileName | None = None
    ) -> Any:
        """An unfitted library estimator for `config` (§16.1)."""
        if config.task not in self.settings.models[config.name].tasks:
            raise ValueError(f"{config.name} does not support {config.task}")
        params = self.params(config, profile)
        return FACTORIES[config.name](config.task, params, config.seed, config.n_jobs)
