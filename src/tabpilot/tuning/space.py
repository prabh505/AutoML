"""MVP search spaces (PRD §17.2). Trial 0 of every study uses the library defaults instead."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna

from tabpilot.schemas import ExecutionProfileName, ModelName, Task

Params = dict[str, Any]
BOOSTING_ROUNDS = 2000
CATBOOST_ITERATIONS = 1500


def _linear(trial: optuna.Trial, task: Task, profile: ExecutionProfileName) -> Params:
    if task == Task.REGRESSION:
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _random_forest(trial: optuna.Trial, task: Task, profile: ExecutionProfileName) -> Params:
    deep = 20 if profile == ExecutionProfileName.HOSTED else None
    return {
        "max_depth": trial.suggest_categorical("max_depth", [6, 10, 16, deep]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
    }


def _lightgbm(trial: optuna.Trial, task: Task, profile: ExecutionProfileName) -> Params:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "n_estimators": BOOSTING_ROUNDS,  # early stopping, patience 50 (TabularModel)
    }


def _xgboost(trial: optuna.Trial, task: Task, profile: ExecutionProfileName) -> Params:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": BOOSTING_ROUNDS,
    }


def _catboost(trial: optuna.Trial, task: Task, profile: ExecutionProfileName) -> Params:
    return {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "iterations": CATBOOST_ITERATIONS,
    }


def _hist_gb(trial: optuna.Trial, task: Task, profile: ExecutionProfileName) -> Params:
    """Not in §17.2 (fallback-only model); mirrors the LightGBM space (DEC-031)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
    }


SPACES: dict[ModelName, Callable[[optuna.Trial, Task, ExecutionProfileName], Params]] = {
    ModelName.LINEAR: _linear,
    ModelName.RANDOM_FOREST: _random_forest,
    ModelName.LIGHTGBM: _lightgbm,
    ModelName.XGBOOST: _xgboost,
    ModelName.CATBOOST: _catboost,
    ModelName.HIST_GB: _hist_gb,
}
