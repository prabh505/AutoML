"""Metric registry (PRD §10.2–§10.4; I-4).

Conventions:
- Classification: `y_true` holds class indices 0..k-1 in the order of `Problem.classes`.
  `y_pred` is a probability matrix (n, k) whose columns follow that same order. Binary metrics
  use the column of `Problem.positive_index`, never "column 1" by assumption.
- Regression: `y_true` and `y_pred` are values on the original target scale. Callers invert any
  target transform first (§10.4).
- Probabilities are clipped to [1e-15, 1 − 1e-15] before log loss (§10.3).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn import metrics as skm

from tabpilot.schemas import METRIC_DIRECTION, PRIMARY_METRICS, MetricName, MetricSpec, Task

EPS = 1e-15
AP_POSITIVE_SHARE = 0.10
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Problem:
    """Task facts every metric needs: task, class order, and the positive class (binary)."""

    task: Task
    classes: tuple[str, ...] = ()
    positive_index: int | None = None

    def __post_init__(self) -> None:
        if self.task == Task.REGRESSION:
            if self.classes or self.positive_index is not None:
                raise ValueError("regression problems have no classes")
            return
        if len(self.classes) < 2:
            raise ValueError("classification needs at least 2 classes")
        if self.task == Task.BINARY and (
            len(self.classes) != 2 or self.positive_index not in (0, 1)
        ):
            raise ValueError("binary problems need 2 classes and a positive index")

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def spec(name: MetricName) -> MetricSpec:
    """The metric's spec with its fixed direction."""
    return MetricSpec(name=name, direction=METRIC_DIRECTION[name])


def default_primary(task: Task, positive_share: float | None = None) -> MetricName:
    """§10.2 defaults: binary ROC-AUC (AP if positive share < 10%), multiclass log loss,
    regression RMSE.
    """
    if task == Task.BINARY:
        if positive_share is not None and positive_share < AP_POSITIVE_SHARE:
            return MetricName.AVERAGE_PRECISION
        return MetricName.ROC_AUC
    return MetricName.LOG_LOSS if task == Task.MULTICLASS else MetricName.RMSE


SECONDARY: dict[Task, tuple[MetricName, ...]] = {
    Task.BINARY: (
        MetricName.ROC_AUC,
        MetricName.AVERAGE_PRECISION,
        MetricName.LOG_LOSS,
        MetricName.BRIER,
        MetricName.ACCURACY,
        MetricName.F1,
    ),
    Task.MULTICLASS: (
        MetricName.ACCURACY,
        MetricName.MACRO_F1,
        MetricName.BALANCED_ACCURACY,
        MetricName.OVR_MACRO_ROC_AUC,
    ),
    Task.REGRESSION: (MetricName.MAE, MetricName.R2, MetricName.RMSLE),
}


def allowed_primary(task: Task) -> frozenset[MetricName]:
    """User-selectable primary metrics for `task` (§10.2)."""
    return PRIMARY_METRICS[task]


def _clip(p: FloatArray) -> FloatArray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    normalized: FloatArray = clipped / clipped.sum(axis=1, keepdims=True)
    return normalized


def _positive(p: FloatArray, problem: Problem) -> FloatArray:
    assert problem.positive_index is not None  # noqa: S101 - guaranteed by Problem
    return np.asarray(p, dtype=np.float64)[:, problem.positive_index]


def _is_positive(y: IntArray, problem: Problem) -> IntArray:
    flags: IntArray = (np.asarray(y) == problem.positive_index).astype(np.int64)
    return flags


def log_loss(y: IntArray, p: FloatArray, problem: Problem) -> float:
    """Mean negative log-likelihood over all `problem.classes`, even ones absent in `y` (§10.3)."""
    probs = _clip(p)
    return float(-np.mean(np.log(probs[np.arange(len(y)), np.asarray(y)])))


def _ovr_macro_auc(y: IntArray, p: FloatArray, problem: Problem) -> float:
    """One-vs-rest macro ROC-AUC over the classes present in `y` (both outcomes needed)."""
    scores = []
    for k in range(problem.n_classes):
        target = (np.asarray(y) == k).astype(int)
        if 0 < target.sum() < len(target):
            scores.append(skm.roc_auc_score(target, np.asarray(p)[:, k]))
    return float(np.mean(scores)) if scores else float("nan")


def _rmsle(y: FloatArray, pred: FloatArray) -> float:
    if np.min(y) < 0:
        raise ValueError("RMSLE requires a non-negative target")
    return float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(y)) ** 2)))


def _balanced_accuracy(y: IntArray, p: FloatArray) -> float:
    """Balanced accuracy; a fold may legitimately miss a predicted class (§10.3)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(skm.balanced_accuracy_score(y, np.argmax(p, axis=1)))


def threshold_metrics(
    y: IntArray, p: FloatArray, problem: Problem, threshold: float = 0.5
) -> tuple[float, float]:
    """Accuracy and F1 for the positive class at `threshold` (binary, §10.2)."""
    truth = _is_positive(y, problem)
    pred = (_positive(p, problem) >= threshold).astype(int)
    return float(skm.accuracy_score(truth, pred)), float(skm.f1_score(truth, pred, zero_division=0))


def f1_optimal_threshold(y: IntArray, p: FloatArray, problem: Problem) -> float:
    """Threshold maximizing positive-class F1, chosen on OOF predictions (§10.2)."""
    truth = _is_positive(y, problem)
    precision, recall, thresholds = skm.precision_recall_curve(truth, _positive(p, problem))
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)  # fmt: skip
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]) if len(thresholds) else 0.5


def _roc_auc(y: IntArray, p: FloatArray, pr: Problem) -> float:
    return float(skm.roc_auc_score(_is_positive(y, pr), _positive(p, pr)))


def _average_precision(y: IntArray, p: FloatArray, pr: Problem) -> float:
    return float(skm.average_precision_score(_is_positive(y, pr), _positive(p, pr)))


def _brier(y: IntArray, p: FloatArray, pr: Problem) -> float:
    return float(np.mean((_positive(p, pr) - _is_positive(y, pr)) ** 2))


def _accuracy(y: IntArray, p: FloatArray, pr: Problem) -> float:
    if pr.task == Task.MULTICLASS:
        return float(skm.accuracy_score(y, np.argmax(p, axis=1)))
    return threshold_metrics(y, p, pr)[0]


def _f1(y: IntArray, p: FloatArray, pr: Problem) -> float:
    return threshold_metrics(y, p, pr)[1]


def _macro_f1(y: IntArray, p: FloatArray, pr: Problem) -> float:
    labels = list(range(pr.n_classes))
    pred = np.argmax(p, axis=1)
    return float(skm.f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def _rmse(y: FloatArray, pred: FloatArray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _mae(y: FloatArray, pred: FloatArray) -> float:
    return float(np.mean(np.abs(y - pred)))


def _r2(y: FloatArray, pred: FloatArray) -> float:
    return float(skm.r2_score(y, pred))


_CLASSIFICATION: dict[MetricName, Callable[[IntArray, FloatArray, Problem], float]] = {
    MetricName.ROC_AUC: _roc_auc,
    MetricName.AVERAGE_PRECISION: _average_precision,
    MetricName.LOG_LOSS: log_loss,
    MetricName.BRIER: _brier,
    MetricName.ACCURACY: _accuracy,
    MetricName.F1: _f1,
    MetricName.MACRO_F1: _macro_f1,
    MetricName.BALANCED_ACCURACY: lambda y, p, _pr: _balanced_accuracy(y, p),
    MetricName.OVR_MACRO_ROC_AUC: _ovr_macro_auc,
}
_REGRESSION: dict[MetricName, Callable[[FloatArray, FloatArray], float]] = {
    MetricName.RMSE: _rmse,
    MetricName.MAE: _mae,
    MetricName.RMSLE: _rmsle,
    MetricName.R2: _r2,
}
BINARY_ONLY = frozenset(
    {MetricName.ROC_AUC, MetricName.AVERAGE_PRECISION, MetricName.BRIER, MetricName.F1}
)


def score(
    name: MetricName, y_true: NDArray[np.generic], y_pred: NDArray[np.generic], problem: Problem
) -> float:
    """Compute metric `name` (§10.2) under the conventions in the module docstring."""
    if problem.task == Task.REGRESSION:
        if name not in _REGRESSION:
            raise ValueError(f"{name} is not a regression metric")
        pred = np.asarray(y_pred, dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            raise ValueError("predictions must be finite")
        return _REGRESSION[name](np.asarray(y_true, dtype=np.float64), pred)
    if name not in _CLASSIFICATION:
        raise ValueError(f"{name} is not a classification metric")
    proba = np.asarray(y_pred, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != problem.n_classes or not np.all(np.isfinite(proba)):
        raise ValueError("classification predictions must be a finite (n, n_classes) matrix")
    if name in BINARY_ONLY and problem.task != Task.BINARY:
        raise ValueError(f"{name} is a binary metric")
    return _CLASSIFICATION[name](np.asarray(y_true, dtype=np.int64), proba, problem)


def selection_score(name: MetricName, value: float) -> float:
    """Higher is always better (§10.4)."""
    return spec(name).selection_score(value)
