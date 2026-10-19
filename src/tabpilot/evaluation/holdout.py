"""One-time holdout evaluation (PRD §11.2 L-7, §18.3, FR-EVL-02).

The finalizer freezes selection, then opens the `HoldoutGuard` exactly once and scores both the
selected pipeline and the reference baseline from that single access (PI-14). Classification
scores carry a seeded bootstrap 95% interval (1,000 resamples of HOLD rows). Results are reported
separately and are never passed to selection.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tabpilot.evaluation.metrics import Problem, score
from tabpilot.evaluation.predict import predict_aligned
from tabpilot.schemas import MetricName, SplitSpec, Task
from tabpilot.schemas.experiment import HoldoutResult, HoldoutScores
from tabpilot.splitting.guard import HoldoutGuard

BOOTSTRAP_RESAMPLES = 1000
CALLER = "evaluation.holdout"
NO_HOLDOUT_REASON = "no holdout: fewer than 2,000 valid rows; CV only (§11.5)"


def make_guard(split: SplitSpec, frozen: Callable[[], bool]) -> HoldoutGuard:
    """The run's guard over the split's HOLD rows; opens only when `frozen()` is True."""
    return HoldoutGuard(np.asarray(split.holdout_indices, dtype=np.int64), frozen)


def bootstrap_ci(
    y: NDArray[Any],
    pred: NDArray[np.float64],
    problem: Problem,
    metric: MetricName,
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float] | None:
    """Percentile 95% interval over row resamples. Resamples where the metric is undefined
    (e.g. one class only) are skipped; None when no resample is usable.
    """
    rng = np.random.default_rng(seed)
    values = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(resamples):
            idx = rng.integers(0, len(y), len(y))
            try:
                value = score(metric, y[idx], pred[idx], problem)
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
    if not values:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def _scores(
    model: Any, X: pd.DataFrame, y: NDArray[Any], problem: Problem, *, metric: MetricName, seed: int
) -> HoldoutScores:
    pred, _ = predict_aligned(model, X, problem)
    scored = pred[:, 0] if problem.task == Task.REGRESSION else pred
    out: HoldoutScores = {metric.value: score(metric, y, scored, problem)}
    if problem.task != Task.REGRESSION:
        ci = bootstrap_ci(y, scored, problem, metric, seed=seed)
        if ci is not None:
            out["ci95"] = ci
    return out


def evaluate_holdout(
    guard: HoldoutGuard | None,
    final_model: Any,
    baseline_model: Any | None,
    *,
    frame: pd.DataFrame,
    y: NDArray[Any],
    problem: Problem,
    metric: MetricName,
    seed: int,
) -> HoldoutResult:
    """Score the final model and the baseline on HOLD once (T-LEAK-03). Without a holdout the
    result says why.
    """
    if guard is None or guard.size == 0:
        return HoldoutResult(present=False, reason=NO_HOLDOUT_REASON)
    rows = guard.open(CALLER)
    X, y_hold = frame.iloc[rows], y[rows]
    final = _scores(final_model, X, y_hold, problem, metric=metric, seed=seed)
    baseline = (_scores(baseline_model, X, y_hold, problem, metric=metric, seed=seed)
                if baseline_model is not None else None)  # fmt: skip
    return HoldoutResult(present=True, final=final, baseline=baseline,
                         accessed_at=datetime.now(UTC))  # fmt: skip
