"""Predictions aligned to the problem's class order (PRD §10.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tabpilot.evaluation.metrics import Problem
from tabpilot.schemas import Task


def predict_aligned(
    model: Any, X: pd.DataFrame, problem: Problem
) -> tuple[NDArray[np.float64], bool]:
    """(n, n_outputs) predictions; probability columns aligned to `problem.classes` by the model's
    `classes_` (codes), never assumed. The flag is True when a class was absent from training
    (its column is 0, and log loss clips it, §10.3).
    """
    if problem.task == Task.REGRESSION:
        return np.asarray(model.predict(X), dtype=np.float64).reshape(len(X), 1), False
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(model.classes_).astype(np.int64)
    full = np.zeros((len(X), problem.n_classes), dtype=np.float64)
    full[:, classes] = proba
    return full, len(classes) < problem.n_classes
