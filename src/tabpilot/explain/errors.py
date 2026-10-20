"""Error analysis on DEV out-of-fold predictions (PRD §21.4; FR-EXP-01).

Classification: confusion matrix, per-class precision/recall, and worst segments ranked by error
rate. Regression: residuals, mean absolute error by predicted-value decile, and worst segments
ranked by mean absolute error. Worst segments come from the top 3 categorical features (in the
caller's importance order): categories with ≥ 30 DEV rows. Only OOF predictions are used, never
holdout rows. Segment labels are category values: they are rendered locally in the report and
never placed in the facts table or sent to an LLM (PI-16, I-6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tabpilot.evaluation.metrics import Problem
from tabpilot.schemas import Task

MIN_SEGMENT_ROWS = 30
TOP_CATEGORICAL = 3
SEGMENTS_PER_COLUMN = 5
DECILES = 10
MISSING_LABEL = "(missing)"


@dataclass(frozen=True)
class ClassMetrics:
    """Precision and recall of one class on OOF predictions."""

    label: str
    precision: float
    recall: float
    support: int


@dataclass(frozen=True)
class Segment:
    """One category of a categorical feature and its OOF error (rank 1 = worst)."""

    column: str
    rank: int
    label: str
    n: int
    error: float


@dataclass(frozen=True)
class DecileError:
    """Mean absolute error of the rows whose prediction falls in [lower, upper]."""

    lower: float
    upper: float
    mae: float
    n: int


@dataclass
class ErrorAnalysis:
    """§21.4 results. `error_name` says what `overall_error` and segment errors measure."""

    task: Task
    error_name: str
    overall_error: float
    labels: list[str] = field(default_factory=list)
    confusion: NDArray[np.int64] | None = None
    per_class: list[ClassMetrics] = field(default_factory=list)
    actual: NDArray[np.float64] | None = None
    predicted: NDArray[np.float64] | None = None
    deciles: list[DecileError] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)

    @property
    def confusion_normalized(self) -> NDArray[np.float64] | None:
        """Rows sum to 1 (true-class recall view); empty rows stay 0."""
        if self.confusion is None:
            return None
        totals = self.confusion.sum(axis=1, keepdims=True).astype(np.float64)
        out = np.zeros_like(totals * self.confusion, dtype=np.float64)
        return np.asarray(
            np.divide(self.confusion, totals, out=out, where=totals > 0), dtype=np.float64
        )

    @property
    def residuals(self) -> NDArray[np.float64] | None:
        if self.actual is None or self.predicted is None:
            return None
        return np.asarray(self.actual - self.predicted, dtype=np.float64)


def _labels(column: pd.Series) -> NDArray[Any]:
    values = column.astype("string").fillna(MISSING_LABEL)
    return np.asarray(values.to_numpy(dtype=object))


def worst_segments(
    errors: NDArray[np.float64], frame: pd.DataFrame, categorical: Sequence[str]
) -> list[Segment]:
    """Categories (≥ 30 rows) of the first 3 listed categorical columns, worst error first."""
    out: list[Segment] = []
    columns = [c for c in categorical if c in frame.columns][:TOP_CATEGORICAL]
    for column in columns:
        labels = _labels(frame[column])
        stats = []
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            n = int(mask.sum())
            if n >= MIN_SEGMENT_ROWS:
                stats.append((float(errors[mask].mean()), str(label), n))
        stats.sort(key=lambda s: (-s[0], s[1]))
        out.extend(
            Segment(column, rank, label, n, err)
            for rank, (err, label, n) in enumerate(stats[:SEGMENTS_PER_COLUMN], start=1)
        )
    return out


def _deciles(pred: NDArray[np.float64], abs_err: NDArray[np.float64]) -> list[DecileError]:
    edges = np.quantile(pred, np.linspace(0.0, 1.0, DECILES + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], pred, side="right"), 0, DECILES - 1)
    out = []
    for b in range(DECILES):
        mask = bins == b
        if mask.any():
            out.append(DecileError(float(pred[mask].min()), float(pred[mask].max()),
                                   float(abs_err[mask].mean()), int(mask.sum())))  # fmt: skip
    return out


def error_analysis(
    oof: NDArray[Any],
    y: NDArray[Any],
    dev_frame: pd.DataFrame,
    problem: Problem,
    categorical: Sequence[str] = (),
) -> ErrorAnalysis:
    """§21.4 on OOF predictions aligned with `y` and `dev_frame` (DEV rows only)."""
    pred_raw = np.asarray(oof, dtype=np.float64)
    if len(pred_raw) != len(y) or len(y) != len(dev_frame):
        raise ValueError("oof, y and dev_frame must have the same rows")
    if problem.task == Task.REGRESSION:
        pred = pred_raw.reshape(len(pred_raw), -1)[:, 0]
        actual = np.asarray(y, dtype=np.float64)
        abs_err = np.abs(actual - pred)
        return ErrorAnalysis(task=problem.task, error_name="mean absolute error",
                             overall_error=float(abs_err.mean()), actual=actual, predicted=pred,
                             deciles=_deciles(pred, abs_err),
                             segments=worst_segments(abs_err, dev_frame, categorical))  # fmt: skip
    truth = np.asarray(y, dtype=np.int64)
    k = problem.n_classes
    if problem.task == Task.BINARY and pred_raw.ndim == 1:
        pred_raw = np.column_stack([1 - pred_raw, pred_raw])
    predicted = np.argmax(pred_raw, axis=1)
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    per_class = []
    for c, label in enumerate(problem.classes):
        tp = int(confusion[c, c])
        col, row = int(confusion[:, c].sum()), int(confusion[c, :].sum())
        per_class.append(ClassMetrics(label, tp / col if col else 0.0, tp / row if row else 0.0,
                                      row))  # fmt: skip
    wrong = (predicted != truth).astype(np.float64)
    return ErrorAnalysis(task=problem.task, error_name="error rate",
                         overall_error=float(wrong.mean()), labels=list(problem.classes),
                         confusion=confusion, per_class=per_class,
                         segments=worst_segments(wrong, dev_frame, categorical))  # fmt: skip
