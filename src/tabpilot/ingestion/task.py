"""Task detection and the binary positive-label rule (PRD §10.1, FR-TGT-02; PI-37).

Runs on the target values after missing-target rows are removed. The user always confirms the
task, and changing it re-runs target validation (`validate_target(..., task)`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabpilot.schemas import TaskDetection
from tabpilot.schemas._base import Task

MAX_CLASSES = 50
AMBIGUOUS_MAX = 20
AMBIGUOUS_MULTICLASS_DEFAULT_MAX = 10
BALANCED_LOW, BALANCED_HIGH = 0.45, 0.55
PREFERRED_POSITIVE = ("1", "true", "yes")


def positive_label(values: pd.Series) -> str:
    """Minority class; if balanced (45–55%), `1`/`true`/`yes` when present, else the
    lexicographically larger label (§10.1). The user may change it.
    """
    counts = values.astype("string").value_counts()
    labels = sorted(str(label) for label in counts.index)
    shares = counts / counts.sum()
    minority = str(shares.idxmin())
    if not (BALANCED_LOW <= float(shares.min()) <= BALANCED_HIGH):
        return minority
    for preferred in PREFERRED_POSITIVE:
        for label in labels:
            if label.lower() == preferred:
                return label
    return labels[-1]


def _integral(nums: pd.Series) -> bool:
    finite = nums[np.isfinite(nums)]
    return bool((finite == np.floor(finite)).all())


def detect_task(target: pd.Series) -> TaskDetection:
    """Detect binary / multiclass / regression with a confidence and reason (§10.1)."""
    values = target.dropna()
    distinct = int(values.nunique())
    if distinct == 2:
        return TaskDetection(task=Task.BINARY, confidence="high",
                             reason="exactly 2 distinct values",
                             positive_label=positive_label(values))  # fmt: skip
    if not pd.api.types.is_numeric_dtype(values):
        reason = f"text labels with {distinct} distinct values"
        if distinct > MAX_CLASSES:
            reason += f" (more than {MAX_CLASSES} classes is not supported)"
        return TaskDetection(task=Task.MULTICLASS, confidence="high", reason=reason)
    nums = values.astype("float64")
    if not _integral(nums):
        return TaskDetection(task=Task.REGRESSION, confidence="high",
                             reason="numbers with fractional values")  # fmt: skip
    # Integer-valued numbers (including floats such as 1.0, 2.0: PI-37).
    if distinct > MAX_CLASSES:
        return TaskDetection(task=Task.REGRESSION, confidence="high",
                             reason=f"integer with {distinct} distinct values")  # fmt: skip
    reason = f"integer with {distinct} distinct values"
    if distinct <= AMBIGUOUS_MAX and distinct <= AMBIGUOUS_MULTICLASS_DEFAULT_MAX:
        return TaskDetection(task=Task.MULTICLASS, confidence="ambiguous", reason=reason,
                             alternatives=[Task.REGRESSION])  # fmt: skip
    return TaskDetection(task=Task.REGRESSION, confidence="ambiguous", reason=reason,
                         alternatives=[Task.MULTICLASS])  # fmt: skip
