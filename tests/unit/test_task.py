"""P05 task detection tests: T-TGT-07…10 (§10.1; PI-37)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabpilot.ingestion.task import detect_task, positive_label
from tabpilot.schemas import Task


@pytest.mark.unit
@pytest.mark.parametrize(("values", "positive"), [
    (["Yes"] * 30 + ["No"] * 70, "Yes"),            # minority
    (["a"] * 50 + ["b"] * 50, "b"),                 # balanced → lexicographically larger
    (["no"] * 52 + ["yes"] * 48, "yes"),            # balanced, `yes` preferred
    (["0"] * 50 + ["1"] * 50, "1"),
    (["False"] * 46 + ["True"] * 54, "True"),
    (["z"] * 46 + ["0"] * 54, "z"),                 # balanced, no preferred token
])  # fmt: skip
def test_T_TGT_07_binary_and_positive_label(values: list[str], positive: str) -> None:
    det = detect_task(pd.Series(values, dtype="category"))
    assert (det.task, det.confidence, det.positive_label) == (Task.BINARY, "high", positive)
    assert positive_label(pd.Series(values)) == positive
    assert detect_task(pd.Series([0.0, 1.0] * 20)).task == Task.BINARY


@pytest.mark.unit
def test_T_TGT_08_multiclass_labels() -> None:
    det = detect_task(pd.Series(list("ABCDE") * 20, dtype="category"))
    assert (det.task, det.confidence, det.positive_label) == (Task.MULTICLASS, "high", None)
    many = detect_task(pd.Series([f"k{i}" for i in range(60)] * 2))
    assert many.task == Task.MULTICLASS
    assert "not supported" in many.reason  # validation then raises E-TARGET-TOO-MANY-CLASSES


@pytest.mark.unit
@pytest.mark.parametrize("values", [
    np.linspace(0, 10, 200), np.arange(200) * 1.0, np.r_[np.arange(60), np.arange(60)],
])  # fmt: skip
def test_T_TGT_09_regression(values: np.ndarray) -> None:
    det = detect_task(pd.Series(values))
    assert (det.task, det.confidence) == (Task.REGRESSION, "high")


@pytest.mark.unit
@pytest.mark.parametrize(("distinct", "task", "alternative"), [
    (3, Task.MULTICLASS, Task.REGRESSION), (10, Task.MULTICLASS, Task.REGRESSION),
    (11, Task.REGRESSION, Task.MULTICLASS), (20, Task.REGRESSION, Task.MULTICLASS),
    (21, Task.REGRESSION, Task.MULTICLASS), (50, Task.REGRESSION, Task.MULTICLASS),
])  # fmt: skip
def test_T_TGT_10_ambiguous_integers(distinct: int, task: Task, alternative: Task) -> None:
    for values in (np.arange(400) % distinct, (np.arange(400) % distinct).astype(float)):  # PI-37
        det = detect_task(pd.Series(values))
        assert (det.task, det.confidence, det.alternatives) == (task, "ambiguous", [alternative])
        assert f"{distinct} distinct" in det.reason
