"""P05 metric tests: hand-computed references, class alignment, directions (§10.2–§10.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tabpilot.evaluation.metrics import (
    SECONDARY,
    Problem,
    allowed_primary,
    default_primary,
    f1_optimal_threshold,
    score,
    selection_score,
    spec,
    threshold_metrics,
)
from tabpilot.schemas import Direction, MetricName, Task

BIN = Problem(Task.BINARY, ("No", "Yes"), positive_index=1)
BIN_REVERSED = Problem(Task.BINARY, ("Yes", "No"), positive_index=0)
MULTI = Problem(Task.MULTICLASS, ("a", "b", "c"))
REG = Problem(Task.REGRESSION)


@pytest.mark.unit
def test_binary_metrics_hand_computed() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    # ROC-AUC: positive scores {0.35, 0.8} vs negatives {0.1, 0.4}: 3 of 4 pairs ordered.
    assert score(MetricName.ROC_AUC, y, p, BIN) == pytest.approx(0.75)
    expected_ll = -np.mean(np.log([0.9, 0.6, 0.35, 0.8]))
    assert score(MetricName.LOG_LOSS, y, p, BIN) == pytest.approx(expected_ll)
    assert score(MetricName.BRIER, y, p, BIN) == pytest.approx(np.mean([0.01, 0.16, 0.4225, 0.04]))
    assert score(MetricName.ACCURACY, y, p, BIN) == pytest.approx(0.75)  # threshold 0.5
    assert score(MetricName.F1, y, p, BIN) == pytest.approx(2 / 3)
    # AP: ranking 0.8(+), 0.4(-), 0.35(+), 0.1(-) → (1/1 + 2/3) / 2
    assert score(MetricName.AVERAGE_PRECISION, y, p, BIN) == pytest.approx((1 + 2 / 3) / 2)


@pytest.mark.unit
def test_positive_column_follows_class_order_not_position() -> None:
    """AP/ROC-AUC use the positive class column, never 'column 1' by assumption."""
    y = np.array([1, 1, 0, 0])  # in BIN_REVERSED, index 0 = "Yes" is positive
    p = np.array([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]])
    assert score(MetricName.ROC_AUC, y, p, BIN_REVERSED) == pytest.approx(0.75)


@pytest.mark.unit
def test_multiclass_log_loss_with_missing_class_and_clipping() -> None:
    y = np.array([0, 1, 1])  # class "c" absent in this fold
    p = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    expected = -np.mean(np.log([1 - 2e-15, 0.5, 1e-15]))
    assert score(MetricName.LOG_LOSS, y, p, MULTI) == pytest.approx(expected, rel=1e-6)
    assert math.isfinite(score(MetricName.LOG_LOSS, y, p, MULTI))
    assert score(MetricName.MACRO_F1, y, p, MULTI) == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert score(MetricName.BALANCED_ACCURACY, y, p, MULTI) == pytest.approx(0.75)
    assert score(MetricName.ACCURACY, y, p, MULTI) == pytest.approx(2 / 3)
    auc = score(MetricName.OVR_MACRO_ROC_AUC, y, p, MULTI)  # classes a, b present
    assert 0 <= auc <= 1


@pytest.mark.unit
def test_regression_metrics_hand_computed() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 2.0, 2.0, 5.0])
    assert score(MetricName.RMSE, y, pred, REG) == pytest.approx(math.sqrt((0.25 + 0 + 1 + 1) / 4))
    assert score(MetricName.MAE, y, pred, REG) == pytest.approx(0.625)
    assert score(MetricName.R2, y, pred, REG) == pytest.approx(1 - 2.25 / 5)
    rmsle = math.sqrt(np.mean((np.log1p(pred) - np.log1p(y)) ** 2))
    assert score(MetricName.RMSLE, y, pred, REG) == pytest.approx(rmsle)
    with pytest.raises(ValueError, match="non-negative"):
        score(MetricName.RMSLE, -y, pred, REG)


@pytest.mark.unit
def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        score(MetricName.RMSE, np.ones(2), np.array([1.0, np.nan]), REG)
    with pytest.raises(ValueError, match="matrix"):
        score(MetricName.LOG_LOSS, np.zeros(2), np.ones((2, 2)), MULTI)
    with pytest.raises(ValueError, match="binary metric"):
        score(MetricName.ROC_AUC, np.zeros(3), np.full((3, 3), 1 / 3), MULTI)
    with pytest.raises(ValueError, match="regression metric"):
        score(MetricName.LOG_LOSS, np.zeros(2), np.zeros(2), REG)
    with pytest.raises(ValueError, match="classification metric"):
        score(MetricName.RMSE, np.zeros(2), np.full((2, 2), 0.5), BIN)
    with pytest.raises(ValueError, match="positive index"):
        Problem(Task.BINARY, ("a", "b"))
    with pytest.raises(ValueError, match="at least 2"):
        Problem(Task.MULTICLASS, ("a",))
    with pytest.raises(ValueError, match="no classes"):
        Problem(Task.REGRESSION, ("a", "b"))


@pytest.mark.unit
def test_directions_and_selection_score() -> None:
    """Every metric's direction, written out (P05 self-review)."""
    maximize = {MetricName.ROC_AUC, MetricName.AVERAGE_PRECISION, MetricName.ACCURACY,
                MetricName.F1, MetricName.MACRO_F1, MetricName.BALANCED_ACCURACY,
                MetricName.OVR_MACRO_ROC_AUC, MetricName.R2}  # fmt: skip
    for name in MetricName:
        expected = Direction.MAXIMIZE if name in maximize else Direction.MINIMIZE
        assert spec(name).direction == expected, name
    assert selection_score(MetricName.ROC_AUC, 0.8) == 0.8
    assert selection_score(MetricName.RMSE, 3.0) == -3.0
    assert selection_score(MetricName.LOG_LOSS, 0.2) > selection_score(MetricName.LOG_LOSS, 0.3)


@pytest.mark.unit
def test_default_and_allowed_metrics() -> None:
    assert default_primary(Task.BINARY, positive_share=0.30) == MetricName.ROC_AUC
    assert default_primary(Task.BINARY, positive_share=0.03) == MetricName.AVERAGE_PRECISION
    assert default_primary(Task.MULTICLASS) == MetricName.LOG_LOSS
    assert default_primary(Task.REGRESSION) == MetricName.RMSE
    assert MetricName.RMSLE in allowed_primary(Task.REGRESSION)
    assert MetricName.ACCURACY not in allowed_primary(Task.BINARY)
    assert set(SECONDARY) == set(Task)


@pytest.mark.unit
def test_f1_optimal_threshold_on_oof() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int)
    pos = np.clip(0.3 * y + rng.normal(0.2, 0.1, 400), 0, 1)
    p = np.c_[1 - pos, pos]
    t = f1_optimal_threshold(y, p, BIN)
    assert 0 < t < 1
    assert threshold_metrics(y, p, BIN, t)[1] >= threshold_metrics(y, p, BIN, 0.5)[1]
    constant = np.c_[np.full(4, 0.5), np.full(4, 0.5)]
    assert f1_optimal_threshold(np.array([0, 1, 0, 1]), constant, BIN) == pytest.approx(0.5)
