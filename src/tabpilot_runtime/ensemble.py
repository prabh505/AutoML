"""`VotingEnsemble`: fixed-weight blend of fitted pipelines (PRD §20.3, §20.5; A-001).

Classification averages member probabilities after aligning each member's columns to
`classes_` by label (never by position), then renormalizes rows to sum to 1. Regression averages
predictions. Weights are fixed at construction (learned on OOF by the ensembling stage).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if


def _is_classifier(model: VotingEnsemble) -> bool:
    return model.task == "classification"


class VotingEnsemble(BaseEstimator):  # type: ignore[misc]
    """Weighted soft-voting over `estimators` (already fitted when `prefit=True`)."""

    def __init__(
        self,
        estimators: list[Any],
        weights: list[float],
        task: str = "classification",
        classes: list[Any] | None = None,
        prefit: bool = True,
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.task = task
        self.classes = classes
        self.prefit = prefit

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier" if self.task == "classification" else "regressor"
        return tags

    def _check(self) -> np.ndarray:
        if self.task not in ("classification", "regression"):
            raise ValueError(f"unknown task {self.task!r}")
        weights = np.asarray(self.weights, dtype=float)
        if len(weights) != len(self.estimators) or len(weights) == 0 or np.any(weights < 0):
            raise ValueError("one non-negative weight per estimator is required")
        if weights.sum() <= 0:
            raise ValueError("weights must not all be zero")
        return np.asarray(weights / weights.sum(), dtype=float)

    def fit(self, X: Any, y: Any) -> VotingEnsemble:
        """Refit clones of the members on (X, y), unless they are prefit (the default)."""
        self.weights_ = self._check()
        self.estimators_ = (list(self.estimators) if self.prefit
                            else [clone(e).fit(X, y) for e in self.estimators])  # fmt: skip
        if self.task == "classification":
            labels = self.classes if self.classes is not None else sorted(
                {c for e in self.estimators_ for c in np.asarray(e.classes_).tolist()})  # fmt: skip
            self.classes_ = np.asarray(labels)
        return self

    def _fitted(self) -> None:
        """Prefit members need no data: an unfitted prefit ensemble initializes itself."""
        if hasattr(self, "estimators_"):
            return
        if not self.prefit:
            raise ValueError("VotingEnsemble is not fitted")
        self.fit(None, None)

    @available_if(_is_classifier)  # type: ignore[untyped-decorator]
    def predict_proba(self, X: Any) -> np.ndarray:
        self._fitted()
        index = {label: i for i, label in enumerate(self.classes_.tolist())}
        out = np.zeros((len(X), len(self.classes_)), dtype=float)
        for weight, est in zip(self.weights_, self.estimators_, strict=True):
            proba = np.asarray(est.predict_proba(X), dtype=float)
            for j, label in enumerate(np.asarray(est.classes_).tolist()):
                out[:, index[label]] += weight * proba[:, j]
        totals = out.sum(axis=1, keepdims=True)
        safe = np.full_like(out, 1.0 / out.shape[1])
        return np.asarray(np.divide(out, totals, out=safe, where=totals > 0), dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        self._fitted()
        if self.task == "classification":
            return np.asarray(self.classes_[np.argmax(self.predict_proba(X), axis=1)])
        preds = np.column_stack([np.asarray(e.predict(X), dtype=float) for e in self.estimators_])
        return np.asarray(preds @ self.weights_)
