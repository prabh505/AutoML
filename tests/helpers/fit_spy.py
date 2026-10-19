"""`FitSpy`: records which rows each wrapped transformer/estimator was fitted on (T-LEAK-07).

Wrap any scikit-learn step: `FitSpy(SimpleImputer(), key="imputer")`. Each `fit` (and
`fit_transform`) records `list(X.index)` under `key` in `FIT_LOG`. Inputs must be pandas objects
so row identity survives (`set_output(transform="pandas")`). The log is stored by key, so it
survives `sklearn.base.clone`, which deep-copies parameters.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

FIT_LOG: defaultdict[str, list[list[Any]]] = defaultdict(list)


def reset_fit_log() -> None:
    """Forget every recorded fit."""
    FIT_LOG.clear()


def fitted_rows(key: str) -> list[set[Any]]:
    """Row-index sets recorded for `key`, one per fit call, in call order."""
    return [set(rows) for rows in FIT_LOG[key]]


class FitSpy(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Transparent wrapper that records fit-row indices of `inner`."""

    def __init__(self, inner: Any, key: str) -> None:
        self.inner = inner
        self.key = key

    def _record(self, X: Any) -> None:
        if not isinstance(X, pd.DataFrame | pd.Series):
            raise TypeError("FitSpy needs pandas input to know row identity")
        FIT_LOG[self.key].append(list(X.index))

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> FitSpy:
        self._record(X)
        self.inner_ = clone(self.inner).fit(X, y, **fit_params)
        return self

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        self._record(X)
        self.inner_ = clone(self.inner)
        if hasattr(self.inner_, "fit_transform"):
            return self.inner_.fit_transform(X, y, **fit_params)
        return self.inner_.fit(X, y, **fit_params).transform(X)

    def transform(self, X: Any) -> Any:
        return self.inner_.transform(X)

    def predict(self, X: Any) -> Any:
        return self.inner_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.inner_.predict_proba(X)

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        return self.inner_.get_feature_names_out(input_features)

    def set_output(self, *, transform: str | None = None) -> FitSpy:
        if hasattr(self.inner, "set_output"):
            self.inner.set_output(transform=transform)
        return self

    @property
    def classes_(self) -> Any:
        return self.inner_.classes_

    def __getattr__(self, name: str) -> Any:
        """Expose the wrapped step's fitted attributes (`failed_features_`, `best_iteration_`…)."""
        if name.endswith("_") and not name.startswith("_") and "inner_" in self.__dict__:
            return getattr(self.__dict__["inner_"], name)
        raise AttributeError(name)
