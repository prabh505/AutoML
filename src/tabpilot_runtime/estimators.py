"""The model step of every pipeline (PRD §15.1, §16.1–§16.3).

`TabularModel` wraps one library estimator and adds what the pipeline needs around it:

- numeric columns → float32 (§15.3); raw categorical columns (CatBoost) stay strings;
- categorical declarations per library: LightGBM `categorical_feature`, CatBoost `cat_features`,
  selected by output-name prefix because DSL output names are resolved only at fit;
- early stopping for boosting models on an internal 90/10 split of the (already preprocessed)
  training fold, stratified when possible, with a fixed seed. The outer validation fold is never
  seen. The best iteration is recorded in `best_iteration_` (§16.3, §11.10);
- XGBoost binary `scale_pos_weight` = negatives / positives of the training fold (§16.2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import train_test_split
from sklearn.utils.metaestimators import available_if

FAMILIES = frozenset({"linear", "random_forest", "lightgbm", "xgboost", "catboost", "hist_gb"})
EARLY_STOPPING_FAMILIES = frozenset({"lightgbm", "xgboost", "catboost"})
MIN_EARLY_STOPPING_ROWS = 20


def _has_proba(model: TabularModel) -> bool:
    return bool(is_classifier(model.estimator))


class TabularModel(BaseEstimator):  # type: ignore[misc]
    """Library estimator + float32 casting, categorical declaration, and early stopping."""

    def __init__(
        self,
        estimator: Any,
        family: str,
        *,
        categorical_prefixes: tuple[str, ...] = (),
        early_stopping: bool = False,
        early_stopping_fraction: float = 0.1,
        early_stopping_rounds: int = 50,
        balance_scale_pos_weight: bool = False,
        seed: int = 0,
    ) -> None:
        self.estimator = estimator
        self.family = family
        self.categorical_prefixes = categorical_prefixes
        self.early_stopping = early_stopping
        self.early_stopping_fraction = early_stopping_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.balance_scale_pos_weight = balance_scale_pos_weight
        self.seed = seed

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier" if is_classifier(self.estimator) else "regressor"
        return tags

    def _categorical(self, X: pd.DataFrame) -> list[str]:
        prefixes = self.categorical_prefixes
        return [c for c in X.columns if prefixes and str(c).startswith(prefixes)]

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = set(self.raw_categorical_) if self.family == "catboost" else set()
        out = X.copy()
        for col in out.columns:
            if col in raw:
                out[col] = out[col].astype("string").fillna("__missing__").astype(object)
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
        return out

    def _fit_kwargs(self) -> dict[str, Any]:
        if self.family == "lightgbm" and self.raw_categorical_:
            return {"categorical_feature": list(self.raw_categorical_)}
        if self.family == "catboost":
            return {"cat_features": list(self.raw_categorical_), "verbose": False}
        if self.family == "xgboost":
            return {"verbose": False}
        return {}

    def _inner_split(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        stratify = None
        if is_classifier(self.estimator):
            _, counts = np.unique(y, return_counts=True)
            if counts.min() >= 2:
                stratify = y
        X_fit, X_es, y_fit, y_es = train_test_split(
            X, y, test_size=self.early_stopping_fraction, random_state=self.seed, stratify=stratify
        )
        return X_fit, X_es, np.asarray(y_fit), np.asarray(y_es)

    def _fit_with_early_stopping(self, est: Any, X: pd.DataFrame, y: np.ndarray) -> int:
        X_fit, X_es, y_fit, y_es = self._inner_split(X, y)
        self.early_stopping_rows_ = list(X_es.index)
        kwargs = self._fit_kwargs()
        rounds = self.early_stopping_rounds
        if self.family == "lightgbm":
            import lightgbm  # noqa: PLC0415 - optional library, imported only when used

            stop = lightgbm.early_stopping(rounds, verbose=False)
            est.fit(X_fit, y_fit, eval_X=(X_es,), eval_y=(y_es,), callbacks=[stop], **kwargs)
            return int(est.best_iteration_ or est.n_estimators)
        if self.family == "xgboost":
            est.set_params(early_stopping_rounds=rounds)
            est.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], **kwargs)
            return int(est.best_iteration) + 1
        est.fit(X_fit, y_fit, eval_set=(X_es, y_es), early_stopping_rounds=rounds, **kwargs)
        return int(est.get_best_iteration()) + 1

    def fit(self, X: pd.DataFrame, y: Any) -> TabularModel:
        """Fit on the preprocessed training fold (§16.3)."""
        if self.family not in FAMILIES:
            raise ValueError(f"unknown model family {self.family!r}")
        y_arr = np.asarray(y)
        self.raw_categorical_ = self._categorical(X)
        X_ready = self._prepare(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        est = clone(self.estimator)
        if self.balance_scale_pos_weight and self.family == "xgboost":
            positives = float(np.sum(y_arr == 1))
            est.set_params(scale_pos_weight=(len(y_arr) - positives) / max(positives, 1.0))
        self.early_stopping_rows_ = []
        self.best_iteration_: int | None = None
        use_es = (self.early_stopping and self.family in EARLY_STOPPING_FAMILIES
                  and len(y_arr) >= MIN_EARLY_STOPPING_ROWS)  # fmt: skip
        if use_es:
            self.best_iteration_ = self._fit_with_early_stopping(est, X_ready, y_arr)
        else:
            est.fit(X_ready, y_arr, **self._fit_kwargs())
        self.estimator_ = est
        if is_classifier(self.estimator):
            self.classes_ = np.asarray(est.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator_.predict(self._prepare(X))).reshape(len(X), -1).squeeze(1)

    @available_if(_has_proba)  # type: ignore[untyped-decorator]
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator_.predict_proba(self._prepare(X)))
