"""scikit-learn transformers pickled into `model.joblib` (PRD §14.2, §14.5, §15; A-001).

All learned state is set in `fit` and `transform` never refits (L-4). Every class is
`clone`-safe: constructor arguments are stored unchanged and only `*_`-suffixed attributes hold
fitted state.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from tabpilot_runtime.dsl_ops import NEGATIVE_TO_NAN, OPS, negative_count

LLM_PREFIX = "llm__"
RARE = "__rare__"
DEGENERATE_NAN_SHARE = 0.99
SECONDS_PER_100K = 2.0  # §14.5 per-feature cap on transform
MIN_CAP_S = 0.5  # floor so fixed per-call overhead on small frames is not an overrun (DEC-029)


class _FrameTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Base for transformers that take and return pandas DataFrames."""

    feature_names_out_: list[str]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    def set_output(self, *, transform: str | None = None) -> _FrameTransformer:
        return self  # always returns pandas


class RareCategoryGrouper(_FrameTransformer):
    """Categories with share < `min_share` or count < `min_count` on the training rows, and
    unseen categories, become `__rare__` (§14.2, §15.3). Missing stays missing.
    """

    def __init__(self, columns: list[str], min_share: float = 0.01, min_count: int = 10) -> None:
        self.columns = columns
        self.min_share = min_share
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y: Any = None) -> RareCategoryGrouper:
        self.keep_: dict[str, list[str]] = {}
        for col in self.columns:
            counts = X[col].astype("string").value_counts(dropna=True)
            ok = (counts >= self.min_count) & (counts / max(len(X), 1) >= self.min_share)
            self.keep_[col] = sorted(str(v) for v in counts.index[ok])
        self.feature_names_out_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, keep in self.keep_.items():
            values = out[col].astype("string")
            grouped = values.where(values.isin(keep) | values.isna(), RARE)
            out[col] = pd.Categorical(grouped, categories=[*keep, RARE])
        return out


class FrequencyEncoder(_FrameTransformer):
    """Replaces each listed categorical with its training-fold frequency; unseen → 0 (§14.2)."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> FrequencyEncoder:
        self.freq_ = {
            col: X[col].astype("string").value_counts(normalize=True).to_dict()
            for col in self.columns
        }
        self.feature_names_out_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, freq in self.freq_.items():
            mapped = out[col].astype("string").map(freq)
            out[col] = mapped.astype("float64").fillna(0.0).astype(np.float32)
        return out


class BaselineFeatureTransformer(_FrameTransformer):
    """Deterministic baseline features (§14.2; A-006).

    - Datetime columns → year, month, day, dayofweek, [hour], is_weekend, days_since_min (the
      minimum is learned on the training rows). The original column is dropped.
    - Missing indicators `<col>__missing` (uint8) for `missing_indicator_columns`.
    - Rare-category grouping for `categorical_columns`.
    """

    def __init__(
        self,
        datetime_columns: list[str],
        missing_indicator_columns: list[str],
        categorical_columns: list[str],
        rare_min_share: float = 0.01,
        rare_min_count: int = 10,
    ) -> None:
        self.datetime_columns = datetime_columns
        self.missing_indicator_columns = missing_indicator_columns
        self.categorical_columns = categorical_columns
        self.rare_min_share = rare_min_share
        self.rare_min_count = rare_min_count

    def fit(self, X: pd.DataFrame, y: Any = None) -> BaselineFeatureTransformer:
        self.min_date_: dict[str, Any] = {}
        self.has_time_: dict[str, bool] = {}
        for col in self.datetime_columns:
            values = pd.to_datetime(X[col], errors="coerce")
            self.min_date_[col] = values.min()
            times = values.dropna()
            self.has_time_[col] = bool((times != times.dt.normalize()).any())
        self.rare_ = RareCategoryGrouper(self.categorical_columns, self.rare_min_share,
                                         self.rare_min_count).fit(X)  # fmt: skip
        self.feature_names_out_ = list(self._transform(X).columns)
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self.rare_.transform(X)
        for col in self.missing_indicator_columns:
            out[f"{col}__missing"] = X[col].isna().to_numpy().astype(np.uint8)
        for col in self.datetime_columns:
            values = pd.to_datetime(X[col], errors="coerce")
            parts: dict[str, pd.Series] = {
                "year": values.dt.year,
                "month": values.dt.month,
                "day": values.dt.day,
                "dayofweek": values.dt.dayofweek,
            }
            if self.has_time_[col]:
                parts["hour"] = values.dt.hour
            parts["is_weekend"] = (values.dt.dayofweek >= 5).where(values.notna())
            base = self.min_date_[col]
            parts["days_since_min"] = (values - base).dt.total_seconds() / 86_400.0
            for name, part in parts.items():
                out[f"{col}__{name}"] = pd.to_numeric(part, errors="coerce").astype(np.float32)
            out = out.drop(columns=[col])
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform(X)


class DSLFeatureTransformer(_FrameTransformer):
    """Executes validated Feature DSL specs with first-party op code (§14.3–§14.6).

    `specs` are plain dicts (`id`, `name`, `op`, `inputs`, `params`) that were validated by
    `tabpilot.features.validate_specs`. Outputs are appended as `llm__<name>` (collisions get
    `__2`, …) in spec order: numeric → float32, categorical → category, boolean → uint8.

    A feature that raises, is constant or ≥ 99% NaN on the training rows, or exceeds the
    per-feature time cap is dropped and listed in `failed_features_`. Features depending on a
    dropped feature are dropped too (§14.6). For `log1p`/`sqrt`, `negative_inputs_` counts the
    training rows whose negative input became NaN (A-007).

    The time cap is soft: elapsed time is measured after each op returns, nothing is interrupted.
    That is sufficient because every op is first-party vectorized numpy/pandas code with no loops
    over user input, so it cannot hang; an overrun at `transform` is recorded in `slow_features_`
    for the CV runner to drop the feature and restart (§14.6).
    """

    def __init__(
        self,
        specs: list[dict[str, Any]],
        seconds_per_100k: float = SECONDS_PER_100K,
        min_cap_s: float = MIN_CAP_S,
    ) -> None:
        self.specs = specs
        self.seconds_per_100k = seconds_per_100k
        self.min_cap_s = min_cap_s

    def _cap(self, n_rows: int) -> float:
        return max(self.min_cap_s, self.seconds_per_100k * n_rows / 100_000)

    def _column_name(self, name: str, taken: set[str]) -> str:
        base = candidate = f"{LLM_PREFIX}{name}"
        n = 1
        while candidate in taken:
            n += 1
            candidate = f"{base}__{n}"
        return candidate

    def _inputs(self, spec: dict[str, Any], frame: pd.DataFrame) -> list[pd.Series]:
        return [
            frame[self.resolved_[ref]] if ref in self.resolved_ else frame[ref]
            for ref in spec["inputs"]
        ]

    def _cast(
        self, values: np.ndarray, kind: str, state: dict[str, Any]
    ) -> np.ndarray | pd.Categorical:
        if kind == "numeric":
            return values.astype(np.float32)
        if kind == "boolean":
            return values.astype(np.uint8)
        if "edges" in state:  # bin_quantile: categories fixed by the fitted edges
            return pd.Categorical(
                values, categories=[f"bin_{i}" for i in range(len(state["edges"]) + 1)]
            )
        return pd.Categorical(values)

    def fit(self, X: pd.DataFrame, y: Any = None) -> DSLFeatureTransformer:
        frame = X.copy()
        self.resolved_: dict[str, str] = {}
        self.state_: dict[str, dict[str, Any]] = {}
        self.failed_features_: dict[str, str] = {}
        self.active_: list[dict[str, Any]] = []
        self.negative_inputs_: dict[str, int] = {}
        cap = self._cap(len(X))
        taken = set(frame.columns)
        for spec in self.specs:
            name = spec["name"]
            missing_dep = [ref for ref in spec["inputs"] if ref in self.failed_features_]
            if missing_dep:
                self.failed_features_[name] = f"depends on failed feature {missing_dep[0]}"
                continue
            transform, fit_state, kind = OPS[spec["op"]]
            params = dict(spec.get("params") or {})
            try:
                cols = self._inputs(spec, frame)
                start = time.perf_counter()
                state = fit_state(cols, params) if fit_state else {}
                values = transform(cols, params, state)
                elapsed = time.perf_counter() - start
            except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
                self.failed_features_[name] = f"failed: {type(exc).__name__}"
                continue
            if elapsed > cap:
                self.failed_features_[name] = f"failed: exceeded {cap:.3f} s time cap"
                continue
            series = pd.Series(values)
            if series.isna().mean() >= DEGENERATE_NAN_SHARE or series.nunique(dropna=True) <= 1:
                self.failed_features_[name] = "degenerate"
                continue
            column = self._column_name(name, taken)
            taken.add(column)
            self.resolved_[name] = column
            self.state_[name] = state
            if spec["op"] in NEGATIVE_TO_NAN:
                self.negative_inputs_[name] = negative_count(cols)
            frame[column] = self._cast(values, kind, state)
            self.active_.append(spec)
        self.feature_names_out_ = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        self.slow_features_: list[str] = []
        cap = self._cap(len(X))
        for spec in self.active_:
            transform, _, kind = OPS[spec["op"]]
            start = time.perf_counter()
            values = transform(self._inputs(spec, frame), dict(spec.get("params") or {}),
                               self.state_[spec["name"]])  # fmt: skip
            if time.perf_counter() - start > cap:
                self.slow_features_.append(spec["name"])
            frame[self.resolved_[spec["name"]]] = self._cast(
                values, kind, self.state_[spec["name"]]
            )
        return frame
