"""Input-side pipeline steps: name mapping, type casting, column selection (PRD §15.1, §15.3).

These run first in every exported pipeline, so they enforce the inference contract: input columns
may arrive under their original or normalized names, extra columns are ignored, and missing
columns raise `PredictionSchemaError` listing them (never silently filled).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tabpilot_runtime.dsl_ops import FALSE_TOKENS, TRUE_TOKENS
from tabpilot_runtime.transformers import _FrameTransformer

KINDS = frozenset({"numeric", "categorical", "boolean", "datetime"})


class PredictionSchemaError(ValueError):
    """Input columns the fitted pipeline needs are missing (§15.3)."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        super().__init__(f"missing input columns: {', '.join(self.missing)}")


class ColumnNameMapper(_FrameTransformer):
    """Original → normalized column names; output has exactly the mapped columns, in map order.

    A column is accepted under its normalized name or its original name (in that order).
    """

    def __init__(self, column_map: dict[str, str]) -> None:
        self.column_map = column_map

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        picked: dict[str, pd.Series] = {}
        missing: list[str] = []
        for original, normalized in self.column_map.items():
            if normalized in X.columns:
                picked[normalized] = X[normalized]
            elif original in X.columns:
                picked[normalized] = X[original]
            else:
                missing.append(original)
        if missing:
            raise PredictionSchemaError(missing)
        return pd.DataFrame(picked, index=X.index)

    def fit(self, X: pd.DataFrame, y: Any = None) -> ColumnNameMapper:
        self.feature_names_out_ = list(self._select(X).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._select(X)


def _datetime(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return values
    # "mixed" parses element by element, so one row and a batch parse identically (T-PRE-03).
    return pd.to_datetime(values, errors="coerce", format="mixed")


class TypeCaster(_FrameTransformer):
    """Casts each column to its schema kind (§15.1): numeric → float64 with ±inf → NaN,
    categorical → `category` of strings, boolean → kept (tokens or 0/1, see `BooleanEncoder`),
    datetime → datetime64. Unparseable values become missing.
    """

    def __init__(self, kinds: dict[str, str]) -> None:
        self.kinds = kinds

    def fit(self, X: pd.DataFrame, y: Any = None) -> TypeCaster:
        unknown = sorted(set(self.kinds.values()) - KINDS)
        if unknown:
            raise ValueError(f"unknown column kinds: {unknown}")
        self.feature_names_out_ = list(self.kinds)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, Any] = {}
        for col, kind in self.kinds.items():
            values = X[col]
            if kind == "numeric":
                num = pd.to_numeric(values, errors="coerce").astype("float64")
                out[col] = num.replace([np.inf, -np.inf], np.nan)
            elif kind == "categorical":
                out[col] = values.astype("string").astype("category")
            elif kind == "datetime":
                out[col] = _datetime(values)
            elif pd.api.types.is_numeric_dtype(values.dtype):
                out[col] = values.astype("float64")
            else:
                out[col] = values.astype("string")
        return pd.DataFrame(out, index=X.index)


class ColumnSelector(_FrameTransformer):
    """Fixes the feature list and order at fit; reindexes at transform (§15.3)."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> ColumnSelector:
        self.columns_ = list(X.columns)
        self.feature_names_out_ = list(self.columns_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise PredictionSchemaError(missing)
        return X[self.columns_]


class BooleanEncoder(_FrameTransformer):
    """Boolean columns → float32 1/0 (§15.2 boolean branch). Tokens map with `TRUE_TOKENS` /
    `FALSE_TOKENS` (case-insensitive); numbers map to `x != 0`; anything else stays missing.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> BooleanEncoder:
        self.feature_names_out_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, Any] = {}
        tokens = {**dict.fromkeys(TRUE_TOKENS, 1.0), **dict.fromkeys(FALSE_TOKENS, 0.0)}
        for col in X.columns:
            values = X[col]
            if pd.api.types.is_numeric_dtype(values.dtype):
                num = values.astype("float64")
                mapped = (num != 0).astype("float64").where(num.notna())
            else:
                lowered = values.astype("string").str.strip().str.lower()
                mapped = lowered.map(tokens).astype("float64")
            out[col] = mapped.astype(np.float32)
        return pd.DataFrame(out, index=X.index)
