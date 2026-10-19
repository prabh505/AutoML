"""Feature DSL operation implementations (PRD §14.3, §14.5; A-007).

Op dispatch is a fixed mapping from op name to a function defined in this module. Spec fields
are data. Nothing here builds expressions from strings or looks attributes up by name.

NaN semantics (§14.5): arithmetic propagates NaN. `ratio` with |denominator| < eps → NaN.
`log1p`/`sqrt` of negatives → NaN (counted). Unseen group keys → NaN. `equals` on a missing input
→ 0. Stateful ops (`clip`, `bin_quantile`, `group_*`) learn state only in `fit_state`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

MISSING_KEY = "__missing__"
# Copies of `tabpilot.ingestion.inference` tokens (this package cannot import `tabpilot`; a unit
# test keeps them equal). Boolean columns arrive as `category` of their original tokens.
TRUE_TOKENS = frozenset({"true", "yes", "y", "t", "1"})
FALSE_TOKENS = frozenset({"false", "no", "n", "f", "0"})
Columns = list[pd.Series]
State = dict[str, Any]


def _num(s: pd.Series) -> np.ndarray:
    """Numeric view of a column; boolean tokens map to 1/0, anything else non-numeric → NaN."""
    if not pd.api.types.is_numeric_dtype(s.dtype):
        lowered = s.astype("string").str.strip().str.lower()
        tokens = lowered.map(
            {**dict.fromkeys(TRUE_TOKENS, "1"), **dict.fromkeys(FALSE_TOKENS, "0")}
        )
        s = tokens.fillna(lowered)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def negative_count(cols: Columns) -> int:
    """Rows whose first input is < 0: they become NaN in `log1p`/`sqrt` (A-007 counter)."""
    return int(np.sum(_num(cols[0]) < 0))


def _f64(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _keys(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna(MISSING_KEY).astype(object)


# ------------------------------------------------------------------ stateless ops
def add(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _f64(_num(cols[0]) + _num(cols[1]))


def subtract(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _f64(_num(cols[0]) - _num(cols[1]))


def multiply(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _f64(_num(cols[0]) * _num(cols[1]))


def ratio(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    num, den = _num(cols[0]), _num(cols[1])
    eps = float(params.get("eps", 1e-6))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = _f64(num / den)
    out[np.abs(den) < eps] = np.nan
    return out


def log1p(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    x = _num(cols[0])
    out = np.full_like(x, np.nan)
    ok = x >= 0
    out[ok] = np.log1p(x[ok])
    return out


def sqrt(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    x = _num(cols[0])
    out = np.full_like(x, np.nan)
    ok = x >= 0
    out[ok] = np.sqrt(x[ok])
    return out


def square(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _num(cols[0]) ** 2


def is_missing(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return cols[0].isna().to_numpy().astype(np.uint8)


def equals(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    x = _num(cols[0])
    return np.asarray(x == float(params["value"]), dtype=np.uint8)


def count_missing(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _f64(np.sum([c.isna().to_numpy() for c in cols], axis=0))


def row_sum(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _f64(np.sum([_num(c) for c in cols], axis=0))


def row_mean(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _f64(np.mean([_num(c) for c in cols], axis=0))


def concat_categories(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    parts = [_keys(c) for c in cols]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + "|" + part
    return joined.to_numpy(dtype=object)


def date_diff_days(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    a = pd.to_datetime(cols[0], errors="coerce")
    b = pd.to_datetime(cols[1], errors="coerce")
    return ((a - b).dt.total_seconds() / 86_400.0).to_numpy(dtype=np.float64, na_value=np.nan)


# ------------------------------------------------------------------ stateful ops
def clip_fit(cols: Columns, params: dict[str, Any]) -> State:
    x = _num(cols[0])
    lower = float(np.nanquantile(x, float(params["lower_q"])))
    upper = float(np.nanquantile(x, float(params["upper_q"])))
    return {"lower": lower, "upper": upper}


def clip(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _f64(np.clip(_num(cols[0]), state["lower"], state["upper"]))


def bin_quantile_fit(cols: Columns, params: dict[str, Any]) -> State:
    x = _num(cols[0])
    qs = np.linspace(0, 1, int(params["n_bins"]) + 1)[1:-1]
    return {"edges": [float(e) for e in np.unique(np.nanquantile(x, qs))]}


def bin_quantile(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    x = _num(cols[0])
    codes = np.searchsorted(np.asarray(state["edges"]), x, side="right")
    out = np.array([f"bin_{c}" for c in codes], dtype=object)
    out[np.isnan(x)] = None
    return out


def _group_fit(cols: Columns, stat: str) -> State:
    keys = _keys(cols[0])
    if stat == "count":
        return {"map": keys.value_counts().astype(float).to_dict()}
    values = pd.Series(_num(cols[1]), index=keys.index)
    grouped = values.groupby(keys)
    table = grouped.mean() if stat == "mean" else grouped.std(ddof=1)
    return {"map": {k: float(v) for k, v in table.items()}}


def _lookup(cols: Columns, state: State) -> np.ndarray:
    keys = _keys(cols[0])
    return keys.map(state["map"]).to_numpy(dtype=np.float64, na_value=np.nan)


def group_mean_fit(cols: Columns, params: dict[str, Any]) -> State:
    return _group_fit(cols, "mean")


def group_std_fit(cols: Columns, params: dict[str, Any]) -> State:
    return _group_fit(cols, "std")


def group_count_fit(cols: Columns, params: dict[str, Any]) -> State:
    return _group_fit(cols, "count")


def group_lookup(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _lookup(cols, state)


def group_deviation(cols: Columns, params: dict[str, Any], state: State) -> np.ndarray:
    return _f64(_num(cols[1]) - _lookup(cols, state))


TransformFn = Callable[[Columns, dict[str, Any], State], np.ndarray]
FitFn = Callable[[Columns, dict[str, Any]], State]

# op → (transform, fit-or-None, output kind). Output kind ∈ numeric | categorical | boolean.
OPS: dict[str, tuple[TransformFn, FitFn | None, str]] = {
    "add": (add, None, "numeric"),
    "subtract": (subtract, None, "numeric"),
    "multiply": (multiply, None, "numeric"),
    "ratio": (ratio, None, "numeric"),
    "log1p": (log1p, None, "numeric"),
    "sqrt": (sqrt, None, "numeric"),
    "square": (square, None, "numeric"),
    "clip": (clip, clip_fit, "numeric"),
    "bin_quantile": (bin_quantile, bin_quantile_fit, "categorical"),
    "is_missing": (is_missing, None, "boolean"),
    "equals": (equals, None, "boolean"),
    "count_missing": (count_missing, None, "numeric"),
    "row_sum": (row_sum, None, "numeric"),
    "row_mean": (row_mean, None, "numeric"),
    "concat_categories": (concat_categories, None, "categorical"),
    "group_mean": (group_lookup, group_mean_fit, "numeric"),
    "group_std": (group_lookup, group_std_fit, "numeric"),
    "group_count": (group_lookup, group_count_fit, "numeric"),
    "group_deviation": (group_deviation, group_mean_fit, "numeric"),
    "date_diff_days": (date_diff_days, None, "numeric"),
}
NEGATIVE_TO_NAN = frozenset({"log1p", "sqrt"})
STATEFUL = frozenset(op for op, (_, fit, _) in OPS.items() if fit is not None)
