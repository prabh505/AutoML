"""Per-column statistics, associations, correlations, and structure checks (PRD §9.2).

Everything here receives DEV rows only (see `profiling.profile`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score

from tabpilot.ingestion.inference import ID_NAME
from tabpilot.schemas import SemanticType, Task
from tabpilot.schemas.dataset import ColumnInfo
from tabpilot.schemas.profile import Association, ColumnProfile, CorrelatedPair

QUANTILE_SAMPLE = 100_000
ASSOCIATION_SAMPLE = 50_000
CORRELATION_SAMPLE = 20_000
CORRELATION_COLUMNS = 100
CORRELATION_THRESHOLD = 0.95
MI_TOP_CATEGORIES = 50
RARE_SHARE = 0.01
TOP_K = 10
TEMPORAL_RHO = 0.3
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def scalar(value: object) -> float:
    """A pandas/numpy reduction result as a Python float (NaN if not numeric)."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _finite(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def numeric_values(values: pd.Series, semantic: SemanticType) -> pd.Series:
    """Float view of a numeric or datetime column (datetimes as days since epoch)."""
    if semantic == SemanticType.DATETIME:
        days = values.astype("datetime64[ns]").astype("int64") / 86_400e9
        return days.where(values.notna()).astype("float64")
    return values.astype("float64")


def numeric_stats(col: ColumnInfo, values: pd.Series, rng: np.random.Generator) -> dict[str, Any]:
    """Distribution statistics for a numeric column (§9.2)."""
    x = values.astype("float64").dropna()
    if x.empty:
        return {}
    qx = x
    if len(x) > QUANTILE_SAMPLE:
        qx = x.sample(QUANTILE_SAMPLE, random_state=int(rng.integers(2**31)))
    p1, p5, p25, p50, p75, p95, p99 = (float(v) for v in np.quantile(qx, QUANTILES))
    iqr = p75 - p25
    outside_iqr = ((x < p25 - 1.5 * iqr) | (x > p75 + 1.5 * iqr)).mean()
    return {
        "mean": float(x.mean()), "std": _finite(float(x.std(ddof=1))) if len(x) > 1 else 0.0,
        "min": float(x.min()), "p1": p1, "p5": p5, "p25": p25, "p50": p50, "p75": p75,
        "p95": p95, "p99": p99, "max": float(x.max()), "skew": _finite(scalar(x.skew())),
        "kurtosis": _finite(scalar(x.kurt())), "zero_rate": float((x == 0).mean()),
        "negative_rate": float((x < 0).mean()), "outlier_rate_iqr": float(outside_iqr),
        "outlier_rate_p1_p99": float(((x < p1) | (x > p99)).mean()),
        "possibly_categorical": "possibly_categorical" in col.hints,
    }  # fmt: skip


def categorical_stats(values: pd.Series) -> dict[str, Any]:
    """Counts, entropy, top-k shares, rare share (§9.2). Labels stay in the full profile."""
    counts = values.astype("string").value_counts(dropna=True)
    if counts.empty:
        return {}
    shares = counts / counts.sum()
    entropy = float(-(shares * np.log2(shares)).sum())
    return {
        "entropy_bits": max(entropy, 0.0),
        "top_categories": [str(v) for v in counts.index[:TOP_K]],
        "top_shares": [round(float(s), 6) for s in shares.iloc[:TOP_K]],
        "rare_share": min(1.0, float(shares[shares < RARE_SHARE].sum())),
    }


def datetime_stats(values: pd.Series) -> dict[str, Any]:
    """Range and order facts for a datetime column (§9.2)."""
    x = values.dropna()
    if x.empty:
        return {}
    lo, hi = x.min(), x.max()
    return {
        "min_date": lo.isoformat(), "max_date": hi.isoformat(),
        "span_days": float((hi - lo).total_seconds() / 86_400),
        "monotonic_with_row_order": bool(x.is_monotonic_increasing or x.is_monotonic_decreasing),
    }  # fmt: skip


def association(
    col: ColumnInfo, values: pd.Series, y: pd.Series, task: Task, seed: int
) -> Association | None:
    """Feature–target association (§9.2). Classification AUC is direction-free, max(a, 1 − a)."""
    categorical = col.semantic_type in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN)
    if categorical:
        labels = values.astype("string").fillna("__missing__")
        top = labels.value_counts().index[:MI_TOP_CATEGORIES]
        codes = pd.Categorical(labels.where(labels.isin(top), "__other__")).codes.reshape(-1, 1)
        if task == Task.REGRESSION:
            mi = mutual_info_regression(codes, y.astype("float64"), discrete_features=True,
                                        random_state=seed)  # fmt: skip
        else:
            target = pd.factorize(y.astype("string"))[0]
            mi = mutual_info_classif(codes, target, discrete_features=True, random_state=seed)
        return Association(kind="mi", value=round(float(mi[0]), 6))
    x = numeric_values(values, col.semantic_type)
    mask = x.notna().to_numpy()
    if mask.sum() < 3:
        return None
    xv = x.to_numpy()[mask]
    if task == Task.REGRESSION:
        rho = sps.spearmanr(xv, y.astype("float64").to_numpy()[mask]).statistic
        if not math.isfinite(rho):
            return None
        return Association(kind="spearman", value=round(float(rho), 6))
    target = pd.factorize(y.astype("string"))[0][mask]
    aucs = []
    for k in np.unique(target):
        hit = target == k
        if 0 < hit.sum() < len(hit):
            auc = float(roc_auc_score(hit, xv))
            aucs.append(max(auc, 1 - auc))
        if task == Task.BINARY:
            break
    return Association(kind="auc", value=round(float(np.mean(aucs)), 6)) if aucs else None


def association_strength(assoc: Association | None) -> float:
    """Comparable strength for ranking columns (§9.4 ordering)."""
    if assoc is None:
        return 0.0
    return abs(assoc.value - 0.5) * 2 if assoc.kind == "auc" else abs(assoc.value)


def correlated_pairs(
    frame: pd.DataFrame, numeric: list[str], rng: np.random.Generator
) -> list[CorrelatedPair]:
    """Numeric pairs with |Spearman ρ| ≥ 0.95 among the 100 highest-variance columns (§9.2)."""
    if len(numeric) < 2:
        return []
    sample = frame[numeric]
    if len(sample) > CORRELATION_SAMPLE:
        sample = sample.sample(CORRELATION_SAMPLE, random_state=int(rng.integers(2**31)))
    data = sample.astype("float64")
    top = data.var().sort_values(ascending=False).index[:CORRELATION_COLUMNS]
    rho = data[top].corr(method="spearman")
    pairs = []
    cols = list(rho.columns)
    matrix = rho.to_numpy(dtype=np.float64)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b, value = cols[j], float(matrix[i, j])
            if math.isfinite(value) and abs(value) >= CORRELATION_THRESHOLD:
                pairs.append(CorrelatedPair(a=a, b=b, rho=round(float(value), 6)))
    return sorted(pairs, key=lambda p: (-abs(p.rho), p.a, p.b))


def group_candidates(frame: pd.DataFrame, columns: list[ColumnInfo]) -> list[str]:
    """ID-like names with 2 ≤ mean rows per value ≤ rows/10 (§9.2)."""
    n = len(frame)
    found = []
    for c in columns:
        distinct = frame[c.name].nunique()
        if distinct and ID_NAME.search(c.name) and 2 <= n / distinct <= n / 10:
            found.append(c.name)
    return found


def temporal_columns(
    frame: pd.DataFrame, columns: list[ColumnInfo], y: pd.Series, task: Task
) -> list[str]:
    """Datetimes monotonic with row order, or monotonic ID-like columns correlated with the
    target (|ρ| ≥ 0.3) (§9.2).
    """
    target = y.astype("float64") if task == Task.REGRESSION else pd.Series(
        pd.factorize(y.astype("string"))[0], index=y.index, dtype="float64")  # fmt: skip
    found = []
    for c in columns:
        values = frame[c.name].dropna()
        if values.empty or values.nunique() < 3:
            continue
        monotonic = values.is_monotonic_increasing or values.is_monotonic_decreasing
        if c.semantic_type == SemanticType.DATETIME and monotonic:
            found.append(c.name)
        elif (
            (c.semantic_type == SemanticType.IDENTIFIER or ID_NAME.search(c.name))
            and monotonic
            and pd.api.types.is_numeric_dtype(values)
        ):
            rho = sps.spearmanr(values.astype("float64"), target.loc[values.index]).statistic
            if math.isfinite(rho) and abs(rho) >= TEMPORAL_RHO:
                found.append(c.name)
    return found


def column_profile(col: ColumnInfo, values: pd.Series, extra: dict[str, Any]) -> ColumnProfile:
    """Assemble one `ColumnProfile`."""
    return ColumnProfile(name=col.name, type=col.semantic_type,
                         missing_rate=float(values.isna().mean()) if len(values) else 0.0,
                         distinct=int(values.nunique()), **extra)  # fmt: skip
