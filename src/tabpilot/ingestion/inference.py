"""Type inference, column checks, and downcasting (PRD §8.5–§8.6; PI-41, PI-42; A-006).

Input columns are strings (string-first parsing) or, for Parquet, already-typed columns.
Every coercion, conversion, and exclusion is counted and returned, so nothing is discarded
silently (§8.6 no-silent-discard rule).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tabpilot.schemas import ColumnRole, SemanticType, WarningCode

NA_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "#n/a"})  # case-insensitive
BOOL_TOKENS = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"})
TRUE_TOKENS = frozenset({"true", "yes", "y", "t", "1"})
ID_NAME = re.compile(r"(^|_)(id|uuid|guid|key|index|row|record|code|number|no)($|_)")
THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d",
                "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y %H:%M",
                "%m/%d/%Y %H:%M")  # fmt: skip
PARSE_SHARE = 0.95
ID_DISTINCT = 0.95
INT_ID_DISTINCT = 0.99
KEY_LENGTH_CV = 0.1
TEXT_MEAN_LEN = 50
TEXT_DISTINCT = 0.5
POSSIBLY_CATEGORICAL_MAX = 15
EXCLUDE_MISSING = 0.95
HIGH_MISSING = 0.30
NEAR_CONSTANT = 0.995
HIGH_CARDINALITY = 50
VERY_HIGH_CARDINALITY_RATIO = 0.5
FLOAT32_SAFE = 1e7


@dataclass
class ColumnResult:
    """Typed values plus the decisions made for one column."""

    values: pd.Series
    semantic_type: SemanticType
    role: ColumnRole = ColumnRole.FEATURE
    exclusion_reason: str | None = None
    coerced_to_missing: int = 0
    inf_converted: int = 0
    hints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def clean_strings(raw: pd.Series) -> pd.Series:
    """Strip whitespace and turn NA tokens into missing (string-first parsing, §8.3 step 6)."""
    s = raw.astype("string").str.strip()
    return s.mask(s.str.lower().isin(NA_TOKENS))


def _to_numeric(s: pd.Series) -> pd.Series:
    nums = pd.to_numeric(s, errors="coerce")
    nonnull = s.dropna()
    if len(nonnull) and nums.notna().sum() / len(nonnull) < PARSE_SHARE:
        has_thousands = nonnull.str.contains(",", regex=False)
        if has_thousands.any() and nonnull[has_thousands].str.match(THOUSANDS).all():
            nums = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
    return nums.astype("float64")


def _parse_datetime(s: pd.Series) -> tuple[pd.Series, float]:
    """Best single format among the candidates (never per-element guessing, §8.5 rule 5)."""
    nonnull = s.dropna()
    best: tuple[pd.Series, float] = (pd.Series(pd.NaT, index=s.index), 0.0)
    if nonnull.empty:
        return best
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        share = parsed.notna().sum() / len(nonnull)
        if share > best[1]:
            best = (parsed, share)
        if share == 1.0:
            break
    return best


def _is_key_like(nonnull: pd.Series) -> bool:
    lengths = nonnull.str.len().astype(float)
    no_spaces = not nonnull.str.contains(r"\s", regex=True).any()
    mean = lengths.mean()
    return bool(no_spaces and mean > 0 and lengths.std(ddof=0) / mean <= KEY_LENGTH_CV)


def _downcast_numeric(values: pd.Series) -> pd.Series:
    """float32 unless large values have fractional parts; ints → smallest signed int (§8.5)."""
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values.astype("float32")
    integral = bool((finite == np.floor(finite)).all())
    if integral and values.notna().all():
        return pd.to_numeric(values.astype("int64"), downcast="integer")
    big_fraction = ((finite.abs() > FLOAT32_SAFE) & (finite != np.floor(finite))).any()
    if big_fraction or (integral and finite.abs().max() >= 2**24):
        return values.astype("float64")
    return values.astype("float32")


def _numeric(name: str, s: pd.Series, nonnull_count: int) -> ColumnResult:
    nums = _to_numeric(s) if s.dtype == "string" else s.astype("float64")
    coerced = int(s.notna().sum() - nums.notna().sum())
    inf_mask = np.isinf(nums)
    inf_count = int(inf_mask.sum())
    nums = nums.mask(inf_mask)
    result = ColumnResult(nums, SemanticType.NUMERIC, coerced_to_missing=coerced,
                          inf_converted=inf_count)  # fmt: skip
    if coerced:
        result.warnings.append(f"{WarningCode.COERCED.value}:{name}:{coerced}")
    if inf_count:
        result.warnings.append(f"{WarningCode.INF.value}:{name}:{inf_count}")
    finite = nums.dropna()
    integral = bool(len(finite)) and bool((finite == np.floor(finite)).all())
    distinct = finite.nunique()
    # PI-41: rules 7 and 10 refine columns that rule 4 classified as numeric.
    if (
        integral
        and nonnull_count
        and distinct / nonnull_count >= INT_ID_DISTINCT
        and (
            ID_NAME.search(name) or finite.is_monotonic_increasing or finite.is_monotonic_decreasing
        )
    ):
        return ColumnResult(nums, SemanticType.IDENTIFIER, ColumnRole.EXCLUDED,
                            "identifier_integer_unique", coerced, inf_count,
                            warnings=result.warnings)  # fmt: skip
    if integral and distinct <= POSSIBLY_CATEGORICAL_MAX:
        result.hints.append("possibly_categorical")
    return result


def infer_column(name: str, raw: pd.Series) -> ColumnResult:  # noqa: PLR0911 - one exit per §8.5 rule
    """Apply the §8.5 rules in order (with PI-41 refinements) to one column."""
    if pd.api.types.is_bool_dtype(raw):
        return ColumnResult(raw.astype("string").astype("category"), SemanticType.BOOLEAN)
    if pd.api.types.is_datetime64_any_dtype(raw):
        return ColumnResult(raw, SemanticType.DATETIME)
    typed_numeric = pd.api.types.is_numeric_dtype(raw)
    s = raw if typed_numeric else clean_strings(raw)
    nonnull = s.dropna()
    n = len(nonnull)
    if n == 0:  # rule 1
        return ColumnResult(s, SemanticType.CONSTANT, ColumnRole.EXCLUDED, "all_missing")
    if nonnull.nunique() == 1:  # rule 2
        return ColumnResult(s, SemanticType.CONSTANT, ColumnRole.EXCLUDED, "constant")
    lowered = nonnull.astype("string").str.lower()
    if lowered.nunique() == 2 and set(lowered.unique()) <= BOOL_TOKENS:  # rule 3
        # Original tokens are kept (target labels such as "Yes" matter); preprocessing maps
        # them to 0/1 with TRUE_TOKENS (§15.2 boolean branch).
        return ColumnResult(s, SemanticType.BOOLEAN)
    if typed_numeric:  # rule 4 (Parquet numeric)
        return _numeric(name, s, n)
    if pd.to_numeric(nonnull, errors="coerce").notna().sum() / n >= PARSE_SHARE or (
        _to_numeric(nonnull).notna().sum() / n >= PARSE_SHARE
    ):
        return _numeric(name, s, n)  # rule 4
    parsed, share = _parse_datetime(s)
    if share >= PARSE_SHARE:  # rule 5
        coerced = int(n - parsed.notna().sum())
        result = ColumnResult(parsed, SemanticType.DATETIME, coerced_to_missing=coerced)
        if coerced:
            result.warnings.append(f"{WarningCode.COERCED.value}:{name}:{coerced}")
        return result
    distinct_ratio = nonnull.nunique() / n
    if distinct_ratio >= ID_DISTINCT and (ID_NAME.search(name) or _is_key_like(nonnull)):
        return ColumnResult(
            s,
            SemanticType.IDENTIFIER,
            ColumnRole.EXCLUDED,
            f"identifier_unique_ratio_{distinct_ratio:.2f}",
        )  # rule 6
    if nonnull.str.len().mean() >= TEXT_MEAN_LEN and distinct_ratio >= TEXT_DISTINCT:
        return ColumnResult(s, SemanticType.TEXT, ColumnRole.EXCLUDED, "free_text")  # rule 8
    return ColumnResult(s, SemanticType.CATEGORICAL)  # rule 9


def apply_column_checks(name: str, result: ColumnResult, n_rows: int) -> ColumnResult:
    """§8.6 checks: missing rate, near-constant, cardinality (A-006, PI-42)."""
    values = result.values
    missing = float(values.isna().mean()) if n_rows else 0.0
    if result.role == ColumnRole.FEATURE:
        if missing > EXCLUDE_MISSING:
            result.role, result.exclusion_reason = ColumnRole.EXCLUDED, "missing_rate_gt_95"
        elif missing >= HIGH_MISSING:
            result.warnings.append(f"{WarningCode.HIGH_MISSING.value}:{name}")
    if result.semantic_type in (SemanticType.CONSTANT, SemanticType.TEXT):
        return result
    top_share = values.value_counts(dropna=False, normalize=True).iloc[0] if n_rows else 0.0
    if top_share >= NEAR_CONSTANT:
        result.warnings.append(f"{WarningCode.NEAR_CONSTANT.value}:{name}")
    if result.semantic_type == SemanticType.CATEGORICAL:
        distinct = values.nunique()
        nonnull = int(values.notna().sum())
        if distinct > HIGH_CARDINALITY:
            result.hints.append("high_cardinality")
        if nonnull and distinct / nonnull > VERY_HIGH_CARDINALITY_RATIO:
            result.warnings.append(f"{WarningCode.HIGH_CARDINALITY.value}:{name}")
    return result


def finalize_dtype(result: ColumnResult) -> pd.Series:
    """Memory-lean dtype per semantic type (§8.5 downcasting)."""
    if result.semantic_type == SemanticType.NUMERIC:
        return _downcast_numeric(result.values)
    if result.semantic_type in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN):
        return result.values.astype("category")
    return result.values
