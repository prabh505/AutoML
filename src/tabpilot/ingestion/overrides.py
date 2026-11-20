"""Validation of user column-type overrides (FR-ING-04 Val/Err).

"Numeric → categorical always allowed; text → numeric only if ≥ 95% parse". The same 95% rule is
applied to every type that needs parsing (numeric, datetime); boolean needs at most two distinct
values that are boolean tokens or numbers. A rejected override returns an inline message for the
UI; the message names the column and the parse share, never a cell value (I-11).
"""

from __future__ import annotations

import warnings

import pandas as pd

from tabpilot.ingestion.inference import BOOL_TOKENS
from tabpilot.schemas import SemanticType

MIN_PARSE_SHARE = 0.95
OVERRIDE_TYPES = (SemanticType.NUMERIC, SemanticType.CATEGORICAL, SemanticType.BOOLEAN,
                  SemanticType.DATETIME)  # fmt: skip


def _share(parsed: pd.Series, present: pd.Series) -> float:
    return float(parsed.notna().sum()) / len(present) if len(present) else 0.0


def check_override(column: str, values: pd.Series, new_type: SemanticType) -> str | None:
    """Return why `values` cannot be read as `new_type`, or None when the override is valid."""
    if new_type not in OVERRIDE_TYPES:
        return f"Column '{column}' cannot be set to {new_type.value}; choose one of " + ", ".join(
            t.value for t in OVERRIDE_TYPES
        )
    present = values.dropna()
    if new_type == SemanticType.CATEGORICAL:
        return None
    if new_type == SemanticType.NUMERIC:
        share = _share(pd.to_numeric(present.astype("string").str.strip(), errors="coerce"),
                       present)  # fmt: skip
    elif new_type == SemanticType.DATETIME:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(present.astype("string"), errors="coerce", format="mixed")
        share = _share(parsed, present)
    else:
        lowered = present.astype("string").str.strip().str.lower()
        numeric = pd.to_numeric(lowered, errors="coerce")
        tokens = lowered.isin(BOOL_TOKENS) | numeric.notna()
        if lowered.nunique() > 2 or not bool(tokens.all()):
            return (f"Column '{column}' cannot be boolean: it needs at most two distinct values "
                    "such as yes/no, true/false or 1/0.")  # fmt: skip
        return None
    if share < MIN_PARSE_SHARE:
        return (f"Only {share:.0%} of the values in column '{column}' can be read as "
                f"{new_type.value}; at least {MIN_PARSE_SHARE:.0%} must parse.")  # fmt: skip
    return None
