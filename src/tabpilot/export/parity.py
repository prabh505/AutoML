"""Synthetic parity fixture and R1 verification (PRD §22.5, §22.6, §22.8; T-ART-02, T-ART-06).

The fixture holds 50 synthetic rows under the original column names:

- numeric: uniform between the DEV p5 and p95 of the column;
- categorical: categories the fitted encoders already know (they ship inside `model.joblib`, so
  nothing new is exposed); when no encoder keeps categories (CatBoost), neutral placeholders;
- boolean: `true`/`false`; datetime: uniform dates within the DEV range;
- 10% of each column's cells are missing.

A synthetic row whose canonical form equals any DEV/HOLD row is redrawn, so the fixture never
reproduces a real row (T-ART-06). R1: the model is serialized, reloaded with `loader`, and must
reproduce the expected outputs exactly for labels and within 1e-9 for probabilities and values.
A failure re-serializes once; a second failure is E-EXPORT-PARITY (§22.8).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tabpilot.schemas import ErrorCode, ProfileSummary, TabPilotError
from tabpilot_runtime import ColumnNameMapper, TypeCaster
from tabpilot_runtime.bundle import ParityReport, compare, predict_outputs, read_table

PARITY_ROWS = 50
MISSING_SHARE = 0.10
REDRAWS = 100
JOBLIB_PROTOCOL = 5
PLACEHOLDER_CATEGORIES = ("category_a", "category_b", "category_c")
SPECIAL_TOKENS = frozenset({"__rare__", "__missing__", "__other__", "nan", "<NA>"})
DEFAULT_DATE = date(2020, 1, 1)
NUMERIC_DIGITS = 12


def walk(obj: Any) -> Iterable[Any]:
    """Every object in a (fitted) estimator tree, including fitted sub-estimators."""
    stack, seen = [obj], set()
    while stack:
        item = stack.pop()
        if id(item) in seen or item is None or isinstance(item, str | int | float):
            continue
        seen.add(id(item))
        yield item
        if isinstance(item, list | tuple):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif callable(getattr(item, "get_params", None)) and not isinstance(item, type):
            stack.extend(item.get_params(deep=False).values())
            fitted = getattr(item, "__dict__", {})
            stack.extend(fitted.get(a) for a in ("transformers_", "estimators_", "regressor_"))


def column_rules(model: Any) -> tuple[dict[str, str], dict[str, str]]:
    """(original → normalized column map, normalized → kind) merged over every member."""
    column_map: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for item in walk(model):
        if isinstance(item, ColumnNameMapper):
            column_map.update(item.column_map)
        elif isinstance(item, TypeCaster):
            kinds.update(item.kinds)
    return column_map, kinds


def known_categories(model: Any) -> dict[str, list[str]]:
    """Categories stored by fitted encoders, by (normalized) input column."""
    found: dict[str, set[str]] = {}
    for item in walk(model):
        names = getattr(item, "feature_names_in_", None)
        categories = getattr(item, "categories_", None)
        if names is not None and isinstance(categories, list) and len(categories) == len(names):
            for name, cats in zip(names, categories, strict=True):
                found.setdefault(str(name), set()).update(str(c) for c in cats)
        freq = getattr(item, "freq_", None)
        if isinstance(freq, dict):
            for name, table in freq.items():
                if isinstance(table, dict):
                    found.setdefault(str(name), set()).update(str(c) for c in table)
    return {k: sorted(v - SPECIAL_TOKENS) for k, v in found.items() if v - SPECIAL_TOKENS}


def _canonical(values: pd.Series, kind: str) -> pd.Series:
    if kind == "numeric":
        num = pd.to_numeric(values, errors="coerce")
        # 12 significant digits: a copy written by another CSV writer can differ in the last ulp
        return num.map(lambda v: "" if pd.isna(v) else format(float(v), f".{NUMERIC_DIGITS}g"))
    if kind == "datetime":
        when = pd.to_datetime(values, errors="coerce", format="mixed")
        return when.map(lambda v: "" if pd.isna(v) else v.date().isoformat())
    return values.astype("string").fillna("").str.strip().str.lower()


def row_hashes(frame: pd.DataFrame, kinds: dict[str, str]) -> list[str]:
    """SHA-256 of each row's canonical feature values (normalized column names), in row order.
    Columns missing from `frame` are skipped; with none present the list is empty.
    """
    columns = [c for c in kinds if c in frame.columns]
    if not columns:
        return []
    parts = pd.concat([_canonical(frame[c], kinds[c]) for c in columns], axis=1)
    joined = parts.astype(str).agg("\x1f".join, axis=1)
    return [hashlib.sha256(row.encode("utf-8")).hexdigest() for row in joined]


def _numeric(stats: Any, rng: np.random.Generator, n: int) -> list[str]:
    low = getattr(stats, "p5", None) if stats is not None else None
    high = getattr(stats, "p95", None) if stats is not None else None
    if low is None or high is None:
        low = getattr(stats, "min", None) if stats is not None else None
        high = getattr(stats, "max", None) if stats is not None else None
    low, high = (0.0, 1.0) if low is None or high is None else (float(low), float(high))
    return [repr(float(v)) for v in rng.uniform(low, max(high, low), n)]


def _dates(stats: Any, rng: np.random.Generator, n: int) -> list[str]:
    start = DEFAULT_DATE
    span = 365
    if stats is not None and stats.min_date and stats.max_date:
        start = pd.Timestamp(stats.min_date).date()
        span = max(1, (pd.Timestamp(stats.max_date).date() - start).days)
    return [(start + timedelta(days=int(d))).isoformat() for d in rng.integers(0, span + 1, n)]


def _column(kind: str, stats: Any, categories: list[str], rng: np.random.Generator,
            n: int) -> list[str | None]:  # fmt: skip
    values: list[str | None]
    if kind == "numeric":
        values = list(_numeric(stats, rng, n))
    elif kind == "datetime":
        values = list(_dates(stats, rng, n))
    elif kind == "boolean":
        values = [("true", "false")[int(i)] for i in rng.integers(0, 2, n)]
    else:
        pool = categories or list(PLACEHOLDER_CATEGORIES)
        values = [pool[int(i)] for i in rng.integers(0, len(pool), n)]
    for i in np.flatnonzero(rng.random(n) < MISSING_SHARE):
        values[int(i)] = None
    return values


def fixture_frame(
    model: Any,
    profile: ProfileSummary | None,
    *,
    seed: int,
    forbidden: Collection[str] = frozenset(),
    n: int = PARITY_ROWS,
) -> pd.DataFrame:
    """§22.5 synthetic rows under the original column names (text cells, None = missing)."""
    column_map, kinds = column_rules(model)
    categories = known_categories(model)
    stats = {c.name: c for c in profile.columns} if profile is not None else {}
    rng = np.random.default_rng(seed)

    def draw(rows: int) -> pd.DataFrame:
        return pd.DataFrame({orig: _column(kinds.get(norm, "categorical"), stats.get(norm),
                                           categories.get(norm, []), rng, rows)
                             for orig, norm in column_map.items()})  # fmt: skip

    frame = draw(n)
    normalized = dict(column_map)
    for _ in range(REDRAWS):
        hashes = row_hashes(frame.rename(columns=normalized), kinds)
        clash = [i for i, h in enumerate(hashes) if h in forbidden]
        if not clash:
            return frame
        frame.iloc[clash] = draw(len(clash)).to_numpy()
    raise RuntimeError("could not draw synthetic parity rows that differ from every real row")


def write_fixture(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Write the fixture CSV and return it exactly as `read_table` reads it back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return read_table(path)


@dataclass(frozen=True)
class R1Result:
    """R1 outcome; `rebuilt` is True when the first serialization failed parity."""

    report: ParityReport
    rebuilt: bool


Loader = Callable[[Path], Any]


def serialize_and_verify(
    model: Any,
    model_path: Path,
    frame: pd.DataFrame,
    expected: dict[str, Any],
    *,
    run_id: str,
    loader: Loader = joblib.load,
) -> R1Result:
    """Dump with joblib (protocol 5), reload with `loader`, compare; rebuild once (§22.8)."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    task, classes = expected["task"], list(expected["classes"])
    report = None
    for attempt in range(2):
        joblib.dump(model, model_path, protocol=JOBLIB_PROTOCOL)
        report = compare(expected, predict_outputs(loader(model_path), frame, task, classes))
        if report.ok:
            return R1Result(report, rebuilt=attempt == 1)
    model_path.unlink(missing_ok=True)
    raise TabPilotError(ErrorCode.EXPORT_PARITY, run_id=run_id)
