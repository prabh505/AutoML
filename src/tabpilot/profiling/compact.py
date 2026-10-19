"""LLM-safe compact profile (PRD §9.4, §26.2, FR-PRF-03; A-005; I-6).

Content rule: only normalized column names, types, aggregate numbers, and flags. No category
labels, no example rows, no values of identifier columns. Numeric order statistics of feature
columns are allowed aggregates in standard mode (A-005). Minimal mode keeps only skew and a range
class (as a `range:*` flag).

Size rule: ≤ `max_chars` of canonical JSON (default 14,000). The §9.4 truncation algorithm
downgrades the lowest-ranked entries to `short`, then moves them into `rest_summary`.
Ordering is deterministic (association strength desc, then name).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from tabpilot.profiling.stats import association_strength
from tabpilot.schemas import (
    ColumnRole,
    CompactProfile,
    DatasetMetadata,
    ProfileSummary,
    SemanticType,
    Task,
)
from tabpilot.schemas.profile import ColumnProfile, CompactColumn, CompactTarget, RestSummary

MAX_CHARS = 14_000
DETAIL_COLUMNS = 150
REST_NAMES_LIMIT = 50
SIGNIFICANT = 6


def canonical_json(compact: CompactProfile) -> str:
    """The exact text sent to the LLM and shown in the inspector (FR-PRF-03)."""
    payload = compact.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _round(x: float | None) -> float | None:
    return None if x is None else float(f"{x:.{SIGNIFICANT}g}")


def _range_flag(col: ColumnProfile) -> str | None:
    if col.min is None or col.max is None:
        return None
    if col.min >= 0 and col.max <= 1:
        return "range:unit_interval"
    return "range:nonnegative" if col.min >= 0 else "range:signed"


def _flags(col: ColumnProfile, hints: list[str], leak: str | None, target: str) -> list[str]:
    flags = [h for h in hints if h in ("possibly_categorical", "high_cardinality")]
    if leak:
        flags.append(f"leak_{leak}")
    if target and target.rstrip("_") in col.name:
        flags.append("name_contains_target")
    range_flag = _range_flag(col)
    if range_flag:
        flags.append(range_flag)
    return sorted(flags)


def _entry(
    col: ColumnProfile, level: Literal["full", "short"], mode: str, flags: list[str]
) -> CompactColumn:
    base: dict[str, Any] = {
        "name": col.name, "type": col.type, "level": level,
        "missing": _round(col.missing_rate), "assoc": col.association,
    }  # fmt: skip
    if level == "short":
        return CompactColumn(**base, flags=flags)
    if col.type == SemanticType.NUMERIC:
        if mode == "standard" and col.min is not None:
            quantiles = (col.min, col.p5, col.p25, col.p50, col.p75, col.p95, col.max)
            base["q"] = [_round(v) for v in quantiles]
        base["skew"] = _round(col.skew)
        base["distinct"] = col.distinct
    elif col.type in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN):
        base["distinct"] = col.distinct
        base["entropy_bits"] = _round(col.entropy_bits)
        base["top_shares"] = [_round(s) for s in (col.top_shares or [])] or None
    else:  # datetime: span only; dates are cell values
        base["distinct"] = col.distinct
    return CompactColumn(**base, flags=flags)


def _target(profile: ProfileSummary) -> CompactTarget:
    t = profile.target
    if profile.task == Task.REGRESSION:
        return CompactTarget(skew=_round(t.skew), nonnegative=None if t.min is None else t.min >= 0)
    n_classes = len(t.class_counts or {})
    share = _round(t.positive_share) if profile.task == Task.BINARY else None
    return CompactTarget(positive_share=share, imbalance_ratio=_round(t.imbalance_ratio),
                         n_classes=n_classes)  # fmt: skip


def _rest(columns: list[ColumnProfile]) -> RestSummary | None:
    if not columns:
        return None
    counts: dict[SemanticType, int] = {}
    for c in columns:
        counts[c.type] = counts.get(c.type, 0) + 1
    return RestSummary(
        n_columns=len(columns), counts_by_type=counts,
        mean_missing=_round(sum(c.missing_rate for c in columns) / len(columns)) or 0.0,
        names=sorted(c.name for c in columns) if len(columns) <= REST_NAMES_LIMIT else None,
    )  # fmt: skip


def build_compact_profile(
    profile: ProfileSummary,
    metadata: DatasetMetadata,
    mode: Literal["standard", "minimal"] = "standard",
    max_chars: int = MAX_CHARS,
) -> CompactProfile:
    """Build the bounded, label-free planner view of the DEV profile (§9.4)."""
    hints = {c.name: c.hints for c in metadata.columns}
    excluded = {c.name for c in metadata.columns if c.role == ColumnRole.EXCLUDED}
    leak = {f.column: f.level for f in profile.leakage}
    target = metadata.target.name if metadata.target else ""
    ranked = sorted((c for c in profile.columns if c.name not in excluded),
                    key=lambda c: (-association_strength(c.association), c.name))  # fmt: skip
    detail, rest = ranked[:DETAIL_COLUMNS], ranked[DETAIL_COLUMNS:]
    levels: list[Literal["full", "short"]] = ["full"] * len(detail)
    codes = sorted({w.split(":", 1)[0] for w in profile.warnings})

    def build() -> CompactProfile:
        entries = [
            _entry(c, levels[i], mode, _flags(c, hints.get(c.name, []), leak.get(c.name), target))
            for i, c in enumerate(detail)
        ]
        return CompactProfile(
            task=profile.task, n_rows_dev=profile.n_rows_dev, n_features=len(ranked),
            target=_target(profile), columns=entries, rest_summary=_rest(rest),
            warnings=codes, truncated=bool(rest) or "short" in levels, mode=mode,
        )  # fmt: skip

    compact = build()
    while len(canonical_json(compact)) > max_chars:
        if "full" in levels:
            levels[len(levels) - 1 - levels[::-1].index("full")] = "short"
        elif detail:
            rest.insert(0, detail.pop())
            levels.pop()
        else:
            break
        compact = build()
    return compact
