"""`profile()`: the DEV-only profiling engine (PRD §9, FR-PRF-01/02; A-002, A-004).

The signature takes the full validated frame **and** the `SplitSpec`, and selects DEV rows
internally via `split.dev_indices`. Callers cannot hand it holdout rows by mistake (P06), and
it never reads `holdout_indices` (AST-enforced).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tabpilot.logging import get_logger
from tabpilot.profiling import leakage, stats
from tabpilot.profiling.budget import ProfileBudget
from tabpilot.schemas import (
    ColumnRole,
    DatasetMetadata,
    ExecutionProfileName,
    ProfileSummary,
    SemanticType,
    SplitSpec,
    Task,
    WarningCode,
)
from tabpilot.schemas.dataset import ColumnInfo
from tabpilot.schemas.profile import TargetProfile

log = get_logger("profile")
SMALL_DATA_ROWS = 500
IMBALANCE_SHARE = 0.05
HIGH_MISSING = (0.30, 0.95)
WIDE_RATIO = 5
STAT_GROUPS = ("numeric", "categorical", "datetime", "association", "correlation", "leakage")


def _target_profile(y: pd.Series, task: Task, positive_label: str | None) -> TargetProfile:
    if task == Task.REGRESSION:
        x = y.astype("float64")
        return TargetProfile(mean=float(x.mean()), std=float(x.std()),
                             skew=stats.scalar(x.skew()), min=float(x.min()))  # fmt: skip
    counts = y.astype("string").value_counts()
    shares = counts / counts.sum()
    positive = float(shares.get(positive_label, 0.0)) if positive_label is not None else None
    return TargetProfile(class_counts={str(k): int(v) for k, v in counts.items()},
                         imbalance_ratio=float(counts.max() / counts.min()),
                         positive_share=positive)  # fmt: skip


def _column_stats(
    dev: pd.DataFrame, features: list[ColumnInfo], budget: ProfileBudget, rng: np.random.Generator
) -> dict[str, dict[str, Any]]:
    """Numeric, categorical, and datetime statistics under per-group caps (§9.2, §9.5)."""
    extras: dict[str, dict[str, Any]] = {c.name: {} for c in features}
    groups: tuple[tuple[str, set[SemanticType], Callable[[ColumnInfo], dict[str, Any]]], ...] = (
        ("numeric", {SemanticType.NUMERIC}, lambda c: stats.numeric_stats(c, dev[c.name], rng)),
        ("categorical", {SemanticType.CATEGORICAL, SemanticType.BOOLEAN},
         lambda c: stats.categorical_stats(dev[c.name])),
        ("datetime", {SemanticType.DATETIME}, lambda c: stats.datetime_stats(dev[c.name])),
    )  # fmt: skip
    for group, types, fn in groups:
        timer = budget.group(group)
        for col in (c for c in features if c.semantic_type in types):
            if timer.expired():
                break
            extras[col.name].update(fn(col))
    return extras


def _warnings(
    dev: pd.DataFrame,
    features: list[ColumnInfo],
    summary_parts: dict[str, Any],
) -> list[str]:
    """Dataset suitability warnings (§9.3)."""
    warnings: list[str] = []
    if len(dev) < SMALL_DATA_ROWS:
        warnings.append(WarningCode.SMALL_DATA.value)
    class_counts = summary_parts["target"].class_counts
    if class_counts:
        counts = np.array(list(class_counts.values()))
        if counts.min() / counts.sum() < IMBALANCE_SHARE:
            warnings.append(WarningCode.IMBALANCE.value)
    for col in features:
        missing = float(dev[col.name].isna().mean())
        if HIGH_MISSING[0] <= missing <= HIGH_MISSING[1]:
            warnings.append(f"{WarningCode.HIGH_MISSING.value}:{col.name}")
    for f in summary_parts["leakage"]:
        near = f.level == "near_deterministic"
        code = WarningCode.LEAK_NEAR_DETERMINISTIC if near else WarningCode.LEAK_STRONG
        warnings.append(f"{code.value}:{f.column}")
    warnings += [f"{WarningCode.GROUP_SUSPECTED.value}:{g}" for g in summary_parts["groups"]]
    warnings += [f"{WarningCode.TEMPORAL.value}:{t}" for t in summary_parts["temporal"]]
    if summary_parts["duplicates"]:
        warnings.append(f"{WarningCode.DUPLICATES.value}:{summary_parts['duplicates']}")
    if summary_parts["conflicting"] > 0:
        warnings.append(
            f"{WarningCode.CONFLICTING_DUPLICATES.value}:{summary_parts['conflicting']}"
        )
    if len(features) > len(dev) / WIDE_RATIO:
        warnings.append(WarningCode.WIDE.value)
    return warnings


def profile(
    frame: pd.DataFrame,
    split: SplitSpec,
    metadata: DatasetMetadata,
    task: Task,
    *,
    execution_profile: ExecutionProfileName,
    time_budget_s: float,
    seed_root: int = 42,
    positive_label: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ProfileSummary:
    """Profile DEV rows (§9.2), detect leakage/structure (§11.7, §9.3), and emit warnings."""
    if metadata.target is None:
        raise ValueError("profile() needs metadata with a validated target (apply_target)")
    target = metadata.target.name
    dev = frame.iloc[list(split.dev_indices)]
    seed = seed_root + 3  # seed_profile (§11.3)
    rng = np.random.default_rng(seed)
    budget = ProfileBudget.for_run(execution_profile, time_budget_s, len(STAT_GROUPS), clock=clock)
    features = [c for c in metadata.columns if c.role == ColumnRole.FEATURE and c.name != target]
    y = dev[target]
    extras = _column_stats(dev, features, budget, rng)

    assoc_rows = dev
    if len(dev) > stats.ASSOCIATION_SAMPLE:
        assoc_rows = dev.sample(stats.ASSOCIATION_SAMPLE, random_state=int(rng.integers(2**31)))
    timer = budget.group("association")
    for col in features:
        if timer.expired():
            break
        extras[col.name]["association"] = stats.association(
            col, assoc_rows[col.name], assoc_rows[target], task, seed
        )
    numeric = [c.name for c in features if c.semantic_type == SemanticType.NUMERIC]
    correlation_timer = budget.group("correlation")
    pairs = [] if correlation_timer.expired() else stats.correlated_pairs(dev, numeric, rng)
    leak_timer = budget.group("leakage", budget.leakage_cap_s)
    findings, scores = leakage.scan(dev, target, metadata, task, seed, leak_timer)
    for name, value in scores.items():
        extras[name]["leakage_score"] = round(value, 6)

    others = [c for c in metadata.columns if c.name != target]
    duplicates = int(dev.duplicated().sum())
    parts: dict[str, Any] = {
        "target": _target_profile(y, task, positive_label), "leakage": findings,
        "groups": stats.group_candidates(dev, others),
        "temporal": stats.temporal_columns(dev, others, y, task), "duplicates": duplicates,
        "conflicting": int(dev.drop(columns=[target]).duplicated().sum()) - duplicates,
    }  # fmt: skip
    warnings = _warnings(dev, features, parts)
    summary = ProfileSummary(
        task=task, n_rows_dev=len(dev), n_features=len(features),
        memory_mb=round(float(dev.memory_usage(deep=True).sum()) / 1e6, 3),
        duplicate_rows=duplicates, conflicting_duplicate_rows=max(parts["conflicting"], 0),
        columns=[stats.column_profile(c, dev[c.name], extras[c.name]) for c in features],
        target=parts["target"], correlated_pairs=pairs, leakage=findings,
        group_candidates=parts["groups"], temporal_columns=parts["temporal"],
        warnings=warnings, truncated_stats=list(budget.truncated),
    )  # fmt: skip
    log.info("profile_completed", rows=len(dev), features=len(features), warnings=len(warnings),
             truncated=len(budget.truncated))  # fmt: skip
    return summary
