"""Target validation and the only automatic row removal (PRD §8.7, FR-TGT-01; A-008; PI-47).

`validate_target` is pure: it checks the target and reports what would be removed.
`apply_target` removes missing-target rows (and, with explicit consent, rows of classes with
fewer than 2 rows), and records every removal in the metadata.

Class checks need the task (PI-47). With `task=None` only task-independent rules run (used
before task detection). Task detection (P05) re-runs validation with the confirmed task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from tabpilot.ingestion.loader import LoadedDataset
from tabpilot.schemas import (
    ColumnRole,
    ErrorCode,
    SemanticType,
    TabPilotError,
    TargetReport,
    Task,
    WarningCode,
)
from tabpilot.schemas.dataset import NumericSummary, RowRemoval, TargetInfo

MIN_ROWS = 50
MIN_CLASS_ROWS = 2
MAX_CLASSES = 50
CONFIRM_MISSING_SHARE = 0.5
UNSUPPORTED = {SemanticType.TEXT, SemanticType.IDENTIFIER, SemanticType.DATETIME}


@dataclass(frozen=True)
class TargetCheck:
    """A validated target: the report plus the row mask that would be kept."""

    report: TargetReport
    keep: np.ndarray
    task: Task | None


def _labels(values: pd.Series) -> pd.Series:
    return values.astype("string")


def validate_target(
    loaded: LoadedDataset, target: str, task: Task | None = None, drop_tiny_classes: bool = False
) -> TargetCheck:
    """Check `target` (normalized name) under §8.7; raise the matching E-TARGET-* code."""
    column = next((c for c in loaded.metadata.columns if c.name == target), None)
    if column is None:
        raise KeyError(f"unknown target column {target!r}")
    values = loaded.frame[target]
    missing = values.isna().to_numpy()
    n_missing = int(missing.sum())
    if n_missing == len(values):
        raise TabPilotError(ErrorCode.TARGET_ALL_MISSING, target=column.original)
    present = values[~missing]
    n_unique = int(present.nunique())
    if n_unique <= 1:
        raise TabPilotError(ErrorCode.TARGET_CONSTANT, target=column.original)
    if column.semantic_type in UNSUPPORTED or (
        task == Task.REGRESSION and column.semantic_type != SemanticType.NUMERIC
    ):
        raise TabPilotError(ErrorCode.TARGET_UNSUPPORTED_TYPE, target=column.original,
                            semantic_type=column.semantic_type.value)  # fmt: skip

    keep = ~missing
    warnings: list[str] = []
    if n_missing:
        warnings.append(f"{WarningCode.TARGET_MISSING.value}:{target}:{n_missing}")
    class_counts: dict[str, int] | None = None
    distribution: NumericSummary | None = None
    if task in (Task.BINARY, Task.MULTICLASS) or (
        task is None and column.semantic_type != SemanticType.NUMERIC
    ):
        counts = _labels(present).value_counts()
        if task is not None and len(counts) > MAX_CLASSES:
            raise TabPilotError(ErrorCode.TARGET_TOO_MANY_CLASSES, target=column.original,
                                count=len(counts), limit=MAX_CLASSES)  # fmt: skip
        tiny = counts[counts < MIN_CLASS_ROWS]
        if task is not None and len(tiny):
            if not drop_tiny_classes:
                raise TabPilotError(ErrorCode.TARGET_CLASS_TOO_SMALL, target=column.original,
                                    n_classes=len(tiny), minimum=MIN_CLASS_ROWS)  # fmt: skip
            keep &= ~_labels(values).isin(tiny.index).fillna(False).to_numpy()
            counts = counts[counts >= MIN_CLASS_ROWS]
        class_counts = {str(k): int(v) for k, v in counts.items()}
    else:
        nums = present.astype("float64")
        arr = nums.to_numpy(dtype="float64")
        distribution = NumericSummary(mean=float(arr.mean()), std=float(arr.std()),
                                      min=float(arr.min()), max=float(arr.max()),
                                      skew=float(cast(float, nums.skew())))  # fmt: skip
    n_kept = int(keep.sum())
    if n_kept < MIN_ROWS:
        raise TabPilotError(ErrorCode.TARGET_TOO_FEW_ROWS, target=column.original, rows=n_kept,
                            minimum=MIN_ROWS)  # fmt: skip
    report = TargetReport(
        target=target, n_missing=n_missing, n_rows_removed=len(values) - n_kept,
        n_unique=n_unique, class_counts=class_counts, distribution=distribution,
        warnings=warnings, requires_confirmation=n_missing > CONFIRM_MISSING_SHARE * len(values),
    )  # fmt: skip
    return TargetCheck(report=report, keep=keep, task=task)


def _conflicting_duplicates(frame: pd.DataFrame, target: str) -> int:
    """Rows whose features duplicate an earlier row but whose target differs (§8.6)."""
    features = frame.drop(columns=[target])
    feature_dupes = int(features.duplicated().sum())
    full_dupes = int(frame.duplicated().sum())
    return feature_dupes - full_dupes


def apply_target(loaded: LoadedDataset, check: TargetCheck) -> LoadedDataset:
    """Remove rows per `check` and record target info and removals in the metadata."""
    target = check.report.target
    frame = loaded.frame.loc[check.keep].reset_index(drop=True)
    meta = loaded.metadata
    removals = [RowRemoval(reason="target_missing", count=check.report.n_missing)]
    tiny_removed = check.report.n_rows_removed - check.report.n_missing
    if tiny_removed:
        removals.append(RowRemoval(reason="class_too_small_with_consent", count=tiny_removed))
    columns = [
        c.model_copy(update={"role": ColumnRole.TARGET, "exclusion_reason": None})
        if c.name == target
        else c
        for c in meta.columns
    ]
    warnings = [
        w for w in meta.warnings if not w.startswith(WarningCode.CONFLICTING_DUPLICATES.value)
    ]
    warnings += check.report.warnings
    conflicting = _conflicting_duplicates(frame, target)
    if conflicting:
        warnings.append(f"{WarningCode.CONFLICTING_DUPLICATES.value}:{conflicting}")
    metadata = meta.model_copy(update={
        "n_rows_valid": len(frame), "row_removals": removals, "columns": columns,
        "target": TargetInfo(name=target, task=check.task, class_counts=check.report.class_counts),
        "warnings": warnings,
    })  # fmt: skip
    return LoadedDataset(frame=frame, metadata=type(meta).model_validate(metadata.model_dump()),
                         parse_report=loaded.parse_report)  # fmt: skip
