"""Split protocol: holdout first, then one fixed fold assignment (PRD §11.1–§11.6; A-003).

Row indices are positions in the validated frame (missing-target rows already removed),
0..n_valid-1. `split_id` is the SHA-256 of the canonical split, so identical inputs and seed give
an identical `split_id` in any process.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from tabpilot.schemas import ErrorCode, SplitSpec, TabPilotError, Task, WarningCode

HOLDOUT_MIN_ROWS = 2000  # D-06 default; compared with n_valid (A-003)
HOLDOUT_FRACTION = 0.2
DEFAULT_K = 5
REPEATS = 3
QUANTILE_BINS = 10
DUPLICATE_GROUPING_SHARE = 0.01
GROUP_HOLDOUT_ATTEMPTS = 20
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SplitResult:
    """The split plus warnings to show the user (e.g. W-NO-HOLDOUT)."""

    spec: SplitSpec
    warnings: list[str] = field(default_factory=list)


def _is_classification(task: Task) -> bool:
    return task in (Task.BINARY, Task.MULTICLASS)


def duplicate_groups(features: pd.DataFrame) -> IntArray | None:
    """Synthetic group ids for exact duplicate rows when they exceed 1% (§11.6), else None."""
    hashes = pd.util.hash_pandas_object(features, index=False).to_numpy()
    _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    n_duplicates = len(hashes) - len(first)
    if n_duplicates <= DUPLICATE_GROUPING_SHARE * len(hashes):
        return None
    return inverse.astype(np.int64)


def _quantile_bins(y: NDArray[np.float64]) -> IntArray:
    """Regression stratification labels: 10 quantile bins, merging duplicate edges (§11.3)."""
    for bins in range(QUANTILE_BINS, 1, -1):
        labels = pd.qcut(y, q=bins, labels=False, duplicates="drop")
        counts = np.bincount(labels)
        if counts.min() >= 2:
            return np.asarray(labels, dtype=np.int64)
    return np.zeros(len(y), dtype=np.int64)


def _holdout(
    y: NDArray[np.generic], task: Task, groups: IntArray | None, seed: int
) -> tuple[IntArray, IntArray, list[str]]:
    """Return (dev, hold) row positions for a 20% holdout (§11.1, §11.3)."""
    rows = np.arange(len(y))
    warnings: list[str] = []
    if groups is None:
        strata = y if _is_classification(task) else _quantile_bins(y.astype(np.float64))
        dev, hold = train_test_split(rows, test_size=HOLDOUT_FRACTION, random_state=seed,
                                     stratify=strata)  # fmt: skip
        return np.sort(dev), np.sort(hold), warnings
    for attempt in range(GROUP_HOLDOUT_ATTEMPTS):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=HOLDOUT_FRACTION, random_state=seed + attempt
        )
        dev, hold = next(splitter.split(rows, groups=groups))
        if not _is_classification(task) or (
            set(np.unique(y[dev])) == set(np.unique(y)) == set(np.unique(y[hold]))
        ):
            return np.sort(dev), np.sort(hold), warnings
    # No group shuffle kept every class on both sides: use one stratified group fold (~20%).
    sgkf = StratifiedGroupKFold(
        n_splits=round(1 / HOLDOUT_FRACTION), shuffle=True, random_state=seed
    )
    dev, hold = next(sgkf.split(rows, y, groups=groups))
    warnings.append(WarningCode.GROUP_HOLDOUT_FALLBACK.value)
    return np.sort(dev), np.sort(hold), warnings


def _k_for(
    y_dev: NDArray[np.generic], task: Task, groups: IntArray | None, k: int, target: str
) -> int:
    """K reduction on DEV counts (§11.3, PI-29)."""
    if _is_classification(task):
        smallest = int(pd.Series(y_dev).value_counts().min())
        if smallest < 2:
            n_tiny = int((pd.Series(y_dev).value_counts() < 2).sum())
            raise TabPilotError(ErrorCode.TARGET_CLASS_TOO_SMALL, target=target, n_classes=n_tiny,
                                minimum=2)  # fmt: skip
        k = min(k, max(2, smallest))
    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups < 2 * k:
            k = max(2, min(k, n_groups // 2))
    return k


def _assign(n: int, splits: Iterator[tuple[IntArray, IntArray]]) -> list[int]:
    fold_of = np.full(n, -1, dtype=np.int64)
    for fold, (_, val) in enumerate(splits):
        fold_of[val] = fold
    if (fold_of < 0).any():
        raise RuntimeError("splitter left rows without a fold")
    return [int(v) for v in fold_of]


def _split_id(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dev_folds(
    y_dev: NDArray[np.generic],
    task: Task,
    *,
    g_dev: IntArray | None,
    k: int,
    seed: int,
    small: bool,
) -> tuple[str, list[list[int]], list[str]]:
    """Fold strategy and assignment on DEV (§11.3)."""
    n = len(y_dev)
    pos = np.arange(n)
    clf = _is_classification(task)
    if g_dev is not None:
        notes = [WarningCode.NO_REPEATED_CV.value] if small else []
        if clf:
            sgkf = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
            return "stratified_group_kfold", [_assign(n, sgkf.split(pos, y_dev, g_dev))], notes
        gkf = GroupKFold(k, shuffle=True, random_state=seed)
        return "group_kfold", [_assign(n, gkf.split(pos, y_dev, g_dev))], notes
    if small:
        cls = RepeatedStratifiedKFold if clf else RepeatedKFold
        splits = list(cls(n_splits=k, n_repeats=REPEATS, random_state=seed).split(pos, y_dev))
        assignments = [_assign(n, iter(splits[r * k : (r + 1) * k])) for r in range(REPEATS)]
        name = "repeated_stratified_kfold" if clf else "repeated_kfold"
        return name, assignments, []
    if clf:
        skf = StratifiedKFold(k, shuffle=True, random_state=seed)
        return "stratified_kfold", [_assign(n, skf.split(pos, y_dev))], []
    return "kfold", [_assign(n, KFold(k, shuffle=True, random_state=seed).split(pos))], []


def build_split(
    y: pd.Series,
    task: Task,
    seed_root: int = 42,
    *,
    group: pd.Series | None = None,
    features: pd.DataFrame | None = None,
    k: int = DEFAULT_K,
) -> SplitResult:
    """Build the run's holdout and fold assignment (FR-SPL-01, §11.1–§11.6).

    `group` is the user-declared group column; without one, exact duplicate rows (> 1%) of
    `features` are grouped (§11.6). `k` may be lowered by the caller (e.g. to 3 when the hosted
    estimator predicts an overrun, §11.3).
    """
    seed = seed_root  # seed_split = seed_root (§11.3)
    values = y.to_numpy()
    labels = pd.factorize(values)[0].astype(np.int64) if _is_classification(task) else values
    n_valid = len(values)
    warnings: list[str] = []
    groups: IntArray | None = None
    if group is not None:
        groups = pd.factorize(group.to_numpy())[0].astype(np.int64)
    elif features is not None:
        groups = duplicate_groups(features)
    duplicate_grouping = group is None and groups is not None

    if n_valid >= HOLDOUT_MIN_ROWS:
        dev, hold, extra = _holdout(labels, task, groups, seed)
        warnings += extra
    else:
        dev, hold = np.arange(n_valid), np.array([], dtype=np.int64)
        warnings.append(WarningCode.NO_HOLDOUT.value)
    g_dev = groups[dev] if groups is not None else None
    k = _k_for(labels[dev], task, g_dev, k, str(y.name))
    strategy, assignments, extra = _dev_folds(
        labels[dev], task, g_dev=g_dev, k=k, seed=seed, small=n_valid < HOLDOUT_MIN_ROWS
    )
    warnings += extra
    hold_list, dev_list = [int(i) for i in hold], [int(i) for i in dev]
    group_column = None if group is None else str(group.name)
    payload: dict[str, object] = {
        "strategy": strategy, "k": k, "n_repeats": len(assignments), "seed": seed,
        "n_valid": n_valid, "holdout_indices": hold_list, "dev_indices": dev_list,
        "fold_assignment": assignments, "group_column": group_column,
        "duplicate_grouping": duplicate_grouping,
    }  # fmt: skip
    spec = SplitSpec(
        split_id=_split_id(payload), strategy=strategy, k=k, n_repeats=len(assignments),
        seed=seed, n_valid=n_valid, holdout_present=bool(hold_list),
        holdout_fraction=HOLDOUT_FRACTION if hold_list else 0.0, holdout_indices=hold_list,
        dev_indices=dev_list, fold_assignment=assignments, group_column=group_column,
        duplicate_grouping=duplicate_grouping,
    )  # fmt: skip
    return SplitResult(spec=spec, warnings=warnings)


def iter_folds(spec: SplitSpec) -> Iterator[tuple[int, int, IntArray, IntArray]]:
    """Yield (repeat, fold, train_rows, valid_rows) as row indices into the validated frame."""
    dev = np.asarray(spec.dev_indices, dtype=np.int64)
    for repeat, assignment in enumerate(spec.fold_assignment):
        folds = np.asarray(assignment, dtype=np.int64)
        for fold in range(spec.k):
            yield repeat, fold, dev[folds != fold], dev[folds == fold]


def average_repeats(oof_per_repeat: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    """OOF for repeated CV: per-row mean across repeats (§11.4)."""
    if not oof_per_repeat:
        raise ValueError("no OOF predictions")
    stacked = np.stack(oof_per_repeat)
    return np.asarray(stacked.mean(axis=0), dtype=np.float32)
