"""P05 split and guard tests: T-LEAK-01, T-LEAK-03 (guard level), T-LEAK-04, §11.3 branches."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.helpers.errors import raises_error

from tabpilot.config import load_settings
from tabpilot.ingestion import apply_target, load_dataset, validate_target
from tabpilot.schemas import ErrorCode, SplitSpec, Task
from tabpilot.splitting import average_repeats, build_split, iter_folds
from tabpilot.splitting.guard import HoldoutAlreadyOpened, HoldoutGuard, HoldoutSealed
from tabpilot.splitting.strategy import duplicate_groups

RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
ROOT = Path(__file__).resolve().parents[2]


def fixture(name: str, target: str, task: Task) -> pd.DataFrame:
    s = load_settings({})
    ld = load_dataset(ROOT / "fixtures" / "datasets" / name, name, s, RID)
    return apply_target(ld, validate_target(ld, target, task)).frame


def check_partition(spec: SplitSpec) -> None:
    rows = set(spec.dev_indices) | set(spec.holdout_indices)
    assert rows == set(range(spec.n_valid))
    for _, _, train, valid in iter_folds(spec):
        assert set(train).isdisjoint(valid)
        assert set(train) | set(valid) == set(spec.dev_indices)
        assert set(valid).isdisjoint(spec.holdout_indices)


# ---------------------------------------------------------------- strategy branches (§11.3)
@pytest.mark.unit
@pytest.mark.parametrize(("name", "target", "task", "strategy", "k", "repeats", "holdout"), [
    ("binary_basic.csv", "churn", Task.BINARY, "stratified_kfold", 5, 1, 500),
    ("regression_basic.csv", "price", Task.REGRESSION, "kfold", 5, 1, 600),
    ("binary_small.csv", "target_", Task.BINARY, "repeated_stratified_kfold", 5, 3, 0),
    ("multiclass_rare.csv", "color", Task.MULTICLASS, "repeated_stratified_kfold", 3, 3, 0),
])  # fmt: skip
def test_strategy_branches(  # noqa: PLR0917 - parametrized case fields
    name: str, target: str, task: Task, strategy: str, k: int, repeats: int, holdout: int
) -> None:
    frame = fixture(name, target, task)
    result = build_split(frame[target], task)
    spec = result.spec
    assert (spec.strategy, spec.k, spec.n_repeats, len(spec.holdout_indices)) == (
        strategy, k, repeats, holdout)  # fmt: skip
    assert ("W-NO-HOLDOUT" in result.warnings) == (holdout == 0)
    check_partition(spec)


@pytest.mark.unit
def test_small_regression_uses_repeated_kfold() -> None:
    y = pd.Series(np.random.default_rng(0).normal(size=300), name="y")
    spec = build_split(y, Task.REGRESSION).spec
    assert (spec.strategy, spec.n_repeats, spec.holdout_present) == ("repeated_kfold", 3, False)
    assert spec.fold_assignment[0] != spec.fold_assignment[1]


@pytest.mark.unit
def test_holdout_is_stratified() -> None:
    frame = fixture("binary_imbalanced.csv", "is_fraud", Task.BINARY)
    spec = build_split(frame["is_fraud"], Task.BINARY).spec
    y = frame["is_fraud"].astype(str).to_numpy()
    assert (y[spec.holdout_indices] == "1").sum() == 30  # exactly 3% of 1,000
    for repeat, fold, _, valid in iter_folds(spec):
        assert (y[valid] == "1").sum() in (24, 24)  # 120 DEV positives / 5 folds
        assert repeat == 0
        assert 0 <= fold < 5


@pytest.mark.unit
def test_regression_holdout_stratified_on_quantile_bins() -> None:
    y = pd.Series(np.r_[np.zeros(1000), np.linspace(1, 100, 1500)], name="y")  # duplicate edges
    spec = build_split(y, Task.REGRESSION).spec
    hold = y.to_numpy()[spec.holdout_indices]
    assert len(hold) == 500
    assert 150 <= (hold == 0).sum() <= 250  # ≈ 40% zeros, like the full data


@pytest.mark.unit
def test_threshold_uses_valid_rows_A003() -> None:
    y = pd.Series(["a", "b"] * 1100, name="y")  # n_valid 2,200 → holdout; DEV 1,760
    spec = build_split(y, Task.BINARY).spec
    assert spec.holdout_present
    assert spec.n_repeats == 1
    assert len(spec.dev_indices) == 1760


@pytest.mark.unit
def test_k_reduction_and_too_small_class() -> None:
    y = pd.Series(["a"] * 150 + ["b"] * 147 + ["r"] * 3, name="color")
    assert build_split(y, Task.MULTICLASS).spec.k == 3
    one = pd.Series(["a"] * 150 + ["b"] * 149 + ["r"], name="color")
    with raises_error(ErrorCode.TARGET_CLASS_TOO_SMALL, target="color", n_classes=1, minimum=2):
        build_split(one, Task.MULTICLASS)
    assert build_split(pd.Series(["a", "b"] * 1000, name="y"), Task.BINARY, k=3).spec.k == 3


# ---------------------------------------------------------------- groups and duplicates
@pytest.mark.unit
def test_T_LEAK_04_group_integrity() -> None:
    frame = fixture("grouped.csv", "readmitted", Task.BINARY)
    result = build_split(frame["readmitted"], Task.BINARY, group=frame["patient_id"])
    spec = result.spec
    assert spec.strategy == "stratified_group_kfold"
    assert spec.group_column == "patient_id"
    groups = frame["patient_id"].to_numpy()
    assert set(groups[spec.dev_indices]).isdisjoint(groups[spec.holdout_indices])
    seen: dict[int, int] = {}
    for _, fold, _, valid in iter_folds(spec):
        for g in set(groups[valid]):
            assert seen.setdefault(g, fold) == fold, "group appears in two folds"
    check_partition(spec)


@pytest.mark.unit
def test_group_regression_and_few_groups() -> None:
    rng = np.random.default_rng(1)
    groups = pd.Series(np.repeat(np.arange(6), 50), name="g")
    y = pd.Series(rng.normal(size=300), name="y")
    result = build_split(y, Task.REGRESSION, group=groups)
    assert result.spec.strategy == "group_kfold"
    assert result.spec.k == 3  # 6 groups < 2·K → K = 3
    assert "W-NO-REPEATED-CV" in result.warnings


@pytest.mark.unit
def test_group_holdout_fallback_warns_when_a_class_cannot_straddle() -> None:
    """A class confined to one group can never be on both sides; the user is told (§11.3)."""
    groups = pd.Series(np.repeat(np.arange(500), 5), name="g")
    labels = np.array(["a"] * 2500, dtype=object)
    labels[:5] = "b"  # class b lives in group 0 only
    result = build_split(pd.Series(labels, name="y"), Task.BINARY, group=groups)
    assert "W-GROUP-HOLDOUT-FALLBACK" in result.warnings
    spec = result.spec
    assert set(groups.to_numpy()[spec.dev_indices]).isdisjoint(
        groups.to_numpy()[spec.holdout_indices]
    )


@pytest.mark.unit
def test_group_holdout_keeps_classes_on_both_sides() -> None:
    groups = pd.Series(np.repeat(np.arange(500), 5), name="g")
    labels = np.where(np.arange(2500) % 10 < 3, "b", "a").astype(object)
    result = build_split(pd.Series(labels, name="y"), Task.BINARY, group=groups)
    assert "W-GROUP-HOLDOUT-FALLBACK" not in result.warnings
    assert set(labels[result.spec.holdout_indices]) == {"a", "b"}


@pytest.mark.unit
def test_duplicate_grouping_keeps_duplicates_together() -> None:
    frame = fixture("quality_issues.csv", "outcome", Task.BINARY)
    features = frame.drop(columns=["outcome"])
    result = build_split(frame["outcome"], Task.BINARY, features=features)
    spec = result.spec
    assert spec.duplicate_grouping
    keys = pd.util.hash_pandas_object(features, index=False).to_numpy()
    fold_of = dict(zip(spec.dev_indices, spec.fold_assignment[0], strict=True))
    for key in set(keys):
        rows = np.flatnonzero(keys == key)
        if len(rows) > 1:
            assert len({fold_of[r] for r in rows}) == 1
    no_dupes = pd.DataFrame({"a": range(100), "b": range(100)})
    assert duplicate_groups(no_dupes) is None


@pytest.mark.unit
def test_duplicates_never_straddle_dev_and_holdout() -> None:
    base = pd.DataFrame({"x": np.arange(1900) % 1700, "z": np.arange(1900) % 1700})  # 200 dupes
    y = pd.Series(["a", "b"] * 1100, name="y")
    features = pd.concat([base, base.iloc[:300]], ignore_index=True)
    spec = build_split(y, Task.BINARY, features=features).spec
    keys = pd.util.hash_pandas_object(features, index=False).to_numpy()
    assert spec.duplicate_grouping
    assert set(keys[spec.dev_indices]).isdisjoint(keys[spec.holdout_indices])


# ---------------------------------------------------------------- determinism
@pytest.mark.unit
def test_split_is_deterministic_across_processes() -> None:
    y = pd.Series(["a", "b", "c"] * 900, name="y")
    first = build_split(y, Task.MULTICLASS, seed_root=7).spec
    assert build_split(y, Task.MULTICLASS, seed_root=7).spec == first
    assert build_split(y, Task.MULTICLASS, seed_root=8).spec.split_id != first.split_id
    code = ("import pandas as pd; from tabpilot.splitting import build_split; "
            "from tabpilot.schemas import Task; "
            "print(build_split(pd.Series(['a','b','c']*900, name='y'), Task.MULTICLASS, 7).spec.split_id)")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == first.split_id


@pytest.mark.unit
def test_repeated_cv_oof_averaging() -> None:
    reps = [np.array([[0.2], [0.4]], dtype=np.float32), np.array([[0.4], [0.8]], dtype=np.float32)]
    np.testing.assert_allclose(average_repeats(reps), [[0.3], [0.6]])
    assert average_repeats(reps).dtype == np.float32
    with pytest.raises(ValueError, match="no OOF"):
        average_repeats([])


# ---------------------------------------------------------------- holdout guard
@pytest.mark.unit
def test_T_LEAK_01_holdout_sealed_until_frozen() -> None:
    frozen = {"value": False}
    guard = HoldoutGuard(np.array([3017, 5029, 7043]), lambda: frozen["value"])
    with pytest.raises(HoldoutSealed):
        guard.open("tuning")
    assert guard.access_log == []
    assert [e.caller for e in guard.refused_log] == ["tuning"]
    assert "3017" not in repr(guard)


@pytest.mark.unit
def test_T_LEAK_03_guard_opens_exactly_once() -> None:
    frozen = {"value": False}
    guard = HoldoutGuard(np.array([3, 5, 7]), lambda: frozen["value"])
    frozen["value"] = True
    hold = guard.open("finalize")
    np.testing.assert_array_equal(hold, [3, 5, 7])
    with pytest.raises(ValueError, match="read-only"):
        hold[0] = 99
    with pytest.raises(HoldoutAlreadyOpened):
        guard.open("finalize")
    assert len(guard.access_log) == 1
    assert guard.access_log[0].granted
    assert len(guard.refused_log) == 1
    assert guard.size == 3
