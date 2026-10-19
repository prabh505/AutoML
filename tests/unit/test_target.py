"""P04 target validation tests: T-TGT-01…06 and row-removal recording (§8.7; A-008)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.helpers.errors import raises_error

from tabpilot.config import Settings, load_settings
from tabpilot.ingestion import apply_target, load_dataset, validate_target
from tabpilot.ingestion.loader import LoadedDataset
from tabpilot.schemas import ColumnRole, ErrorCode, Task

RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
FIX = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def s() -> Settings:
    return load_settings({})


def csv(tmp_path: Path, frame: pd.DataFrame, s: Settings) -> LoadedDataset:
    path = tmp_path / "t.csv"
    frame.to_csv(path, index=False)
    return load_dataset(path, "t.csv", s, RID)


def base(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({"x": rng.normal(size=n).round(3), "z": rng.integers(0, 50, n)})


@pytest.mark.unit
def test_T_TGT_01_constant(tmp_path: Path, s: Settings) -> None:
    ld = csv(tmp_path, base().assign(y="same"), s)
    with raises_error(ErrorCode.TARGET_CONSTANT, target="y"):
        validate_target(ld, "y")


@pytest.mark.unit
def test_T_TGT_02_all_missing(tmp_path: Path, s: Settings) -> None:
    ld = csv(tmp_path, base().assign(y=""), s)
    with raises_error(ErrorCode.TARGET_ALL_MISSING, target="y"):
        validate_target(ld, "y")


@pytest.mark.unit
def test_T_TGT_03_too_few_rows(tmp_path: Path, s: Settings) -> None:
    y = ["a", "b"] * 24 + [""] * 152
    ld = csv(tmp_path, base().assign(y=y), s)
    with raises_error(ErrorCode.TARGET_TOO_FEW_ROWS, target="y", rows=48, minimum=50):
        validate_target(ld, "y")


@pytest.mark.unit
def test_T_TGT_04_class_too_small_and_consent(tmp_path: Path, s: Settings) -> None:
    y = ["a"] * 100 + ["b"] * 99 + ["c"]
    ld = csv(tmp_path, base().assign(y=y), s)
    validate_target(ld, "y")  # task unknown: class rules not yet applied (PI-47)
    with raises_error(ErrorCode.TARGET_CLASS_TOO_SMALL, target="y", n_classes=1, minimum=2):
        validate_target(ld, "y", Task.MULTICLASS)
    check = validate_target(ld, "y", Task.MULTICLASS, drop_tiny_classes=True)
    applied = apply_target(ld, check)
    assert applied.metadata.n_rows_valid == 199
    assert {r.reason: r.count for r in applied.metadata.row_removals} == {
        "target_missing": 0, "class_too_small_with_consent": 1}  # fmt: skip


@pytest.mark.unit
def test_T_TGT_05_too_many_classes(tmp_path: Path, s: Settings) -> None:
    y = [f"k{i % 60}" for i in range(200)]
    ld = csv(tmp_path, base().assign(y=y), s)
    with raises_error(ErrorCode.TARGET_TOO_MANY_CLASSES, target="y", count=60, limit=50):
        validate_target(ld, "y", Task.MULTICLASS)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["text", "identifier", "datetime"])
def test_T_TGT_06_unsupported_type(kind: str, tmp_path: Path, s: Settings) -> None:
    n = 200
    values = {
        "text": [f"long free text sentence number {i} with many words inside it" for i in range(n)],
        "identifier": [f"ID{i:06d}" for i in range(n)],
        "datetime": [f"2024-01-{i % 28 + 1:02d}" for i in range(n)],
    }[kind]
    ld = csv(tmp_path, base(n).assign(y=values), s)
    with raises_error(ErrorCode.TARGET_UNSUPPORTED_TYPE, target="y", semantic_type=kind):
        validate_target(ld, "y")


@pytest.mark.unit
def test_regression_requires_numeric_target(tmp_path: Path, s: Settings) -> None:
    ld = csv(tmp_path, base().assign(y=["a", "b"] * 100), s)
    with raises_error(ErrorCode.TARGET_UNSUPPORTED_TYPE, semantic_type="categorical"):
        validate_target(ld, "y", Task.REGRESSION)


@pytest.mark.unit
def test_missing_target_rows_removed_and_recorded(tmp_path: Path, s: Settings) -> None:
    """The only automatic row removal; counted and recorded (FR-TGT-01, A-008)."""
    y = ["Yes", "No"] * 90 + [""] * 20
    ld = csv(tmp_path, base().assign(y=y), s)
    check = validate_target(ld, "y", Task.BINARY)
    assert (check.report.n_missing, check.report.n_rows_removed) == (20, 20)
    assert check.report.requires_confirmation is False
    assert check.report.class_counts == {"Yes": 90, "No": 90}
    applied = apply_target(ld, check)
    meta = applied.metadata
    assert len(applied.frame) == meta.n_rows_valid == 180
    assert meta.n_rows_raw == 200
    assert {"reason": "target_missing", "count": 20} in [r.model_dump() for r in meta.row_removals]
    assert "W-TARGET-MISSING:y:20" in meta.warnings
    assert {c.name: c.role for c in meta.columns}["y"] == ColumnRole.TARGET
    assert meta.target is not None
    assert meta.target.task == Task.BINARY


@pytest.mark.unit
def test_more_than_half_missing_requires_confirmation(tmp_path: Path, s: Settings) -> None:
    """A-008: > 50% missing target → W-TARGET-MISSING + confirmation (not an error)."""
    y = ["a", "b"] * 40 + [""] * 120
    ld = csv(tmp_path, base().assign(y=y), s)
    check = validate_target(ld, "y")
    assert check.report.requires_confirmation is True
    assert check.report.n_missing == 120


@pytest.mark.unit
def test_regression_target_distribution(s: Settings) -> None:
    ld = load_dataset(FIX / "datasets" / "regression_skewed.csv", "r.csv", s, RID)
    report = validate_target(ld, "claim_amount", Task.REGRESSION).report
    assert report.class_counts is None
    assert report.distribution is not None
    assert report.distribution.skew is not None
    assert report.distribution.skew > 2


@pytest.mark.unit
def test_conflicting_duplicates_recorded(s: Settings) -> None:
    ld = load_dataset(FIX / "datasets" / "quality_issues.csv", "q.csv", s, RID)
    applied = apply_target(ld, validate_target(ld, "outcome", Task.BINARY))
    assert "W-CONFLICTING-DUPLICATES:5" in applied.metadata.warnings


@pytest.mark.unit
def test_unknown_target_name(s: Settings) -> None:
    ld = load_dataset(FIX / "datasets" / "binary_small.csv", "b.csv", s, RID)
    with pytest.raises(KeyError):
        validate_target(ld, "nope")
