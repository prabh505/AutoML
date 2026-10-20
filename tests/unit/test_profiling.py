"""P06 tests: T-PRF-01…04, T-LEAK-05/06, T-PRIV-01, DEV-only access, structure detection."""

from __future__ import annotations

import json
import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from tests.helpers.guarded_frame import guard

from tabpilot.config import load_settings
from tabpilot.ingestion import apply_target, load_dataset, validate_target
from tabpilot.ingestion.loader import LoadedDataset
from tabpilot.ingestion.task import detect_task
from tabpilot.profiling import (
    apply_leakage_exclusions,
    build_compact_profile,
    canonical_json,
    profile,
)
from tabpilot.profiling.budget import ProfileBudget
from tabpilot.schemas import (
    ColumnOverride,
    ColumnRole,
    ExecutionProfileName,
    ProfileSummary,
    SplitSpec,
    Task,
)
from tabpilot.splitting import build_split

RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
NUMBER_TOKEN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
FIX = Path(__file__).resolve().parents[2] / "fixtures" / "datasets"
HOSTED = ExecutionProfileName.HOSTED


def prepare(
    name: str, target: str, path: Path | None = None
) -> tuple[LoadedDataset, SplitSpec, Task, str | None]:
    s = load_settings({})
    ld = load_dataset(path or FIX / name, name, s, RID)
    det = detect_task(ld.frame[target].dropna())
    ld = apply_target(ld, validate_target(ld, target, det.task))
    spec = build_split(ld.frame[target], det.task, features=ld.frame.drop(columns=[target])).spec
    return ld, spec, det.task, det.positive_label


def run(name: str, target: str, **kw: Any) -> tuple[ProfileSummary, LoadedDataset, SplitSpec]:
    ld, spec, task, pos = prepare(name, target)
    summary = profile(ld.frame, spec, ld.metadata, task, execution_profile=HOSTED, time_budget_s=960,
                      positive_label=pos, **kw)  # fmt: skip
    return summary, ld, spec


@pytest.fixture(scope="module")
def basic() -> tuple[ProfileSummary, LoadedDataset, SplitSpec]:
    return run("binary_basic.csv", "churn")


# ---------------------------------------------------------------- T-PRF-01 statistics
@pytest.mark.unit
def test_T_PRF_01_statistics_match_dev_rows(
    basic: tuple[ProfileSummary, LoadedDataset, SplitSpec],
) -> None:
    summary, ld, spec = basic
    dev = ld.frame.iloc[spec.dev_indices]
    cols = {c.name: c for c in summary.columns}
    assert summary.n_rows_dev == len(dev) == 2000
    tenure = dev["tenure_months"].astype(float)
    assert cols["tenure_months"].p50 == pytest.approx(float(np.quantile(tenure, 0.5)))
    assert cols["tenure_months"].mean == pytest.approx(tenure.mean())
    assert cols["tenure_months"].min == tenure.min()
    assert cols["income"].missing_rate == pytest.approx(dev["income"].isna().mean())
    contract = dev["contract"].astype(str).value_counts(normalize=True)
    assert cols["contract"].top_shares == pytest.approx(list(contract.round(6)))
    assert cols["contract"].entropy_bits == pytest.approx(
        float(-(contract * np.log2(contract)).sum())
    )
    assert cols["signup_date"].span_days is not None
    assert cols["signup_date"].min_date is not None
    assert summary.target.class_counts == dev["churn"].astype(str).value_counts().to_dict()
    assert summary.target.positive_share == pytest.approx((dev["churn"] == "Yes").mean())
    # Associations follow the generating process: informative > noise.
    strength = {n: c.association.value for n, c in cols.items() if c.association}
    assert strength["tenure_months"] > 0.6
    assert strength["noise_1"] < 0.56
    assert cols["contract"].association is not None
    assert cols["contract"].association.kind == "mi"


@pytest.mark.unit
def test_T_PRF_01_regression_and_correlations(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    frame = pd.DataFrame({"a": x, "a_copy": 2 * x + 1e-3 * rng.normal(size=600),
                          "b": rng.normal(size=600), "y": 3 * x + rng.normal(size=600)})  # fmt: skip
    path = tmp_path / "r.csv"
    frame.to_csv(path, index=False)
    ld, spec, task, _ = prepare("r.csv", "y", path)
    summary = profile(
        ld.frame, spec, ld.metadata, task, execution_profile=HOSTED, time_budget_s=960
    )
    assert [(p.a, p.b) for p in summary.correlated_pairs] in ([("a", "a_copy")], [("a_copy", "a")])
    assoc = {c.name: c.association for c in summary.columns}
    assert assoc["a"] is not None
    assert assoc["a"].kind == "spearman"
    assert assoc["a"].value > 0.8
    assert summary.target.skew is not None


# ---------------------------------------------------------------- T-PRF-02 time caps
class StepClock:
    """Fake clock that advances `step` seconds on every reading."""

    def __init__(self, step: float) -> None:
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


@pytest.mark.unit
def test_T_PRF_02_time_caps_truncate_and_record() -> None:
    summary, _, _ = run("binary_basic.csv", "churn", clock=StepClock(30.0))
    assert set(summary.truncated_stats) >= {"numeric", "association", "leakage"}
    cols = {c.name: c for c in summary.columns}
    assert all(c.mean is None for c in cols.values())  # numeric group cut before any column
    assert summary.leakage == []


@pytest.mark.unit
def test_T_PRF_02_budget_rules() -> None:
    b = ProfileBudget.for_run(HOSTED, time_budget_s=960, n_groups=6)
    assert b.total_s == 60.0  # max(5% of 960 = 48, 60)
    assert b.group_cap_s == pytest.approx(10.0)  # 60 / 6 < 20
    big = ProfileBudget.for_run(ExecutionProfileName.LOCAL, time_budget_s=7200, n_groups=6)
    assert big.total_s == 360.0
    assert big.group_cap_s == 60.0
    assert big.leakage_cap_s == 60.0


# ---------------------------------------------------------------- T-PRF-03 compact size
@pytest.mark.unit
def test_T_PRF_03_compact_truncation_on_wide() -> None:
    summary, ld, _ = run("wide.csv", "label")
    compact = build_compact_profile(summary, ld.metadata)
    text = canonical_json(compact)
    assert len(text) <= 14_000
    assert compact.truncated
    assert compact.rest_summary is not None
    assert compact.rest_summary.n_columns + len(compact.columns) == 189
    assert compact.n_features == 189
    assert "W-WIDE" in compact.warnings
    order = [c.name for c in compact.columns[:3]]
    assert set(order) <= {"v001", "v002", "v003"} or order[0] in {
        "v001",
        "v002",
    }  # informative first
    tiny = build_compact_profile(summary, ld.metadata, max_chars=3_000)
    assert len(canonical_json(tiny)) <= 3_000
    assert all(c.level == "short" for c in tiny.columns)


# ---------------------------------------------------------------- T-PRF-04 determinism, modes
@pytest.mark.unit
def test_T_PRF_04_determinism_and_minimal_mode(
    basic: tuple[ProfileSummary, LoadedDataset, SplitSpec],
) -> None:
    summary, ld, _ = basic
    again, _, _ = run("binary_basic.csv", "churn")
    standard = canonical_json(build_compact_profile(summary, ld.metadata))
    assert standard == canonical_json(build_compact_profile(again, ld.metadata))
    minimal = build_compact_profile(summary, ld.metadata, mode="minimal")
    assert all(c.q is None for c in minimal.columns)
    assert any(f.startswith("range:") for c in minimal.columns for f in c.flags)
    assert json.loads(canonical_json(minimal))["mode"] == "minimal"
    code = ("from tests.unit.test_profiling import run; from tabpilot.profiling import "
            "build_compact_profile, canonical_json; s, ld, _ = run('binary_basic.csv', 'churn'); "
            "print(canonical_json(build_compact_profile(s, ld.metadata)))")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[2])  # fmt: skip
    assert out.stdout.strip() == standard


# ---------------------------------------------------------------- leakage
@pytest.mark.unit
def test_T_LEAK_05_near_deterministic_column_excluded_with_notice() -> None:
    summary, ld, _ = run("leaky_binary.csv", "defaulted")
    near = [f for f in summary.leakage if f.level == "near_deterministic"]
    assert [f.column for f in near] == ["status_code"]
    assert "W-LEAK-NEAR-DETERMINISTIC:status_code" in summary.warnings
    meta = apply_leakage_exclusions(ld.metadata, summary.leakage, [])
    col = {c.name: c for c in meta.columns}["status_code"]
    assert (col.role, col.exclusion_reason) == (ColumnRole.EXCLUDED, "leakage_near_deterministic")
    compact = build_compact_profile(summary, meta)
    assert "status_code" not in {c.name for c in compact.columns}
    assert "W-LEAK-NEAR-DETERMINISTIC" in compact.warnings
    readd = apply_leakage_exclusions(ld.metadata, summary.leakage,
                                     [ColumnOverride(column="status_code", role=ColumnRole.FEATURE)])  # fmt: skip
    assert {c.name: c.role for c in readd.columns}["status_code"] == ColumnRole.FEATURE


@pytest.mark.unit
def test_T_LEAK_06_strong_column_kept_and_warned() -> None:
    summary, ld, _ = run("leaky_binary.csv", "defaulted")
    strong = {f.column: f for f in summary.leakage if f.level == "strong"}
    assert set(strong) == {"risk_score"}
    assert 0.90 <= strong["risk_score"].score < 0.95
    assert "W-LEAK-STRONG:risk_score" in summary.warnings
    meta = apply_leakage_exclusions(ld.metadata, summary.leakage, [])
    assert {c.name: c.role for c in meta.columns}["risk_score"] == ColumnRole.FEATURE
    flags = {c.name: c.flags for c in build_compact_profile(summary, meta).columns}
    assert "leak_strong" in flags["risk_score"]


# ---------------------------------------------------------------- DEV-only access
@pytest.mark.unit
def test_profile_reads_dev_rows_only() -> None:
    ld, spec, task, pos = prepare("binary_basic.csv", "churn")
    hold_rows = ld.frame.index[spec.holdout_indices]  # test code may look; profile() may not
    guarded = guard(ld.frame, hold_rows)
    summary = profile(guarded, spec, ld.metadata, task, execution_profile=HOSTED, time_budget_s=960,
                      positive_label=pos)  # fmt: skip
    assert summary.n_rows_dev == 2000


# ---------------------------------------------------------------- structure and warnings
@pytest.mark.unit
def test_group_and_temporal_detection(tmp_path: Path) -> None:
    summary, _, _ = run("grouped.csv", "readmitted")
    assert summary.group_candidates == ["patient_id"]
    assert "W-GROUP-SUSPECTED:patient_id" in summary.warnings
    n = 800
    rng = np.random.default_rng(3)
    order = np.arange(n)
    frame = pd.DataFrame({
        "event_date": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "row_id": order, "x": rng.normal(size=n),
        "y": (order / n + rng.normal(0, 0.2, n) > 0.5).astype(int),
    })  # fmt: skip
    path = tmp_path / "t.csv"
    frame.to_csv(path, index=False)
    ld, spec, task, _ = prepare("t.csv", "y", path)
    temporal = profile(
        ld.frame, spec, ld.metadata, task, execution_profile=HOSTED, time_budget_s=960
    )
    assert set(temporal.temporal_columns) == {"event_date", "row_id"}
    assert {"W-TEMPORAL:event_date", "W-TEMPORAL:row_id"} <= set(temporal.warnings)


@pytest.mark.unit
def test_dataset_warnings() -> None:
    small, _, _ = run("binary_small.csv", "target_")
    assert "W-SMALL-DATA" in small.warnings
    imbalanced, _, _ = run("binary_imbalanced.csv", "is_fraud")
    assert "W-IMBALANCE" in imbalanced.warnings
    quality, _, _ = run("quality_issues.csv", "outcome")
    assert "W-HIGH-MISSING:missing_60" in quality.warnings
    assert quality.duplicate_rows == 20
    assert quality.conflicting_duplicate_rows == 5


@pytest.mark.unit
def test_profile_requires_validated_target() -> None:
    s = load_settings({})
    ld = load_dataset(FIX / "binary_small.csv", "b", s, RID)
    spec = build_split(ld.frame["target_"], Task.BINARY).spec
    with pytest.raises(ValueError, match="validated target"):
        profile(
            ld.frame, spec, ld.metadata, Task.BINARY, execution_profile=HOSTED, time_budget_s=960
        )


# ---------------------------------------------------------------- T-PRIV-01 (property-based)
def _strings(node: Any) -> set[str]:
    """Every string key and value in a JSON document."""
    if isinstance(node, dict):
        return set(node) | {s for v in node.values() for s in _strings(v)}
    if isinstance(node, list):
        return {s for v in node for s in _strings(v)}
    return {node} if isinstance(node, str) else set()


LABEL = st.text(alphabet=string.ascii_uppercase + string.digits, min_size=6, max_size=10)


@pytest.mark.unit
@settings(
    max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(labels=st.lists(LABEL, min_size=3, max_size=8, unique=True), seed=st.integers(0, 10_000))
@example(labels=["000000", "000001", "000002", "00000A", "0000000"], seed=1568)  # KI-012/021
def test_T_PRIV_01_no_raw_values_in_compact_profile(
    labels: list[str], seed: int, tmp_path: Path
) -> None:
    rng = np.random.default_rng(seed)
    n = 300
    ids = rng.permutation(10**6)[:n] + 123_456  # ≥ 6 significant digits
    frame = pd.DataFrame({
        "account_id": ids, "segment": rng.choice(labels, n), "flag": rng.choice(["Y", "N"], n),
        "amount": rng.normal(100, 10, n).round(2), "target": rng.choice(["good", "bad"], n),
    })  # fmt: skip
    path = tmp_path / f"p{seed}.csv"
    frame.to_csv(path, index=False)
    ld, spec, task, pos = prepare(path.name, "target_", path)
    summary = profile(ld.frame, spec, ld.metadata, task, execution_profile=HOSTED, time_budget_s=960,
                      positive_label=pos)  # fmt: skip
    for mode in ("standard", "minimal"):
        text = canonical_json(build_compact_profile(summary, ld.metadata, mode=mode))
        strings = _strings(json.loads(text))
        for label in [*labels, "Y", "N", "good", "bad"]:
            assert label not in strings, f"category label {label!r} leaked"
            assert f'"{label}"' not in text
        # whole numeric tokens only: a share such as 0.166667 is not the identifier 166667
        numbers = {float(t) for t in NUMBER_TOKEN.findall(text)}
        for value in ids:
            assert float(value) not in numbers, "identifier value leaked"
