"""P04 feasibility tests: T-RES-01 (estimator) and T-RES-02 (simplifications, rejection)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers.errors import raises_error

from tabpilot.config import Settings, load_settings
from tabpilot.ingestion import apply_target, estimate_feasibility, load_dataset, raise_if_infeasible
from tabpilot.ingestion.target import validate_target
from tabpilot.resources.estimator import Plan, estimate_peak_mb
from tabpilot.schemas import (
    DatasetMetadata,
    ErrorCode,
    ModelName,
    RunConfig,
    Simplification,
    Task,
)

RID = "01JAB3K8X2V7Q9M4T6R1Z0YH5C"
FIX = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def hosted() -> Settings:
    return load_settings({"TABPILOT_PROFILE": "hosted"})


def meta(rel: str, target: str, task: Task, s: Settings) -> DatasetMetadata:
    ld = load_dataset(FIX / rel, "x.csv", s, RID)
    return apply_target(ld, validate_target(ld, target, task)).metadata


def config(task: Task = Task.BINARY, ensemble: bool = True) -> RunConfig:
    metric = "roc_auc" if task == Task.BINARY else "log_loss"
    return RunConfig(run_id=RID, target="churn", task=task, primary_metric=metric,
                     time_budget_s=960, execution_profile="hosted",
                     enable_ensemble=ensemble)  # fmt: skip


def scaled(m: DatasetMetadata, rows: int) -> DatasetMetadata:
    return m.model_copy(update={"n_rows_raw": rows, "n_rows_valid": rows})


@pytest.mark.unit
def test_T_RES_01_estimator_behaviour(hosted: Settings) -> None:
    m = meta("datasets/binary_basic.csv", "churn", Task.BINARY, hosted)
    report = estimate_feasibility(m, config(), hosted)
    assert report.status == "ok"
    assert report.limit_mb == 600.0
    assert 250 * 1.5 < report.estimated_peak_mb < 600
    plan = Plan(models=(ModelName.LINEAR, ModelName.LIGHTGBM, ModelName.RANDOM_FOREST))
    sizes = [estimate_peak_mb(scaled(m, r), plan, hosted) for r in (1_000, 10_000, 100_000)]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[2]
    no_rf = Plan(models=(ModelName.LINEAR, ModelName.LIGHTGBM))
    assert estimate_peak_mb(scaled(m, 100_000), no_rf, hosted) < sizes[2]
    assert estimate_peak_mb(m, plan, hosted) == pytest.approx(estimate_peak_mb(m, plan, hosted))


@pytest.mark.unit
def test_T_RES_02_simplifications_in_order(hosted: Settings) -> None:
    m = scaled(meta("datasets/binary_basic.csv", "churn", Task.BINARY, hosted), 100_000)
    report = estimate_feasibility(m, config(), hosted)
    assert report.status == "simplified"
    order = list(Simplification)
    assert report.simplifications == sorted(report.simplifications, key=order.index)
    assert Simplification.DISABLE_RANDOM_FOREST in report.simplifications
    assert Simplification.REDUCE_TUNING_TRIALS not in report.simplifications  # no memory effect
    assert report.estimated_peak_mb <= report.limit_mb
    raise_if_infeasible(report)  # does not raise


@pytest.mark.unit
def test_T_RES_02_rejection(hosted: Settings) -> None:
    m = meta("datasets/wide.csv", "label", Task.BINARY, hosted)
    report = estimate_feasibility(scaled(m, 5_000_000), config(), hosted)
    assert report.status == "rejected"
    with raises_error(ErrorCode.RESOURCE_INFEASIBLE) as info:
        raise_if_infeasible(report)
    assert info.value.params["limit"] == 600
    assert "TABPILOT_PROFILE=local" in info.value.user_message


@pytest.mark.unit
def test_local_limit_scales_with_ram() -> None:
    s = load_settings({"TABPILOT_PROFILE": "local"})
    m = meta("datasets/binary_small.csv", "target_", Task.BINARY, s)  # "target" is reserved (§8.4)
    report = estimate_feasibility(m, config(), s, physical_ram_mb=16_000)
    assert report.limit_mb == 11_200.0
    assert report.status == "ok"


@pytest.mark.unit
def test_disabling_ensemble_reduces_oof(hosted: Settings) -> None:
    m = meta("datasets/multiclass_basic.csv", "species", Task.MULTICLASS, hosted)
    with_ens = estimate_feasibility(m, config(Task.MULTICLASS), hosted).estimated_peak_mb
    without = estimate_feasibility(
        m, config(Task.MULTICLASS, ensemble=False), hosted
    ).estimated_peak_mb
    assert without <= with_ens
