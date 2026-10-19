"""P09 resource tests: time ledger (§29.3 + A-004), RSS monitor (§29.6), T-RES-03."""

from __future__ import annotations

import time
from typing import Any

import pytest
from tests.helpers.clock import FakeClock
from tests.unit.test_training import synthetic_dev

from tabpilot.config import load_settings
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import build_pipeline
from tabpilot.resources import MemoryLimitExceeded, ResourceMonitor, TimeBudget, process_rss_mb
from tabpilot.schemas import CandidateSpec, MetricName, ModelName, Stage, Task
from tabpilot.schemas.experiment import ModelConfig
from tabpilot.training import run_cv

REG = ModelRegistry(load_settings({}))


def ledger(total: float = 1000.0) -> tuple[TimeBudget, FakeClock]:
    clock = FakeClock()
    return TimeBudget(total, trials_max=60, clock=clock.monotonic), clock


# ---------------------------------------------------------------- §29.3 ledger (A-004)
@pytest.mark.unit
def test_cumulative_boundaries_follow_A_004_shares() -> None:
    budget, _ = ledger(1000)
    assert budget.boundary("ingest") == 60.0  # 5% = 50 s, but at least 60 s
    assert budget.boundary("plan") == pytest.approx(110.0)
    assert budget.boundary("train") == pytest.approx(510.0)
    assert budget.boundary("reflect") == pytest.approx(660.0)
    assert budget.boundary("ensemble") == pytest.approx(700.0)  # the 70% stop point
    assert budget.boundary("reserved") == 1000.0
    big, _ = ledger(4000)
    assert big.boundary("ingest") == 200.0
    assert big.boundary("ensemble") == 2800.0
    with pytest.raises(ValueError, match="unknown"):
        budget.boundary("nope")
    with pytest.raises(ValueError, match="positive"):
        TimeBudget(0, 1)


@pytest.mark.unit
def test_unused_time_rolls_forward_and_reserved_slice_is_kept() -> None:
    budget, clock = ledger(1000)
    clock.advance(20)  # ingest + plan finished early
    assert budget.remaining(Stage.TRAIN_CANDIDATES) == pytest.approx(490.0)
    clock.advance(600)  # training overran its boundary
    assert budget.expired(Stage.TRAIN_CANDIDATES)
    assert budget.remaining(Stage.ENSEMBLE) == pytest.approx(80.0)  # 620 s of 700 s used
    clock.advance(90)
    assert budget.expired(Stage.ENSEMBLE)  # non-reserved work stops at 70%
    assert not budget.expired(Stage.FINALIZE)
    assert budget.remaining(Stage.REPORT) == pytest.approx(290.0)
    assert budget.deadline(Stage.EXPORT) == budget.started + 1000


@pytest.mark.unit
def test_stage_timings_and_ledger_contract() -> None:
    budget, clock = ledger(500)
    budget.begin(Stage.INGEST)
    clock.advance(3)
    budget.end(Stage.INGEST)
    budget.end(Stage.SPLIT)  # end without begin records a zero-length stage
    budget.use_trials(7)
    record = budget.ledger()
    assert record.stage_seconds[Stage.INGEST] == 3.0
    assert record.stage_seconds[Stage.SPLIT] == 0.0
    assert (record.trials_used, record.trials_max, record.time_budget_s) == (7, 60, 500)
    assert budget.trials_left() == 53


# ---------------------------------------------------------------- §29.6 RSS monitor
class FakeRss:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.mark.unit
def test_soft_limit_warns_and_collects_once(monkeypatch: pytest.MonkeyPatch) -> None:
    collected: list[int] = []
    monkeypatch.setattr("tabpilot.resources.monitor.gc.collect", lambda: collected.append(1) or 0)
    rss = FakeRss(100)
    clock = FakeClock()
    mon = ResourceMonitor(500, 700, rss_fn=rss, clock=clock.monotonic)
    mon.sample()
    rss.value = 600
    mon.sample()
    mon.sample()
    assert mon.soft_exceeded
    assert collected == [1]
    mon.check()  # soft only: no abort
    assert mon.peak_mb == 600
    assert mon.hard_strikes == 0
    assert [s.rss_mb for s in mon.samples] == [100]  # recorded every 5 s
    clock.advance(5)
    mon.sample()
    assert [s.rss_mb for s in mon.samples] == [100, 600]


@pytest.mark.unit
def test_hard_limit_aborts_at_check_and_two_strikes_stop_training() -> None:
    rss = FakeRss(800)
    mon = ResourceMonitor(500, 700, rss_fn=rss)
    with pytest.raises(MemoryLimitExceeded, match="700 MB"):
        mon.check()
    assert not mon.stop_training
    with pytest.raises(MemoryLimitExceeded):
        mon.check()
    assert mon.stop_training
    rss.value = 100
    mon.check()  # below limits again: nothing pending
    with pytest.raises(ValueError, match="soft_mb"):
        ResourceMonitor(700, 500)


@pytest.mark.unit
def test_monitor_thread_samples_real_rss() -> None:
    mon = ResourceMonitor(1e6, 2e6, interval_s=0.01).start()
    time.sleep(0.1)
    mon.stop()
    assert mon.peak_mb > 10
    assert process_rss_mb() > 10


@pytest.mark.unit
def test_T_RES_03_memory_hog_candidate_aborted_run_continues() -> None:
    """A candidate that pushes RSS over the hard limit is `failed: memory`; others complete."""
    dev, split, schema = synthetic_dev()
    rss = FakeRss(100)
    mon = ResourceMonitor(500, 700, rss_fn=rss)

    def build(cand: CandidateSpec, *a: Any, **kw: Any) -> Any:
        rss.value = 900 if cand.model.name == ModelName.RANDOM_FOREST else 100  # the hog
        return build_pipeline(cand, *a, **kw)

    results = {}
    for name in (ModelName.LINEAR, ModelName.RANDOM_FOREST, ModelName.LIGHTGBM):
        cand = CandidateSpec(candidate_id="c", iteration=0, feature_set="baseline",
                             model=ModelConfig(name=name, task=Task.BINARY, seed=1, n_jobs=1))  # fmt: skip
        results[name] = run_cv(cand, dev, split, schema=schema, registry=REG,
                               metric=MetricName.ROC_AUC, budget_hook=mon.check, build=build).result  # fmt: skip
    assert results[ModelName.RANDOM_FOREST].status == "failed"
    assert (results[ModelName.RANDOM_FOREST].error or "").startswith("memory: RSS above")
    assert results[ModelName.LINEAR].status == "ok"
    assert results[ModelName.LIGHTGBM].status == "ok"
    assert mon.hard_strikes == 1


@pytest.mark.unit
def test_jump_straight_past_hard_limit_also_sets_soft_flag() -> None:
    mon = ResourceMonitor(500, 700, rss_fn=FakeRss(900))
    mon.sample()
    assert mon.soft_exceeded
