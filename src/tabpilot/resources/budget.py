"""Time-budget ledger (PRD §29.3 as amended by A-004).

Stage groups get cumulative deadlines: ingest 5% (at least 60 s), planning 5%, training and tuning
40%, reflection 15%, ensemble 5%; the remaining 30% is reserved for finalize, explain, report and
export. A group may run until its cumulative boundary, so time an earlier group did not use rolls
forward automatically. Non-reserved work stops at the 70% stop point and never consumes the
reserved slice.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tabpilot.schemas import Stage
from tabpilot.schemas.run import BudgetLedger

GROUP_SHARES: tuple[tuple[str, float], ...] = (("ingest", 0.05), ("plan", 0.05), ("train", 0.40),
                                               ("reflect", 0.15), ("ensemble", 0.05))  # fmt: skip
RESERVED_SHARE = 0.30
STOP_FRACTION = 0.70
INGEST_MIN_S = 60.0
STAGE_GROUP: dict[Stage, str] = {
    Stage.INGEST: "ingest", Stage.SPLIT: "ingest", Stage.PROFILE: "ingest",
    Stage.PLAN: "plan", Stage.VALIDATE_PLAN: "plan",
    Stage.BUILD_FEATURES: "train", Stage.TRAIN_CANDIDATES: "train", Stage.EVALUATE: "train",
    Stage.REFLECT: "reflect", Stage.APPLY_CHANGES: "reflect", Stage.ENSEMBLE: "ensemble",
    Stage.FINALIZE: "reserved", Stage.EXPLAIN: "reserved", Stage.REPORT: "reserved",
    Stage.EXPORT: "reserved", Stage.RECOVER: "reserved", Stage.RUN: "reserved",
}  # fmt: skip


@dataclass
class StageTiming:
    """Start/end of one stage, in seconds since the run started."""

    start: float
    end: float | None = None

    @property
    def duration(self) -> float:
        return 0.0 if self.end is None else self.end - self.start


@dataclass
class TimeBudget:
    """The run's time ledger. `clock` is injectable for tests (monotonic seconds)."""

    total_s: float
    trials_max: int
    clock: Callable[[], float] = time.monotonic
    started: float = field(init=False)
    timings: dict[Stage, StageTiming] = field(default_factory=dict)
    trials_used: int = 0

    def __post_init__(self) -> None:
        if self.total_s <= 0:
            raise ValueError("time budget must be positive")
        self.started = self.clock()

    def elapsed(self) -> float:
        return self.clock() - self.started

    def boundary(self, group: str) -> float:
        """Seconds since start at which `group` must stop (cumulative, §29.3)."""
        if group == "reserved":
            return self.total_s
        stop = STOP_FRACTION * self.total_s
        cumulative = 0.0
        for name, share in GROUP_SHARES:
            cumulative += share * self.total_s
            if name == "ingest":
                cumulative = max(cumulative, min(INGEST_MIN_S, stop))
            if name == group:
                return min(cumulative, stop)
        raise ValueError(f"unknown stage group {group!r}")

    def remaining(self, stage: Stage) -> float:
        """Seconds left for `stage` before its group boundary (≥ 0)."""
        return max(0.0, self.boundary(STAGE_GROUP[stage]) - self.elapsed())

    def deadline(self, stage: Stage) -> float:
        """Absolute clock value at which `stage` must stop."""
        return self.started + self.boundary(STAGE_GROUP[stage])

    def expired(self, stage: Stage) -> bool:
        return self.remaining(stage) <= 0.0

    def begin(self, stage: Stage) -> None:
        self.timings[stage] = StageTiming(self.elapsed())

    def end(self, stage: Stage) -> None:
        timing = self.timings.setdefault(stage, StageTiming(self.elapsed()))
        timing.end = self.elapsed()

    def trials_left(self) -> int:
        return max(0, self.trials_max - self.trials_used)

    def use_trials(self, n: int) -> None:
        self.trials_used += n

    def ledger(self) -> BudgetLedger:
        """The `budget_ledger/1.0` contract for run records."""
        return BudgetLedger(
            time_budget_s=int(self.total_s),
            elapsed_s=round(self.elapsed(), 3),
            stage_seconds={s: round(t.duration, 3) for s, t in self.timings.items()},
            trials_used=self.trials_used,
            trials_max=self.trials_max,
        )
