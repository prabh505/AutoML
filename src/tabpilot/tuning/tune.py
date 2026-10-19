"""Budgeted Optuna tuning per model (PRD §17.1, §17.3–§17.6; A-004).

Each trial is a full `run_cv` on the run's folds (I-3), so HOLD is never read (L-5). Pruning uses
the running mean of fold selection scores. In `trials` mode the wall clock is ignored and results
are reproducible; in `time` mode the number of completed trials depends on the hardware (§17.5).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import optuna

from tabpilot.evaluation.metrics import selection_score
from tabpilot.logging import get_logger
from tabpilot.models import ModelRegistry
from tabpilot.preprocessing import PipelineSchema
from tabpilot.schemas import CandidateSpec, ExecutionProfileName, MetricName, SplitSpec
from tabpilot.training import BudgetExceeded, CancelToken, CVRun, DevData, run_cv
from tabpilot.tuning.space import SPACES

TuningMode = Literal["trials", "time"]
MAX_CONSECUTIVE_FAILURES = 5
SMALL_DATA_ROWS = 2000
SMALL_DATA_TRIALS = 20
SUBSAMPLE_ROWS = 50_000
TRIALS_PER_MODEL = {ExecutionProfileName.HOSTED: 30, ExecutionProfileName.LOCAL: 100}
HOSTED_TRIAL_TIMEOUT_S = 120.0  # A-004 (PRD 60 s)
LOCAL_TRIAL_TIMEOUT_S = 600.0  # A-004 (PRD 300 s)
TRIAL_TIMEOUT_SHARE = 0.25

log = get_logger("tune")


@dataclass(frozen=True)
class TuningLimits:
    """Per-model limits (§17.3 with A-004 timeouts)."""

    max_trials: int
    per_trial_timeout_s: float
    mode: TuningMode = "time"
    train_row_cap: int | None = None


def limits_for(
    profile: ExecutionProfileName,
    allocation_s: float,
    n_dev_rows: int,
    trials_left: int,
    *,
    mode: TuningMode = "time",
    max_trials: int | None = None,
) -> TuningLimits:
    """§17.3 limits for one model: trial cap (small data ≤ 20, never above the run's remaining
    trials), per-trial timeout, and within-fold subsampling for large hosted DEV sets (§17.6).
    """
    cap = TRIALS_PER_MODEL[profile] if max_trials is None else max_trials
    if n_dev_rows < SMALL_DATA_ROWS:
        cap = min(cap, SMALL_DATA_TRIALS)
    if profile == ExecutionProfileName.HOSTED:
        timeout = min(HOSTED_TRIAL_TIMEOUT_S, TRIAL_TIMEOUT_SHARE * allocation_s)
    else:
        timeout = LOCAL_TRIAL_TIMEOUT_S
    subsample = (SUBSAMPLE_ROWS if profile == ExecutionProfileName.HOSTED
                 and n_dev_rows > SUBSAMPLE_ROWS else None)  # fmt: skip
    return TuningLimits(max(0, min(cap, trials_left)), timeout, mode, subsample)


@dataclass
class TrialRecord:
    """One line of `trials.jsonl` (§30.1). Params and scores only, never data values."""

    model: str
    number: int
    params: dict[str, Any]
    state: Literal["complete", "pruned", "fail"]
    fold_scores: list[float]
    value: float | None
    seconds: float
    error: str | None = None


@dataclass
class TuningResult:
    """Outcome of one model's study."""

    best: CVRun
    best_params: dict[str, Any]
    trials: list[TrialRecord] = field(default_factory=list)
    elapsed_s: float = 0.0
    stopped: str = "max_trials"

    @property
    def completed(self) -> int:
        return sum(t.state == "complete" for t in self.trials)

    @property
    def pruned(self) -> int:
        return sum(t.state == "pruned" for t in self.trials)

    @property
    def failed(self) -> int:
        return sum(t.state == "fail" for t in self.trials)


def _with_params(candidate: CandidateSpec, params: dict[str, Any]) -> CandidateSpec:
    model = candidate.model.model_copy(update={"params": params})
    return candidate.model_copy(update={"model": model})


def tune(  # noqa: PLR0912, PLR0915 - the §17.4 loop
    candidate: CandidateSpec,
    dev: DevData,
    split: SplitSpec,
    *,
    schema: PipelineSchema,
    registry: ModelRegistry,
    metric: MetricName,
    allocation_s: float,
    limits: TuningLimits,
    seed: int,
    profile: ExecutionProfileName,
    cancel: CancelToken | None = None,
    extra_hook: Callable[[], None] | None = None,
    stop: Callable[[], bool] | None = None,
    on_trial: Callable[[TrialRecord], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> TuningResult:
    """Tune `candidate.model` within `allocation_s` (§17.4). Trial 0 = library defaults; the best
    completed trial wins (ties: earlier trial). Without a completed trial the default-config CV
    result is used (§17.5). `stop()` ends the study before any further trial once trial 0 ran.

    Time limits are cooperative: they are checked between folds, so a single slow fold can
    overrun the allocation by up to one fold's fit time. The hard wall-clock kill is the worker's
    job (P15, §13.7); DEC-032.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    timed = limits.mode == "time"
    start = clock()
    deadline = start + allocation_s
    name, task = candidate.model.name, candidate.model.task
    records: list[TrialRecord] = []
    runs: dict[int, CVRun] = {}
    failures, stopped = 0, "max_trials"
    while len(records) < limits.max_trials:
        if cancel is not None and cancel.is_set():
            stopped = "cancelled"
            break
        if timed and clock() >= deadline:
            stopped = "allocation"
            break
        if stop is not None and stop() and records:
            stopped = "resources"  # e.g. RSS above the soft limit: no extra trials (§29.6)
            break
        if failures >= MAX_CONSECUTIVE_FAILURES:
            stopped = "consecutive_failures"
            break
        trial = study.ask()
        params = {} if trial.number == 0 else SPACES[name](trial, task, profile)
        trial_deadline = min(deadline, clock() + limits.per_trial_timeout_s)

        def hook(trial_deadline: float = trial_deadline) -> None:
            if extra_hook is not None:
                extra_hook()
            if timed and clock() > trial_deadline:
                raise BudgetExceeded

        def on_fold(scores: list[float], trial: optuna.Trial = trial) -> None:
            running = sum(selection_score(metric, s) for s in scores) / len(scores)
            trial.report(running, step=len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned

        began = clock()
        record = TrialRecord(name.value, trial.number, params, "fail", [], None, 0.0)
        try:
            run = run_cv(_with_params(candidate, params), dev, split, schema=schema,
                         registry=registry, metric=metric, cancel=cancel, budget_hook=hook,
                         on_fold=on_fold, fallback=False, profile=profile,
                         train_row_cap=limits.train_row_cap)  # fmt: skip
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            record.state, failures = "pruned", 0
        else:
            result = run.result
            if result.status == "ok":
                value = selection_score(metric, sum(result.fold_scores) / len(result.fold_scores))
                study.tell(trial, value)
                record.state, record.value, failures = "complete", value, 0
                record.fold_scores = list(result.fold_scores)
                runs[trial.number] = run
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                record.error = (
                    "trial timeout" if result.status == "skipped_budget" else result.error
                )
                failures += 1
        record.seconds = round(clock() - began, 3)
        records.append(record)
        if on_trial is not None:
            on_trial(record)
    best_number = _best_trial(records)
    if best_number is None:
        log.warning("tuning_no_completed_trial", model=name.value)

        def bounded() -> None:  # the default-config run still respects the allocation
            if extra_hook is not None:
                extra_hook()
            if timed and clock() > deadline:
                raise BudgetExceeded

        best = run_cv(_with_params(candidate, {}), dev, split, schema=schema,
                      registry=registry, metric=metric, cancel=cancel, budget_hook=bounded,
                      profile=profile)  # fmt: skip
        params = {}
    else:
        params = next(r.params for r in records if r.number == best_number)
        best = runs[best_number]
        if limits.train_row_cap is not None:  # §17.6: recompute on full training folds
            best = run_cv(_with_params(candidate, params), dev, split, schema=schema,
                          registry=registry, metric=metric, cancel=cancel,
                          budget_hook=extra_hook, profile=profile)  # fmt: skip
    log.info("tuning_completed", model=name.value, trials=len(records), stopped=stopped)
    return TuningResult(best, params, records, round(clock() - start, 3), stopped)


def _best_trial(records: list[TrialRecord]) -> int | None:
    """Highest objective; ties go to the earlier trial number (§17.4)."""
    done = [r for r in records if r.state == "complete" and r.value is not None
            and math.isfinite(r.value)]  # fmt: skip
    if not done:
        return None
    return max(done, key=lambda r: (r.value, -r.number)).number
