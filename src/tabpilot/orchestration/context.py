"""Worker-local run context and registry (PRD §13.3).

`RunState` stays small and serializable; everything large or live (data frames, the split,
fitted pipelines, OOF arrays, the LLM router, the time ledger, the resource monitor) lives in
one `RunContext` per run, registered by run id for the graph nodes to look up.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from tabpilot.config import Settings, load_settings
from tabpilot.evaluation.selection import SelectionFreeze
from tabpilot.llm import BudgetTracker, LLMRouter
from tabpilot.models import ModelRegistry
from tabpilot.resources import ResourceMonitor, TimeBudget
from tabpilot.schemas import (
    EventType,
    ModelName,
    RunConfig,
    RunEvent,
    Simplification,
    Stage,
    TabPilotError,
)
from tabpilot.schemas.experiment import Seeds
from tabpilot.tracking import RunRecorder
from tabpilot.training import BudgetExceeded, Cancelled, CancelToken, CVRun
from tabpilot.tuning import TuningMode

EventListener = Callable[[RunEvent], None]


class _NeverCancelled:
    def is_set(self) -> bool:
        return False


@dataclass
class WorkerOptions:
    """How the worker runs (CLI/test knobs that are not part of `RunConfig`)."""

    tuning_mode: TuningMode = "time"
    trials_per_model: int | None = None
    models: tuple[ModelName, ...] = ()
    llm_stub_mode: str = "ok"
    include_oof: bool = False


@dataclass
class RunContext:
    """Live objects of one run. Created once per worker; nodes read and update it."""

    config: RunConfig
    run_dir: Path
    input_path: Path
    display_name: str
    settings: Settings
    recorder: RunRecorder
    budget: TimeBudget
    monitor: ResourceMonitor
    registry: ModelRegistry
    seeds: Seeds
    n_jobs: int
    options: WorkerOptions = field(default_factory=WorkerOptions)
    cancel: CancelToken = field(default_factory=_NeverCancelled)
    router: LLMRouter | None = None
    llm_budget: BudgetTracker | None = None
    simplifications: set[Simplification] = field(default_factory=set)
    # data (pre-graph stages)
    loaded: Any = None
    split: Any = None
    guard: Any = None
    freeze: SelectionFreeze = field(default_factory=SelectionFreeze)
    summary: Any = None
    problem: Any = None  # Problem, set by prepare()
    dev: Any = None  # DevData, set by prepare()
    schema: Any = None  # PipelineSchema, set by prepare()
    feasibility: Any = None
    # planning
    planner_input: Any = None
    plan_received: dict[str, Any] | None = None
    pending_plan: Any = None
    # training and reflection
    runs: dict[str, CVRun] = field(default_factory=dict)
    rejected: set[str] = field(default_factory=set)
    pending_proposal: Any = None
    pending_change: Any = None
    round_budget: Any = None
    reflection_seconds: float = 0.0
    initial_trials: int = 0
    soft_warned: bool = False
    time_hit: bool = False
    fallback_used: bool = False
    # finalize and later
    selection: Any = None
    selection_ids: frozenset[str] = frozenset()  # entries the last `selection` was made over
    winner_oof_path: str | None = None
    final_model: Any = None
    baseline_model: Any = None
    baseline_name: str | None = None
    winner_run: CVRun | None = None
    ensemble_members: list[Any] = field(default_factory=list)
    holdout: Any = None
    explanation: Any = None
    errors: Any = None
    report: Any = None
    bundle: Any = None
    failure: TabPilotError | None = None
    route: str = ""  # the next node, decided by the node that just ran
    last_round_accepted: bool = False
    listeners: list[EventListener] = field(default_factory=list)
    current_stage: Stage = Stage.RUN

    @property
    def run_id(self) -> str:
        return self.config.run_id

    @property
    def metric(self) -> Any:
        return self.config.primary_metric

    @classmethod
    def create(cls, config: RunConfig, *, run_dir: Path, input_path: Path, display_name: str,
               options: WorkerOptions | None = None,
               cancel: CancelToken | None = None) -> RunContext:  # fmt: skip
        """Settings, ledger, monitor and recorder for `config` (the worker's first step)."""
        settings = load_settings({**os.environ, "TABPILOT_PROFILE": config.execution_profile})
        limits = settings.limits
        soft, hard = limits.rss_limits_mb(psutil.virtual_memory().total / 2**20)
        return cls(
            config=config, run_dir=run_dir, input_path=input_path, display_name=display_name,
            settings=settings, recorder=RunRecorder(run_dir, config.run_id),
            budget=TimeBudget(float(config.time_budget_s), limits.max_trials_per_run),
            monitor=ResourceMonitor(soft, hard), registry=ModelRegistry(settings),
            seeds=Seeds.derive(config.seed_root),
            n_jobs=limits.resolved_n_jobs(psutil.cpu_count(logical=False) or 1),
            options=options or WorkerOptions(), cancel=cancel or _NeverCancelled(),
        )  # fmt: skip

    # ------------------------------------------------------------ events and stages
    def emit(self, stage: Stage, kind: EventType, **data: Any) -> RunEvent:
        """Record an event (after the action it describes) and pass it to listeners."""
        event = self.recorder.event(stage, kind, **data)
        for listener in self.listeners:
            listener(event)
        return event

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled

    @contextmanager
    def stage(self, stage: Stage) -> Iterator[None]:
        """Cancel check, ledger timing, and started/completed events around a stage."""
        self.check_cancel()
        self.budget.begin(stage)
        self.current_stage = stage
        self.emit(stage, EventType.STAGE_STARTED)
        yield
        self.budget.end(stage)
        self.emit(stage, EventType.STAGE_COMPLETED,
                  seconds=round(self.budget.timings[stage].duration, 3))  # fmt: skip

    def skip(self, stage: Stage, reason: str) -> None:
        self.emit(stage, EventType.STAGE_SKIPPED, reason=reason)

    def train_hook(self, stage: Stage = Stage.TRAIN_CANDIDATES) -> Callable[[], None]:
        """Between folds: memory check, cancellation, and (time mode) the stage boundary."""
        timed = self.options.tuning_mode == "time"

        def hook() -> None:
            self.monitor.check()
            self.check_cancel()
            if timed and self.budget.expired(stage):
                self.time_hit = True
                raise BudgetExceeded

        return hook

    def available_models(self) -> list[ModelName]:
        models = self.registry.available(self.config.execution_profile)
        if self.options.models:
            models = [m for m in models if m in self.options.models]
        if Simplification.DISABLE_RANDOM_FOREST in self.simplifications:
            models = [m for m in models if m != ModelName.RANDOM_FOREST]
        return models


_REGISTRY: dict[str, RunContext] = {}


def register(ctx: RunContext) -> RunContext:
    _REGISTRY[ctx.run_id] = ctx
    return ctx


def context(run_id: str) -> RunContext:
    """The live context of `run_id` (nodes receive only the small `RunState`)."""
    return _REGISTRY[run_id]


def release(run_id: str) -> None:
    _REGISTRY.pop(run_id, None)
