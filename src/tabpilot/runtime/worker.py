"""Run worker: one process per run, started with the `spawn` context (PRD §6.1, §28.3).

Protocol (worker → manager, over a queue of `(kind, payload)` tuples):

- `("event", RunEvent JSON)` after each completed action (§27.4);
- `("heartbeat", {"rss_mb", "elapsed_s"})` every 2 s from a daemon thread;
- `("preview", canonical planner-input JSON)` in preview mode;
- `("result", RunResult JSON)` exactly once, last.

The worker reads its data from the run directory only (§7.1). Cancellation is cooperative
(checked between folds, trials and nodes); the manager escalates with SIGTERM/SIGKILL.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import psutil

from tabpilot.llm import BudgetTracker, LLMRouter
from tabpilot.llm.consent import ConsentStore
from tabpilot.llm.factory import build_router, default_budget
from tabpilot.llm.records import LLMCallRecorder
from tabpilot.llm.stub import StubConsent, StubProvider
from tabpilot.orchestration.context import RunContext, WorkerOptions, register, release
from tabpilot.orchestration.graph import run_graph
from tabpilot.orchestration.prepare import prepare
from tabpilot.orchestration.records import write_records
from tabpilot.planning import preview_planner_input
from tabpilot.schemas import (
    ErrorCode,
    EventType,
    RunConfig,
    RunResult,
    RunState,
    Stage,
    StageError,
    TabPilotError,
)
from tabpilot.schemas.errors import render_message
from tabpilot.training import Cancelled

HEARTBEAT_S = 2.0
WorkerMode = Literal["run", "preview"]


class Outbox(Protocol):
    def put(self, item: Any) -> None: ...


class Flag(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class WorkerSpec:
    """Everything a worker needs; picklable for the `spawn` context."""

    config_json: str
    run_dir: str
    input_path: str
    display_name: str
    options: WorkerOptions
    mode: WorkerMode = "run"


def _router(ctx: RunContext) -> tuple[LLMRouter | None, BudgetTracker | None]:
    """The run's LLM router: none (`off`), the offline stub (`stub`), or the consented
    providers (`providers`). Router events are recorded under the current stage.
    """
    cfg = ctx.config
    if cfg.llm_mode == "off":
        return None, None
    budget = BudgetTracker(default_budget(cfg.time_budget_s, ctx.settings.limits.llm_calls_per_run))

    def events(kind: EventType, data: dict[str, Any]) -> None:
        ctx.emit(ctx.current_stage, kind, **data)

    records = ctx.run_dir / "records"
    if cfg.llm_mode == "stub":
        router = LLMRouter([StubProvider(ctx.options.llm_stub_mode)], consent=StubConsent([]),
                           recorder=LLMCallRecorder(records), events=events,
                           sleep=lambda _s: None)  # fmt: skip
        return router, budget
    consent = ConsentStore.granted_for(list(cfg.consented_providers))
    return build_router(ctx.settings, os.environ, consent=consent, records_dir=records,
                        events=events), budget  # fmt: skip


def _heartbeat(out: Outbox, stop: threading.Event, started: float) -> None:
    process = psutil.Process()
    while not stop.wait(HEARTBEAT_S):
        out.put(("heartbeat", {"rss_mb": round(process.memory_info().rss / 2**20, 1),
                               "elapsed_s": round(time.monotonic() - started, 1)}))  # fmt: skip


def _delete_inputs(run_dir: Path) -> None:
    """§24.7: raw input and the normalized copy are removed right after export."""
    shutil.rmtree(run_dir / "input", ignore_errors=True)


def _failed(config: RunConfig, started: float, code: ErrorCode, message: str,
            stage: Stage | None) -> RunResult:  # fmt: skip
    return RunResult(run_id=config.run_id, status="failed",
                     duration_s=round(time.monotonic() - started, 3), error_code=code,
                     error_message=message[:1000], stopped_after_stage=stage)  # fmt: skip


def execute_run(spec: WorkerSpec, out: Outbox, cancel: Flag) -> RunResult:
    """Run (or preview) one run; never raises — every outcome becomes a `RunResult`."""
    started = time.monotonic()
    config = RunConfig.model_validate_json(spec.config_json)
    ctx = RunContext.create(config, run_dir=Path(spec.run_dir), input_path=Path(spec.input_path),
                            display_name=spec.display_name, options=spec.options,
                            cancel=cancel)  # fmt: skip
    ctx.listeners.append(lambda event: out.put(("event", event.model_dump_json())))
    register(ctx)
    ctx.monitor.start()
    state: RunState | None = None
    try:
        ctx.router, ctx.llm_budget = _router(ctx)
        ctx.emit(Stage.RUN, EventType.STAGE_STARTED, profile=config.execution_profile)
        state = prepare(ctx)
        if spec.mode == "preview":
            out.put(("preview", preview_planner_input(ctx.planner_input)))
            return RunResult(run_id=config.run_id, status="cancelled",
                             duration_s=round(time.monotonic() - started, 3))  # fmt: skip
        state = run_graph(state)
        if ctx.failure is not None or state.status == "failed":
            failure = ctx.failure or TabPilotError(ErrorCode.NO_VALID_MODEL, reasons="unknown")
            return _failed(config, started, failure.code, failure.user_message,
                           ctx.current_stage)  # fmt: skip
        write_records(state, ctx)
        _delete_inputs(ctx.run_dir)
        partial = state.status == "completed_partial"
        ctx.emit(Stage.RUN, EventType.RUN_COMPLETED, status=state.status,
                 winner=state.best_candidate_id)  # fmt: skip
        return RunResult(run_id=config.run_id, status="completed_partial" if partial
                         else "completed", duration_s=round(time.monotonic() - started, 3),
                         stopped_after_stage=Stage.TRAIN_CANDIDATES if partial else None,
                         final=state.final)  # fmt: skip
    except Cancelled:
        ctx.emit(Stage.RUN, EventType.RUN_CANCELLED, at_stage=ctx.current_stage.value)
        return RunResult(run_id=config.run_id, status="cancelled",
                         duration_s=round(time.monotonic() - started, 3),
                         stopped_after_stage=ctx.current_stage)  # fmt: skip
    except TabPilotError as exc:
        ctx.emit(Stage.RUN, EventType.RUN_FAILED, error_code=exc.code.value)
        return _failed(config, started, exc.code, exc.user_message, ctx.current_stage)
    except Exception as exc:
        error = StageError(stage=ctx.current_stage, error_class=type(exc).__name__[:100],
                           message=type(exc).__name__, recoverable=False)  # fmt: skip
        ctx.emit(Stage.RUN, EventType.RUN_FAILED, error_code=ErrorCode.WORKER_LOST.value,
                 error_class=error.error_class, at_stage=error.stage.value)  # fmt: skip
        return _failed(config, started, ErrorCode.WORKER_LOST,
                       render_message(ErrorCode.WORKER_LOST, {}), ctx.current_stage)  # fmt: skip
    finally:
        ctx.monitor.stop()
        release(config.run_id)


def worker_main(spec: WorkerSpec, out: Outbox, cancel: Flag) -> None:
    """Process entrypoint: heartbeats, the run, then exactly one result message."""
    stop = threading.Event()
    beat = threading.Thread(target=_heartbeat, args=(out, stop, time.monotonic()), daemon=True)
    beat.start()
    try:
        result = execute_run(spec, out, cancel)
    finally:
        stop.set()
    out.put(("result", result.model_dump_json()))
