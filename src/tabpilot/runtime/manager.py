"""Run Manager: slot lock, worker spawn, event relay, cancellation, timeouts, cleanup
(PRD §6.3, §13.7, §24.7, §28; FR-RUN-01…03).

One worker per run, started with the `spawn` multiprocessing context (D-02 spike: a spawned
child starts, communicates over queues and is killable here). `mode="thread"` is the documented
fallback behind the same interface: cancellation is cooperative only and there is no hard kill.

Lifecycle of a run, as seen by the manager:

```
start → slot acquired → worker running ──result──► completed / completed_partial / failed
                          │  ├─ cancel(): event, then SIGTERM after 10 s, SIGKILL 5 s later
                          │  ├─ no heartbeat for 20 s, or the process dies → failed (E-WORKER-LOST)
                          │  └─ 130% of the time budget → SIGKILL → failed (E-WORKER-LOST)
                          └─ failed / cancelled → run directory deleted (§24.7)
```

The manager never interprets ML results: it relays events, enforces the process-level limits
and cleans up.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import psutil
from ulid import ULID

from tabpilot.config import Settings
from tabpilot.ingestion import StoredUpload, accept_upload
from tabpilot.logging import get_logger
from tabpilot.orchestration.context import WorkerOptions
from tabpilot.runtime.cleanup import (
    IDLE_TTL_S,
    MAX_AGE_S,
    SWEEP_INTERVAL_S,
    delete_run_dir,
    sweep,
)
from tabpilot.runtime.slots import Slot, SlotBusyError, acquire_slot
from tabpilot.runtime.worker import WorkerSpec, worker_main
from tabpilot.schemas import (
    ErrorCode,
    RunConfig,
    RunEvent,
    RunResult,
    Stage,
)
from tabpilot.schemas.errors import render_message
from tabpilot.schemas.run import MIN_TIME_BUDGET_S

log = get_logger("run_manager")

HEARTBEAT_TIMEOUT_S = 20.0
CANCEL_GRACE_S = 10.0
TERM_GRACE_S = 5.0
HARD_KILL_FACTOR = 1.3
POLL_S = 0.2
Mode = str  # "process" | "thread"

__all__ = [
    "InvalidConfigError",
    "RunManager",
    "RunStatus",
    "SlotBusyError",
    "UnknownRunError",
    "pid_exists",
]


class InvalidConfigError(ValueError):
    """`RunConfig` is outside the active profile's limits (§28.2)."""


class UnknownRunError(KeyError):
    """No run with this id is known to the manager (§28.2)."""


@dataclass(frozen=True)
class RunStatus:
    """What the UI shows for a run."""

    run_id: str
    state: str  # running | completed | completed_partial | failed | cancelled | deleted
    last_heartbeat_s: float | None = None
    rss_mb: float | None = None
    elapsed_s: float | None = None


@dataclass
class _Run:
    config: RunConfig
    run_dir: Path
    slot: Slot
    started: float
    events: list[RunEvent] = field(default_factory=list)
    result: RunResult | None = None
    process: Any = None
    thread: threading.Thread | None = None
    cancel: Any = None
    out: Any = None
    monitor: threading.Thread | None = None
    last_heartbeat: float = 0.0
    rss_mb: float | None = None
    worker_elapsed_s: float | None = None
    cancel_requested_at: float | None = None
    finished_at: float | None = None
    deleted: bool = False
    pid: int | None = None
    exit_code: int | None = None  # negative: killed by that signal (e.g. -15 SIGTERM, -9 SIGKILL)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RunManager:
    """Starts, watches, cancels and cleans up run workers (§28.2)."""

    def __init__(
        self,
        root: Path,
        settings: Settings,
        *,
        mode: Mode = "process",
        heartbeat_timeout_s: float = HEARTBEAT_TIMEOUT_S,
        cancel_grace_s: float = CANCEL_GRACE_S,
        term_grace_s: float = TERM_GRACE_S,
        hard_kill_factor: float = HARD_KILL_FACTOR,
        delete_failed: bool = True,
        worker: Callable[..., None] = worker_main,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        if mode not in ("process", "thread"):
            raise ValueError(f"unknown run mode {mode!r}")
        self.root = root
        self.settings = settings
        self.mode = mode
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.cancel_grace_s = cancel_grace_s
        self.term_grace_s = term_grace_s
        self.hard_kill_factor = hard_kill_factor
        self.delete_failed = delete_failed  # False: the CLI keeps a failed run's records
        self._worker = worker  # replaceable by tests (must be picklable for `spawn`)
        self._clock = clock
        self._wall = wall
        self._runs: dict[str, _Run] = {}
        self._errors: dict[str, RunResult] = {}  # sanitized records of deleted failed runs
        self._lock = threading.Lock()
        self._ctx = mp.get_context("spawn")
        self._sweeper: threading.Timer | None = None
        root.mkdir(parents=True, exist_ok=True, mode=0o700)

    # ------------------------------------------------------------ validation helpers
    def validate_upload(self, file: BinaryIO, filename: str) -> StoredUpload:
        """§28.2: validate and store an upload under a new run id (E-FILE-*)."""
        settings = self.settings.model_copy(update={"data_dir": self.root.parent})
        stored = accept_upload(file, filename, settings, run_id=str(ULID()))
        return stored

    def check_config(self, cfg: RunConfig) -> None:
        """Validate `cfg` against the active profile's limits (raises `InvalidConfigError`)."""
        limits = self.settings.limits
        if cfg.execution_profile != self.settings.profile:
            raise InvalidConfigError(
                f"execution profile {cfg.execution_profile} does not match the running "
                f"profile {self.settings.profile}")  # fmt: skip
        if not MIN_TIME_BUDGET_S <= cfg.time_budget_s <= limits.time_budget_max_s:
            raise InvalidConfigError(
                f"time budget {cfg.time_budget_s} s is outside {MIN_TIME_BUDGET_S}–"
                f"{limits.time_budget_max_s} s")  # fmt: skip

    # ------------------------------------------------------------ start
    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def start(
        self,
        cfg: RunConfig,
        *,
        input_path: Path | None = None,
        display_name: str | None = None,
        options: WorkerOptions | None = None,
        run_dir: Path | None = None,
    ) -> str:
        """§28.2: acquire a slot and spawn the worker. A second `start` for a known run id
        returns that id without starting anything (idempotency).
        """
        with self._lock:
            if cfg.run_id in self._runs:
                return cfg.run_id
            self.check_config(cfg)
            directory = run_dir or self.run_dir(cfg.run_id)
            source = input_path or self._stored_input(directory)
            slot = acquire_slot(self.root, self.settings.limits.max_concurrent_runs)
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                run = _Run(cfg, directory, slot, started=self._clock())
                self._spawn(run, source, display_name or source.name, options or WorkerOptions())
            except BaseException:
                slot.release()
                raise
            self._runs[cfg.run_id] = run
        run.monitor = threading.Thread(target=self._watch, args=(run,), daemon=True,
                                       name=f"run-{cfg.run_id[-6:]}")  # fmt: skip
        run.monitor.start()
        return cfg.run_id

    @staticmethod
    def _stored_input(directory: Path) -> Path:
        found = sorted((directory / "input").glob("raw.*"))
        if not found:
            raise InvalidConfigError("the run has no stored upload")
        return found[0]

    def _spawn(self, run: _Run, source: Path, display_name: str, options: WorkerOptions) -> None:
        spec = WorkerSpec(config_json=run.config.model_dump_json(), run_dir=str(run.run_dir),
                          input_path=str(source), display_name=display_name,
                          options=options)  # fmt: skip
        run.last_heartbeat = self._clock()
        if self.mode == "process":
            run.out = self._ctx.Queue()
            run.cancel = self._ctx.Event()
            run.process = self._ctx.Process(
                target=self._worker,
                args=(spec, run.out, run.cancel),
                name=f"worker-{run.config.run_id[-6:]}",
                daemon=True,
            )
            run.process.start()
            run.pid = run.process.pid
        else:
            run.out = queue.Queue()
            run.cancel = threading.Event()
            run.thread = threading.Thread(
                target=self._worker,
                args=(spec, run.out, run.cancel),
                daemon=True,
                name=f"worker-{run.config.run_id[-6:]}",
            )
            run.thread.start()

    # ------------------------------------------------------------ monitoring
    def _alive(self, run: _Run) -> bool:
        if run.process is not None:
            return bool(run.process.is_alive())
        return run.thread is not None and run.thread.is_alive()

    def _drain(self, run: _Run) -> None:
        while True:
            try:
                kind, payload = run.out.get_nowait()
            except (queue.Empty, OSError, EOFError):
                return
            if kind == "event":
                run.events.append(RunEvent.model_validate_json(payload))
            elif kind == "heartbeat":
                run.last_heartbeat = self._clock()
                run.rss_mb = payload.get("rss_mb")
                run.worker_elapsed_s = payload.get("elapsed_s")
            elif kind == "result":
                run.result = RunResult.model_validate_json(payload)

    def _watch(self, run: _Run) -> None:
        """Relay messages and enforce heartbeat, cancel and hard-kill limits until the run ends."""
        budget = float(run.config.time_budget_s)
        while True:
            self._drain(run)
            now = self._clock()
            if run.result is not None:
                break
            if not self._alive(run):
                self._drain(run)  # the result may have arrived just before exit
                if run.result is None:
                    self._lost(run, "the worker exited without a result")
                break
            if run.cancel_requested_at is not None:
                self._escalate(run, now)
            if now - run.started > self.hard_kill_factor * budget:
                self._kill(run)
                self._lost(run, "hard wall-clock limit (130%) reached")
                break
            if now - run.last_heartbeat > self.heartbeat_timeout_s:
                self._kill(run)
                self._lost(run, "no heartbeat")
                break
            time.sleep(POLL_S)
        self._finish(run)

    def _lost(self, run: _Run, why: str) -> None:
        log.warning("worker_lost", run_id=run.config.run_id, why=why)
        duration = round(self._clock() - run.started, 3)
        run.result = RunResult(
            run_id=run.config.run_id,
            status="failed",
            duration_s=duration,
            error_code=ErrorCode.WORKER_LOST,
            error_message=render_message(ErrorCode.WORKER_LOST, {}),
            stopped_after_stage=self._last_stage(run),
        )

    @staticmethod
    def _last_stage(run: _Run) -> Stage | None:
        return next((e.stage for e in reversed(run.events)), None)

    def _escalate(self, run: _Run, now: float) -> None:
        """§13.7: cooperative first; SIGTERM after 10 s; SIGKILL 5 s later."""
        assert run.cancel_requested_at is not None  # noqa: S101
        waited = now - run.cancel_requested_at
        if run.process is None:
            return  # thread mode: cooperative only
        if waited > self.cancel_grace_s + self.term_grace_s:
            self._kill(run)
        elif waited > self.cancel_grace_s:
            run.process.terminate()

    def _kill(self, run: _Run) -> None:
        """SIGKILL the worker and every child process it started."""
        if run.process is None or run.pid is None:
            return
        try:
            parent = psutil.Process(run.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        run.process.join(timeout=5)

    def _finish(self, run: _Run) -> None:
        """Release the slot, join the worker, and apply the §24.7 cleanup for this outcome."""
        if run.process is not None:
            run.process.join(timeout=self.term_grace_s)
            if run.process.is_alive():
                self._kill(run)
            run.exit_code = run.process.exitcode
            run.process.close()
            run.process = None
        run.slot.release()
        result = run.result
        assert result is not None  # noqa: S101
        run.finished_at = self._wall()
        if result.status in ("failed", "cancelled"):
            self._errors[run.config.run_id] = result
            if self.delete_failed:
                delete_run_dir(run.run_dir)

    # ------------------------------------------------------------ public API (§28.2)
    def events(self, run_id: str, since_seq: int = 0) -> list[RunEvent]:
        """Events with `seq >= since_seq` (events survive the deletion of a failed run)."""
        return [e for e in list(self._get(run_id).events) if e.seq >= since_seq]

    def worker_exit_code(self, run_id: str) -> int | None:
        """How the worker ended: 0 clean, positive = exit status, negative = killed by signal."""
        return self._get(run_id).exit_code

    def worker_pid(self, run_id: str) -> int | None:
        """PID of the worker process (None in thread mode); tests check it is gone after cancel."""
        return self._get(run_id).pid

    def cancel(self, run_id: str) -> None:
        """Request cancellation; repeated calls are a no-op. Blocks up to 15 s for the stop."""
        run = self._get(run_id)
        with run.lock:
            if run.result is not None or run.cancel_requested_at is not None:
                return
            run.cancel_requested_at = self._clock()
            run.cancel.set()
        self.wait(run_id, timeout=self.cancel_grace_s + self.term_grace_s + 5.0)

    def wait(self, run_id: str, timeout: float | None = None) -> RunResult | None:
        """Block until the run has finished (monitor done) or `timeout` seconds pass."""
        run = self._get(run_id)
        if run.monitor is not None:
            run.monitor.join(timeout)
        return run.result

    def result(self, run_id: str) -> RunResult | None:
        run = self._get(run_id)
        return run.result if run.monitor is None or not run.monitor.is_alive() else None

    def status(self, run_id: str) -> RunStatus:
        run = self._get(run_id)
        if run.deleted:
            return RunStatus(run_id, "deleted")
        running = run.monitor is not None and run.monitor.is_alive()
        state = "running" if running or run.result is None else run.result.status
        return RunStatus(run_id, state, round(self._clock() - run.last_heartbeat, 1) if running
                         else None, run.rss_mb, run.worker_elapsed_s)  # fmt: skip

    def delete(self, run_id: str) -> None:
        """Delete the run directory now (§24.7 "Delete now"); repeated delete is a no-op."""
        run = self._runs.get(run_id)
        if run is None:
            if run_id in self._errors:
                return
            raise UnknownRunError(run_id)
        if run.monitor is not None and run.monitor.is_alive():
            self.cancel(run_id)
        delete_run_dir(run.run_dir)
        run.deleted = True

    def _get(self, run_id: str) -> _Run:
        try:
            return self._runs[run_id]
        except KeyError:
            raise UnknownRunError(run_id) from None

    # ------------------------------------------------------------ sweeping (§24.7)
    def reap_idle(self) -> list[str]:
        """Delete finished runs idle longer than the profile's TTL (hosted 30 min)."""
        ttl = IDLE_TTL_S[self.settings.profile]
        if ttl is None:
            return []
        gone = []
        for run_id, run in list(self._runs.items()):
            done = run.finished_at is not None and not run.deleted
            if done and self._wall() - run.finished_at > ttl:  # type: ignore[operator]
                delete_run_dir(run.run_dir)
                run.deleted = True
                gone.append(run_id)
        return gone

    def sweep(self) -> list[str]:
        """Delete stale run directories (hosted 2 h / local 7 days); skips active runs."""
        active = {r.run_dir.name for r in self._runs.values()
                  if r.monitor is not None and r.monitor.is_alive()}  # fmt: skip
        return sweep(self.root, MAX_AGE_S[self.settings.profile], now=self._wall, skip=active)

    def start_sweeper(self) -> None:
        """Sweep now and every 10 minutes (on app start, §24.7)."""

        def tick() -> None:
            self.reap_idle()
            self.sweep()
            self._sweeper = threading.Timer(SWEEP_INTERVAL_S, tick)
            self._sweeper.daemon = True
            self._sweeper.start()

        tick()

    def shutdown(self) -> None:
        """Stop the sweeper and any running workers (no orphan processes)."""
        if self._sweeper is not None:
            self._sweeper.cancel()
        for run_id, run in list(self._runs.items()):
            if run.monitor is not None and run.monitor.is_alive():
                self.cancel(run_id)


def pid_exists(pid: int | None) -> bool:
    """Whether `pid` is a live (non-zombie) process; used by tests and the sweeper."""
    if pid is None or not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
