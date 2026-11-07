"""Controlled worker behaviours for `RunManager` lifecycle tests (P15).

Each function has the worker signature `(spec, out, cancel)` and is module-level, so the
`spawn` context can pickle it by name. They speak the real worker protocol (events, heartbeats,
result) but do no ML.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabpilot.runtime.worker import WorkerSpec
from tabpilot.schemas import (
    ErrorCode,
    EventType,
    RunConfig,
    RunEvent,
    RunResult,
    Stage,
)
from tabpilot.schemas.experiment import HoldoutResult
from tabpilot.schemas.run import FinalResult

BEAT_S = 0.2


def _event(
    spec: WorkerSpec, seq: int, kind: EventType, stage: Stage = Stage.RUN, **data: Any
) -> str:
    config = RunConfig.model_validate_json(spec.config_json)
    return RunEvent(run_id=config.run_id, seq=seq, ts=datetime.now(UTC), stage=stage, type=kind,
                    data=data).model_dump_json()  # fmt: skip


def _result(spec: WorkerSpec, status: str, **kw: Any) -> str:
    config = RunConfig.model_validate_json(spec.config_json)
    return RunResult(run_id=config.run_id, status=status, duration_s=0.1, **kw).model_dump_json()  # type: ignore[arg-type]


def _beat(out: Any) -> None:
    out.put(("heartbeat", {"rss_mb": 12.5, "elapsed_s": 0.1}))


def _loop(out: Any, cancel: Any, *, stop_on_cancel: bool) -> bool:
    """Heartbeat until cancelled (returns True) or forever (stubborn workers)."""
    while True:
        _beat(out)
        if stop_on_cancel and cancel.is_set():
            return True
        time.sleep(BEAT_S)


def quick_ok(spec: WorkerSpec, out: Any, cancel: Any) -> None:
    """Finish at once with a completed result and a bundle file in the run directory."""
    run_dir = Path(spec.run_dir)
    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    (run_dir / "bundle.zip").write_bytes(b"zip")
    out.put(("event", _event(spec, 0, EventType.STAGE_STARTED, Stage.INGEST)))
    out.put(("event", _event(spec, 1, EventType.RUN_COMPLETED)))
    final = FinalResult(selected_entry_id="0-linear-00000000", bundle_path="bundle.zip",
                        holdout=HoldoutResult(present=False, reason="test"))  # fmt: skip
    out.put(("result", _result(spec, "completed", final=final)))


def quick_fail(spec: WorkerSpec, out: Any, cancel: Any) -> None:
    """Finish at once with a failed result (E-NO-VALID-MODEL)."""
    (Path(spec.run_dir) / "input").mkdir(parents=True, exist_ok=True)
    out.put(("event", _event(spec, 0, EventType.RUN_FAILED, error_code="E-NO-VALID-MODEL")))
    out.put(("result", _result(spec, "failed", error_code=ErrorCode.NO_VALID_MODEL,
                               error_message="No model trained successfully.")))  # fmt: skip


def polite(spec: WorkerSpec, out: Any, cancel: Any) -> None:
    """Heartbeat; stop cooperatively when the cancel event is set."""
    out.put(("event", _event(spec, 0, EventType.STAGE_STARTED, Stage.TRAIN_CANDIDATES)))
    _loop(out, cancel, stop_on_cancel=True)
    out.put(("event", _event(spec, 1, EventType.RUN_CANCELLED)))
    out.put(("result", _result(spec, "cancelled", stopped_after_stage=Stage.TRAIN_CANDIDATES)))


def ignores_cancel(spec: WorkerSpec, out: Any, cancel: Any) -> None:
    """Ignores the cancel event but keeps the default SIGTERM behaviour (dies on SIGTERM)."""
    out.put(("event", _event(spec, 0, EventType.STAGE_STARTED, Stage.TRAIN_CANDIDATES)))
    _loop(out, cancel, stop_on_cancel=False)


def stubborn(spec: WorkerSpec, out: Any, cancel: Any) -> None:
    """Ignores the cancel event and SIGTERM: only SIGKILL stops it."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    out.put(("event", _event(spec, 0, EventType.STAGE_STARTED, Stage.TRAIN_CANDIDATES)))
    _loop(out, cancel, stop_on_cancel=False)


def crash(spec: WorkerSpec, out: Any, cancel: Any) -> None:
    """Emit one event, then die abruptly (like an OOM kill)."""
    out.put(("event", _event(spec, 0, EventType.STAGE_STARTED, Stage.TRAIN_CANDIDATES)))
    time.sleep(0.5)  # let the queue's feeder thread flush
    os._exit(9)


def silent(spec: WorkerSpec, out: Any, cancel: Any) -> None:
    """Alive but sending no heartbeats (a hung process)."""
    out.put(("event", _event(spec, 0, EventType.STAGE_STARTED, Stage.TRAIN_CANDIDATES)))
    time.sleep(120)


def spawner(spec: WorkerSpec, out: Any, cancel: Any) -> None:
    """Starts a child process (its pid is in an event), then heartbeats until killed."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    out.put(("event", _event(spec, 0, EventType.STAGE_STARTED, Stage.TRAIN_CANDIDATES,
                             child_pid=child.pid)))  # fmt: skip
    _loop(out, cancel, stop_on_cancel=False)
