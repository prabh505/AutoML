"""Run lifecycle: run manager, worker, events, cancel, slot lock, cleanup (PRD §28, FR-RUN-*).

Interface: RunManager.start/cancel/status/cleanup. Depends on orchestration.

Must not: ML logic.
"""

from tabpilot.runtime.manager import (
    InvalidConfigError,
    RunManager,
    RunStatus,
    SlotBusyError,
    UnknownRunError,
    pid_exists,
)

__all__ = [
    "InvalidConfigError",
    "RunManager",
    "RunStatus",
    "SlotBusyError",
    "UnknownRunError",
    "pid_exists",
]
