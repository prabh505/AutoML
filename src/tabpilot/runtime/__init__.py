"""Run lifecycle: run manager, worker, events, cancel, slot lock, cleanup (PRD §28, FR-RUN-*).

Interface: RunManager.start/cancel/status/cleanup. Depends on orchestration.

Must not: ML logic.
"""

from tabpilot.runtime.manager import (
    InvalidConfigError,
    RunManager,
    RunStatus,
    SlotBusyError,
    SlotNotice,
    TargetChoice,
    UnknownRunError,
    UploadValidation,
    pid_exists,
)
from tabpilot.runtime.results import Bundle, Figure, RunArtifacts, read_artifacts

__all__ = [
    "Bundle",
    "Figure",
    "InvalidConfigError",
    "RunArtifacts",
    "RunManager",
    "RunStatus",
    "SlotBusyError",
    "SlotNotice",
    "TargetChoice",
    "UnknownRunError",
    "UploadValidation",
    "pid_exists",
    "read_artifacts",
]
