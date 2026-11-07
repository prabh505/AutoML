"""Run-directory cleanup lifecycle (PRD §24.7, FR-RUN-02; T-SEC-06, T-SEC-07).

| Event                         | Action |
| completed / partial           | raw input deleted at export (worker); bundle + report kept |
| failed / cancelled            | whole run directory deleted (within 15 s) |
| "Delete now"                  | whole run directory deleted |
| idle TTL (hosted 30 min)      | finished run directory deleted |
| sweeper (start, every 10 min) | run directories older than 2 h hosted / 7 days local |
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Collection
from pathlib import Path

from tabpilot.logging import get_logger
from tabpilot.schemas import ExecutionProfileName

log = get_logger("cleanup")

SWEEP_INTERVAL_S = 600.0
IDLE_TTL_S = {ExecutionProfileName.HOSTED: 30 * 60.0, ExecutionProfileName.LOCAL: None}
MAX_AGE_S = {ExecutionProfileName.HOSTED: 2 * 3600.0, ExecutionProfileName.LOCAL: 7 * 86400.0}
RESERVED = {".locks"}


def delete_run_dir(path: Path) -> bool:
    """`shutil.rmtree` with error logging; returns False when something is left behind (the
    sweeper retries, §24.7). Names only are logged, never file contents (I-11).
    """

    def on_error(_func: Callable[..., object], failed: str, _exc: object) -> None:
        log.warning("cleanup_delete_failed", path=Path(failed).name)

    if path.exists():
        shutil.rmtree(path, onerror=on_error)
    return not path.exists()


def sweep(
    root: Path,
    max_age_s: float,
    *,
    now: Callable[[], float] = time.time,
    skip: Collection[str] = (),
) -> list[str]:
    """Delete run directories whose last modification is older than `max_age_s`; directories
    named in `skip` (runs that are still active) are never touched.
    """
    removed: list[str] = []
    if not root.is_dir():
        return removed
    cutoff = now() - max_age_s
    for child in sorted(root.iterdir()):
        if child.name in RESERVED or child.name in skip or not child.is_dir():
            continue
        try:
            stale = child.stat().st_mtime < cutoff
        except OSError:
            continue
        if stale and delete_run_dir(child):
            removed.append(child.name)
    if removed:
        log.info("sweeper_removed", count=len(removed))
    return removed
