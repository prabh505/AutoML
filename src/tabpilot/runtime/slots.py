"""Run slots with an OS file lock (PRD FR-RUN-03, §28.2 `SlotBusy`).

At most `limits.max_concurrent_runs` runs are active per app instance. Each slot is a lock file
under the runs root held with `flock`: the lock is released when the holder closes it, and the
OS releases it if the holder process dies, so a crashed app never leaves a slot stuck.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path

LOCK_DIR = ".locks"


class SlotBusyError(RuntimeError):
    """Another analysis holds every slot (FR-RUN-03)."""

    def __init__(self, max_runs: int) -> None:
        super().__init__(f"Another analysis is running ({max_runs} concurrent run(s) allowed).")
        self.max_runs = max_runs


@dataclass
class Slot:
    """A held slot; `release()` is idempotent."""

    index: int
    _fd: int | None

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


def acquire_slot(root: Path, max_runs: int) -> Slot:
    """Take the first free slot or raise `SlotBusyError` (never blocks)."""
    lock_dir = root / LOCK_DIR
    lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for index in range(max_runs):
        fd = os.open(lock_dir / f"slot{index}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        return Slot(index, fd)
    raise SlotBusyError(max_runs)


def slots_in_use(root: Path, max_runs: int) -> int:
    """How many slots are currently held (by any process), for the wait notice (§27.1)."""
    busy = 0
    for index in range(max_runs):
        path = root / LOCK_DIR / f"slot{index}.lock"
        if not path.exists():
            continue
        fd = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            busy += 1
        finally:
            os.close(fd)
    return busy
