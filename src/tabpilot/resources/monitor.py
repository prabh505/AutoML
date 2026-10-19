"""RSS resource monitor (PRD §29.6).

A daemon thread samples the process RSS (including child processes) every `interval_s`. Above the
soft limit it records a warning, calls `gc.collect()`, and sets `soft_exceeded` so the caller can
skip optional stages after the current fold. Above the hard limit it raises a pending abort that
`check()` delivers as `MemoryLimitExceeded` at the next fold boundary; the CV runner then marks
the candidate `failed: memory`. After two hard-limit aborts `stop_training` is set.
"""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import psutil

BYTES_PER_MB = 2**20
RECORD_EVERY_S = 5.0
MAX_HARD_STRIKES = 2


class MemoryLimitExceeded(Exception):  # noqa: N818 - mirrors BudgetExceeded
    """RSS crossed the hard limit while a candidate was running (§29.6)."""


def process_rss_mb() -> float:
    """RSS of this process and its children, in MB."""
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            continue
    return total / BYTES_PER_MB


@dataclass
class Sample:
    """One RSS reading, seconds since the monitor started."""

    t: float
    rss_mb: float


@dataclass
class ResourceMonitor:
    """Soft/hard RSS limits with cooperative candidate abort (§29.6)."""

    soft_mb: float
    hard_mb: float
    interval_s: float = 1.0
    rss_fn: Callable[[], float] = process_rss_mb
    clock: Callable[[], float] = time.monotonic
    samples: list[Sample] = field(default_factory=list)
    peak_mb: float = 0.0
    soft_exceeded: bool = False
    hard_strikes: int = 0
    _pending_abort: bool = False
    _last_record: float = float("-inf")
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _started: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.soft_mb < self.hard_mb:
            raise ValueError("need 0 < soft_mb < hard_mb")
        self._started = self.clock()

    @property
    def stop_training(self) -> bool:
        """Two hard-limit aborts: stop training and finalize with the best valid result."""
        return self.hard_strikes >= MAX_HARD_STRIKES

    def sample(self) -> float:
        """Take one reading and apply the limits. Called by the thread; callable directly."""
        rss = self.rss_fn()
        now = self.clock() - self._started
        self.peak_mb = max(self.peak_mb, rss)
        if now - self._last_record >= RECORD_EVERY_S:
            self.samples.append(Sample(round(now, 3), round(rss, 1)))
            self._last_record = now
        if rss >= self.hard_mb:
            self._pending_abort = True
        elif rss >= self.soft_mb and not self.soft_exceeded:
            self.soft_exceeded = True
            gc.collect()
        return rss

    def check(self) -> None:
        """Deliver a pending hard-limit abort (fold boundary). Frees memory first."""
        self.sample()
        if self._pending_abort:
            self._pending_abort = False
            self.hard_strikes += 1
            gc.collect()
            raise MemoryLimitExceeded(f"RSS above the {self.hard_mb:.0f} MB hard limit")

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def start(self) -> ResourceMonitor:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rss-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
            self._thread = None
