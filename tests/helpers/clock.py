"""Fake clock and sleeper for retry, backoff, and budget tests (no real waiting)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeClock:
    """Monotonic fake time. `sleep` advances time and records the requested duration."""

    now: float = 1_000.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot sleep a negative duration")
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds
