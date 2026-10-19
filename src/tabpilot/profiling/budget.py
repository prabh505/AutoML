"""Cooperative time caps for profiling (PRD §9.5; A-004; PI-27).

Each statistic group runs under a cap, checked between columns. When a group runs out of time,
the remaining columns get `None` for that statistic and the profile records the group in
`truncated_stats`. The clock is injectable so tests can force timeouts.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tabpilot.schemas import ExecutionProfileName

GROUP_CAP_S = {ExecutionProfileName.HOSTED: 20.0, ExecutionProfileName.LOCAL: 120.0}  # A-004
LEAKAGE_CAP_S = 60.0  # A-004 (§9.2: 30 s × 2)
MIN_TOTAL_S = 60.0  # A-004: max(5% of budget, 60 s)
TOTAL_SHARE = 0.05


@dataclass
class ProfileBudget:
    """Total and per-group time caps with a monotonic clock."""

    total_s: float
    group_cap_s: float
    leakage_cap_s: float = LEAKAGE_CAP_S
    clock: Callable[[], float] = time.monotonic
    truncated: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._start = self.clock()

    @classmethod
    def for_run(
        cls, profile: ExecutionProfileName, time_budget_s: float, n_groups: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> ProfileBudget:  # fmt: skip
        """Caps for a run: total = max(5% budget, 60 s); group caps scaled to fit (PI-27)."""
        total = max(TOTAL_SHARE * time_budget_s, MIN_TOTAL_S)
        group = min(GROUP_CAP_S[profile], total / max(n_groups, 1))
        return cls(total_s=total, group_cap_s=group, leakage_cap_s=min(LEAKAGE_CAP_S, total),
                   clock=clock)  # fmt: skip

    def remaining(self) -> float:
        """Seconds left in the total profiling budget."""
        return self.total_s - (self.clock() - self._start)

    def group(self, name: str, cap_s: float | None = None) -> GroupTimer:
        """Start a timer for one statistic group."""
        return GroupTimer(self, name, min(cap_s or self.group_cap_s, max(self.remaining(), 0.0)))


@dataclass
class GroupTimer:
    """Cooperative timer for one group; `expired()` is checked between columns."""

    budget: ProfileBudget
    name: str
    cap_s: float

    def __post_init__(self) -> None:
        self._start = self.budget.clock()
        self._reported = False

    def expired(self) -> bool:
        """True once the cap is exceeded. The first expiry is recorded in `truncated_stats`."""
        over = self.budget.clock() - self._start > self.cap_s
        if over and not self._reported:
            self.budget.truncated.append(self.name)
            self._reported = True
        return over
