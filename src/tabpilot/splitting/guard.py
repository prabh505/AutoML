"""`HoldoutGuard`: the only way to obtain holdout rows (PRD §11.2 L-2, L-7; I-1; PI-14).

- `open(caller)` raises `HoldoutSealed` unless the frozen-state callable returns True.
- The holdout can be opened exactly once per run. The Finalizer scores both the selected
  pipeline and the reference baseline from that single access (PI-14). A second `open`
  raises `HoldoutAlreadyOpened`.
- Every call is logged: granted accesses in `access_log` (exactly one per run, T-LEAK-03),
  refused attempts in `refused_log` (DEC-027).

Only `tabpilot.evaluation.holdout`, `tabpilot.orchestration`, and the split builder may import
this module (import contract R2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray


class HoldoutSealed(RuntimeError):  # noqa: N818 - PRD name (§25.3)
    """The holdout was requested before selection was frozen."""


class HoldoutAlreadyOpened(RuntimeError):  # noqa: N818 - mirrors HoldoutSealed
    """The holdout was requested a second time."""


@dataclass(frozen=True)
class AccessEvent:
    """One call to `HoldoutGuard.open`."""

    caller: str
    ts: datetime
    granted: bool
    reason: str


class HoldoutGuard:
    """Holds HOLD row indices until selection is frozen; hands them out once."""

    def __init__(self, indices: NDArray[np.int64], state_ref: Callable[[], bool]) -> None:
        self._indices = np.array(indices, dtype=np.int64, copy=True)
        self._indices.setflags(write=False)
        self._frozen = state_ref
        self._access_log: list[AccessEvent] = []
        self._refused_log: list[AccessEvent] = []

    def __repr__(self) -> str:
        return f"HoldoutGuard(n={len(self._indices)}, opened={bool(self._access_log)})"

    @property
    def size(self) -> int:
        """Number of holdout rows. The size is not sensitive; the indices are."""
        return len(self._indices)

    def open(self, caller: str) -> NDArray[np.int64]:
        """Return the holdout row indices (read-only) once, after selection is frozen."""
        now = datetime.now(UTC)
        if not self._frozen():
            self._refused_log.append(AccessEvent(caller, now, False, "selection not frozen"))
            raise HoldoutSealed(f"{caller}: holdout is sealed until selection is frozen")
        if self._access_log:
            self._refused_log.append(AccessEvent(caller, now, False, "already opened"))
            raise HoldoutAlreadyOpened(f"{caller}: holdout was already opened by "
                                       f"{self._access_log[0].caller}")  # fmt: skip
        self._access_log.append(AccessEvent(caller, now, True, "granted"))
        return self._indices

    @property
    def access_log(self) -> list[AccessEvent]:
        """Granted accesses (length ≤ 1)."""
        return list(self._access_log)

    @property
    def refused_log(self) -> list[AccessEvent]:
        """Refused attempts (before freeze, or a second open)."""
        return list(self._refused_log)
