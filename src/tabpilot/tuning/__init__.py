"""Optuna hyperparameter optimization (PRD §17, FR-TUN-01).

Interface: tune(...). Depends on training.

Must not: HoldoutGuard.
"""

from tabpilot.tuning.space import SPACES
from tabpilot.tuning.tune import (
    TrialRecord,
    TuningLimits,
    TuningMode,
    TuningResult,
    limits_for,
    tune,
)

__all__ = [
    "SPACES",
    "TrialRecord",
    "TuningLimits",
    "TuningMode",
    "TuningResult",
    "limits_for",
    "tune",
]
