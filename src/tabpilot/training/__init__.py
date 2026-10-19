"""Cross-validation runner and OOF store (PRD §11.4, §16, FR-TRN-01).

Interface: run_cv(...) -> CVRun (its `.result` is the TrainingResult contract). Depends on
preprocessing, evaluation.metrics, resources.

Must not: HoldoutGuard.
"""

from tabpilot.training.cv import (
    BudgetExceeded,
    Cancelled,
    CancelToken,
    CVRun,
    DevData,
    OOFStore,
    entry_id_for,
    run_cv,
    sanitize_error,
)
from tabpilot.training.guards import rf_resource_guard

__all__ = [
    "BudgetExceeded",
    "CVRun",
    "CancelToken",
    "Cancelled",
    "DevData",
    "OOFStore",
    "entry_id_for",
    "rf_resource_guard",
    "run_cv",
    "sanitize_error",
]
