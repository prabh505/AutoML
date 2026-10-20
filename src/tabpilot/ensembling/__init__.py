"""Cross-fitted hill-climbing ensembles (PRD §20, FR-ENS-01).

Interface: cross_fitted_ensemble(...). Depends on evaluation.

Must not: HoldoutGuard.
"""

from tabpilot.ensembling.ensemble import (
    Candidate,
    blend,
    build_ensemble,
    cross_fitted_ensemble,
    eligible,
    fold_partition,
    hill_climb,
    oof_fold_scores,
)

__all__ = [
    "Candidate",
    "blend",
    "build_ensemble",
    "cross_fitted_ensemble",
    "eligible",
    "fold_partition",
    "hill_climb",
    "oof_fold_scores",
]
