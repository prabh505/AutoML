"""Holdout + folds (PRD §11, FR-SPL-01).

Interface: build_split -> SplitSpec; HoldoutGuard. Depends on schemas.

Must not: profiling, LLM; looking at feature values except the group column and target.
"""

from tabpilot.splitting.strategy import SplitResult, average_repeats, build_split, iter_folds

# HoldoutGuard is deliberately not re-exported: import it from tabpilot.splitting.guard, which
# only the finalizer path may do (import contract R2).
__all__ = ["SplitResult", "average_repeats", "build_split", "iter_folds"]
