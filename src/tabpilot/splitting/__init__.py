"""Holdout + folds (PRD §11, FR-SPL-01).

Interface: build_split -> SplitSpec; HoldoutGuard. Depends on schemas.

Must not: profiling, LLM; looking at feature values except the group column and target.
"""
