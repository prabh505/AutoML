"""Cross-validation runner and OOF store (PRD §11.4, §16, FR-TRN-01).

Interface: run_cv(...) -> TrainingResult. Depends on preprocessing, evaluation.metrics,
resources.

Must not: HoldoutGuard.
"""
