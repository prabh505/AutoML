"""Metrics, leaderboard, selection, holdout evaluation (PRD §10, §18, FR-EVL-*).

Interface: select(...), evaluate_holdout(final, guard). Depends on schemas.

Must not: llm; accepting scores from LLM output.

Submodules are imported directly (`tabpilot.evaluation.selection`, …) so that importing
`tabpilot.evaluation.metrics` from training does not pull in the holdout guard.
"""
