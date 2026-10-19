"""Plans (PRD §12, FR-PLN-*).

Interface: make_plan(ctx) -> (PlannerOutput, PlanValidation, source). Depends on llm,
features.dsl.validate, resources.

Must not: training; executing anything; accessing DataFrames.
"""

from tabpilot.planning.fallback import rule_based_plan

__all__ = ["rule_based_plan"]
