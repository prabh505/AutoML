"""Plans (PRD §12, FR-PLN-*).

Interface: make_plan(ctx) -> (PlannerOutput, PlanValidation, source). Depends on llm,
features.dsl.validate, resources.

Must not: training; executing anything; accessing DataFrames.
"""

from tabpilot.planning.fallback import rule_based_plan
from tabpilot.planning.input import build_planner_input, planner_payload, preview_planner_input
from tabpilot.planning.planner import PlannerContext, make_plan, plan_issue_summary, repair_errors
from tabpilot.planning.templates import compact_schema, load, versions
from tabpilot.planning.validate import PlanContext, parse_plan, validate_plan

__all__ = [
    "PlanContext",
    "PlannerContext",
    "build_planner_input",
    "compact_schema",
    "load",
    "make_plan",
    "parse_plan",
    "plan_issue_summary",
    "planner_payload",
    "preview_planner_input",
    "repair_errors",
    "rule_based_plan",
    "validate_plan",
    "versions",
]
