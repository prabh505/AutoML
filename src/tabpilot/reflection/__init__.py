"""Reflection proposals and paired-fold acceptance (PRD §19, FR-RFL-01).

Interface: reflect(...), accept(...). Depends on llm, planning (validator).

Must not: HoldoutGuard; exceeding round/time/trial budgets.
"""

from tabpilot.reflection.accept import MIN_REL_GAIN, accept, paired_decision
from tabpilot.reflection.budget import RoundBudget, RoundOutcome, reflection_budget, should_stop
from tabpilot.reflection.input import (
    build_reflection_input,
    raw_importances,
    source_column,
    source_importances,
)
from tabpilot.reflection.propose import (
    ChangeSet,
    ReflectionContext,
    apply_changes,
    change_set_id,
    propose,
    retrain_parents,
    validate_proposal,
)
from tabpilot.reflection.templates import load, versions

__all__ = [
    "MIN_REL_GAIN",
    "ChangeSet",
    "ReflectionContext",
    "RoundBudget",
    "RoundOutcome",
    "accept",
    "apply_changes",
    "build_reflection_input",
    "change_set_id",
    "load",
    "paired_decision",
    "propose",
    "raw_importances",
    "reflection_budget",
    "retrain_parents",
    "should_stop",
    "source_column",
    "source_importances",
    "validate_proposal",
    "versions",
]
