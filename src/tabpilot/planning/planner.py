"""LLM planner with one repair call and a guaranteed fallback (PRD §12.4–§12.9; FR-PLN-01/03).

```
plan call (temperature 0) → validate
  unparseable / schema-invalid → one repair call with field-path errors (≤ 1,500 chars) → validate
  still unusable, semantically invalid, or any router failure → rule-based plan (§12.8)
```

At most two planner LLM calls per run (§12.5). Every path returns a validated plan (I-8).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from tabpilot.llm import BudgetTracker, LLMRouter, LLMUnavailable
from tabpilot.logging import get_logger
from tabpilot.planning.fallback import rule_based_plan
from tabpilot.planning.input import planner_payload
from tabpilot.planning.templates import PLANNER_REPAIR, PLANNER_SYSTEM, load
from tabpilot.planning.validate import PlanContext, validate_plan
from tabpilot.schemas import (
    EventType,
    LLMRequest,
    PlannerInput,
    PlannerOutput,
    PlanValidation,
    ProfileSummary,
)
from tabpilot.schemas.llm import LLMPurpose
from tabpilot.schemas.plan import PlanIssue

PLAN_MAX_OUTPUT_TOKENS = 2000
REPAIR_ERROR_CHARS = 1500
PlanSource = Literal["llm", "rule_based"]
EventSink = Callable[[EventType, dict[str, Any]], None]

log = get_logger("plan")


@dataclass
class PlannerContext:
    """Everything one planning step needs."""

    planner_input: PlannerInput
    plan_context: PlanContext
    summary: ProfileSummary  # for the rule-based fallback
    router: LLMRouter | None
    budget: BudgetTracker | None
    events: EventSink | None = None


def _emit(ctx: PlannerContext, kind: EventType, **data: Any) -> None:
    if ctx.events is not None:
        ctx.events(kind, data)


def repair_errors(issues: list[PlanIssue]) -> str:
    """Field paths and messages for the repair call, at most 1,500 characters (§12.5)."""
    lines = [f"{i.path or '(root)'}: {i.code}: {i.message}" for i in issues]
    text = "\n".join(lines)
    return text if len(text) <= REPAIR_ERROR_CHARS else text[: REPAIR_ERROR_CHARS - 1] + "…"


def _call(
    router: LLMRouter, budget: BudgetTracker, purpose: LLMPurpose, template: str,
    user_json: dict[str, Any],
) -> Any:  # fmt: skip
    tpl = load(template)
    request = LLMRequest(purpose=purpose, system=tpl.text, user_json=user_json,
                         response_schema=PlannerOutput, max_output_tokens=PLAN_MAX_OUTPUT_TOKENS,
                         temperature=0.0, timeout_s=budget.budget.per_call_timeout_s)  # fmt: skip
    return router.complete(request, budget, template=tpl.id)


def _validated(ctx: PlannerContext, raw: Any) -> tuple[PlannerOutput | None, PlanValidation]:
    plan, validation = validate_plan(raw, ctx.plan_context)
    _emit(ctx, EventType.PLAN_VALIDATED, status=validation.status,
          issues=[i.code for i in validation.issues])  # fmt: skip
    return plan, validation


def _fallback(ctx: PlannerContext, reason: str) -> tuple[PlannerOutput, PlanValidation, PlanSource]:
    _emit(ctx, EventType.PLANNER_FALLBACK, reason=reason)
    log.warning("planner_fallback", reason=reason)
    pc = ctx.plan_context
    plan = rule_based_plan(ctx.summary, pc.task, pc.available_models)
    repaired, validation = validate_plan(plan, pc)
    if repaired is None or validation.status == "invalid":  # §12.8 plans are valid by design
        raise RuntimeError("rule-based plan failed validation")
    return repaired, validation, "rule_based"


def _raw(response: Any) -> Any:
    return response.parsed if response.parsed is not None else response.content


def make_plan(ctx: PlannerContext) -> tuple[PlannerOutput, PlanValidation, PlanSource]:
    """§12.4–§12.9: always returns a validated plan and its source."""
    router, budget = ctx.router, ctx.budget
    if router is None or budget is None or not router.usable():
        return _fallback(ctx, "no LLM provider available")
    payload = planner_payload(ctx.planner_input)
    try:
        first = _call(router, budget, "plan", PLANNER_SYSTEM, payload)
        _emit(ctx, EventType.PLAN_RECEIVED, source="llm", provider=first.provider)
        plan, validation = _validated(ctx, _raw(first))
        if plan is None:  # unparseable or schema-invalid: one repair call (§12.5)
            repair = {"planner_input": payload, "previous_response": first.content,
                      "errors": repair_errors(validation.issues)}  # fmt: skip
            second = _call(router, budget, "plan_repair", PLANNER_REPAIR, repair)
            _emit(ctx, EventType.PLAN_RECEIVED, source="llm_repair", provider=second.provider)
            plan, validation = _validated(ctx, _raw(second))
    except LLMUnavailable as exc:
        return _fallback(ctx, exc.reason)
    if plan is None:
        return _fallback(ctx, "plan still invalid after one repair")
    if validation.status == "invalid":
        severe = sorted({i.code for i in validation.issues if i.severity == "severe"})
        return _fallback(ctx, "plan invalid: " + ", ".join(severe))
    return plan, validation, "llm"


def plan_issue_summary(validation: PlanValidation) -> str:
    """One-line UI summary (§12.7)."""
    fixes = [i for i in validation.issues if i.severity != "info"]
    if validation.status == "valid":
        return "Plan accepted"
    if validation.status == "repaired":
        return f"Plan accepted with {len(fixes)} fixes: " + json.dumps([i.code for i in fixes])
    return "LLM plan rejected; using rule-based plan"
