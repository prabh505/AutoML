"""Reflection proposals: one call, one repair, validation, change application (PRD §19.3, §19.7).

Structural bounds come from the `ReflectionProposal` contract (task, metric, folds, holdout,
seeds and budgets are not fields). `validate_proposal` adds the run-dependent checks and drops
every change it cannot apply safely; it never raises. Feature specs reuse the P07 validator
through the P11 plan context, with the currently active LLM features as earlier specs, so the
12-feature cap holds at all times (A-010).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from tabpilot.evaluation.metrics import selection_score
from tabpilot.features import validate_specs
from tabpilot.llm import BudgetTracker, LLMRouter, LLMUnavailable
from tabpilot.logging import get_logger
from tabpilot.planning.planner import repair_errors
from tabpilot.planning.validate import PlanContext
from tabpilot.reflection.templates import REFLECTOR_REPAIR, REFLECTOR_SYSTEM, load
from tabpilot.schemas import (
    CandidateSpec,
    ClassWeight,
    EventType,
    LLMRequest,
    MetricName,
    ModelName,
    PlannerOutput,
    ReflectionInput,
    ReflectionProposal,
    TargetTransform,
    Task,
)
from tabpilot.schemas.experiment import LeaderboardEntry
from tabpilot.schemas.features import MAX_ACTIVE_LLM_FEATURES, FeatureSpec
from tabpilot.schemas.llm import LLMPurpose
from tabpilot.schemas.plan import PlanIssue

REFLECT_MAX_OUTPUT_TOKENS = 1500
RETRAIN_FAMILIES = 2
LOW_ENCODINGS = frozenset({"onehot", "ordinal"})
ProposalStatus = Literal["ok", "empty", "invalid", "failed"]
EventSink = Callable[[EventType, dict[str, Any]], None]

log = get_logger("reflect")


@dataclass
class ReflectionContext:
    """Everything one reflection proposal needs."""

    reflection_input: ReflectionInput
    plan_context: PlanContext
    active_features: list[FeatureSpec]
    router: LLMRouter | None
    budget: BudgetTracker | None
    events: EventSink | None = None


@dataclass
class ChangeSet:
    """Children to retrain and the plan-level effects of an accepted proposal."""

    change_set_id: str
    children: list[CandidateSpec]
    allocation: dict[ModelName, float] | None = None
    dropped_models: list[ModelName] = field(default_factory=list)


def _emit(ctx: ReflectionContext, kind: EventType, **data: Any) -> None:
    if ctx.events is not None:
        ctx.events(kind, data)


# ---------------------------------------------------------------- validation
def _check_add(change: Any, ctx: ReflectionContext, issues: list[str]) -> Any | None:
    allowed = min(ctx.reflection_input.remaining_budget.features_allowed,
                  MAX_ACTIVE_LLM_FEATURES - len(ctx.active_features))  # fmt: skip
    combined = [*ctx.active_features, *change.features]
    accepted, found = validate_specs(combined, ctx.plan_context.feature_schema(),
                                     max_count=MAX_ACTIVE_LLM_FEATURES)  # fmt: skip
    new = {id(f) for f in change.features}  # by identity: a new spec may reuse an active name
    new_names = {f.name for f in change.features}
    issues.extend(f"add_features: {i.code} ({i.feature})" for i in found
                  if i.fatal and i.feature in new_names)  # fmt: skip
    accepted_new = [f for f in accepted if id(f) in new]
    kept = accepted_new[: max(allowed, 0)]
    if len(kept) < len(accepted_new):
        issues.append(f"add_features: capped at {allowed} (12 active features at most)")
    return change.model_copy(update={"features": kept}) if kept else None


def _check(  # noqa: PLR0911 - one exit per §19.3 change type
    change: Any, ctx: ReflectionContext, plan: PlannerOutput, issues: list[str]
) -> Any | None:
    pc = ctx.plan_context
    plan_models = [m.name for m in plan.models]
    kind = change.type
    if kind == "add_features":
        return _check_add(change, ctx, issues)
    if kind == "remove_features":
        active = {f.name for f in ctx.active_features}
        names = [n for n in change.feature_names if n in active]
        if len(names) < len(change.feature_names):
            issues.append("remove_features: unknown feature names dropped")
        return change.model_copy(update={"feature_names": names}) if names else None
    if kind == "change_encoding":
        cols = [c for c in change.columns if c in pc.feature_types]
        if len(cols) < len(change.columns):
            issues.append("change_encoding: unknown columns dropped")
        return change.model_copy(update={"columns": cols}) if cols else None
    if kind == "toggle_class_weight":
        if pc.task == Task.REGRESSION or change.model not in plan_models:
            issues.append("toggle_class_weight: classification models in the plan only")
            return None
        return change
    if kind == "change_target_transform":
        nonneg = pc.target_min is not None and pc.target_min >= 0
        ok = pc.task == Task.REGRESSION and (change.value == TargetTransform.NONE or nonneg)
        if not ok:
            issues.append("change_target_transform: regression with a non-negative target only")
            return None
        return change
    if kind == "reallocate_tuning":
        alloc = {m: v for m, v in change.allocation.items() if m in plan_models}
        if len(alloc) < len(change.allocation):
            issues.append("reallocate_tuning: models outside the plan dropped")
        return change.model_copy(update={"allocation": alloc}) if alloc else None
    if change.model not in plan_models or len(plan_models) <= 1:
        issues.append("drop_model: must name a plan model and leave at least one")
        return None
    return change


def validate_proposal(
    proposal: ReflectionProposal, ctx: ReflectionContext
) -> tuple[ReflectionProposal, list[str]]:
    """§19.3 run-dependent checks. Invalid changes are removed and listed; never raises."""
    issues: list[str] = []
    plan = ctx.reflection_input.current_plan
    kept: list[Any] = []
    dropped_models = 0
    for change in proposal.changes:
        try:
            checked = _check(change, ctx, plan, issues)
        except Exception as exc:  # the validator never raises
            issues.append(f"{change.type}: validator error {type(exc).__name__}")
            checked = None
        if checked is not None and checked.type == "drop_model":
            if dropped_models + 1 >= len(plan.models):
                issues.append("drop_model: at least one model must remain")
                continue
            dropped_models += 1
        if checked is not None:
            kept.append(checked)
    return proposal.model_copy(update={"changes": kept}), issues


# ---------------------------------------------------------------- the call
def _call(router: LLMRouter, budget: BudgetTracker, purpose: LLMPurpose, template: str,
          user_json: dict[str, Any]) -> Any:  # fmt: skip
    tpl = load(template)
    request = LLMRequest(purpose=purpose, system=tpl.text, user_json=user_json,
                         response_schema=ReflectionProposal, temperature=0.0,
                         max_output_tokens=REFLECT_MAX_OUTPUT_TOKENS,
                         timeout_s=budget.budget.per_call_timeout_s)  # fmt: skip
    return router.complete(request, budget, template=tpl.id)


def _parse(raw: Any) -> tuple[ReflectionProposal | None, list[PlanIssue]]:
    try:
        return ReflectionProposal.model_validate(raw), []
    except ValidationError as exc:
        return None, [PlanIssue(code=str(e["type"])[:60], severity="severe",
                                message=str(e["msg"])[:300],
                                path=".".join(str(p) for p in e["loc"])[:120] or None)
                      for e in exc.errors()]  # fmt: skip


def propose(ctx: ReflectionContext) -> tuple[ReflectionProposal | None, ProposalStatus, list[str]]:
    """One reflector call, one repair for unparseable/schema-invalid output, then validation."""
    router, budget = ctx.router, ctx.budget
    if router is None or budget is None or not router.usable():
        return None, "failed", ["no LLM provider available"]
    payload = ctx.reflection_input.model_dump(mode="json", exclude_none=True)
    try:
        first = _call(router, budget, "reflect", REFLECTOR_SYSTEM, payload)
        proposal, errors = _parse(first.parsed if first.parsed is not None else first.content)
        if proposal is None:
            repair = {"reflection_input": payload, "previous_response": first.content,
                      "errors": repair_errors(errors)}  # fmt: skip
            second = _call(router, budget, "reflect_repair", REFLECTOR_REPAIR, repair)
            proposal, errors = _parse(
                second.parsed if second.parsed is not None else second.content
            )
    except LLMUnavailable as exc:
        return None, "failed", [exc.reason]
    if proposal is None:
        return None, "invalid", [f"{e.path}: {e.code}" for e in errors]
    _emit(ctx, EventType.REFLECTION_PROPOSED, changes=[c.type for c in proposal.changes])
    if not proposal.changes:
        return proposal, "empty", []
    validated, issues = validate_proposal(proposal, ctx)
    if not validated.changes:
        return validated, "invalid", issues
    return validated, "ok", issues


# ---------------------------------------------------------------- application
def change_set_id(proposal: ReflectionProposal) -> str:
    """Stable 8-hex id of the validated changes."""
    payload = proposal.model_dump_json(include={"changes"})
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def retrain_parents(
    entries: Sequence[LeaderboardEntry], metric: MetricName, proposal: ReflectionProposal
) -> list[LeaderboardEntry]:
    """§19.4: the best entry of each of the top 2 families, plus explicitly targeted models;
    dropped models are excluded. Ties break by entry id.
    """
    ok = [e for e in entries if e.status == "ok" and e.comparable and e.mean is not None
          and e.model != "ensemble"]  # fmt: skip
    ok.sort(key=lambda e: (-selection_score(metric, e.mean or 0.0), e.entry_id))
    best: dict[str, LeaderboardEntry] = {}
    for e in ok:
        best.setdefault(str(e.model), e)
    families = list(best)[:RETRAIN_FAMILIES]
    targeted = {c.model.value for c in proposal.changes if c.type == "toggle_class_weight"}
    dropped = {c.model.value for c in proposal.changes if c.type == "drop_model"}
    chosen = [f for f in best if (f in families or f in targeted) and f not in dropped]
    return [best[f] for f in chosen]


def _child(parent: CandidateSpec, parent_entry: str, proposal: ReflectionProposal,
           set_id: str) -> CandidateSpec | None:  # fmt: skip
    features = list(parent.features)
    encoding, model, transform = parent.encoding, parent.model, parent.target_transform
    for c in proposal.changes:
        if c.type == "remove_features":
            features = [f for f in features if f.name not in set(c.feature_names)]
        elif c.type == "add_features":
            features += [f for f in c.features if f.name not in {g.name for g in features}]
        elif c.type == "change_encoding":
            field_name = "low_cardinality" if c.encoding in LOW_ENCODINGS else "high_cardinality"
            encoding = encoding.model_copy(update={field_name: c.encoding})
        elif c.type == "toggle_class_weight" and c.model == model.name:
            model = model.model_copy(update={"class_weight": ClassWeight(c.value)})
        elif c.type == "change_target_transform":
            transform = c.value
    features = features[:MAX_ACTIVE_LLM_FEATURES]
    if (features, encoding, model, transform) == (parent.features, parent.encoding, parent.model,
                                                  parent.target_transform):  # fmt: skip
        return None  # nothing changes for this candidate
    iteration = min(parent.iteration + 1, 2)
    return parent.model_copy(update={
        "candidate_id": f"{iteration}-{model.name.value}-{set_id}", "iteration": iteration,
        "features": features, "feature_set": "baseline+llm" if features else "baseline",
        "encoding": encoding, "model": model, "target_transform": transform,
        "parent_entry_id": parent_entry, "change_set_id": set_id,
    })  # fmt: skip


def apply_changes(
    proposal: ReflectionProposal, parents: Sequence[tuple[str, CandidateSpec]]
) -> ChangeSet:
    """Child candidates (with lineage) for each `(parent entry id, parent candidate)`, plus the
    plan-level allocation and dropped models (§19.7).
    """
    set_id = change_set_id(proposal)
    children = [child for entry_id, cand in parents
                if (child := _child(cand, entry_id, proposal, set_id)) is not None]  # fmt: skip
    allocation = None
    for c in proposal.changes:
        if c.type == "reallocate_tuning":
            total = math.fsum(c.allocation.values())
            allocation = {m: v / total for m, v in c.allocation.items()} if total > 0 else None
    dropped = [c.model for c in proposal.changes if c.type == "drop_model"]
    return ChangeSet(set_id, children, allocation, dropped)
