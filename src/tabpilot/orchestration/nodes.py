"""Graph nodes (PRD §13.2, §13.4, §13.7).

Every node reads the small `RunState`, works on the run's `RunContext`, emits events only after
the action it describes has completed (§27.4), and returns a partial state update. Routing
decisions are made inside the nodes and stored in `ctx.route`; the edge functions only read it,
so the graph stays deterministic and every skip is recorded as an event.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from tabpilot import __version__
from tabpilot.ensembling import Candidate, build_ensemble
from tabpilot.evaluation.holdout import evaluate_holdout
from tabpilot.evaluation.leaderboard import not_comparable, threshold_report
from tabpilot.evaluation.metrics import selection_score
from tabpilot.evaluation.metrics import spec as metric_spec
from tabpilot.evaluation.selection import NoValidModelError, select
from tabpilot.explain import (
    EXPLAIN_TIMEOUT_S,
    ExplainData,
    ExplainSettings,
    error_analysis,
    explain,
)
from tabpilot.export import ExportSettings, export_bundle
from tabpilot.features.provenance import provenance
from tabpilot.orchestration.context import RunContext, context
from tabpilot.orchestration.training import (
    entry_for,
    initial_candidates,
    train_children,
    train_initial,
)
from tabpilot.planning import PlanContext, PlannerContext, make_plan, planner_payload
from tabpilot.planning import versions as planner_versions
from tabpilot.reflection import (
    ReflectionContext,
    accept,
    apply_changes,
    build_reflection_input,
    propose,
    reflection_budget,
    retrain_parents,
)
from tabpilot.reflection import versions as reflector_versions
from tabpilot.reporting import ReportState, build_report
from tabpilot.reporting import versions as narrator_versions
from tabpilot.schemas import (
    CandidateSpec,
    ColumnRole,
    ErrorCode,
    EventType,
    ExecutionProfileName,
    FeatureSpec,
    ModelName,
    RunState,
    SemanticType,
    Stage,
    StageError,
    TabPilotError,
    Task,
)
from tabpilot.schemas.experiment import (
    Budgets,
    ExperimentConfig,
    LeaderboardEntry,
    LLMSettingsSummary,
    ModelConfig,
    SplitSummary,
)
from tabpilot.schemas.features import FeatureProvenance
from tabpilot.schemas.reflection import ReflectionRecord, RemainingBudget
from tabpilot.schemas.run import FinalResult
from tabpilot.splitting import iter_folds
from tabpilot.training import CVRun, OOFStore, refit_final, run_cv
from tabpilot.training.refit import baseline_candidate, refit_params
from tabpilot_runtime import VotingEnsemble

LIBRARIES = ("scikit-learn", "lightgbm", "optuna", "pandas", "numpy", "xgboost", "catboost")
MAX_ACTIVE_FEATURES = 12
MAX_NEW_FEATURES = 6
MB = 1024 * 1024
MIN_REPORT_LLM_S = 30.0  # below this the narrator uses the template (reserved slice, §29.3)


# ---------------------------------------------------------------- helpers
def _ctx(state: RunState) -> RunContext:
    return context(state.run_id)


def ledger(ctx: RunContext) -> Any:
    """The `budget_ledger/1.0` snapshot including LLM usage."""
    current = ctx.budget.ledger()
    return ctx.llm_budget.add_to(current) if ctx.llm_budget is not None else current


def _emit(ctx: RunContext, stage: Stage, kind: EventType, data: dict[str, Any]) -> None:
    ctx.emit(stage, kind, **data)


def _llm_on(ctx: RunContext) -> bool:
    return ctx.router is not None and ctx.llm_budget is not None and bool(ctx.router.usable())


def eligible(state: RunState, ctx: RunContext) -> list[LeaderboardEntry]:
    """Incumbent-eligible entries: ok, comparable, not a rejected reflection child (§19.7)."""
    return [e for e in state.leaderboard if e.status == "ok" and e.comparable
            and e.mean is not None and e.entry_id not in ctx.rejected
            and e.model != "ensemble"]  # fmt: skip


def library_versions() -> dict[str, str]:
    found = {}
    for name in LIBRARIES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def experiment_config(state: RunState, ctx: RunContext) -> ExperimentConfig:
    """`experiment/1.0` facts of this run (report, records, bundle)."""
    cfg, split = state.config, ctx.split
    llm_used = cfg.llm_mode != "off"
    versions = ({**planner_versions(), **reflector_versions(), **narrator_versions()}
                if llm_used else {})  # fmt: skip
    return ExperimentConfig(
        run_id=cfg.run_id, tabpilot_version=__version__, execution_profile=cfg.execution_profile,
        task=cfg.task, target=cfg.target, positive_label=cfg.positive_label,
        primary_metric=metric_spec(cfg.primary_metric),
        split=SplitSummary(split_id=split.split_id, strategy=str(split.strategy), k=split.k,
                           n_repeats=split.n_repeats, holdout_fraction=split.holdout_fraction,
                           holdout_present=split.holdout_present, group_column=split.group_column,
                           duplicate_grouping=split.duplicate_grouping),
        seeds=ctx.seeds,
        budgets=Budgets(time_s=cfg.time_budget_s, max_trials=ctx.budget.trials_max,
                        llm_calls=ctx.llm_budget.budget.max_calls if ctx.llm_budget else 0,
                        reflection_rounds=cfg.reflection_max_rounds),
        llm=LLMSettingsSummary(enabled=llm_used, order=list(cfg.provider_order)),
        library_versions=library_versions(), prompt_template_versions=versions,
    )  # fmt: skip


def feature_provenance(state: RunState, ctx: RunContext) -> list[FeatureProvenance]:
    """One record per LLM feature (first iteration that used it); failures from the winner."""
    seen: dict[str, tuple[FeatureSpec, int]] = {}
    for cand in state.candidates:
        for spec in cand.features:
            seen.setdefault(spec.name, (spec, cand.iteration))
    dropped = ctx.winner_run.dropped_features if ctx.winner_run is not None else {}
    records = []
    for spec, iteration in seen.values():
        record = provenance([spec], [], None, "llm_plan" if iteration == 0 else "llm_reflection",
                            iteration)[0]  # fmt: skip
        if spec.name in dropped:
            reason = dropped[spec.name]
            status = "degenerate" if reason == "degenerate" else "failed"
            record = record.model_copy(update={"status": status, "failure_reason": reason[:300]})
        records.append(record)
    return records


def _stage_error(stage: Stage, exc: BaseException, recoverable: bool) -> StageError:
    code = exc.code if isinstance(exc, TabPilotError) else None
    return StageError(stage=stage, error_code=code, error_class=type(exc).__name__[:100],
                      message=type(exc).__name__, recoverable=recoverable)  # fmt: skip


# ---------------------------------------------------------------- plan
def plan_context(state: RunState, ctx: RunContext) -> PlanContext:
    meta = state.dataset_meta
    feature_types = {c.name: c.semantic_type for c in meta.columns
                     if c.role == ColumnRole.FEATURE}  # fmt: skip
    excluded = frozenset(c.name for c in meta.columns if c.role != ColumnRole.FEATURE)
    minimums = {c.name: c.min for c in ctx.summary.columns if c.min is not None}
    target_min = ctx.summary.target.min if state.config.task == Task.REGRESSION else None
    return PlanContext(task=state.config.task, target=state.config.target,
                       primary_metric=state.config.primary_metric, feature_types=feature_types,
                       available_models=ctx.available_models(), k=ctx.split.k,
                       excluded=excluded, target_min=target_min, minimums=minimums,
                       n_rows=len(ctx.split.dev_indices))  # fmt: skip


def plan_node(state: RunState) -> dict[str, Any]:
    """Planner call with one repair and the rule-based fallback (§12.4–§12.9)."""
    ctx = _ctx(state)
    with ctx.stage(Stage.PLAN):
        received: list[Any] = []
        planner = PlannerContext(ctx.planner_input, plan_context(state, ctx), ctx.summary,
                                 ctx.router if _llm_on(ctx) else None, ctx.llm_budget,
                                 events=lambda kind, data: _emit(ctx, Stage.PLAN, kind, data),
                                 on_response=received.append)  # fmt: skip
        ctx.pending_plan = make_plan(planner)
        ctx.plan_received = received[0] if received and isinstance(received[0], dict) else None
    return {"budget": ledger(ctx)}


def validate_plan_node(state: RunState) -> dict[str, Any]:
    """Commit the validated plan (validation itself ran with the planner, §12.6)."""
    ctx = _ctx(state)
    with ctx.stage(Stage.VALIDATE_PLAN):
        plan, validation, source = ctx.pending_plan
    return {"plan": plan, "plan_validation": validation, "plan_source": source,
            "budget": ledger(ctx)}  # fmt: skip


# ---------------------------------------------------------------- features and training
def build_features_node(state: RunState) -> dict[str, Any]:
    """Iteration 0: one candidate per planned model with the plan's features. Later
    iterations: the children added by `apply_changes` (already validated).
    """
    ctx = _ctx(state)
    with ctx.stage(Stage.BUILD_FEATURES):
        candidates = list(state.candidates)
        if state.iteration == 0 and not candidates:
            assert state.plan is not None  # noqa: S101 - set by validate_plan
            candidates = initial_candidates(state.plan, ctx)
    return {"candidates": candidates, "budget": ledger(ctx)}


def _fallback_candidate(state: RunState, ctx: RunContext) -> CandidateSpec:
    """§13.2 recover → train: one default-parameter model of the simplest available family."""
    available = ctx.available_models() or [ModelName.LINEAR]
    name = ModelName.LINEAR if ModelName.LINEAR in available else available[0]
    config = ModelConfig(name=name, task=state.config.task, seed=ctx.seeds.model, n_jobs=ctx.n_jobs)
    return CandidateSpec(candidate_id=f"0-{name.value}-fallback", iteration=0, model=config,
                         feature_set="baseline")  # fmt: skip


def train_candidates_node(state: RunState) -> dict[str, Any]:
    """Train what this iteration still needs; unexpected errors route to `recover`."""
    ctx = _ctx(state)
    trained = {e.entry_id for e in state.leaderboard}
    entries: list[LeaderboardEntry] = []
    ctx.route = "evaluate"
    errors = list(state.errors)
    try:
        with ctx.stage(Stage.TRAIN_CANDIDATES):
            if ctx.fallback_used and not any(e.status == "ok" for e in state.leaderboard):
                cand = _fallback_candidate(state, ctx)
                cv = run_cv(cand, ctx.dev, ctx.split, schema=ctx.schema, registry=ctx.registry,
                            metric=ctx.metric, cancel=ctx.cancel, budget_hook=ctx.train_hook(),
                            profile=ctx.config.execution_profile)  # fmt: skip
                ctx.runs[cv.result.entry_id] = cv
                entries.append(entry_for(ctx, cv.result, cv.candidate, cv))
                ctx.emit(Stage.TRAIN_CANDIDATES, EventType.CANDIDATE_COMPLETED
                         if cv.result.status == "ok" else EventType.CANDIDATE_FAILED,
                         model=cand.model.name.value, entry_id=cv.result.entry_id,
                         status=cv.result.status, fallback=True)  # fmt: skip
            elif state.iteration == 0:
                assert state.plan is not None  # noqa: S101
                todo = [c for c in state.candidates if c.iteration == 0]
                entries = train_initial(ctx, state.plan, todo) if not trained else []
            else:
                children = [c for c in state.candidates if c.iteration == state.iteration
                            and c.parent_entry_id is not None]  # fmt: skip
                trials = ctx.round_budget.trials if ctx.round_budget is not None else 0
                entries = train_children(ctx, children, trials)
    except (TabPilotError, MemoryError, ValueError, RuntimeError) as exc:
        errors.append(_stage_error(Stage.TRAIN_CANDIDATES, exc, recoverable=True))
        ctx.emit(Stage.TRAIN_CANDIDATES, EventType.STAGE_FAILED, error_class=type(exc).__name__)
        ctx.route = "recover"
    return {"leaderboard": [*state.leaderboard, *entries], "errors": errors,
            "budget": ledger(ctx)}  # fmt: skip


# ---------------------------------------------------------------- evaluate and reflection
def _record_round(state: RunState, ctx: RunContext) -> list[ReflectionRecord]:
    """Paired-fold decisions for the children of the round that just trained (§19.5)."""
    proposal, change = ctx.pending_change
    by_id = {e.entry_id: e for e in state.leaderboard}
    decisions = []
    for child in (e for e in state.leaderboard if e.change_set_id == change.change_set_id):
        parent = by_id.get(child.parent_entry_id or "")
        if parent is None:
            continue
        decision = accept(parent, child)
        decisions.append(decision)
        if not decision.accepted:
            ctx.rejected.add(child.entry_id)
    accepted = any(d.accepted for d in decisions)
    reason = ("accepted by the paired-fold rule" if accepted
              else "no change met the paired-fold rule" if decisions
              else "no candidate was changed by the proposal")  # fmt: skip
    record = ReflectionRecord(round=state.rounds_used, status="accepted" if accepted else
                              "rejected", reason=reason, proposal=proposal,
                              decisions=decisions)  # fmt: skip
    ctx.emit(Stage.EVALUATE, EventType.REFLECTION_DECIDED, round=state.rounds_used,
             accepted=accepted, changes=len(decisions))  # fmt: skip
    ctx.pending_change = None
    ctx.last_round_accepted = accepted
    return [*state.reflection_history, record]


def _reflection_route(state: RunState, ctx: RunContext) -> str:
    """§13.2 evaluate → reflect when rounds remain, the budget allows and an LLM is usable."""
    cfg = state.config
    if state.rounds_used >= cfg.reflection_max_rounds:
        return "ensemble"
    if state.rounds_used > 0 and not ctx.last_round_accepted:
        return "ensemble"  # §19.6: no change accepted in the last round
    if not _llm_on(ctx):
        ctx.emit(Stage.REFLECT, EventType.REFLECTION_SKIPPED, reason="llm_unavailable")
        return "ensemble"
    if ctx.monitor.soft_exceeded:
        ctx.emit(Stage.REFLECT, EventType.REFLECTION_SKIPPED, reason="memory")
        return "ensemble"
    rb = reflection_budget(ctx.budget.total_s, ctx.budget.elapsed(), ctx.reflection_seconds,
                           max(1, ctx.initial_trials))  # fmt: skip
    if rb.skipped:
        ctx.emit(Stage.REFLECT, EventType.REFLECTION_SKIPPED, reason=rb.skip_reason)
        return "ensemble"
    ctx.round_budget = rb
    return "reflect"


def evaluate_node(state: RunState) -> dict[str, Any]:
    """Leaderboard, comparability, incumbent; decide reflect / ensemble / recover."""
    ctx = _ctx(state)
    history = list(state.reflection_history)
    with ctx.stage(Stage.EVALUATE):
        if ctx.pending_change is not None:
            history = _record_round(state, ctx)
        if not_comparable(state.leaderboard):
            ctx.emit(Stage.EVALUATE, EventType.WARNING_RAISED, code="not_comparable")
        ranked = eligible(state, ctx)
        try:
            ctx.selection = select(ranked)
            ctx.selection_ids = frozenset(e.entry_id for e in ranked)
            best = ctx.selection.winner
        except NoValidModelError:
            ctx.selection, ctx.selection_ids = None, frozenset()
            best = None
        ctx.emit(Stage.EVALUATE, EventType.LEADERBOARD_UPDATED, n=len(state.leaderboard),
                 winner=best)  # fmt: skip
    probe = state.model_copy(update={"reflection_history": history})
    ctx.route = "recover" if best is None else _reflection_route(probe, ctx)
    return {"best_candidate_id": best, "reflection_history": history, "budget": ledger(ctx)}


def reflect_node(state: RunState) -> dict[str, Any]:
    """One reflector proposal (§19.2–§19.3); route to apply_changes only for a valid one."""
    ctx = _ctx(state)
    rb = ctx.round_budget
    start = ctx.budget.elapsed()
    with ctx.stage(Stage.REFLECT):
        active = {f.name: f for c in state.candidates for f in c.features}
        remaining = RemainingBudget(
            seconds=round(rb.seconds, 1),
            trials=rb.trials,
            features_allowed=max(0, min(MAX_NEW_FEATURES, MAX_ACTIVE_FEATURES - len(active))),
        )
        assert state.plan is not None  # noqa: S101
        ri = build_reflection_input(task=state.config.task, metric=state.config.primary_metric,
                                    entries=eligible(state, ctx), importances=[],
                                    provenance=feature_provenance(state, ctx), plan=state.plan,
                                    remaining=remaining)  # fmt: skip
        rctx = ReflectionContext(
            ri,
            plan_context(state, ctx),
            list(active.values()),
            ctx.router,
            ctx.llm_budget,
            events=lambda kind, data: _emit(ctx, Stage.REFLECT, kind, data),
        )
        proposal, status, issues = propose(rctx)
    ctx.reflection_seconds += ctx.budget.elapsed() - start
    history = list(state.reflection_history)
    if status == "ok" and proposal is not None:
        ctx.pending_proposal = proposal
        ctx.route = "apply_changes"
    else:
        reason = {"empty": "proposal empty", "invalid": "proposal invalid after one repair",
                  "failed": "LLM unavailable"}.get(status, status)  # fmt: skip
        history.append(ReflectionRecord(round=state.rounds_used + 1, status=status,
                                        reason=reason, proposal=proposal,
                                        issues=[i[:200] for i in issues][:20]))  # fmt: skip
        ctx.emit(Stage.REFLECT, EventType.REFLECTION_DECIDED, round=state.rounds_used + 1,
                 accepted=False, status=status)  # fmt: skip
        ctx.route = "ensemble"
    return {"reflection_history": history, "budget": ledger(ctx)}


def apply_changes_node(state: RunState) -> dict[str, Any]:
    """Children of the top-2 families plus targeted models, with lineage (§19.4, §19.7)."""
    ctx = _ctx(state)
    with ctx.stage(Stage.APPLY_CHANGES):
        proposal = ctx.pending_proposal
        parents = retrain_parents(eligible(state, ctx), state.config.primary_metric, proposal)
        pairs = [(e.entry_id, ctx.runs[e.entry_id].candidate) for e in parents
                 if e.entry_id in ctx.runs]  # fmt: skip
        change = apply_changes(proposal, pairs)
        ctx.pending_change = (proposal, change)
        ctx.pending_proposal = None
    rounds = state.rounds_used + 1
    return {"candidates": [*state.candidates, *change.children], "rounds_used": rounds,
            "iteration": min(rounds, 2), "budget": ledger(ctx)}  # fmt: skip


# ---------------------------------------------------------------- recover / fail
def recover_node(state: RunState) -> dict[str, Any]:
    """Route after a training error or no valid candidate (§13.2)."""
    ctx = _ctx(state)
    with ctx.stage(Stage.RECOVER):
        valid = eligible(state, ctx)
        if valid:
            ctx.route = "finalize"
        elif (not ctx.fallback_used and not ctx.monitor.stop_training
              and not ctx.budget.expired(Stage.TRAIN_CANDIDATES)):  # fmt: skip
            ctx.fallback_used = True
            ctx.route = "train_candidates"
        else:
            ctx.route = "fail"
    return {"budget": ledger(ctx)}


def fail_node(state: RunState) -> dict[str, Any]:
    """E-NO-VALID-MODEL with each model's last reason (§8.9)."""
    ctx = _ctx(state)
    reasons = "; ".join(f"{e.model}: {e.warnings[-1] if e.warnings else e.status}"
                        for e in state.leaderboard)[:400] or "no model was trained"  # fmt: skip
    ctx.failure = ctx.failure or TabPilotError(ErrorCode.NO_VALID_MODEL, reasons=reasons)
    ctx.emit(Stage.RUN, EventType.RUN_FAILED, error_code=ctx.failure.code.value)
    return {"status": "failed", "budget": ledger(ctx)}


# ---------------------------------------------------------------- ensemble and finalize
def ensemble_node(state: RunState) -> dict[str, Any]:
    """Cross-fitted ensemble of the eligible OOF predictions (§20), when enabled and affordable."""
    ctx = _ctx(state)
    if not state.config.enable_ensemble:
        ctx.skip(Stage.ENSEMBLE, "disabled")
        return {"budget": ledger(ctx)}
    if ctx.budget.expired(Stage.ENSEMBLE) or ctx.monitor.soft_exceeded:
        ctx.skip(Stage.ENSEMBLE, "budget" if ctx.budget.expired(Stage.ENSEMBLE) else "memory")
        return {"budget": ledger(ctx)}
    with ctx.stage(Stage.ENSEMBLE):
        cands = [Candidate(e, ctx.runs[e.entry_id].oof) for e in eligible(state, ctx)
                 if e.entry_id in ctx.runs]  # fmt: skip
        rows = np.asarray(ctx.split.dev_indices)
        limit = ctx.settings.limits.max_bundle_mb * MB
        result = build_ensemble(cands, ctx.dev.y[rows], ctx.split, ctx.problem, ctx.metric,
                                artifact_limit_bytes=limit)  # fmt: skip
        ctx.emit(Stage.ENSEMBLE, EventType.ENSEMBLE_DECIDED, kept=result.kept,
                 reason=result.reason, members=len(result.members))  # fmt: skip
    return {"ensemble": result, "budget": ledger(ctx)}


def _refit(ctx: RunContext, run: CVRun) -> Any:
    return refit_final(run.candidate, run.best_iterations, ctx.dev, ctx.split, schema=ctx.schema,
                       registry=ctx.registry, profile=ctx.config.execution_profile)  # fmt: skip


def finalize_node(state: RunState) -> dict[str, Any]:
    """Freeze selection, refit on DEV (next-ranked once on failure), open the holdout once."""
    ctx = _ctx(state)
    with ctx.stage(Stage.FINALIZE):
        ranked = eligible(state, ctx)
        ids = frozenset(e.entry_id for e in ranked)
        if ctx.selection is not None and ctx.selection_ids == ids and not ctx.freeze.frozen:
            selection = ctx.selection  # the incumbent chosen by the last evaluate (same entries)
        else:
            selection = select(ranked, ctx.freeze)
        runners = sorted(
            (e for e in ranked if e.entry_id != selection.winner),
            key=lambda e: (-selection_score(ctx.metric, e.mean or 0.0), e.entry_id),
        )
        order = [selection.winner, *[e.entry_id for e in runners]]
        ensemble = state.ensemble if state.ensemble is not None and state.ensemble.kept else None
        final_model, chosen = None, selection.winner
        if ensemble is not None:
            members = [_refit(ctx, ctx.runs[m.entry_id]) for m in ensemble.members]
            final_model = VotingEnsemble(
                members, [m.weight for m in ensemble.members],
                task="regression" if state.config.task == Task.REGRESSION else "classification",
            ).fit(None, None)  # fmt: skip
            chosen = "ensemble"
        else:
            for attempt, entry_id in enumerate(order[:2]):
                try:
                    final_model = _refit(ctx, ctx.runs[entry_id])
                    chosen = entry_id
                    break
                except (ValueError, RuntimeError, MemoryError) as exc:
                    ctx.emit(Stage.FINALIZE, EventType.WARNING_RAISED, code="refit_failed",
                             attempt=attempt, error_class=type(exc).__name__)  # fmt: skip
            if final_model is None:
                raise TabPilotError(ErrorCode.NO_VALID_MODEL, reasons="final refit failed")
        ctx.winner_run = ctx.runs.get(selection.winner)
        winner_oof = ctx.winner_run.oof if ctx.winner_run is not None else None
        if winner_oof is not None:  # FR-TRN-01: the selected candidate's OOF is kept on disk
            store = OOFStore(ctx.run_dir / "oof")
            store.put(selection.winner, winner_oof)
            ctx.winner_oof_path = Path(store.persist(selection.winner)).relative_to(
                ctx.run_dir).as_posix()  # fmt: skip
        base = baseline_candidate(ctx.available_models() or [ModelName.LINEAR],
                                  ctx.runs[selection.winner].candidate.model)  # fmt: skip
        ctx.baseline_name = base.model.name.value
        ctx.baseline_model = refit_final(base, [], ctx.dev, ctx.split, schema=ctx.schema,
                                         registry=ctx.registry,
                                         profile=ctx.config.execution_profile)  # fmt: skip
        ctx.freeze.freeze()
        ctx.selection, ctx.final_model = selection, final_model
        ctx.emit(Stage.FINALIZE, EventType.SELECTION_FROZEN, winner=selection.winner,
                 final=chosen)  # fmt: skip
        holdout = evaluate_holdout(ctx.guard, final_model, ctx.baseline_model,
                                   frame=ctx.loaded.frame, y=ctx.dev.y, problem=ctx.problem,
                                   metric=ctx.metric, seed=ctx.seeds.root)  # fmt: skip
        ctx.holdout = holdout
        ctx.emit(Stage.FINALIZE, EventType.HOLDOUT_EVALUATED, present=holdout.present,
                 accesses=len(ctx.guard.access_log) if ctx.guard else 0,
                 refused=len(ctx.guard.refused_log) if ctx.guard else 0)  # fmt: skip
    final = FinalResult(selected_entry_id=chosen, is_ensemble=ensemble is not None,
                        holdout=holdout)  # fmt: skip
    return {"selection_frozen": True, "final": final, "best_candidate_id": selection.winner,
            "budget": ledger(ctx)}  # fmt: skip


# ---------------------------------------------------------------- explain, report, export
def _winner_oof(ctx: RunContext) -> Any:
    run = ctx.winner_run
    return run.oof if run is not None else None


def explain_node(state: RunState) -> dict[str, Any]:
    """§21.3–§21.5 within min(explain timeout, time left to 100%)."""
    ctx = _ctx(state)
    with ctx.stage(Stage.EXPLAIN):
        rows = np.asarray(ctx.split.dev_indices)
        frame = ctx.dev.frame.iloc[rows]
        position = {row: i for i, row in enumerate(ctx.split.dev_indices)}
        _, _, _, first_valid = next(iter(iter_folds(ctx.split)))
        cap = min(EXPLAIN_TIMEOUT_S[ExecutionProfileName(state.config.execution_profile)],
                  ctx.budget.remaining(Stage.EXPLAIN))  # fmt: skip
        settings = ExplainSettings(
            problem=ctx.problem,
            metric=ctx.metric,
            sources=list(ctx.schema.columns),
            seed=ctx.seeds.explain,
            profile=state.config.execution_profile,
            time_cap_s=cap,
        )
        data = ExplainData(frame, ctx.dev.y[rows], np.asarray([position[r] for r in first_valid]))
        explanation = explain(ctx.final_model, data, settings)
        if explanation.method not in ("shap_tree", "shap_linear", "shap_member_plus_permutation"):
            ctx.emit(Stage.EXPLAIN, EventType.EXPLANATION_FALLBACK, method=explanation.method,
                     reason=explanation.unavailable_reason)  # fmt: skip
        categorical = [c for c, kind in ctx.schema.columns.items()
                       if kind == SemanticType.CATEGORICAL]  # fmt: skip
        ranked = [c for c in explanation.top_features() if c in categorical]
        ordered = ranked + sorted(c for c in categorical if c not in ranked)
        oof = _winner_oof(ctx)
        ctx.explanation = explanation
        ctx.errors = (error_analysis(oof, ctx.dev.y[rows], frame, ctx.problem, ordered)
                      if oof is not None else None)  # fmt: skip
    final = (
        state.final.model_copy(update={"explanation_method": explanation.method})
        if state.final
        else None
    )
    return {"final": final, "budget": ledger(ctx)}


def report_state(state: RunState, ctx: RunContext) -> ReportState:
    assert state.plan is not None  # noqa: S101 - a report only follows a validated plan
    rows = np.asarray(ctx.split.dev_indices)
    thresholds = {
        eid: report
        for eid, cv in ctx.runs.items()
        if (report := threshold_report(cv.oof, ctx.dev.y, ctx.split, ctx.problem))
    }
    return ReportState(
        config=experiment_config(state, ctx), metadata=state.dataset_meta, plan=state.plan,
        entries=list(state.leaderboard), selection=ctx.selection, holdout=ctx.holdout,
        problem=ctx.problem, events=ctx.recorder.read_events(), profile=ctx.summary,
        plan_source=state.plan_source or "rule_based", plan_validation=state.plan_validation,
        winner=ctx.winner_run.candidate if ctx.winner_run else None,
        baseline_model=ctx.baseline_name, provenance=feature_provenance(state, ctx),
        reflections=list(state.reflection_history), ensemble=state.ensemble,
        thresholds=thresholds, ledger=ledger(ctx), explanation=ctx.explanation,
        errors=ctx.errors, winner_oof=_winner_oof(ctx), y_dev=ctx.dev.y[rows],
        status="completed_partial" if ctx.time_hit else "completed",
    )  # fmt: skip


def report_node(state: RunState) -> dict[str, Any]:
    """Facts, narrative (LLM when time and providers allow, else template), HTML/MD/JSON."""
    ctx = _ctx(state)
    with ctx.stage(Stage.REPORT):
        use_llm = _llm_on(ctx) and ctx.budget.remaining(Stage.REPORT) >= MIN_REPORT_LLM_S
        rstate = report_state(state, ctx)
        paths = build_report(rstate, ctx.run_dir / "report",
                             router=ctx.router if use_llm else None,
                             budget=ctx.llm_budget if use_llm else None)  # fmt: skip
        ctx.report = (rstate, paths)
        ctx.emit(Stage.REPORT, EventType.REPORT_RENDERED, narrative=paths.narrative.source,
                 figures=len(paths.figures))  # fmt: skip
    report_paths = [p.relative_to(ctx.run_dir).as_posix()
                    for p in (paths.html, paths.markdown, paths.report_json)]  # fmt: skip
    final = state.final.model_copy(update={"report_paths": report_paths}) if state.final else None
    return {"final": final, "budget": ledger(ctx)}


def export_node(state: RunState) -> dict[str, Any]:
    """Verified bundle (§22); raw input is deleted right after export (§24.7)."""
    ctx = _ctx(state)
    with ctx.stage(Stage.EXPORT):
        rstate, _ = ctx.report
        meta = state.dataset_meta
        original = next(c.original for c in meta.columns if c.name == state.config.target)
        llm_used = state.plan_source == "llm" or (ctx.llm_budget is not None
                                                  and ctx.llm_budget.calls > 0)  # fmt: skip
        settings = ExportSettings(
            out_dir=ctx.run_dir, max_bundle_mb=ctx.settings.limits.max_bundle_mb,
            profile=ExecutionProfileName(state.config.execution_profile),
            target_original=original, dev_positions=list(ctx.split.dev_indices),
            n_rows_valid=meta.n_rows_valid, scan_frame=ctx.loaded.frame,
            lineage=lineage(state, ctx),
            planner_input=planner_payload(ctx.planner_input) if llm_used else None,
            plan_received=ctx.plan_received,
            llm_calls=ctx.run_dir / "records" / "llm_calls.jsonl",
            include_oof=ctx.options.include_oof,
        )  # fmt: skip
        bundle = export_bundle(rstate, ctx.final_model, ctx.run_dir / "report", settings)
        ctx.bundle = bundle
        ctx.emit(Stage.EXPORT, EventType.BUNDLE_EXPORTED, file=bundle.path.name,
                 sha256=bundle.sha256, bytes=bundle.size_bytes,
                 parity_max_abs_diff=bundle.r1.report.max_abs_diff)  # fmt: skip
    final = state.final.model_copy(update={
        "bundle_path": bundle.path.relative_to(ctx.run_dir).as_posix(),
        "bundle_sha256": bundle.sha256}) if state.final else None  # fmt: skip
    status = "completed_partial" if ctx.time_hit else "completed"
    return {"final": final, "status": status, "budget": ledger(ctx)}


def lineage(state: RunState, ctx: RunContext) -> dict[str, Any]:
    """Selection and candidate lineage for `experiment.json` (§19.7)."""
    run = ctx.winner_run
    return {
        "winner": ctx.selection.winner if ctx.selection else None,
        "winner_oof": ctx.winner_oof_path,
        "final": state.final.selected_entry_id if state.final else None,
        "refit_params": refit_params(run.candidate, run.best_iterations) if run else {},
        "cv_best_iterations": run.best_iterations if run else [],
        "baseline_model": ctx.baseline_name, "plan_source": state.plan_source,
        "candidates": [{"candidate_id": c.candidate_id, "iteration": c.iteration,
                        "parent_entry_id": c.parent_entry_id, "change_set_id": c.change_set_id}
                       for c in state.candidates],
    }  # fmt: skip
