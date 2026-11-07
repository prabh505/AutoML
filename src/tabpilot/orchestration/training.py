"""Candidate training for the graph (PRD §16–§18, §19.4, §29.3; FR-TRN-01, FR-TUN-01).

Iteration 0 tunes every planned model within the train-stage time ledger, reserving trial 0
for each later model (§12.8). Reflection children are retrained once with their parent's tuned
parameters on the same folds, within the round budget (§19.4). Every candidate produces one
leaderboard entry, ok or not (§18.4).
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

from tabpilot.evaluation.leaderboard import make_entry, secondary_metrics
from tabpilot.orchestration.context import RunContext
from tabpilot.resources.estimator import SUBSAMPLE_ROWS
from tabpilot.schemas import (
    CandidateSpec,
    EventType,
    PlannerOutput,
    Simplification,
    Stage,
    Task,
)
from tabpilot.schemas.experiment import LeaderboardEntry, ModelConfig, TrainingResult
from tabpilot.training import CVRun, rf_resource_guard, run_cv
from tabpilot.tuning import TuningLimits, limits_for, tune


def initial_candidates(plan: PlannerOutput, ctx: RunContext) -> list[CandidateSpec]:
    """Iteration-0 candidates: one per planned model, with the plan's features (§12.3)."""
    task = ctx.config.task
    out = []
    for planned in sorted(plan.models, key=lambda m: (m.priority, m.name.value)):
        config = ModelConfig(name=planned.name, task=task, seed=ctx.seeds.model,
                             n_jobs=ctx.n_jobs, class_weight=planned.class_weight)  # fmt: skip
        out.append(CandidateSpec(
            candidate_id=f"0-{planned.name.value}", iteration=0, model=config,
            feature_set="baseline+llm" if plan.features else "baseline",
            features=list(plan.features), column_decisions=plan.column_decisions,
            imputation=plan.imputation, encoding=plan.encoding,
            target_transform=plan.target_transform if task == Task.REGRESSION else "none",
        ))  # fmt: skip
    return out


def entry_for(ctx: RunContext, result: TrainingResult, cand: CandidateSpec,
              cv: CVRun | None) -> LeaderboardEntry:  # fmt: skip
    """The leaderboard row for one trained (or skipped) candidate, with its lineage."""
    dev, split, metric = ctx.dev, ctx.split, ctx.metric
    assert dev is not None  # noqa: S101 - set by prepare()
    secondary = (secondary_metrics(cv.oof, dev.y, split, dev.problem, metric)
                 if cv is not None and result.status == "ok" else {})  # fmt: skip
    warnings = list(cv.warnings) if cv else []
    if cv and cv.dropped_features:
        warnings.append("features_dropped")
    entry = make_entry(
        result,
        run_id=ctx.run_id,
        model=cand.model.name,
        iteration=cand.iteration,
        feature_set=cand.feature_set,
        metric=metric,
        split=split,
        run_split_id=split.split_id,
        run_metric=metric,
        secondary=secondary,
        model_size_bytes=cv.model_size_bytes if cv else 0,
        n_features_out=cv.n_features_out if cv else 0,
        warnings=warnings,
    )
    return entry.model_copy(update={"parent_entry_id": cand.parent_entry_id,
                                    "change_set_id": cand.change_set_id})  # fmt: skip


def tuning_limits(ctx: RunContext, alloc: float, n_dev: int, max_trials: int | None, *,
                  reserved: int = 0) -> TuningLimits:  # fmt: skip
    """§17.3 limits with the §29.5 simplifications; `reserved` run trials stay available so
    every later planned model still gets its trial 0 (§12.8).
    """
    left = max(1, ctx.budget.trials_left() - reserved)
    limits = limits_for(ctx.config.execution_profile, alloc, n_dev, left,
                        mode=ctx.options.tuning_mode, max_trials=max_trials)  # fmt: skip
    if Simplification.REDUCE_TUNING_TRIALS in ctx.simplifications:
        limits = replace(limits, max_trials=max(1, limits.max_trials // 2))
    if Simplification.WITHIN_FOLD_SUBSAMPLE in ctx.simplifications and n_dev > SUBSAMPLE_ROWS:
        limits = replace(limits, train_row_cap=SUBSAMPLE_ROWS)
    return limits


def _skipped(ctx: RunContext, cand: CandidateSpec, status: str, reason: str) -> LeaderboardEntry:
    result = TrainingResult(entry_id=f"{cand.iteration}-{cand.model.name.value}-00000000",
                            status=status, error=reason)  # fmt: skip
    ctx.emit(Stage.TRAIN_CANDIDATES, EventType.CANDIDATE_FAILED, model=cand.model.name.value,
             status=status)  # fmt: skip
    return entry_for(ctx, result, cand, None)


def _record(ctx: RunContext, cv: CVRun, trials: int, stopped: str | None) -> LeaderboardEntry:
    ctx.runs[cv.result.entry_id] = cv
    entry = entry_for(ctx, cv.result, cv.candidate, cv)
    kind = EventType.CANDIDATE_COMPLETED if cv.result.status == "ok" else EventType.CANDIDATE_FAILED
    ctx.emit(Stage.TRAIN_CANDIDATES, kind, model=cv.candidate.model.name.value,
             entry_id=cv.result.entry_id, status=cv.result.status, trials=trials,
             stopped=stopped, mean=entry.mean)  # fmt: skip
    return entry


def train_initial(ctx: RunContext, plan: PlannerOutput,
                  candidates: list[CandidateSpec]) -> list[LeaderboardEntry]:  # fmt: skip
    """Tune each iteration-0 candidate within its share of the train-stage time (§17.3)."""
    split = ctx.split
    n_dev = len(split.dev_indices)
    guard = rf_resource_guard(ctx.config.execution_profile, n_dev, ctx.monitor.soft_mb)
    entries: list[LeaderboardEntry] = []
    pending = {c.model.name: plan.tuning_allocation.get(c.model.name, 0.0) for c in candidates}
    for index, cand in enumerate(candidates):
        ctx.check_cancel()
        name = cand.model.name
        later = len(candidates) - index - 1
        share = pending.pop(name)
        rest = share + sum(pending.values())
        alloc = ctx.budget.remaining(Stage.TRAIN_CANDIDATES) * (share / rest if rest else 0.0)
        reason = guard(cand)
        out_of_trials = ctx.budget.trials_left() == 0
        out_of_time = ctx.options.tuning_mode == "time" and ctx.budget.expired(
            Stage.TRAIN_CANDIDATES)  # fmt: skip
        if reason or ctx.monitor.stop_training or out_of_trials or out_of_time:
            if out_of_time:
                ctx.time_hit = True
            status = "skipped_resource" if reason else "skipped_budget"
            note = reason or ("run trial cap reached" if out_of_trials else "time budget reached")
            entries.append(_skipped(ctx, cand, status, note))
            continue
        if ctx.monitor.soft_exceeded and not ctx.soft_warned:
            ctx.soft_warned = True
            ctx.emit(Stage.TRAIN_CANDIDATES, EventType.RESOURCE_WARNING, level="soft",
                     action="no extra tuning trials")  # fmt: skip
        extra = share > 0 and not ctx.monitor.soft_exceeded
        limits = tuning_limits(ctx, alloc, n_dev, ctx.options.trials_per_model if extra else 1,
                               reserved=later)  # fmt: skip
        ctx.emit(Stage.TRAIN_CANDIDATES, EventType.CANDIDATE_STARTED, model=name.value,
                 allocation_s=round(alloc, 1), max_trials=limits.max_trials)  # fmt: skip
        tuned = tune(cand, ctx.dev, split, schema=ctx.schema, registry=ctx.registry,
                     metric=ctx.metric, allocation_s=alloc, limits=limits, seed=ctx.seeds.optuna,
                     profile=ctx.config.execution_profile, cancel=ctx.cancel,
                     extra_hook=ctx.train_hook(),
                     stop=lambda: ctx.monitor.soft_exceeded or ctx.monitor.stop_training,
                     on_trial=lambda r: ctx.recorder.trial(dataclasses.asdict(r)))  # fmt: skip
        ctx.budget.use_trials(len(tuned.trials))
        ctx.initial_trials += len(tuned.trials)
        entries.append(_record(ctx, tuned.best, len(tuned.trials), tuned.stopped))
    return entries


def train_children(ctx: RunContext, children: list[CandidateSpec],
                   trials: int) -> list[LeaderboardEntry]:  # fmt: skip
    """Retrain reflection children once each (parent's tuned parameters, same folds), at most
    `trials` of them, stopping at the reflection-stage boundary (§19.4).
    """
    guard = rf_resource_guard(ctx.config.execution_profile, len(ctx.split.dev_indices),
                              ctx.monitor.soft_mb)  # fmt: skip
    entries = []
    for index, child in enumerate(children):
        ctx.check_cancel()
        if index >= trials or ctx.budget.trials_left() == 0:
            entries.append(_skipped(ctx, child, "skipped_budget", "reflection trial budget"))
            continue
        if ctx.budget.expired(Stage.APPLY_CHANGES) or ctx.monitor.stop_training:
            entries.append(_skipped(ctx, child, "skipped_budget", "reflection time budget"))
            continue
        ctx.emit(Stage.TRAIN_CANDIDATES, EventType.CANDIDATE_STARTED,
                 model=child.model.name.value, parent=child.parent_entry_id)  # fmt: skip
        cv = run_cv(child, ctx.dev, ctx.split, schema=ctx.schema, registry=ctx.registry,
                    metric=ctx.metric, cancel=ctx.cancel,
                    budget_hook=ctx.train_hook(Stage.APPLY_CHANGES), resource_guard=guard,
                    profile=ctx.config.execution_profile)  # fmt: skip
        ctx.budget.use_trials(1)
        entries.append(_record(ctx, cv, 1, None))
    return entries
