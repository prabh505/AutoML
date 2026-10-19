"""Headless CLI: `tabpilot run <file> --target <col> --out <dir>` (PROMPTS P09; FR-PLN-03).

Runs the LLM-free product in-process: ingest → split → profile → rule-based plan → tuned CV per
model → leaderboard and selection → final refit → freeze → one-time holdout → run records. P15
routes the same stages through orchestration.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil
from ulid import ULID

from tabpilot import __version__
from tabpilot.config import Settings, load_settings
from tabpilot.evaluation.holdout import evaluate_holdout, make_guard
from tabpilot.evaluation.leaderboard import make_entry, not_comparable, secondary_metrics
from tabpilot.evaluation.metrics import Problem, default_primary, selection_score
from tabpilot.evaluation.metrics import spec as metric_spec
from tabpilot.evaluation.selection import SelectionFreeze, select
from tabpilot.ingestion import apply_target, load_dataset, validate_target
from tabpilot.ingestion.task import detect_task, positive_label
from tabpilot.models import ModelRegistry
from tabpilot.planning import rule_based_plan
from tabpilot.preprocessing import schema_from
from tabpilot.profiling import apply_leakage_exclusions, profile
from tabpilot.resources import ResourceMonitor, TimeBudget
from tabpilot.schemas import (
    CandidateSpec,
    EventType,
    ExecutionProfileName,
    MetricName,
    ModelName,
    PlannerOutput,
    Stage,
    TabPilotError,
    Task,
)
from tabpilot.schemas.experiment import (
    Budgets,
    EvaluationResult,
    ExperimentConfig,
    LeaderboardEntry,
    LLMSettingsSummary,
    ModelConfig,
    Seeds,
    SplitSummary,
    TrainingResult,
)
from tabpilot.splitting import build_split
from tabpilot.tracking import RunRecorder, dumps
from tabpilot.training import (
    BudgetExceeded,
    CVRun,
    DevData,
    baseline_candidate,
    refit_final,
    refit_params,
    rf_resource_guard,
)
from tabpilot.tuning import TuningMode, limits_for, tune

LIBRARIES = ("scikit-learn", "lightgbm", "optuna", "pandas", "numpy", "xgboost", "catboost")


@dataclass(frozen=True)
class RunArgs:
    """CLI options for `tabpilot run`."""

    file: Path
    target: str
    out: Path
    task: Task | None = None
    budget_s: int | None = None
    seed: int = 42
    profile: ExecutionProfileName = ExecutionProfileName.LOCAL
    tuning_mode: TuningMode = "time"
    trials: int | None = None
    models: tuple[ModelName, ...] = ()


@dataclass
class RunSummary:
    """What the CLI prints and tests inspect."""

    run_dir: Path
    task: Task
    metric: MetricName
    entries: list[LeaderboardEntry]
    winner: str
    holdout: dict[str, Any]


def _library_versions() -> dict[str, str]:
    found = {}
    for name in LIBRARIES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def _target_column(loaded: Any, target: str) -> str:
    """Accept the original or the normalized target name."""
    for col in loaded.metadata.columns:
        if target in (col.original, col.name):
            return str(col.name)
    raise ValueError(f"target column {target!r} is not in the file")


def _problem(frame: Any, target: str, task: Task, positive: str | None) -> Problem:
    if task == Task.REGRESSION:
        return Problem(Task.REGRESSION)
    classes = tuple(sorted(str(v) for v in frame[target].dropna().unique()))
    return Problem(task, classes, classes.index(positive) if task == Task.BINARY else None)


class _Run:
    """State of one CLI run (in-process stages)."""

    def __init__(self, args: RunArgs, settings: Settings) -> None:
        self.args = args
        self.settings = settings
        self.limits = settings.limits
        self.run_id = str(ULID())
        self.dir = args.out
        self.recorder = RunRecorder(self.dir, self.run_id)
        total = args.budget_s or self.limits.time_budget_default_s
        if total > self.limits.time_budget_max_s:
            raise ValueError(f"--budget-s {total} exceeds the {args.profile.value} maximum "
                             f"{self.limits.time_budget_max_s}")  # fmt: skip
        self.budget = TimeBudget(float(total), self.limits.max_trials_per_run)
        soft, hard = self.limits.rss_limits_mb(psutil.virtual_memory().total / 2**20)
        self.monitor = ResourceMonitor(soft, hard)
        self.registry = ModelRegistry(settings)
        self.seeds = Seeds.derive(args.seed)
        self.n_jobs = self.limits.resolved_n_jobs(psutil.cpu_count(logical=False) or 1)
        self.runs: dict[str, CVRun] = {}

    @contextmanager
    def stage(self, stage: Stage) -> Iterator[None]:
        self.budget.begin(stage)
        self.recorder.event(stage, EventType.STAGE_STARTED)
        yield
        self.budget.end(stage)
        self.recorder.event(stage, EventType.STAGE_COMPLETED,
                            seconds=round(self.budget.timings[stage].duration, 3))  # fmt: skip

    def train_hook(self) -> Callable[[], None]:
        timed = self.args.tuning_mode == "time"

        def hook() -> None:
            self.monitor.check()
            if timed and self.budget.expired(Stage.TRAIN_CANDIDATES):
                raise BudgetExceeded

        return hook


def _candidates(plan: PlannerOutput, task: Task, run: _Run) -> list[CandidateSpec]:
    out = []
    for planned in sorted(plan.models, key=lambda m: m.priority):
        config = ModelConfig(name=planned.name, task=task, seed=run.seeds.model, n_jobs=run.n_jobs,
                             class_weight=planned.class_weight)  # fmt: skip
        out.append(CandidateSpec(
            candidate_id=f"0-{planned.name.value}", iteration=0, model=config,
            feature_set="baseline", column_decisions=plan.column_decisions,
            imputation=plan.imputation, encoding=plan.encoding,
            target_transform=plan.target_transform if task == Task.REGRESSION else "none",
        ))  # fmt: skip
    return out


def _train(
    run: _Run,
    *,
    plan: PlannerOutput,
    candidates: list[CandidateSpec],
    dev: DevData,
    split: Any,
    schema: Any,
    metric: MetricName,
) -> list[LeaderboardEntry]:
    profile_name = run.args.profile
    guard = rf_resource_guard(profile_name, len(split.dev_indices), run.monitor.soft_mb)
    entries: list[LeaderboardEntry] = []
    pending = {c.model.name: plan.tuning_allocation.get(c.model.name, 0.0) for c in candidates}
    for cand in candidates:
        name = cand.model.name
        share = pending.pop(name)
        rest = share + sum(pending.values())
        alloc = run.budget.remaining(Stage.TRAIN_CANDIDATES) * (share / rest if rest else 0.0)
        reason = guard(cand)
        if reason or run.monitor.stop_training:
            status = "skipped_resource" if reason else "skipped_budget"
            result = TrainingResult(
                entry_id=f"0-{name.value}-00000000", status=status, error=reason
            )
            entries.append(_entry(run, result=result, cand=cand, dev=dev, split=split,
                                  metric=metric, cv=None))  # fmt: skip
            run.recorder.event(Stage.TRAIN_CANDIDATES, EventType.CANDIDATE_FAILED,
                               model=name.value, status=status)  # fmt: skip
            continue
        max_trials = run.args.trials if share > 0 else 1
        limits = limits_for(profile_name, alloc, len(split.dev_indices),
                            max(1, run.budget.trials_left()), mode=run.args.tuning_mode,
                            max_trials=max_trials)  # fmt: skip
        run.recorder.event(Stage.TRAIN_CANDIDATES, EventType.CANDIDATE_STARTED, model=name.value,
                           allocation_s=round(alloc, 1), max_trials=limits.max_trials)  # fmt: skip
        tuned = tune(cand, dev, split, schema=schema, registry=run.registry, metric=metric,
                     allocation_s=alloc, limits=limits, seed=run.seeds.optuna,
                     profile=profile_name, extra_hook=run.train_hook(),
                     on_trial=lambda r: run.recorder.trial(dataclasses.asdict(r)))  # fmt: skip
        run.budget.use_trials(len(tuned.trials))
        best = tuned.best
        run.runs[best.result.entry_id] = best
        entries.append(_entry(run, result=best.result, cand=best.candidate, dev=dev,
                              split=split, metric=metric, cv=best))  # fmt: skip
        kind = (EventType.CANDIDATE_COMPLETED if best.result.status == "ok"
                else EventType.CANDIDATE_FAILED)  # fmt: skip
        run.recorder.event(Stage.TRAIN_CANDIDATES, kind, model=name.value,
                           entry_id=best.result.entry_id, status=best.result.status,
                           trials=len(tuned.trials), stopped=tuned.stopped)  # fmt: skip
    return entries


def _entry(
    run: _Run,
    *,
    result: TrainingResult,
    cand: CandidateSpec,
    dev: DevData,
    split: Any,
    metric: MetricName,
    cv: CVRun | None,
) -> LeaderboardEntry:
    secondary = (secondary_metrics(cv.oof, dev.y, split, dev.problem, metric)
                 if cv is not None and result.status == "ok" else {})  # fmt: skip
    warnings = list(cv.warnings) if cv else []
    if cv and cv.dropped_features:
        warnings.append("features_dropped")
    return make_entry(result, run_id=run.run_id, model=cand.model.name, iteration=0,
                      feature_set=cand.feature_set, metric=metric, split=split,
                      run_split_id=split.split_id, run_metric=metric, secondary=secondary,
                      model_size_bytes=cv.model_size_bytes if cv else 0,
                      n_features_out=cv.n_features_out if cv else 0, warnings=warnings)  # fmt: skip


def execute(args: RunArgs) -> RunSummary:  # noqa: PLR0915 - the stage sequence, top to bottom
    """Run every stage and write the run directory."""
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"output directory {args.out} is not empty")
    settings = load_settings({**os.environ, "TABPILOT_PROFILE": args.profile.value})
    run = _Run(args, settings)
    run.monitor.start()
    run.recorder.event(Stage.RUN, EventType.STAGE_STARTED, profile=args.profile.value)
    try:
        with run.stage(Stage.INGEST):
            loaded = load_dataset(args.file, args.file.name, settings, run.run_id)
            target = _target_column(loaded, args.target)
            detected = detect_task(loaded.frame[target].dropna())
            task = args.task or detected.task
            loaded = apply_target(loaded, validate_target(loaded, target, task))
            positive = (detected.positive_label or positive_label(loaded.frame[target].dropna())
                        if task == Task.BINARY else None)  # fmt: skip
        with run.stage(Stage.SPLIT):
            split_result = build_split(loaded.frame[target], task, args.seed,
                                       features=loaded.frame.drop(columns=[target]))  # fmt: skip
            split = split_result.spec
            for code in split_result.warnings:
                run.recorder.event(Stage.SPLIT, EventType.WARNING_RAISED, code=str(code))
        with run.stage(Stage.PROFILE):
            summary = profile(loaded.frame, split, loaded.metadata, task,
                              execution_profile=args.profile, time_budget_s=int(run.budget.total_s),
                              seed_root=args.seed, positive_label=positive)  # fmt: skip
            metadata = apply_leakage_exclusions(loaded.metadata, summary.leakage, [])
        with run.stage(Stage.PLAN):
            available = run.registry.available(args.profile)
            if args.models:
                unknown = [m.value for m in args.models if m not in available]
                if unknown:
                    raise ValueError(f"models not available on {args.profile.value}: {unknown}")
                available = [m for m in available if m in args.models]
            plan = rule_based_plan(summary, task, available)
            run.recorder.event(Stage.PLAN, EventType.PLAN_RECEIVED, source="rule_based",
                               models=[m.name.value for m in plan.models])  # fmt: skip
        problem = _problem(loaded.frame, target, task, positive)
        dev = DevData.from_frame(loaded.frame, target, problem)
        schema = schema_from(metadata, summary)
        metric = default_primary(task, summary.target.positive_share)
        with run.stage(Stage.TRAIN_CANDIDATES):
            entries = _train(run, plan=plan, candidates=_candidates(plan, task, run), dev=dev,
                             split=split, schema=schema, metric=metric)  # fmt: skip
        with run.stage(Stage.EVALUATE):
            freeze = SelectionFreeze()
            selection = select(entries, freeze)
            if not_comparable(entries):
                run.recorder.event(Stage.EVALUATE, EventType.WARNING_RAISED, code="not_comparable")
            run.recorder.event(Stage.EVALUATE, EventType.LEADERBOARD_UPDATED, n=len(entries),
                               winner=selection.winner)  # fmt: skip
        with run.stage(Stage.FINALIZE):
            best = run.runs[selection.winner]
            final_model = refit_final(best.candidate, best.best_iterations, dev, split,
                                      schema=schema, registry=run.registry,
                                      profile=args.profile)  # fmt: skip
            base = baseline_candidate(available, best.candidate.model)
            baseline_model = refit_final(base, [], dev, split, schema=schema,
                                         registry=run.registry, profile=args.profile)  # fmt: skip
            freeze.freeze()
            run.recorder.event(Stage.FINALIZE, EventType.SELECTION_FROZEN, winner=selection.winner)
            guard = make_guard(split, freeze.is_frozen) if split.holdout_present else None
            holdout = evaluate_holdout(guard, final_model, baseline_model, frame=loaded.frame,
                                       y=dev.y, problem=problem, metric=metric,
                                       seed=run.seeds.root)  # fmt: skip
            run.recorder.event(Stage.FINALIZE, EventType.HOLDOUT_EVALUATED,
                               present=holdout.present,
                               accesses=len(guard.access_log) if guard else 0)  # fmt: skip
    finally:
        run.monitor.stop()
    evaluation = EvaluationResult(entries=entries, selection=selection, holdout=holdout)
    config = ExperimentConfig(
        run_id=run.run_id, tabpilot_version=__version__, execution_profile=args.profile,
        task=task, target=target, positive_label=positive, primary_metric=metric_spec(metric),
        split=SplitSummary(split_id=split.split_id, strategy=str(split.strategy), k=split.k,
                           n_repeats=split.n_repeats, holdout_fraction=split.holdout_fraction,
                           holdout_present=split.holdout_present, group_column=split.group_column,
                           duplicate_grouping=split.duplicate_grouping),
        seeds=run.seeds,
        budgets=Budgets(time_s=int(run.budget.total_s), max_trials=run.budget.trials_max,
                        llm_calls=0, reflection_rounds=0),
        llm=LLMSettingsSummary(enabled=False),
        library_versions=_library_versions(),
    )  # fmt: skip
    timings = {
        s.value: {"start": round(t.start, 3), "end": None if t.end is None else round(t.end, 3),
                  "duration": round(t.duration, 3)}
        for s, t in run.budget.timings.items()
    }  # fmt: skip
    run.recorder.stage_timings(timings)
    run.recorder.resources(
        [dataclasses.asdict(s) for s in run.monitor.samples], run.monitor.peak_mb
    )
    lineage = {"winner": selection.winner, "refit_params": refit_params(best.candidate,
               best.best_iterations), "cv_best_iterations": best.best_iterations,
               "baseline_model": base.model.name.value, "plan_source": "rule_based"}  # fmt: skip
    run.recorder.experiment({"config": config, "plan": plan, "evaluation": evaluation,
                             "budget": run.budget.ledger(), "lineage": lineage})  # fmt: skip
    for name, payload in (("leaderboard.json", entries), ("selection.json", selection),
                          ("holdout.json", holdout)):  # fmt: skip
        (run.dir / name).write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")
    run.recorder.event(Stage.RUN, EventType.RUN_COMPLETED, winner=selection.winner)
    return RunSummary(run.dir, task, metric, entries, selection.winner,
                      holdout.model_dump(mode="json", exclude_none=True))  # fmt: skip


def _format(summary: RunSummary) -> list[str]:
    lines = [f"run directory: {summary.run_dir}", f"task: {summary.task.value}  "
             f"metric: {summary.metric.value}", "leaderboard (CV on DEV):"]  # fmt: skip

    def rank(e: LeaderboardEntry) -> tuple[bool, float, str]:
        sel = selection_score(summary.metric, e.mean) if e.mean is not None else 0.0
        return (e.status != "ok", -sel, e.entry_id)

    ranked = sorted(summary.entries, key=rank)
    for e in ranked:
        stats = (f"{e.mean:.4f} ± {e.std:.4f} over {len(e.fold_scores)} folds"
                 if e.mean is not None and e.std is not None else e.status)  # fmt: skip
        mark = "*" if e.entry_id == summary.winner else " "
        lines.append(f" {mark} {e.entry_id:<28} {stats}")
    lines.append(f"selected: {summary.winner}")
    hold = summary.holdout
    if hold.get("present"):
        lines.append(f"holdout (independent test, used once): final {hold['final']}  "
                     f"baseline {hold.get('baseline')}")  # fmt: skip
    else:
        lines.append(f"holdout: {hold.get('reason')}")
    return lines


def _models(text: str | None) -> tuple[ModelName, ...]:
    return tuple(ModelName(m.strip()) for m in text.split(",")) if text else ()


def parse(argv: Sequence[str] | None = None) -> RunArgs:
    parser = argparse.ArgumentParser(prog="tabpilot", description="TabPilot headless runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run AutoML on a CSV/Parquet file (rule-based plan)")
    run.add_argument("file", type=Path)
    run.add_argument("--target", required=True)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--task", choices=[t.value for t in Task])
    run.add_argument("--budget-s", type=int)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--profile", choices=[p.value for p in ExecutionProfileName], default="local")
    run.add_argument("--tuning-mode", choices=["trials", "time"], default="time")
    run.add_argument("--trials", type=int, help="trials per tuned model (trials mode)")
    run.add_argument("--models", help="comma-separated subset of the available models")
    ns = parser.parse_args(argv)
    return RunArgs(file=ns.file, target=ns.target, out=ns.out,
                   task=Task(ns.task) if ns.task else None, budget_s=ns.budget_s, seed=ns.seed,
                   profile=ExecutionProfileName(ns.profile), tuning_mode=ns.tuning_mode,
                   trials=ns.trials,
                   models=_models(ns.models))  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; exit 0 on success, 2 on a user-facing error."""
    args = parse(argv)
    try:
        summary = execute(args)
    except TabPilotError as exc:
        sys.stderr.write(f"error: {exc.user_message}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write("\n".join(_format(summary)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
