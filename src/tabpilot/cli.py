"""Headless CLI: `tabpilot run <file> --target <col> --out <dir>` (PROMPTS P09; FR-PLN-03).

Runs the LLM-free product in-process: ingest → split → profile → rule-based plan → tuned CV per
model → leaderboard and selection → final refit → freeze → one-time holdout → explanations and
error analysis → report (template narrative) → verified export bundle → run records. P15
routes the same stages through orchestration.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from ulid import ULID

from tabpilot import __version__
from tabpilot.config import Settings, load_settings
from tabpilot.evaluation.holdout import evaluate_holdout, make_guard
from tabpilot.evaluation.leaderboard import (
    make_entry,
    not_comparable,
    secondary_metrics,
    threshold_report,
)
from tabpilot.evaluation.metrics import Problem, default_primary, selection_score
from tabpilot.evaluation.metrics import spec as metric_spec
from tabpilot.evaluation.selection import NoValidModelError, SelectionFreeze, select
from tabpilot.explain import (
    EXPLAIN_TIMEOUT_S,
    ErrorAnalysis,
    ExplainData,
    ExplainSettings,
    ExplanationResult,
    error_analysis,
    explain,
)
from tabpilot.export import ExportSettings, export_bundle
from tabpilot.ingestion import apply_target, load_dataset, validate_target
from tabpilot.ingestion.task import detect_task, positive_label
from tabpilot.models import ModelRegistry
from tabpilot.planning import rule_based_plan
from tabpilot.preprocessing import PipelineSchema, schema_from
from tabpilot.profiling import apply_leakage_exclusions, profile
from tabpilot.reporting import ReportState, build_report
from tabpilot.resources import ResourceMonitor, TimeBudget
from tabpilot.resources.estimator import (
    ONEHOT_CAP_SIMPLIFIED,
    SUBSAMPLE_ROWS,
    estimate_feasibility,
    raise_if_infeasible,
)
from tabpilot.schemas import (
    CandidateSpec,
    ColumnRole,
    ErrorCode,
    EventType,
    ExecutionProfileName,
    FeasibilityReport,
    MetricName,
    ModelName,
    PlannerOutput,
    RunConfig,
    SemanticType,
    Simplification,
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
from tabpilot.schemas.run import MIN_TIME_BUDGET_S
from tabpilot.splitting import build_split, iter_folds
from tabpilot.tracking import RunRecorder, dumps
from tabpilot.training import (
    BudgetExceeded,
    CVRun,
    DevData,
    OOFStore,
    baseline_candidate,
    refit_final,
    refit_params,
    rf_resource_guard,
)
from tabpilot.tuning import TuningLimits, TuningMode, limits_for, tune

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
    bundle: Path | None = None
    bundle_sha256: str | None = None


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
        if not MIN_TIME_BUDGET_S <= total <= self.limits.time_budget_max_s:
            raise ValueError(f"--budget-s {total} is outside {MIN_TIME_BUDGET_S}–"
                             f"{self.limits.time_budget_max_s} s, the {args.profile.value} "
                             "range (maximum)")  # fmt: skip
        self.budget = TimeBudget(float(total), self.limits.max_trials_per_run)
        soft, hard = self.limits.rss_limits_mb(psutil.virtual_memory().total / 2**20)
        self.monitor = ResourceMonitor(soft, hard)
        self.registry = ModelRegistry(settings)
        self.seeds = Seeds.derive(args.seed)
        self.n_jobs = self.limits.resolved_n_jobs(psutil.cpu_count(logical=False) or 1)
        self.runs: dict[str, CVRun] = {}
        self.freeze = SelectionFreeze()
        self.guard: Any = None  # HoldoutGuard, built at split time (L-2, I-1)
        self.simplifications: set[Simplification] = set()
        self.soft_warned = False

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
    for index, cand in enumerate(candidates):
        name = cand.model.name
        later = len(candidates) - index - 1  # trial 0 is reserved for every planned model
        share = pending.pop(name)
        rest = share + sum(pending.values())
        alloc = run.budget.remaining(Stage.TRAIN_CANDIDATES) * (share / rest if rest else 0.0)
        reason = guard(cand)
        out_of_trials = run.budget.trials_left() == 0  # run-level cap (§17.3), KI-016
        if reason or run.monitor.stop_training or out_of_trials:
            status = "skipped_resource" if reason else "skipped_budget"
            reason = reason or ("run trial cap reached" if out_of_trials else None)
            result = TrainingResult(
                entry_id=f"0-{name.value}-00000000", status=status, error=reason
            )
            entries.append(_entry(run, result=result, cand=cand, dev=dev, split=split,
                                  metric=metric, cv=None))  # fmt: skip
            run.recorder.event(Stage.TRAIN_CANDIDATES, EventType.CANDIDATE_FAILED,
                               model=name.value, status=status)  # fmt: skip
            continue
        if run.monitor.soft_exceeded and not run.soft_warned:
            run.soft_warned = True
            run.recorder.event(Stage.TRAIN_CANDIDATES, EventType.RESOURCE_WARNING,
                               level="soft", action="no extra tuning trials")  # fmt: skip
        extra = share > 0 and not run.monitor.soft_exceeded
        limits = _limits(run, alloc, len(split.dev_indices), run.args.trials if extra else 1,
                         reserved=later)  # fmt: skip
        run.recorder.event(Stage.TRAIN_CANDIDATES, EventType.CANDIDATE_STARTED, model=name.value,
                           allocation_s=round(alloc, 1), max_trials=limits.max_trials)  # fmt: skip
        tuned = tune(cand, dev, split, schema=schema, registry=run.registry, metric=metric,
                     allocation_s=alloc, limits=limits, seed=run.seeds.optuna,
                     profile=profile_name, extra_hook=run.train_hook(),
                     stop=lambda: run.monitor.soft_exceeded or run.monitor.stop_training,
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


def _limits(
    run: _Run, alloc: float, n_dev: int, max_trials: int | None, *, reserved: int = 0
) -> TuningLimits:
    """§17.3 limits with the §29.5 simplifications chosen by the feasibility check. `reserved`
    run trials stay available so every later planned model still gets its trial 0 (§12.8).
    """
    left = max(1, run.budget.trials_left() - reserved)
    limits = limits_for(run.args.profile, alloc, n_dev, left,
                        mode=run.args.tuning_mode, max_trials=max_trials)  # fmt: skip
    if Simplification.REDUCE_TUNING_TRIALS in run.simplifications:
        limits = replace(limits, max_trials=max(1, limits.max_trials // 2))
    if Simplification.WITHIN_FOLD_SUBSAMPLE in run.simplifications and n_dev > SUBSAMPLE_ROWS:
        limits = replace(limits, train_row_cap=SUBSAMPLE_ROWS)
    return limits


def _feasibility(
    run: _Run, loaded: Any, target: str, task: Task, positive: str | None
) -> FeasibilityReport:
    """FR-ING-05: estimate peak memory before any training; reject, or record the §29.5
    simplifications that this run then applies.
    """
    config = RunConfig(run_id=run.run_id, target=target, task=task, positive_label=positive,
                       primary_metric=default_primary(task), time_budget_s=int(run.budget.total_s),
                       seed_root=run.args.seed, llm_mode="off", enable_ensemble=False,
                       execution_profile=run.args.profile)  # fmt: skip
    report = estimate_feasibility(loaded.metadata, config, run.settings,
                                  psutil.virtual_memory().total / 2**20)  # fmt: skip
    raise_if_infeasible(report)
    run.simplifications = set(report.simplifications)
    if report.simplifications:
        run.recorder.event(Stage.INGEST, EventType.WARNING_RAISED, code="feasibility_simplified",
                           simplifications=[s.value for s in report.simplifications])  # fmt: skip
    return report


def _simplified_schema(run: _Run, schema: PipelineSchema) -> PipelineSchema:
    if Simplification.CAP_ONEHOT in run.simplifications:
        return replace(schema, onehot_max_categories=ONEHOT_CAP_SIMPLIFIED)
    return schema


def _explain(
    run: _Run,
    *,
    final_model: Any,
    best: CVRun,
    dev: DevData,
    split: Any,
    schema: PipelineSchema,
    metric: MetricName,
) -> tuple[ExplanationResult, ErrorAnalysis]:
    """§21.3–§21.5 on DEV rows only: explanations under the explain time cap, then error
    analysis on the winner's OOF predictions.
    """
    rows = np.asarray(split.dev_indices)
    frame = dev.frame.iloc[rows]
    position = {row: i for i, row in enumerate(split.dev_indices)}
    _, _, _, first_valid = next(iter(iter_folds(split)))
    cap = min(EXPLAIN_TIMEOUT_S[run.args.profile], run.budget.remaining(Stage.EXPLAIN))
    settings = ExplainSettings(
        problem=dev.problem,
        metric=metric,
        sources=list(schema.columns),
        seed=run.seeds.explain,
        profile=run.args.profile,
        time_cap_s=cap,
    )
    data = ExplainData(frame, dev.y[rows], np.asarray([position[r] for r in first_valid]))
    explanation = explain(final_model, data, settings)
    if explanation.method not in ("shap_tree", "shap_linear", "shap_member_plus_permutation"):
        run.recorder.event(
            Stage.EXPLAIN,
            EventType.EXPLANATION_FALLBACK,
            method=explanation.method,
            reason=explanation.unavailable_reason,
        )
    categorical = [c for c, kind in schema.columns.items() if kind == SemanticType.CATEGORICAL]
    ranked = [c for c in explanation.top_features() if c in categorical]
    ordered = ranked + sorted(c for c in categorical if c not in ranked)
    assert best.oof is not None  # noqa: S101 - an ok winner always has OOF
    return explanation, error_analysis(best.oof, dev.y[rows], frame, dev.problem, ordered)


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
            feasibility = _feasibility(run, loaded, target, task, positive)
        with run.stage(Stage.SPLIT):
            k = 3 if Simplification.REDUCE_FOLDS in run.simplifications else 5
            # duplicate grouping (§11.6) hashes feature columns only: an excluded ID column
            # would make every row unique (KI-013)
            feature_cols = [c.name for c in loaded.metadata.columns if c.role == ColumnRole.FEATURE]
            features = loaded.frame[feature_cols]
            split_result = build_split(loaded.frame[target], task, args.seed, features=features,
                                       k=k)  # fmt: skip
            split = split_result.spec
            if split.holdout_present:  # HOLD is sealed from here until selection is frozen
                run.guard = make_guard(split, run.freeze.is_frozen)
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
            if Simplification.DISABLE_RANDOM_FOREST in run.simplifications:
                available = [m for m in available if m != ModelName.RANDOM_FOREST]
            plan = rule_based_plan(summary, task, available)
            run.recorder.event(Stage.PLAN, EventType.PLAN_RECEIVED, source="rule_based",
                               models=[m.name.value for m in plan.models])  # fmt: skip
        problem = _problem(loaded.frame, target, task, positive)
        dev = DevData.from_frame(loaded.frame, target, problem)
        schema = _simplified_schema(run, schema_from(metadata, summary))
        metric = default_primary(task, summary.target.positive_share)
        with run.stage(Stage.TRAIN_CANDIDATES):
            entries = _train(run, plan=plan, candidates=_candidates(plan, task, run), dev=dev,
                             split=split, schema=schema, metric=metric)  # fmt: skip
        with run.stage(Stage.EVALUATE):
            try:
                selection = select(entries, run.freeze)
            except NoValidModelError as exc:
                reasons = "; ".join(f"{e.model}: {e.warnings[-1] if e.warnings else e.status}"
                                    for e in entries)[:400]  # fmt: skip
                raise TabPilotError(ErrorCode.NO_VALID_MODEL, reasons=reasons) from exc
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
            run.freeze.freeze()
            run.recorder.event(Stage.FINALIZE, EventType.SELECTION_FROZEN, winner=selection.winner)
            guard = run.guard
            holdout = evaluate_holdout(guard, final_model, baseline_model, frame=loaded.frame,
                                       y=dev.y, problem=problem, metric=metric,
                                       seed=run.seeds.root)  # fmt: skip
            run.recorder.event(Stage.FINALIZE, EventType.HOLDOUT_EVALUATED,
                               present=holdout.present,
                               accesses=len(guard.access_log) if guard else 0,
                               refused=len(guard.refused_log) if guard else 0)  # fmt: skip
            store = OOFStore(run.dir / "oof")
            assert best.oof is not None  # noqa: S101 - an ok winner always has OOF
            store.put(selection.winner, best.oof)
            oof_path = Path(store.persist(selection.winner)).relative_to(run.dir).as_posix()
        with run.stage(Stage.EXPLAIN):
            explanation, errors = _explain(
                run,
                final_model=final_model,
                best=best,
                dev=dev,
                split=split,
                schema=schema,
                metric=metric,
            )
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
    rows = np.asarray(split.dev_indices)
    thresholds = {eid: report for eid, cv in run.runs.items()
                  if (report := threshold_report(cv.oof, dev.y, split, problem))}  # fmt: skip
    with run.stage(Stage.REPORT):
        state = ReportState(config=config, metadata=metadata, plan=plan, entries=entries,
                            selection=selection, holdout=holdout, problem=problem,
                            events=run.recorder.read_events(), profile=summary,
                            winner=best.candidate, baseline_model=base.model.name.value,
                            thresholds=thresholds, ledger=run.budget.ledger(),
                            explanation=explanation, errors=errors, winner_oof=best.oof,
                            y_dev=dev.y[rows])  # fmt: skip
        paths = build_report(state, run.dir / "report")
        run.recorder.event(
            Stage.REPORT,
            EventType.REPORT_RENDERED,
            narrative=paths.narrative.source,
            figures=len(paths.figures),
        )
    lineage = {"winner": selection.winner, "refit_params": refit_params(best.candidate,
               best.best_iterations), "cv_best_iterations": best.best_iterations,
               "baseline_model": base.model.name.value, "plan_source": "rule_based",
               "winner_oof": oof_path}  # fmt: skip
    with run.stage(Stage.EXPORT):
        original = next(c.original for c in metadata.columns if c.name == target)
        bundle = export_bundle(state, final_model, run.dir / "report", ExportSettings(
            out_dir=run.dir, max_bundle_mb=run.limits.max_bundle_mb, profile=args.profile,
            target_original=original, dev_positions=list(split.dev_indices),
            n_rows_valid=metadata.n_rows_valid, scan_frame=loaded.frame,
            lineage=lineage))  # fmt: skip
        run.recorder.event(Stage.EXPORT, EventType.BUNDLE_EXPORTED, file=bundle.path.name,
                           sha256=bundle.sha256, bytes=bundle.size_bytes,
                           parity_max_abs_diff=bundle.r1.report.max_abs_diff)  # fmt: skip
    timings = {
        s.value: {"start": round(t.start, 3), "end": None if t.end is None else round(t.end, 3),
                  "duration": round(t.duration, 3)}
        for s, t in run.budget.timings.items()
    }  # fmt: skip
    run.recorder.stage_timings(timings)
    run.recorder.resources(
        [dataclasses.asdict(s) for s in run.monitor.samples], run.monitor.peak_mb
    )
    run.recorder.experiment({"config": config, "plan": plan, "evaluation": evaluation,
                             "budget": run.budget.ledger(), "lineage": lineage,
                             "feasibility": feasibility, "thresholds": thresholds})  # fmt: skip
    for name, payload in (("leaderboard.json", entries), ("selection.json", selection),
                          ("holdout.json", holdout)):  # fmt: skip
        (run.dir / name).write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")
    run.recorder.event(Stage.RUN, EventType.RUN_COMPLETED, winner=selection.winner)
    return RunSummary(run.dir, task, metric, entries, selection.winner,
                      holdout.model_dump(mode="json", exclude_none=True), bundle=bundle.path,
                      bundle_sha256=bundle.sha256)  # fmt: skip


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
    if summary.bundle is not None:
        lines.append(f"bundle: {summary.bundle}  sha256 {summary.bundle_sha256}")
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
