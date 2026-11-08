"""Pre-graph stages: ingest → split → profile, as plain calls in the worker (PRD §13.1).

They are deterministic and linear, emit their own stage events, and seed the initial
`RunState`. The holdout guard is created as soon as the split exists (I-1).
"""

from __future__ import annotations

from dataclasses import replace

import psutil

from tabpilot.evaluation.holdout import make_guard
from tabpilot.evaluation.metrics import Problem
from tabpilot.ingestion import apply_target, load_dataset, validate_target
from tabpilot.ingestion.task import positive_label
from tabpilot.orchestration.context import RunContext
from tabpilot.planning import build_planner_input
from tabpilot.preprocessing import schema_from
from tabpilot.profiling import apply_leakage_exclusions, build_compact_profile, profile
from tabpilot.resources.estimator import (
    ONEHOT_CAP_SIMPLIFIED,
    estimate_feasibility,
    raise_if_infeasible,
)
from tabpilot.schemas import (
    ColumnRole,
    DatasetMetadata,
    EventType,
    RunState,
    Simplification,
    Stage,
    Task,
)
from tabpilot.splitting import build_split
from tabpilot.tracking.records import dumps
from tabpilot.training import DevData

REDUCED_K = 3
DEFAULT_K = 5


def apply_overrides(metadata: DatasetMetadata, ctx: RunContext) -> DatasetMetadata:
    """User column overrides (FR-ING-04): semantic type and feature/excluded role."""
    overrides = {o.column: o for o in ctx.config.column_overrides}
    columns = []
    for col in metadata.columns:
        o = overrides.get(col.name)
        update: dict[str, object] = {}
        if o is not None and col.name != ctx.config.target:
            if o.semantic_type is not None:
                update["semantic_type"] = o.semantic_type
            if o.role == ColumnRole.EXCLUDED:
                update |= {"role": ColumnRole.EXCLUDED, "exclusion_reason": "user_excluded"}
            elif o.role == ColumnRole.FEATURE:
                update |= {"role": ColumnRole.FEATURE, "exclusion_reason": None}
        columns.append(col.model_copy(update=update) if update else col)
    return metadata.model_copy(update={"columns": columns})


def _problem(frame: object, target: str, task: Task, positive: str | None) -> Problem:
    if task == Task.REGRESSION:
        return Problem(Task.REGRESSION)
    classes = tuple(sorted(str(v) for v in frame[target].dropna().unique()))  # type: ignore[index]
    return Problem(task, classes, classes.index(positive) if task == Task.BINARY else None)


def prepare(ctx: RunContext) -> RunState:
    """Run ingest, split and profile; fill `ctx`; return the graph's initial state."""
    cfg = ctx.config
    with ctx.stage(Stage.INGEST):
        loaded = load_dataset(ctx.input_path, ctx.display_name, ctx.settings, cfg.run_id)
        loaded = replace(loaded, metadata=apply_overrides(loaded.metadata, ctx))
        loaded = apply_target(loaded, validate_target(loaded, cfg.target, cfg.task))
        report = estimate_feasibility(loaded.metadata, cfg, ctx.settings,
                                      psutil.virtual_memory().total / 2**20)  # fmt: skip
        raise_if_infeasible(report)
        ctx.feasibility = report
        ctx.simplifications = set(report.simplifications)
        if report.simplifications:
            ctx.emit(Stage.INGEST, EventType.WARNING_RAISED, code="feasibility_simplified",
                     simplifications=[s.value for s in report.simplifications])  # fmt: skip
    positive = cfg.positive_label  # the user's choice, else the §10.1 default for a binary task
    if cfg.task == Task.BINARY and positive is None:
        positive = positive_label(loaded.frame[cfg.target].dropna())
    with ctx.stage(Stage.SPLIT):
        k = REDUCED_K if Simplification.REDUCE_FOLDS in ctx.simplifications else DEFAULT_K
        features = [c.name for c in loaded.metadata.columns if c.role == ColumnRole.FEATURE]
        group = loaded.frame[cfg.group_column] if cfg.group_column else None
        result = build_split(loaded.frame[cfg.target], cfg.task, cfg.seed_root, group=group,
                             features=loaded.frame[features], k=k)  # fmt: skip
        split = result.spec
        if split.holdout_present:  # HOLD is sealed from here until selection is frozen (I-1)
            ctx.guard = make_guard(split, ctx.freeze.is_frozen)
        (ctx.run_dir / "split_spec.json").write_text(dumps(split, indent=2) + "\n", "utf-8")
        for code in result.warnings:
            ctx.emit(Stage.SPLIT, EventType.WARNING_RAISED, code=str(code))
    with ctx.stage(Stage.PROFILE):
        summary = profile(loaded.frame, split, loaded.metadata, cfg.task,
                          execution_profile=cfg.execution_profile,
                          time_budget_s=cfg.time_budget_s, seed_root=cfg.seed_root,
                          positive_label=positive)  # fmt: skip
        re_included = [o for o in cfg.column_overrides if o.role == ColumnRole.FEATURE]
        metadata = apply_leakage_exclusions(loaded.metadata, summary.leakage, re_included)
        compact = build_compact_profile(summary, metadata, cfg.profile_mode)
    problem = _problem(loaded.frame, cfg.target, cfg.task, positive)
    schema = schema_from(metadata, summary)
    if Simplification.CAP_ONEHOT in ctx.simplifications:
        schema = replace(schema, onehot_max_categories=ONEHOT_CAP_SIMPLIFIED)
    ctx.loaded = replace(loaded, metadata=metadata)
    ctx.split, ctx.summary, ctx.problem, ctx.schema = split, summary, problem, schema
    ctx.positive_label = positive
    ctx.dev = DevData.from_frame(loaded.frame, cfg.target, problem)
    ctx.planner_input = build_planner_input(compact, ctx.settings, cfg,
                                            available=ctx.available_models())  # fmt: skip
    return RunState(run_id=cfg.run_id, config=cfg, dataset_meta=metadata,
                    split_spec_ref="split_spec.json", profile_compact=compact,
                    budget=ctx.budget.ledger())  # fmt: skip
