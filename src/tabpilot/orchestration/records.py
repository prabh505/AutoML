"""Run records at the end of a run (PRD §30.1; FR-TRK-01).

`records/experiment.json` (config, plan, evaluation, budget, lineage, feasibility, thresholds,
reflection history, ensemble), the leaderboard/selection/holdout JSON next to it, stage timings
and resource samples. Identifiers, settings and computed numbers only (I-11).
"""

from __future__ import annotations

import dataclasses

from tabpilot.evaluation.leaderboard import threshold_report
from tabpilot.orchestration.context import RunContext
from tabpilot.orchestration.nodes import experiment_config, ledger, lineage
from tabpilot.schemas import RunState
from tabpilot.schemas.experiment import EvaluationResult
from tabpilot.tracking import dumps


def write_records(state: RunState, ctx: RunContext) -> None:
    """Write the records of a run that reached finalize."""
    timings = {
        s.value: {"start": round(t.start, 3), "end": None if t.end is None else round(t.end, 3),
                  "duration": round(t.duration, 3)}
        for s, t in ctx.budget.timings.items()
    }  # fmt: skip
    ctx.recorder.stage_timings(timings)
    ctx.recorder.resources([dataclasses.asdict(s) for s in ctx.monitor.samples],
                           ctx.monitor.peak_mb)  # fmt: skip
    thresholds = {
        eid: report
        for eid, cv in ctx.runs.items()
        if (report := threshold_report(cv.oof, ctx.dev.y, ctx.split, ctx.problem))
    }
    evaluation = EvaluationResult(entries=list(state.leaderboard), selection=ctx.selection,
                                  holdout=ctx.holdout)  # fmt: skip
    ctx.recorder.experiment({
        "config": experiment_config(state, ctx), "plan": state.plan,
        "plan_source": state.plan_source, "plan_validation": state.plan_validation,
        "evaluation": evaluation, "budget": ledger(ctx), "lineage": lineage(state, ctx),
        "feasibility": ctx.feasibility, "thresholds": thresholds,
        "reflection_history": list(state.reflection_history), "ensemble": state.ensemble,
    })  # fmt: skip
    for name, payload in (
        ("leaderboard.json", list(state.leaderboard)),
        ("selection.json", ctx.selection),
        ("holdout.json", ctx.holdout),
    ):
        (ctx.run_dir / name).write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")
