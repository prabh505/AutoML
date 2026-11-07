"""Headless CLI: `tabpilot run <file> --target <col> --out <dir>` (PROMPTS P09, P15; FR-PLN-03).

The CLI validates and stores the upload, builds a `RunConfig`, and runs it through the same
`RunManager` the UI uses: a worker process executes ingest → split → profile and the LangGraph
run graph (plan → train → evaluate → reflect → ensemble → finalize → explain → report →
export). It then prints the leaderboard, selection, holdout and bundle path. `--llm off` makes
no provider calls, `--llm stub` uses the offline stub provider, `--llm providers` uses the
configured providers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ulid import ULID

from tabpilot.config import Settings, load_settings
from tabpilot.evaluation.metrics import default_primary, selection_score
from tabpilot.ingestion import StoredUpload, accept_upload, load_dataset
from tabpilot.ingestion.task import detect_task, positive_label
from tabpilot.models import ModelRegistry
from tabpilot.orchestration.context import WorkerOptions
from tabpilot.runtime import RunManager
from tabpilot.schemas import (
    ExecutionProfileName,
    MetricName,
    ModelName,
    ProviderName,
    RunConfig,
    RunResult,
    TabPilotError,
    Task,
)
from tabpilot.schemas.experiment import LeaderboardEntry
from tabpilot.schemas.run import MIN_TIME_BUDGET_S
from tabpilot.tuning import TuningMode

LLM_MODES = ("off", "stub", "providers")
RUN_MODES = ("process", "thread")


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
    llm: str = "off"
    providers: tuple[ProviderName, ...] = ()
    reflection_rounds: int = 1
    ensemble: bool = True
    run_mode: str = "process"
    stub_mode: str = "ok"


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
    status: str = "completed"


class RunFailedError(RuntimeError):
    """The run ended `failed` or `cancelled`; carries the E-code and its user message."""

    def __init__(self, result: RunResult) -> None:
        self.result = result
        self.code = result.error_code
        super().__init__(result.error_message or f"run {result.status}")


def _target_column(loaded: Any, target: str) -> str:
    """Accept the original or the normalized target name."""
    for col in loaded.metadata.columns:
        if target in (col.original, col.name):
            return str(col.name)
    raise ValueError(f"target column {target!r} is not in the file")


def _check_budget(args: RunArgs, settings: Settings) -> int:
    total = args.budget_s or settings.limits.time_budget_default_s
    maximum = settings.limits.time_budget_max_s
    if not MIN_TIME_BUDGET_S <= total <= maximum:
        raise ValueError(f"--budget-s {total} is outside {MIN_TIME_BUDGET_S}–{maximum} s, the "
                         f"{args.profile.value} range (maximum)")  # fmt: skip
    return total


def _check_models(args: RunArgs, settings: Settings) -> None:
    available = ModelRegistry(settings).available(args.profile)
    unknown = [m.value for m in args.models if m not in available]
    if unknown:
        raise ValueError(f"models not available on {args.profile.value}: {unknown}")


def build_config(args: RunArgs, settings: Settings, stored: StoredUpload) -> RunConfig:
    """Task, target, positive label and metric from a light read of the upload (the UI's
    `validate_upload` + `detect_task` step); the worker re-reads the stored file.
    """
    loaded = load_dataset(stored.path, stored.display_name, settings, stored.run_id)
    target = _target_column(loaded, args.target)
    values = loaded.frame[target].dropna()
    detected = detect_task(values)
    task = args.task or detected.task
    positive = None
    share = None
    if task == Task.BINARY:
        positive = detected.positive_label or positive_label(values)
        share = float((values.astype("string") == str(positive)).mean()) if len(values) else None
    consent = list(args.providers) if args.llm == "providers" else []
    return RunConfig(
        run_id=stored.run_id, target=target, task=task, positive_label=positive,
        primary_metric=default_primary(task, share), time_budget_s=_check_budget(args, settings),
        seed_root=args.seed, llm_mode=args.llm, provider_order=consent,
        consented_providers=consent, reflection_max_rounds=args.reflection_rounds,
        enable_ensemble=args.ensemble, execution_profile=args.profile,
    )  # fmt: skip


def execute(args: RunArgs) -> RunSummary:
    """Run every stage through the `RunManager` and return the summary."""
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"output directory {args.out} is not empty")
    if args.llm not in LLM_MODES or args.run_mode not in RUN_MODES:
        raise ValueError(f"unknown --llm {args.llm!r} or run mode {args.run_mode!r}")
    settings = load_settings({**os.environ, "TABPILOT_PROFILE": args.profile.value})
    _check_budget(args, settings)
    _check_models(args, settings)
    options = WorkerOptions(tuning_mode=args.tuning_mode, trials_per_model=args.trials,
                            models=args.models, llm_stub_mode=args.stub_mode)  # fmt: skip
    with tempfile.TemporaryDirectory(prefix="tabpilot-upload-") as staging:
        with args.file.open("rb") as handle:
            stored = accept_upload(handle, args.file.name,
                                   settings.model_copy(update={"data_dir": Path(staging)}),
                                   run_id=str(ULID()))  # fmt: skip
        config = build_config(args, settings, stored)
        args.out.mkdir(parents=True, exist_ok=True)
        input_path = args.out / "input" / stored.path.name
        input_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stored.path, input_path)
    manager = RunManager(settings.data_dir / "runs", settings, mode=args.run_mode,
                         delete_failed=False)  # fmt: skip
    try:
        manager.start(config, input_path=input_path, display_name=stored.display_name,
                      options=options, run_dir=args.out)  # fmt: skip
        result = manager.wait(config.run_id)
    finally:
        manager.shutdown()
    assert result is not None  # noqa: S101 - wait() returns once the run has finished
    if result.status in ("failed", "cancelled"):
        raise RunFailedError(result)
    return _summary(args.out, config, result)


def _summary(run_dir: Path, config: RunConfig, result: RunResult) -> RunSummary:
    entries = [LeaderboardEntry.model_validate(e)
               for e in json.loads((run_dir / "leaderboard.json").read_text("utf-8"))]  # fmt: skip
    winner = json.loads((run_dir / "selection.json").read_text("utf-8"))["winner"]
    holdout = json.loads((run_dir / "holdout.json").read_text("utf-8"))
    final = result.final
    bundle = run_dir / final.bundle_path if final and final.bundle_path else None
    return RunSummary(run_dir, config.task, config.primary_metric, entries, winner, holdout,
                      bundle=bundle, bundle_sha256=final.bundle_sha256 if final else None,
                      status=result.status)  # fmt: skip


def _format(summary: RunSummary) -> list[str]:
    lines = [f"run directory: {summary.run_dir}", f"task: {summary.task.value}  "
             f"metric: {summary.metric.value}", "leaderboard (CV on DEV):"]  # fmt: skip

    def rank(e: LeaderboardEntry) -> tuple[bool, float, str]:
        sel = selection_score(summary.metric, e.mean) if e.mean is not None else 0.0
        return (e.status != "ok", -sel, e.entry_id)

    for e in sorted(summary.entries, key=rank):
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
    lines.append(f"status: {summary.status}")
    if summary.bundle is not None:
        lines.append(f"bundle: {summary.bundle}  sha256 {summary.bundle_sha256}")
    return lines


def _models(text: str | None) -> tuple[ModelName, ...]:
    return tuple(ModelName(m.strip()) for m in text.split(",")) if text else ()


def _providers(text: str | None) -> tuple[ProviderName, ...]:
    return tuple(ProviderName(p.strip()) for p in text.split(",")) if text else ()


def parse(argv: Sequence[str] | None = None) -> RunArgs:
    parser = argparse.ArgumentParser(prog="tabpilot", description="TabPilot headless runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run AutoML on a CSV/Parquet file")
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
    run.add_argument(
        "--llm",
        choices=LLM_MODES,
        default="off",
        help="off: no provider calls; stub: offline stub; providers: real providers",
    )
    run.add_argument("--providers", help="providers to use and consent to with --llm providers "
                     "(comma-separated, in order)")  # fmt: skip
    run.add_argument("--reflection-rounds", type=int, choices=[0, 1, 2], default=1)
    run.add_argument("--no-ensemble", action="store_true")
    run.add_argument("--run-mode", choices=RUN_MODES, default="process",
                     help="process (default) or the cooperative thread fallback")  # fmt: skip
    ns = parser.parse_args(argv)
    return RunArgs(file=ns.file, target=ns.target, out=ns.out,
                   task=Task(ns.task) if ns.task else None, budget_s=ns.budget_s, seed=ns.seed,
                   profile=ExecutionProfileName(ns.profile), tuning_mode=ns.tuning_mode,
                   trials=ns.trials, models=_models(ns.models), llm=ns.llm,
                   providers=_providers(ns.providers), reflection_rounds=ns.reflection_rounds,
                   ensemble=not ns.no_ensemble, run_mode=ns.run_mode)  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; exit 0 on success, 2 on a user-facing error."""
    args = parse(argv)
    try:
        summary = execute(args)
    except RunFailedError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
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
