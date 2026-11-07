"""Run one worker in-process and collect everything it reports (P15 graph and e2e tests).

`go()` mirrors what the CLI and the UI do: store the upload, build a `RunConfig`, create the run
directory, and call `execute_run` with a collecting outbox. No process is spawned, so tests can
patch the worker's modules and watch events as they are emitted.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tabpilot.cli import RunArgs, build_config
from tabpilot.config import load_settings
from tabpilot.ingestion import accept_upload
from tabpilot.orchestration.context import WorkerOptions
from tabpilot.runtime.worker import WorkerSpec, execute_run
from tabpilot.schemas import (
    ExecutionProfileName,
    ModelName,
    ProviderName,
    RunEvent,
    RunResult,
    Task,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "datasets"
DEFAULT_MODELS = (ModelName.LINEAR, ModelName.LIGHTGBM)


class CancelFlag:
    """A cancel token the test can set (`threading.Event` semantics)."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


@dataclass
class Collector:
    """Outbox that records worker messages and can react to each event as it arrives."""

    on_event: Callable[[RunEvent], None] | None = None
    events: list[RunEvent] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)

    def put(self, item: Any) -> None:
        kind, payload = item
        self.kinds.append(kind)
        if kind == "event":
            event = RunEvent.model_validate_json(payload)
            self.events.append(event)
            if self.on_event is not None:
                self.on_event(event)


@dataclass
class Outcome:
    result: RunResult
    events: list[RunEvent]
    run_dir: Path
    collector: Collector

    def types(self) -> list[str]:
        return [e.type.value for e in self.events]

    def of(self, kind: str) -> list[RunEvent]:
        return [e for e in self.events if e.type.value == kind]


def go(
    tmp_path: Path,
    file: str = "binary_small.csv",
    target: str = "target",
    *,
    llm: str = "off",
    stub_mode: str = "ok",
    rounds: int = 1,
    ensemble: bool = True,
    models: tuple[ModelName, ...] = DEFAULT_MODELS,
    trials: int | None = 2,
    tuning_mode: str = "trials",
    seed: int = 42,
    providers: tuple[ProviderName, ...] = (),
    profile: ExecutionProfileName = ExecutionProfileName.LOCAL,
    task: Task | None = None,
    cancel: CancelFlag | None = None,
    on_event: Callable[[RunEvent], None] | None = None,
    include_oof: bool = False,
) -> Outcome:
    """Execute one run in this process; the run directory is `tmp_path / "run"`."""
    settings = load_settings({"TABPILOT_PROFILE": profile.value})
    source = FIXTURES / file
    args = RunArgs(file=source, target=target, out=tmp_path / "run", task=task, seed=seed,
                   profile=profile, tuning_mode=tuning_mode,  # type: ignore[arg-type]
                   trials=trials, models=models, llm=llm, providers=providers,
                   reflection_rounds=rounds, ensemble=ensemble, stub_mode=stub_mode)  # fmt: skip
    with tempfile.TemporaryDirectory(prefix="tabpilot-test-upload-") as staging:
        with source.open("rb") as handle:
            stored = accept_upload(handle, source.name,
                                   settings.model_copy(update={"data_dir": Path(staging)}))  # fmt: skip
        config = build_config(args, settings, stored)
        run_dir = tmp_path / "run"
        (run_dir / "input").mkdir(parents=True)
        input_path = run_dir / "input" / stored.path.name
        shutil.copyfile(stored.path, input_path)
    options = WorkerOptions(tuning_mode=tuning_mode, trials_per_model=trials,  # type: ignore[arg-type]
                            models=models, llm_stub_mode=stub_mode, include_oof=include_oof)  # fmt: skip
    spec = WorkerSpec(config.model_dump_json(), str(run_dir), str(input_path),
                      stored.display_name, options)  # fmt: skip
    collector = Collector(on_event)
    result = execute_run(spec, collector, cancel or CancelFlag())
    return Outcome(result, collector.events, run_dir, collector)
