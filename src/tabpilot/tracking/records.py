"""Run records writer (PRD §30.1).

`<run_dir>/records/` holds `events.jsonl`, `stage_timings.json`, `trials.jsonl`,
`resources.jsonl`, and `experiment.json`. Records carry identifiers, settings, and computed
numbers only; callers never pass cell values or category labels (I-11). Every file is strict
JSON (NaN/inf are rejected).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tabpilot.schemas import EventType, RunEvent, Stage

RECORD_FILES = ("events.jsonl", "stage_timings.json", "trials.jsonl", "resources.jsonl",
                "experiment.json")  # fmt: skip


def to_jsonable(value: Any) -> Any:
    """Pydantic models → JSON mode dicts; containers recursively; everything else unchanged."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    return value


def dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(to_jsonable(value), sort_keys=True, allow_nan=False, indent=indent,
                      ensure_ascii=False)  # fmt: skip


class RunRecorder:
    """Append-only record files for one run."""

    def __init__(
        self, run_dir: Path, run_id: str, now: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self.dir = run_dir / "records"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._now = now
        self._seq = 0
        for name in ("events.jsonl", "trials.jsonl", "resources.jsonl"):
            (self.dir / name).touch()

    def _append(self, name: str, value: Any) -> None:
        with (self.dir / name).open("a", encoding="utf-8") as fh:
            fh.write(dumps(value) + "\n")

    def _write(self, name: str, value: Any) -> Path:
        path = self.dir / name
        path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def event(self, stage: Stage, type_: EventType, **data: Any) -> RunEvent:
        """Validate and append one `RunEvent` (§26.10)."""
        event = RunEvent(run_id=self.run_id, seq=self._seq, ts=self._now(), stage=stage,
                         type=type_, data=to_jsonable(data))  # fmt: skip
        self._seq += 1
        self._append("events.jsonl", event)
        return event

    def trial(self, record: Mapping[str, Any]) -> None:
        self._append("trials.jsonl", record)

    def resources(self, samples: Iterable[Mapping[str, float]], peak_mb: float) -> None:
        for sample in samples:
            self._append("resources.jsonl", sample)
        self._append("resources.jsonl", {"peak_rss_mb": round(peak_mb, 1)})

    def stage_timings(self, timings: Mapping[str, Mapping[str, float | None]]) -> Path:
        return self._write("stage_timings.json", timings)

    def experiment(self, payload: Mapping[str, Any]) -> Path:
        return self._write("experiment.json", payload)
