"""Read a finished run's outputs for display (PRD §27.1 Results page, §2.4 deliverable 2).

The UI never touches ML objects: it reads the records the worker wrote (leaderboard, selection,
holdout, report JSON, figures, bundle) as schema-validated contracts or plain JSON. The same
reader serves precomputed sample results (D-14), which have the same layout minus the bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tabpilot.schemas import LeaderboardEntry, RunEvent
from tabpilot.schemas.report import FinalReport

PRECOMPUTED_FILE = "precomputed.json"


@dataclass(frozen=True)
class Figure:
    """One report figure and its text alternative (§27.5)."""

    path: Path
    caption: str


@dataclass(frozen=True)
class Bundle:
    """The downloadable ZIP (§22) with its size and checksum."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass
class RunArtifacts:
    """Everything the Results page shows, read from one run directory."""

    root: Path
    leaderboard: list[LeaderboardEntry]
    selection: dict[str, Any]
    holdout: dict[str, Any]
    report: FinalReport | None
    experiment: dict[str, Any]
    figures: list[Figure] = field(default_factory=list)
    report_html: Path | None = None
    report_md: Path | None = None
    bundle: Bundle | None = None
    events: list[RunEvent] = field(default_factory=list)
    precomputed: dict[str, Any] | None = None

    @property
    def selected_entry_id(self) -> str | None:
        final = self.experiment.get("lineage", {}).get("final")
        return str(final) if final else self.selection.get("winner")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def caption_for(name: str) -> str:
    """Readable caption from a figure file name (`shap-bar.png` → `Shap bar`)."""
    return Path(name).stem.replace("-", " ").replace("_", " ").capitalize()


def read_artifacts(root: Path, *, bundle_path: str | None = None,
                   bundle_sha256: str | None = None) -> RunArtifacts | None:  # fmt: skip
    """Load the outputs under `root`; None when the run left no leaderboard (or was deleted)."""
    board = _json(root / "leaderboard.json")
    if not isinstance(board, list):
        return None
    report_json = _json(root / "report" / "report.json")
    figures_dir = root / "report" / "figures"
    figures = [Figure(p, caption_for(p.name)) for p in sorted(figures_dir.glob("*.png"))] \
        if figures_dir.is_dir() else []  # fmt: skip
    bundle = None
    if bundle_path and bundle_sha256 and (root / bundle_path).is_file():
        path = root / bundle_path
        bundle = Bundle(path, bundle_sha256, path.stat().st_size)
    events_file = root / "records" / "events.jsonl"
    events = [RunEvent.model_validate_json(line)
              for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if events_file.is_file() else []  # fmt: skip
    html, md = root / "report" / "report.html", root / "report" / "report.md"
    return RunArtifacts(
        root=root,
        leaderboard=[LeaderboardEntry.model_validate(e) for e in board],
        selection=_json(root / "selection.json") or {},
        holdout=_json(root / "holdout.json") or {},
        report=FinalReport.model_validate(report_json) if isinstance(report_json, dict) else None,
        experiment=_json(root / "records" / "experiment.json") or {},
        figures=figures,
        report_html=html if html.is_file() else None,
        report_md=md if md.is_file() else None,
        bundle=bundle,
        events=events,
        precomputed=_json(root / PRECOMPUTED_FILE),
    )
