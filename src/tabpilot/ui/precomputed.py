"""Precomputed sample results for when the run slot is busy (PRD D-14, §27.3).

Written by `scripts/precompute_samples.py` under `samples/precomputed/<sample_id>/` next to
`config/`. Each result is labeled "precomputed on {date}" wherever it is shown.
"""

from __future__ import annotations

from pathlib import Path

from tabpilot.runtime import RunArtifacts, read_artifacts
from tabpilot.runtime.catalog import samples
from tabpilot.runtime.results import PRECOMPUTED_FILE
from tabpilot.ui import services


def root() -> Path:
    return services.settings().config_dir.parent / "samples" / "precomputed"


def available_samples() -> list[str]:
    """Sample ids with a precomputed result, in catalog order."""
    base = root()
    return [s.sample_id for s in samples() if (base / s.sample_id / PRECOMPUTED_FILE).is_file()]


def load(sample_id: str) -> RunArtifacts | None:
    if sample_id not in available_samples():
        return None
    return read_artifacts(root() / sample_id)
