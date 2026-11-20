"""Precompute labeled sample results for the waiting-for-slot state (PRD D-14, §27.3; P16).

Runs every bundled sample (or the ones named) through the real `RunManager` on the chosen
profile, then copies the displayable outputs to `samples/precomputed/<sample_id>/`:
leaderboard, selection, holdout, the report (HTML, Markdown, JSON, facts, figures),
`records/experiment.json`, `records/events.jsonl`, plus `precomputed.json` with the date and
settings. It does not copy the bundle, the model or any input data.

Usage:
    uv run python scripts/precompute_samples.py                      # all samples, hosted, no LLM
    uv run python scripts/precompute_samples.py --samples passengers --budget-s 240
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from tabpilot.config import load_settings
from tabpilot.runtime import RunManager
from tabpilot.runtime.catalog import metric_options, samples
from tabpilot.runtime.results import PRECOMPUTED_FILE
from tabpilot.schemas import ExecutionProfileName, RunConfig, Task

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "samples" / "precomputed"
COPY_FILES = ("leaderboard.json", "selection.json", "holdout.json", "records/experiment.json",
              "records/events.jsonl", "report/report.html", "report/report.md",
              "report/report.json", "report/facts.json")  # fmt: skip


def run_sample(sample_id: str, out: Path, *, profile: str, budget_s: int, llm: str) -> dict:
    """Run one sample and copy its displayable outputs; returns the label record."""
    settings = load_settings(
        {"TABPILOT_PROFILE": profile, "TABPILOT_CONFIG_DIR": str(ROOT / "config")}
    )
    with tempfile.TemporaryDirectory(prefix="tabpilot-precompute-") as tmp:
        settings = settings.model_copy(update={"data_dir": Path(tmp)})
        manager = RunManager(Path(tmp) / "runs", settings, delete_failed=False)
        sample = next(s for s in samples() if s.sample_id == sample_id)
        upload = manager.validate_sample(sample_id)
        choice = manager.check_target(upload.run_id, sample.target, sample.task)
        _, metric = metric_options(sample.task, choice.positive_share)
        config = RunConfig(run_id=upload.run_id, target=choice.column, task=sample.task,
                           positive_label=choice.positive_label if sample.task == Task.BINARY
                           else None, primary_metric=metric, time_budget_s=budget_s,
                           llm_mode=llm,
                           execution_profile=ExecutionProfileName(profile))  # fmt: skip
        manager.start(config, display_name=sample.filename)
        result = manager.wait(upload.run_id)
        if result is None or result.status not in ("completed", "completed_partial"):
            status = result.status if result else "unknown"
            raise RuntimeError(f"sample {sample_id} did not complete ({status})")
        run_dir = manager.run_dir(upload.run_id)
        target = out / sample_id
        if target.exists():
            shutil.rmtree(target)
        for name in COPY_FILES:
            src = run_dir / name
            if src.is_file():
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target / name)
        figures = run_dir / "report" / "figures"
        if figures.is_dir():
            shutil.copytree(figures, target / "report" / "figures")
        label = {"sample_id": sample_id, "date": datetime.now(UTC).date().isoformat(),
                 "profile": profile, "time_budget_s": budget_s, "llm_mode": llm,
                 "status": result.status, "duration_s": round(result.duration_s, 1),
                 "tabpilot_version": version("tabpilot")}  # fmt: skip
        (target / PRECOMPUTED_FILE).write_text(json.dumps(label, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")  # fmt: skip
        manager.shutdown()
        return label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", nargs="*", default=[s.sample_id for s in samples()])
    parser.add_argument("--profile", choices=["hosted", "local"], default="hosted")
    parser.add_argument("--budget-s", type=int, default=240)
    parser.add_argument("--llm", choices=["off", "stub"], default="off")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for sample_id in args.samples:
        label = run_sample(sample_id, args.out, profile=args.profile, budget_s=args.budget_s,
                           llm=args.llm)  # fmt: skip
        size = sum(p.stat().st_size for p in (args.out / sample_id).rglob("*") if p.is_file())
        sys.stdout.write(
            f"{sample_id}: {label['status']} in {label['duration_s']} s, {size / 1024:.0f} KB\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
