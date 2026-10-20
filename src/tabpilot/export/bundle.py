"""Bundle assembly: layout, checksums, size limits, deterministic ZIP (PRD §22; FR-ART-01).

```
parity fixture (synthetic) → expected outputs → model.joblib + R1 (rebuild once)
→ pipeline.py, requirements.txt, metadata/, report/, REPRODUCIBILITY.md, tabpilot_runtime/
→ raw-row scan → size limit (drop OOF, then figures) → manifest.json → ZIP + SHA-256
```

Manifest checksums are computed after every other file is final. The ZIP is deterministic
(sorted entries, fixed timestamps), so the same files always give the same bundle SHA-256.
"""

from __future__ import annotations

import hashlib
import platform
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import tabpilot_runtime
from tabpilot import __version__
from tabpilot.export.code import render_pipeline
from tabpilot.export.csvsafe import write_csv
from tabpilot.export.parity import (
    Loader,
    R1Result,
    column_rules,
    fixture_frame,
    row_hashes,
    serialize_and_verify,
    walk,
    write_fixture,
)
from tabpilot.export.requirements import requirements_for
from tabpilot.reporting import ReportState
from tabpilot.schemas import ExecutionProfileName, Task
from tabpilot.schemas.report import ArtifactManifest, ManifestFile, ManifestRole, Reproducibility
from tabpilot.tracking.records import dumps
from tabpilot_runtime.bundle import PARITY_TOLERANCE, predict_outputs, read_table

TEMPLATES = Path(__file__).parent / "templates"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
SLUG_CHARS = 40
MB = 1024 * 1024


class BundleTooLargeError(ValueError):
    """The bundle exceeds the profile's limit even without OOF predictions and figures (§22.8)."""


class RawRowError(RuntimeError):
    """A bundle file reproduces a dataset row (T-ART-06); the bundle is never offered."""


@dataclass
class ExportSettings:
    """Inputs the exporter needs besides the report state and the fitted model."""

    out_dir: Path
    max_bundle_mb: float
    profile: ExecutionProfileName
    target_original: str
    dev_positions: list[int]
    n_rows_valid: int
    scan_frame: pd.DataFrame | None = None  # every DEV/HOLD row, for the raw-row scan only
    lineage: dict[str, Any] = field(default_factory=dict)
    planner_input: dict[str, Any] | None = None
    plan_received: dict[str, Any] | None = None
    llm_calls: Path | None = None
    include_oof: bool = False
    loader: Loader = joblib.load
    created_at: datetime | None = None


@dataclass
class BundleResult:
    """The offered bundle and how it was verified."""

    path: Path
    sha256: str
    size_bytes: int
    manifest: ArtifactManifest
    r1: R1Result
    excluded: list[str]


def dataset_slug(display_name: str) -> str:
    """§22.1: normalized original stem, ≤ 40 characters of `[a-z0-9_]`."""
    stem = Path(display_name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:SLUG_CHARS].strip("_")
    return slug or "dataset"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _role(rel: str) -> ManifestRole:
    roles: list[tuple[str, ManifestRole]] = [
        ("model/", "model"), ("report/figures/", "figure"), ("report/", "report"),
        ("metadata/", "metadata"), ("parity/", "parity"), ("tabpilot_runtime/", "runtime"),
        ("pipeline.py", "code"), ("requirements.txt", "requirements"),
        ("REPRODUCIBILITY.md", "docs"), ("oof_predictions.parquet", "oof"),
    ]  # fmt: skip
    for prefix, role in roles:
        if rel == prefix or rel.startswith(prefix):
            return role
    raise ValueError(f"no manifest role for {rel}")


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _leaderboard_csv(state: ReportState, path: Path) -> None:
    headers = ["entry_id", "model", "feature_set", "iteration", "metric", "direction", "mean",
               "std", "sem", "fold_scores", "fit_seconds", "predict_seconds", "status",
               "comparable", "warnings"]  # fmt: skip
    rows = [[e.entry_id, str(e.model), e.feature_set, e.iteration, e.metric.name.value,
             e.metric.direction, e.mean, e.std, e.sem, ";".join(repr(s) for s in e.fold_scores),
             e.fit_seconds, e.predict_seconds, e.status, e.comparable, "; ".join(e.warnings)]
            for e in state.entries]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, headers, rows)


def preprocessing_spec(model: Any, state: ReportState) -> dict[str, Any]:
    """Pipeline step specification: steps, fitted column-transformer branches, decisions."""
    from sklearn.compose import ColumnTransformer  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415

    pipelines = [p for p in walk(model) if isinstance(p, Pipeline)]
    branches = []
    for ct in (c for c in walk(model) if isinstance(c, ColumnTransformer)):
        for name, est, cols in getattr(ct, "transformers_", []):
            kind = est if isinstance(est, str) else type(est).__name__
            branches.append({"name": name, "transformer": kind, "columns": [str(c) for c in cols]})
    cand = state.winner
    return {
        "steps": [[[n, type(s).__name__] for n, s in p.steps] for p in pipelines],
        "column_transformer": branches,
        "imputation": (cand.imputation if cand else state.plan.imputation).model_dump(),
        "encoding": (cand.encoding if cand else state.plan.encoding).model_dump(),
        "target_transform": (cand.target_transform if cand else state.plan.target_transform).value,
    }  # fmt: skip


def _write_metadata(root: Path, state: ReportState, model: Any, s: ExportSettings) -> None:
    meta = root / "metadata"
    _json(meta / "dataset_metadata.json", state.metadata)
    _json(meta / "experiment.json", {
        "config": state.config, "lineage": s.lineage, "entries": state.entries,
        "selection": state.selection, "thresholds": state.thresholds,
        "split": {"split_id": state.config.split.split_id, "n_rows_valid": s.n_rows_valid,
                  "dev_positions": [int(i) for i in s.dev_positions]},
    })  # fmt: skip
    _leaderboard_csv(state, meta / "leaderboard.csv")
    _json(meta / "plan.json", {"source": state.plan_source, "received": s.plan_received,
                               "executed": state.plan})  # fmt: skip
    _json(meta / "plan_validation.json", state.plan_validation)
    if s.planner_input is not None:
        _json(meta / "planner_input.json", s.planner_input)
    if s.llm_calls is not None and s.llm_calls.is_file():
        shutil.copyfile(s.llm_calls, meta / "llm_calls.jsonl")
    _json(meta / "features.json", state.provenance)
    _json(meta / "preprocessing.json", preprocessing_spec(model, state))
    _json(meta / "holdout.json", state.holdout)


def _copy_report(report_dir: Path, root: Path) -> None:
    target = root / "report"
    for name in ("report.html", "report.md", "report.json", "facts.json"):
        if (report_dir / name).is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(report_dir / name, target / name)
    if (report_dir / "figures").is_dir():
        shutil.copytree(report_dir / "figures", target / "figures")


def _copy_runtime(root: Path) -> None:
    source = Path(tabpilot_runtime.__file__).parent
    target = root / "tabpilot_runtime"
    target.mkdir(parents=True)
    for path in sorted(source.glob("*.py")):
        shutil.copyfile(path, target / path.name)


def _write_oof(root: Path, state: ReportState, s: ExportSettings) -> None:
    oof = state.winner_oof
    if oof is None:
        return
    values = np.asarray(oof, dtype=np.float64).reshape(len(oof), -1)
    frame = pd.DataFrame(values, columns=[f"pred_{j}" for j in range(values.shape[1])])
    frame.insert(0, "dev_row", np.asarray(s.dev_positions, dtype=np.int64))
    frame.to_parquet(root / "oof_predictions.parquet", index=False)


def find_raw_rows(root: Path, scan_frame: pd.DataFrame, model: Any) -> list[str]:
    """CSV files in the bundle whose rows hash-equal a DEV/HOLD row (T-ART-06)."""
    column_map, kinds = column_rules(model)
    real = set(row_hashes(scan_frame, kinds))
    hits = []
    for path in root.rglob("*.csv"):
        frame = read_table(path).rename(columns=dict(column_map))
        if all(c in frame.columns for c in kinds) and real & set(row_hashes(frame, kinds)):
            hits.append(path.relative_to(root).as_posix())
    return hits


def _size(root: Path) -> int:
    return sum(p.stat().st_size for p in _files(root))


def _enforce_size(root: Path, limit_bytes: float) -> list[str]:
    """§22.8: drop OOF predictions, then figures; still too large → BundleTooLargeError."""
    excluded: list[str] = []
    for rel, label in (("oof_predictions.parquet", "OOF predictions"),
                       ("report/figures", "report figures")):  # fmt: skip
        if _size(root) <= limit_bytes:
            return excluded
        path = root / rel
        if path.is_dir():
            shutil.rmtree(path)
            excluded.append(label)
        elif path.is_file():
            path.unlink()
            excluded.append(label)
    if _size(root) > limit_bytes:
        raise BundleTooLargeError(
            f"The bundle is {_size(root) / MB:.1f} MB, above the {limit_bytes / MB:.0f} MB limit, "
            "even without OOF predictions and figures.")  # fmt: skip
    return excluded


def _reproducibility(root: Path, state: ReportState, r1: R1Result, requirements: list[str],
                     excluded: list[str]) -> None:  # fmt: skip
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=False,  # noqa: S701 - Markdown of fixed, non-user values
                      undefined=StrictUndefined, keep_trailing_newline=True)  # fmt: skip
    text = env.get_template("REPRODUCIBILITY.md.j2").render(
        run_id=state.config.run_id, parity_rows=r1.report.rows,
        parity_diff=f"{r1.report.max_abs_diff:.3e}", tabpilot_version=__version__,
        python=platform.python_version(), platform=platform.platform(),
        seed_root=state.config.seeds.root, split_id=state.config.split.split_id,
        requirements=requirements, excluded=excluded,
    )  # fmt: skip
    (root / "REPRODUCIBILITY.md").write_text(text, encoding="utf-8")


def _manifest(root: Path, state: ReportState, r1: R1Result, excluded: list[str],
              created_at: datetime) -> ArtifactManifest:  # fmt: skip
    files = [ManifestFile(path=p.relative_to(root).as_posix(), sha256=sha256_file(p),
                          bytes=p.stat().st_size, role=_role(p.relative_to(root).as_posix()))
             for p in _files(root)]  # fmt: skip
    warnings = [f"excluded to fit the size limit: {e}" for e in excluded]
    if r1.rebuilt:
        warnings.append("the first serialization failed parity and was rebuilt")
    return ArtifactManifest(run_id=state.config.run_id, tabpilot_version=__version__,
                            created_at=created_at, python=platform.python_version(),
                            platform=platform.platform(), seed_root=state.config.seeds.root,
                            files=files, warnings=warnings,
                            reproducibility=Reproducibility(
                                r1_verified=True,
                                parity_max_abs_diff=r1.report.max_abs_diff))  # fmt: skip


def _zip(root: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in _files(root):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())


def export_bundle(
    state: ReportState, final_model: Any, report_dir: Path, settings: ExportSettings
) -> BundleResult:
    """§22.1–§22.8. Raises E-EXPORT-PARITY (`TabPilotError`), `BundleTooLargeError` or
    `RawRowError`; on any of them no bundle file is left behind.
    """
    cfg = state.config
    task = cfg.task.value
    classes = list(state.problem.classes) if state.problem.task != Task.REGRESSION else []
    name = f"tabpilot_{dataset_slug(state.metadata.source.display_name)}_{cfg.run_id}"
    root = settings.out_dir / name
    zip_path = settings.out_dir / f"{name}.zip"
    if root.exists() or zip_path.exists():
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True)
    try:
        _, kinds = column_rules(final_model)
        forbidden = (set(row_hashes(settings.scan_frame, kinds))
                     if settings.scan_frame is not None else set())  # fmt: skip
        synthetic = fixture_frame(final_model, state.profile, seed=cfg.seeds.root,
                                  forbidden=forbidden)  # fmt: skip
        parity_frame = write_fixture(synthetic, root / "parity" / "parity_input.csv")
        expected = {"task": task, "classes": classes, "tolerance": PARITY_TOLERANCE,
                    **predict_outputs(final_model, parity_frame, task, classes)}  # fmt: skip
        _json(root / "parity" / "parity_expected.json", expected)
        r1 = serialize_and_verify(final_model, root / "model" / "model.joblib", parity_frame,
                                  expected, run_id=cfg.run_id, loader=settings.loader)  # fmt: skip
        (root / "pipeline.py").write_text(render_pipeline(
            final_model, run_id=cfg.run_id, task=task, target=cfg.target,
            target_original=settings.target_original, classes=classes,
            seeds=cfg.seeds.model_dump()), encoding="utf-8")  # fmt: skip
        requirements = requirements_for(final_model)
        (root / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
        _write_metadata(root, state, final_model, settings)
        _copy_report(report_dir, root)
        _copy_runtime(root)
        if settings.include_oof and settings.profile == ExecutionProfileName.LOCAL:
            _write_oof(root, state, settings)
        if settings.scan_frame is not None:
            hits = find_raw_rows(root, settings.scan_frame, final_model)
            if hits:
                raise RawRowError(f"bundle files reproduce dataset rows: {hits}")
        excluded = _enforce_size(root, settings.max_bundle_mb * MB)
        _reproducibility(root, state, r1, requirements, excluded)
        manifest = _manifest(root, state, r1, excluded,
                             settings.created_at or datetime.now(UTC))  # fmt: skip
        (root / "manifest.json").write_text(dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _zip(root, zip_path)
        size = zip_path.stat().st_size
        if size > settings.max_bundle_mb * MB:
            raise BundleTooLargeError(f"The bundle is {size / MB:.1f} MB, above the "
                                      f"{settings.max_bundle_mb:.0f} MB limit.")  # fmt: skip
    except BaseException:
        zip_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(root, ignore_errors=True)  # the ZIP is the deliverable
    return BundleResult(path=zip_path, sha256=sha256_file(zip_path), size_bytes=size,
                        manifest=manifest, r1=r1, excluded=excluded)  # fmt: skip
