"""Bundle helpers shared by the exporter and the bundled `pipeline.py` (PRD §22.4–§22.5; A-001).

One reader, one prediction routine and one parity comparison, so the parity fixture is read and
scored identically in the worker and in the user's environment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PARITY_TOLERANCE = 1e-9
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value: Any) -> Any:
    """CSV formula-injection guard (PRD §24.9): text starting with `=`, `+`, `-`, `@`, tab or CR
    gets a leading `'`; numbers and None are left as they are.
    """
    if value is None or isinstance(value, bool | int | float):
        return value
    text = str(value)
    return "'" + text if text.startswith(FORMULA_PREFIXES) else text


def write_predictions(path: str | Path, outputs: dict[str, Any], classes: list[str]) -> None:
    """`prediction` (+ `proba_<class>` columns for classification) as a formula-safe CSV."""
    if "values" in outputs:
        frame = pd.DataFrame({"prediction": outputs["values"]})
    else:
        frame = pd.DataFrame({"prediction": [safe_cell(v) for v in outputs["labels"]]})
        proba = np.asarray(outputs["probabilities"], dtype=np.float64)
        for j, label in enumerate(classes):
            frame[safe_cell(f"proba_{label}")] = proba[:, j]
    frame.to_csv(path, index=False, lineterminator="\n")


def unsafe_cell(value: Any) -> Any:
    """Undo `safe_cell`: `'=x` → `=x` (only when the rest starts with a formula character, so
    a genuine leading apostrophe survives)."""
    if isinstance(value, str) and value.startswith("'") and value[1:].startswith(FORMULA_PREFIXES):
        return value[1:]
    return value


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV as text columns (empty cells are missing); the pipeline casts types itself.
    Cells and headers that a writer protected against spreadsheet formulas (§24.9) are restored.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    frame.columns = pd.Index([unsafe_cell(c) for c in frame.columns])
    restored: pd.DataFrame = frame.apply(lambda col: col.map(unsafe_cell))
    return restored


def predict_outputs(model: Any, X: pd.DataFrame, task: str, classes: list[str]) -> dict[str, Any]:
    """Predictions in a JSON-ready form.

    Classification: `labels` (class labels) and `probabilities` with one column per entry of
    `classes`, aligned through the model's `classes_` codes. Regression: `values`.
    """
    if task == "regression":
        return {"values": [float(v) for v in np.asarray(model.predict(X), dtype=np.float64)]}
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    codes = np.asarray(model.classes_).astype(np.int64)
    full = np.zeros((len(X), len(classes)), dtype=np.float64)
    full[:, codes] = proba
    labels = [classes[int(i)] for i in np.argmax(full, axis=1)]
    return {"labels": labels, "probabilities": [[float(v) for v in row] for row in full]}


@dataclass(frozen=True)
class ParityReport:
    """R1 result: exact labels and max |Δ| of probabilities or values."""

    ok: bool
    max_abs_diff: float
    label_mismatches: int
    rows: int


def compare(expected: dict[str, Any], actual: dict[str, Any],
            tolerance: float = PARITY_TOLERANCE) -> ParityReport:  # fmt: skip
    """Compare two `predict_outputs` results (§22.5 parity rule)."""
    key = "values" if "values" in expected else "probabilities"
    want = np.asarray(expected[key], dtype=np.float64)
    got = np.asarray(actual.get(key, []), dtype=np.float64)
    if want.shape != got.shape:
        return ParityReport(False, float("inf"), len(want), len(want))
    diff = float(np.max(np.abs(want - got))) if want.size else 0.0
    mismatches = sum(a != b for a, b in zip(expected.get("labels", []), actual.get("labels", []),
                                            strict=True))  # fmt: skip
    return ParityReport(diff <= tolerance and mismatches == 0, diff, mismatches, len(want))


def check_manifest(root: Path) -> list[str]:
    """Paths whose SHA-256 differs from `manifest.json` (or that are missing). The manifest
    does not list itself. An absent manifest is reported as a problem, never skipped.
    """
    try:
        listed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError):
        return ["manifest.json"]
    bad = []
    for entry in listed:
        path = root / entry["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        if digest != entry["sha256"]:
            bad.append(entry["path"])
    return bad


def verify(bundle_dir: str | Path, model: Any) -> ParityReport:
    """Check the manifest checksums, then score `parity/parity_input.csv` and compare with
    `parity/parity_expected.json` at the fixed 1e-9 tolerance (never one read from the file).
    A failed checksum or an empty fixture is a failed parity, not an exception.
    """
    root = Path(bundle_dir)
    if check_manifest(root):
        return ParityReport(False, float("inf"), 0, 0)
    expected = json.loads((root / "parity" / "parity_expected.json").read_text(encoding="utf-8"))
    frame = read_table(root / "parity" / "parity_input.csv")
    if frame.empty:
        return ParityReport(False, float("inf"), 0, 0)
    actual = predict_outputs(model, frame, expected["task"], list(expected["classes"]))
    return compare(expected, actual)
