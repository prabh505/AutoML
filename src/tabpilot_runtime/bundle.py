"""Bundle helpers shared by the exporter and the bundled `pipeline.py` (PRD §22.4–§22.5; A-001).

One reader, one prediction routine and one parity comparison, so the parity fixture is read and
scored identically in the worker and in the user's environment.
"""

from __future__ import annotations

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


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV as text columns (empty cells are missing); the pipeline casts types itself."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


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


def verify(bundle_dir: str | Path, model: Any) -> ParityReport:
    """Score `parity/parity_input.csv` and compare with `parity/parity_expected.json`."""
    root = Path(bundle_dir)
    expected = json.loads((root / "parity" / "parity_expected.json").read_text(encoding="utf-8"))
    frame = read_table(root / "parity" / "parity_input.csv")
    actual = predict_outputs(model, frame, expected["task"], list(expected["classes"]))
    return compare(expected, actual, float(expected.get("tolerance", PARITY_TOLERANCE)))
