"""Formula-injection-safe CSV writing (PRD §24.9; T-SEC-03).

A cell that begins with `=`, `+`, `-`, `@`, a tab or a carriage return is prefixed with `'`, so
spreadsheet programs show it as text instead of evaluating it. Numbers are written as numbers
(a negative number is data, not a formula).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tabpilot_runtime.bundle import FORMULA_PREFIXES, safe_cell

__all__ = ["FORMULA_PREFIXES", "csv_text", "safe_cell", "write_csv"]


def csv_text(headers: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    """CSV with every header and cell passed through `safe_cell` (minimal quoting, LF endings)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([safe_cell(h) for h in headers])
    for row in rows:
        writer.writerow([safe_cell(c) for c in row])
    return buffer.getvalue()


def write_csv(path: Path, headers: Sequence[Any], rows: Iterable[Sequence[Any]]) -> Path:
    path.write_text(csv_text(headers, rows), encoding="utf-8")
    return path
