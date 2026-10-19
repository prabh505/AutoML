"""Delimiter detection, malformed-record pre-scan, and decimal commas (PRD §8.3; PI-44).

The pre-scan uses the `csv` module, which respects quoting and embedded newlines. It counts the
fields of every record and finds the malformed ones. Parsing then uses the pandas C engine on
the text with exactly those records removed (P04 prompt; DEC-026).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field

from tabpilot.schemas import ErrorCode, TabPilotError

CANDIDATES = (",", ";", "\t", "|")
SAMPLE_CHARS = 64 * 1024
MIN_SCORE = 0.95
AMBIGUITY_GAP = 0.02
MAX_MALFORMED_FRACTION = 0.01
MAX_MALFORMED_ROWS = 1000
DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")
DECIMAL_COMMA_SHARE = 0.8

csv.field_size_limit(16 * 1024 * 1024)


def _lines(text: str) -> list[str]:
    """Split on '\\n' only, keeping terminators, so csv record spans map to physical lines."""
    parts = text.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _score(sample_lines: list[str], delimiter: str) -> float:
    """Fraction of records with the modal field count × (modal count > 1) (§8.3 step 2)."""
    try:
        counts = [len(r) for r in csv.reader(sample_lines, delimiter=delimiter) if r]
    except csv.Error:
        return 0.0
    if not counts:
        return 0.0
    modal, freq = Counter(counts).most_common(1)[0]
    return (freq / len(counts)) if modal > 1 else 0.0


def detect_delimiter(text: str) -> str:
    """Pick the delimiter (§8.3 steps 1–3).

    When no candidate reaches 0.95, the best unambiguous candidate is used and the malformed-line
    rule decides (PI-44). With no multi-field candidate at all → E-DELIMITER-AMBIGUOUS.
    """
    sample = text[:SAMPLE_CHARS]
    if len(text) > SAMPLE_CHARS and "\n" in sample:
        sample = sample[: sample.rindex("\n") + 1]  # whole lines only
    lines = _lines(sample)
    scores = sorted(((_score(lines, d), d) for d in CANDIDATES), reverse=True)
    (best, delim), (second, _) = scores[0], scores[1]
    if best == 0.0 or best - second <= AMBIGUITY_GAP:
        raise TabPilotError(ErrorCode.DELIMITER_AMBIGUOUS)
    return delim


@dataclass(frozen=True)
class PreScan:
    """Records kept after the malformed-record scan."""

    header: list[str]
    good_text: str
    n_records: int
    malformed: int
    malformed_line_numbers: list[int] = field(default_factory=list)


def prescan(text: str, delimiter: str) -> PreScan:
    """Find records whose field count differs from the header's (FR-ING-02).

    ≤ 1% of data rows and ≤ 1,000 rows → skipped and reported (first 5 line numbers);
    otherwise E-PARSE-MALFORMED.
    """
    lines = _lines(text)
    reader = csv.reader(lines, delimiter=delimiter)
    header: list[str] | None = None
    keep = [True] * len(lines)
    bad_lines: list[int] = []
    n_records = 0
    prev = 0
    for record in reader:
        start, end = prev, reader.line_num  # physical lines [start, end)
        prev = end
        if not record:
            continue
        if header is None:
            header = record
            continue
        n_records += 1
        if len(record) != len(header):
            bad_lines.append(start + 1)
            for i in range(start, end):
                keep[i] = False
    if header is None or n_records == 0:
        raise TabPilotError(ErrorCode.FILE_EMPTY)
    bad = len(bad_lines)
    if bad and (bad > MAX_MALFORMED_FRACTION * n_records or bad > MAX_MALFORMED_ROWS):
        raise TabPilotError(ErrorCode.PARSE_MALFORMED, n=bad, pct=round(100 * bad / n_records, 1),
                            line=bad_lines[0])  # fmt: skip
    good = "".join(line for line, k in zip(lines, keep, strict=True) if k) if bad else text
    return PreScan(header=header, good_text=good, n_records=n_records - bad, malformed=bad,
                   malformed_line_numbers=bad_lines[:5])  # fmt: skip


def is_decimal_comma_column(values: list[str]) -> bool:
    """≥ 80% of non-empty tokens look like `12,5` (§8.3 step 4; applies with ';' only)."""
    tokens = [v for v in values if v]
    return bool(tokens) and sum(bool(DECIMAL_COMMA.match(v)) for v in tokens) / len(tokens) >= (
        DECIMAL_COMMA_SHARE
    )
