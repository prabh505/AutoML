"""User-facing error and warning catalog (PRD §8.9, §9.3; A-008).

Every error message has three parts: what happened, where, and what to do. Messages never
include stack traces or file-system paths.
"""

from __future__ import annotations

import string
from enum import StrEnum
from typing import Final

LOCAL_RUN_COMMAND: Final = "TABPILOT_PROFILE=local uv run streamlit run streamlit_app.py"


class ErrorCode(StrEnum):
    """Every user-facing error code (PRD §8.9)."""

    FILE_TOO_LARGE = "E-FILE-TOO-LARGE"
    FILE_TYPE = "E-FILE-TYPE"
    FILE_EMPTY = "E-FILE-EMPTY"
    FILE_CORRUPT = "E-FILE-CORRUPT"
    ARCHIVE_UNSUPPORTED = "E-ARCHIVE-UNSUPPORTED"
    ENCODING = "E-ENCODING"
    DELIMITER_AMBIGUOUS = "E-DELIMITER-AMBIGUOUS"
    PARSE_MALFORMED = "E-PARSE-MALFORMED"
    PARQUET_NESTED = "E-PARQUET-NESTED"
    TOO_MANY_COLUMNS = "E-TOO-MANY-COLUMNS"
    TOO_MANY_ROWS = "E-TOO-MANY-ROWS"
    TARGET_CONSTANT = "E-TARGET-CONSTANT"
    TARGET_ALL_MISSING = "E-TARGET-ALL-MISSING"
    TARGET_TOO_FEW_ROWS = "E-TARGET-TOO-FEW-ROWS"
    TARGET_CLASS_TOO_SMALL = "E-TARGET-CLASS-TOO-SMALL"
    TARGET_TOO_MANY_CLASSES = "E-TARGET-TOO-MANY-CLASSES"
    TARGET_UNSUPPORTED_TYPE = "E-TARGET-UNSUPPORTED-TYPE"
    RESOURCE_INFEASIBLE = "E-RESOURCE-INFEASIBLE"
    NO_VALID_MODEL = "E-NO-VALID-MODEL"
    EXPORT_PARITY = "E-EXPORT-PARITY"
    WORKER_LOST = "E-WORKER-LOST"


ERROR_TEMPLATES: Final[dict[ErrorCode, str]] = {
    ErrorCode.FILE_TOO_LARGE: (
        "The file is {size} MB; this demo accepts up to {limit} MB. Remove unused columns or "
        "rows, or run TabPilot locally (see README) for files up to {local_limit} MB."
    ),
    ErrorCode.FILE_TYPE: "Unsupported file type '{ext}'. Upload .csv, .tsv, .txt or .parquet.",
    ErrorCode.FILE_EMPTY: "The file has no data rows.",
    ErrorCode.FILE_CORRUPT: (
        "The file could not be read as {format}. It may be damaged or saved in another format."
    ),
    ErrorCode.ARCHIVE_UNSUPPORTED: (
        "Compressed archives are not supported. Extract the CSV and upload it directly."
    ),
    ErrorCode.ENCODING: (
        "Text encoding could not be determined. Save the file as UTF-8 and upload again."
    ),
    ErrorCode.DELIMITER_AMBIGUOUS: (
        "We could not tell how columns are separated. Choose the separator:"
    ),
    ErrorCode.PARSE_MALFORMED: (
        "{n} rows ({pct}%) have the wrong number of columns (first at line {line}). Fix these "
        "rows or check the separator."
    ),
    ErrorCode.PARQUET_NESTED: (
        "Columns {cols} contain nested data. Flatten them before uploading."
    ),
    ErrorCode.TOO_MANY_COLUMNS: (
        "{count} columns exceeds the {profile} limit of {limit}. Remove unused columns, or run "
        "TabPilot locally (limit {local_limit} columns)."
    ),
    ErrorCode.TOO_MANY_ROWS: (
        "{count} rows exceeds the {profile} limit of {limit}. Remove rows, or run TabPilot "
        "locally (limit {local_limit} rows)."
    ),
    # E-TARGET-* templates written from PRD §8.7 (PI-09).
    ErrorCode.TARGET_CONSTANT: (
        "The target column '{target}' has only one distinct value, so there is nothing to "
        "predict. Choose another target column."
    ),
    ErrorCode.TARGET_ALL_MISSING: (
        "The target column '{target}' has no values in any row. Choose a column that contains "
        "the values you want to predict."
    ),
    ErrorCode.TARGET_TOO_FEW_ROWS: (
        "Only {rows} rows have a value in target column '{target}'; at least {minimum} are "
        "needed. Add more labeled rows or choose another target column."
    ),
    ErrorCode.TARGET_CLASS_TOO_SMALL: (
        "{n_classes} classes of target column '{target}' have fewer than {minimum} rows. "
        "Remove those rows (TabPilot can do this with your consent) or merge rare classes."
    ),
    ErrorCode.TARGET_TOO_MANY_CLASSES: (
        "Target column '{target}' has {count} classes; the limit is {limit}. If the values are "
        "numeric quantities, choose regression instead."
    ),
    ErrorCode.TARGET_UNSUPPORTED_TYPE: (
        "Target column '{target}' is a {semantic_type} column, which cannot be predicted. "
        "Choose a numeric, categorical or boolean column."
    ),
    ErrorCode.RESOURCE_INFEASIBLE: (
        "Estimated memory {est} MB exceeds the {limit} MB available here even after "
        "simplifications. Use a smaller file or fewer columns, or run TabPilot locally: "
        + LOCAL_RUN_COMMAND
    ),
    ErrorCode.NO_VALID_MODEL: "No model trained successfully. Details: {reasons}.",
    ErrorCode.EXPORT_PARITY: (
        "The exported model did not reproduce its predictions; download disabled. This is a "
        "bug — please report run {run_id}."
    ),
    ErrorCode.WORKER_LOST: (
        "The analysis stopped unexpectedly (it may have run out of memory). Try a smaller file, "
        "fewer columns, or a shorter time budget, or run TabPilot locally."
    ),
}


def template_params(code: ErrorCode) -> frozenset[str]:
    """Return the parameter names that the message template for `code` requires."""
    fields = (name for _, name, _, _ in string.Formatter().parse(ERROR_TEMPLATES[code]))
    return frozenset(name for name in fields if name)


def render_message(code: ErrorCode, params: dict[str, object]) -> str:
    """Render the user message for `code`; raises `ValueError` on missing or unknown params."""
    required = template_params(code)
    missing = required - params.keys()
    unknown = params.keys() - required
    if missing or unknown:
        raise ValueError(
            f"{code.value}: missing params {sorted(missing)}, unknown params {sorted(unknown)}"
        )
    return ERROR_TEMPLATES[code].format(**params)


class TabPilotError(Exception):
    """A user-facing failure carrying an E-code and its message parameters (PRD §8.9)."""

    def __init__(self, code: ErrorCode, **params: object) -> None:
        self.code = code
        self.params = dict(params)
        self.user_message = render_message(code, self.params)
        super().__init__(f"{code.value}: {self.user_message}")


class WarningCode(StrEnum):
    """Every warning code used in the PRD (§8.2, §8.5–§8.7, §9.3, §11.1; A-008)."""

    SMALL_DATA = "W-SMALL-DATA"
    IMBALANCE = "W-IMBALANCE"
    HIGH_MISSING = "W-HIGH-MISSING"
    LEAK_STRONG = "W-LEAK-STRONG"
    LEAK_NEAR_DETERMINISTIC = "W-LEAK-NEAR-DETERMINISTIC"
    GROUP_SUSPECTED = "W-GROUP-SUSPECTED"
    TEMPORAL = "W-TEMPORAL"
    DUPLICATES = "W-DUPLICATES"
    CONFLICTING_DUPLICATES = "W-CONFLICTING-DUPLICATES"
    WIDE = "W-WIDE"
    ENCODING_FALLBACK = "W-ENCODING-FALLBACK"
    COERCED = "W-COERCED"
    INF = "W-INF"
    TARGET_MISSING = "W-TARGET-MISSING"
    NO_HOLDOUT = "W-NO-HOLDOUT"
