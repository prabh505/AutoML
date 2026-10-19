"""Privacy-safe structured JSON logging (PRD §24.10, §30.2–§30.3; I-10, I-11).

Uses the standard library with a JSON formatter (DEC-023: no `structlog` dependency).

- Every record is one JSON object with `ts, level, logger, stage, run_id, event` plus fields.
- A scrubber redacts API-key patterns (`sk-or-…`, `sk-…`, `gsk_…`, `AIza…`) from every
  string as defense in depth.
- Values wrapped with `data_value(...)`, and pandas/numpy containers, are dataset content. They
  are never formatted (no `repr`). In strict mode (tests) logging one raises `LoggingPolicyError`;
  otherwise it is replaced with a placeholder.
- Tracebacks are emitted only when `include_tracebacks=True` (local debug). Otherwise only the
  exception class and a scrubbed message are logged.

This module must not import pandas: `tabpilot.llm` uses it and may not depend on pandas.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import sys
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import IO, Any, Final

ROOT_LOGGER: Final = "tabpilot"
REDACTED_KEY: Final = "[REDACTED-KEY]"
REDACTED_DATA: Final = "[REDACTED-DATA]"

# Order matters: the OpenRouter prefix `sk-or-` must be matched before the generic `sk-`.
KEY_PATTERNS: Final = (
    re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{8,}"),
)
_DATA_CONTAINERS: Final = {
    ("pandas", "DataFrame"), ("pandas", "Series"), ("pandas", "Index"), ("numpy", "ndarray"),
}  # fmt: skip

_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tabpilot_run_id", default=None
)
_strict: bool = False


class LoggingPolicyError(RuntimeError):
    """Raised in strict mode when dataset content is passed to the logger."""


class DataValue:
    """Marks a value as dataset content. The logger never formats it."""

    __slots__ = ("_value",)

    def __init__(self, value: object) -> None:
        self._value = value

    def __repr__(self) -> str:
        return REDACTED_DATA

    __str__ = __repr__


def data_value(value: object) -> DataValue:
    """Tag `value` as dataset content so the logger refuses it (PRD §30.3)."""
    return DataValue(value)


def set_strict(enabled: bool) -> None:
    """Strict mode raises on dataset content instead of redacting it (enabled in tests)."""
    global _strict  # noqa: PLW0603 - process-wide policy switch
    _strict = enabled


def scrub(text: str) -> str:
    """Redact API-key patterns from `text`."""
    for pattern in KEY_PATTERNS:
        text = pattern.sub(REDACTED_KEY, text)
    return text


def _is_data(value: object) -> bool:
    if isinstance(value, DataValue):
        return True
    cls = type(value)
    return (cls.__module__.split(".")[0], cls.__name__) in _DATA_CONTAINERS


def _sanitize(value: object, path: str) -> object:  # noqa: PLR0911 - one return per type
    """Return a JSON-safe, scrubbed copy of `value`; refuse dataset content."""
    if _is_data(value):
        if _strict:
            raise LoggingPolicyError(f"dataset content passed to the logger at '{path}'")
        return REDACTED_DATA
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, Mapping):
        return {scrub(str(k)): _sanitize(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_sanitize(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, datetime):
        return value.isoformat()
    # Unknown objects are summarized by type only; their contents are never formatted.
    return f"<{type(value).__name__}>"


class JsonFormatter(logging.Formatter):
    """Formats records as one scrubbed JSON object per line (PRD §30.2)."""

    def __init__(self, include_tracebacks: bool = False) -> None:
        super().__init__()
        self.include_tracebacks = include_tracebacks

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "stage": getattr(record, "tp_stage", None),
            "run_id": getattr(record, "tp_run_id", None),
            "event": scrub(str(record.msg)),
        }
        fields = getattr(record, "tp_fields", {})
        payload.update({k: v for k, v in fields.items() if k not in payload})
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            payload["error_class"] = type(exc).__name__
            payload["error_message"] = scrub(str(exc))[:500]
            if self.include_tracebacks:
                payload["traceback"] = scrub(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


class StageLogger:
    """Logger bound to a stage. Use `logger.info("event_name", field=value, ...)`."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self._logger = logging.getLogger(f"{ROOT_LOGGER}.{stage}")

    def _log(self, level: int, event: str, exc_info: bool, fields: dict[str, Any]) -> None:
        if not self._logger.isEnabledFor(level):
            return
        clean = {k: _sanitize(v, k) for k, v in fields.items()}
        extra = {"tp_stage": self.stage, "tp_run_id": _run_id.get(), "tp_fields": clean}
        self._logger.log(level, scrub(event), exc_info=exc_info, extra=extra)

    def debug(self, event: str, **fields: Any) -> None:
        """Log at DEBUG."""
        self._log(logging.DEBUG, event, False, fields)

    def info(self, event: str, **fields: Any) -> None:
        """Log at INFO."""
        self._log(logging.INFO, event, False, fields)

    def warning(self, event: str, **fields: Any) -> None:
        """Log at WARNING."""
        self._log(logging.WARNING, event, False, fields)

    def error(self, event: str, exc_info: bool = False, **fields: Any) -> None:
        """Log at ERROR; `exc_info=True` adds the exception class and scrubbed message."""
        self._log(logging.ERROR, event, exc_info, fields)


def get_logger(stage: str) -> StageLogger:
    """Return a structured logger for `stage` (PRD §30.2)."""
    return StageLogger(stage)


@contextlib.contextmanager
def bind_run(run_id: str) -> Iterator[None]:
    """Attach `run_id` to every record logged inside this context."""
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)


def configure_logging(
    level: str = "INFO", stream: IO[str] | None = None, include_tracebacks: bool = False
) -> logging.Handler:
    """Install one JSON handler on the `tabpilot` logger (hosted: INFO; local: DEBUG optional)."""
    root = logging.getLogger(ROOT_LOGGER)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter(include_tracebacks=include_tracebacks))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    return handler
