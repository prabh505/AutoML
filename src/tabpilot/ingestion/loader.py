"""`load_dataset`: stored file → typed, normalized DataFrame + metadata (FR-ING-01…04, §8)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tabpilot.config import Settings
from tabpilot.ingestion import dialect, encoding, inference
from tabpilot.ingestion.normalize import normalize_columns
from tabpilot.ingestion.upload import (
    EXTENSIONS,
    SNIFF_BYTES,
    check_content,
    check_size,
    sanitize_display_name,
)
from tabpilot.logging import get_logger
from tabpilot.schemas import (
    ColumnInfo,
    DatasetMetadata,
    ErrorCode,
    ExecutionProfileName,
    ParseReport,
    TabPilotError,
    WarningCode,
)
from tabpilot.schemas.dataset import SourceInfo

log = get_logger("ingest")


@dataclass(frozen=True)
class LoadedDataset:
    """The typed frame (normalized column names) with its metadata and parse report."""

    frame: pd.DataFrame
    metadata: DatasetMetadata
    parse_report: ParseReport


def _limit_error(code: ErrorCode, count: int, settings: Settings) -> TabPilotError:
    local = settings.profiles[ExecutionProfileName.LOCAL]
    if code == ErrorCode.TOO_MANY_COLUMNS:
        limit, local_limit = settings.limits.max_columns, local.max_columns
    else:
        limit, local_limit = settings.limits.max_rows, local.max_rows
    return TabPilotError(code, count=count, profile=settings.profile.value, limit=limit,
                         local_limit=local_limit)  # fmt: skip


def _check_columns(n: int, settings: Settings) -> None:
    if n > settings.limits.max_columns:
        raise _limit_error(ErrorCode.TOO_MANY_COLUMNS, n, settings)


def _read_parquet(path: Path, settings: Settings) -> tuple[pd.DataFrame, ParseReport]:
    try:
        schema = pq.read_schema(path)
    except (pa.ArrowInvalid, OSError) as exc:
        raise TabPilotError(ErrorCode.FILE_CORRUPT, format="Parquet") from exc
    nested = [f.name for f in schema if pa.types.is_nested(f.type)]
    if nested:
        raise TabPilotError(ErrorCode.PARQUET_NESTED, cols=", ".join(nested))
    _check_columns(len(schema), settings)
    try:
        frame = pq.read_table(path).to_pandas()
    except (pa.ArrowInvalid, OSError) as exc:
        raise TabPilotError(ErrorCode.FILE_CORRUPT, format="Parquet") from exc
    if frame.empty:
        raise TabPilotError(ErrorCode.FILE_EMPTY)
    for name in frame.columns:
        if frame[name].dtype == object:
            frame[name] = frame[name].astype("string")
    return frame, ParseReport(encoding="binary", delimiter=None, quote_char=None,
                              rows_read=len(frame))  # fmt: skip


def _read_text(
    path: Path, settings: Settings, encoding_override: str | None, delimiter_override: str | None
) -> tuple[pd.DataFrame, ParseReport]:
    data = path.read_bytes()
    if not data:
        raise TabPilotError(ErrorCode.FILE_EMPTY)
    enc = (
        encoding.EncodingResult(encoding_override)
        if encoding_override
        else (encoding.detect_encoding(data))
    )
    text = encoding.decode(data, enc).lstrip("﻿")
    if not text.strip():
        raise TabPilotError(ErrorCode.FILE_EMPTY)
    delim = delimiter_override or dialect.detect_delimiter(text)
    scan = dialect.prescan(text, delim)
    _check_columns(len(scan.header), settings)
    placeholder_names = [f"c{i}" for i in range(len(scan.header))]
    frame = pd.read_csv(io.StringIO(scan.good_text), sep=delim, dtype=str, keep_default_na=False,
                        engine="c", header=0, names=placeholder_names)  # fmt: skip
    if len(frame) != scan.n_records:
        raise TabPilotError(ErrorCode.FILE_CORRUPT, format="CSV")
    frame.columns = scan.header
    warnings: list[str] = []
    if enc.fallback:
        warnings.append(encoding.fallback_warning(text))
    decimal_cols: list[str] = []
    if delim == ";":
        for i, name in enumerate(scan.header):
            col = frame.iloc[:, i]
            if dialect.is_decimal_comma_column(col.str.strip().tolist()):
                frame.isetitem(i, col.str.replace(",", ".", regex=False).to_numpy())
                decimal_cols.append(name)
    report = ParseReport(
        encoding=enc.encoding, encoding_fallback=enc.fallback, delimiter=delim,
        decimal_comma_columns=decimal_cols, rows_read=scan.n_records + scan.malformed,
        malformed_lines=scan.malformed, malformed_line_numbers=scan.malformed_line_numbers,
        warnings=warnings,
    )  # fmt: skip
    return frame, report


def load_dataset(
    path: Path,
    display_name: str,
    settings: Settings,
    run_id: str,
    *,
    encoding_override: str | None = None,
    delimiter_override: str | None = None,
) -> LoadedDataset:
    """Parse, normalize, and type a stored upload (FR-ING-01…04).

    The size is checked before any byte is read (T-ING-01). Overrides come from the UI when
    detection is ambiguous (E-DELIMITER-AMBIGUOUS) or the user picks an encoding.
    """
    size = path.stat().st_size
    check_size(size, settings)
    fmt = EXTENSIONS.get(path.suffix.lower())
    if fmt is None:
        raise TabPilotError(ErrorCode.FILE_TYPE, ext=path.suffix.lower() or "(none)")
    with path.open("rb") as fh:  # magic bytes again: defense in depth (FR-ING-01)
        head = fh.read(SNIFF_BYTES)
        fh.seek(max(size - 4, 0))
        check_content(head, fh.read(4), fmt)
    if fmt == "parquet":
        raw, report = _read_parquet(path, settings)
    else:
        raw, report = _read_text(path, settings, encoding_override, delimiter_override)
    if len(raw) > settings.limits.max_rows:
        raise _limit_error(ErrorCode.TOO_MANY_ROWS, len(raw), settings)
    if raw.shape[1] < 2:
        raise TabPilotError(ErrorCode.DELIMITER_AMBIGUOUS)

    originals = [str(c) for c in raw.columns]
    names = normalize_columns(originals)
    n_rows = len(raw)
    typed: dict[str, pd.Series] = {}
    columns: list[ColumnInfo] = []
    warnings = list(report.warnings)
    for i, (original, name) in enumerate(zip(originals, names, strict=True)):
        result = inference.infer_column(name, raw.iloc[:, i].reset_index(drop=True))
        result = inference.apply_column_checks(name, result, n_rows)
        values = inference.finalize_dtype(result)
        typed[name] = values
        warnings += result.warnings
        columns.append(ColumnInfo(
            original=original, name=name, semantic_type=result.semantic_type, role=result.role,
            dtype=str(values.dtype), missing_rate=float(values.isna().mean()) if n_rows else 0.0,
            coerced_to_missing=result.coerced_to_missing + result.inf_converted,
            exclusion_reason=result.exclusion_reason, hints=result.hints,
        ))  # fmt: skip
    frame = pd.DataFrame(typed)
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.append(f"{WarningCode.DUPLICATES.value}:{duplicates}")
    metadata = DatasetMetadata(
        run_id=run_id,
        source=SourceInfo(
            display_name=sanitize_display_name(display_name),
            format=fmt,
            size_bytes=size,
            encoding=report.encoding,
            delimiter=report.delimiter,
            malformed_lines_skipped=report.malformed_lines,
        ),
        n_rows_raw=n_rows,
        n_rows_valid=n_rows,
        n_columns=len(columns),
        columns=columns,
        warnings=warnings,
    )
    log.info("dataset_loaded", rows=n_rows, columns=len(columns), malformed=report.malformed_lines,
             excluded=sum(c.exclusion_reason is not None for c in columns))  # fmt: skip
    return LoadedDataset(frame=frame, metadata=metadata, parse_report=report)
