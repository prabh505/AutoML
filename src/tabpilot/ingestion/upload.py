"""Upload acceptance: size, extension, and content checks before any parsing (FR-ING-01, §24.2).

Storage paths use only the server-generated `run_id`. The original filename is kept only as
sanitized display text.
"""

from __future__ import annotations

import io
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Literal

from ulid import ULID

from tabpilot.config import Settings
from tabpilot.schemas import ErrorCode, ExecutionProfileName, TabPilotError

FileFormat = Literal["csv", "tsv", "txt", "parquet"]
EXTENSIONS: dict[str, FileFormat] = {".csv": "csv", ".tsv": "tsv", ".txt": "txt",
                                     ".parquet": "parquet"}  # fmt: skip
ARCHIVE_EXTENSIONS = {".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar", ".zst"}
MAX_DISPLAY_NAME = 120
BYTES_PER_MB = 1024 * 1024  # Streamlit's upload limit uses the same unit (DEC-026)
SNIFF_BYTES = 8192

MAGIC_ZIP = b"PK\x03\x04"
MAGIC_GZIP = b"\x1f\x8b"
MAGIC_OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
MAGIC_PARQUET = b"PAR1"
UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


@dataclass(frozen=True)
class StoredUpload:
    """An accepted upload stored under the run directory."""

    run_id: str
    path: Path
    display_name: str
    format: FileFormat
    size_bytes: int


def sanitize_display_name(filename: str) -> str:
    """Display-only filename: no path parts or control characters, ≤ 120 chars (§24.2)."""
    name = PureWindowsPath(PurePosixPath(filename.replace("\\", "/")).name).name
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = re.sub(r"\s+", " ", name).strip().lstrip(".")
    if len(name) > MAX_DISPLAY_NAME:
        stem, dot, ext = name.rpartition(".")
        keep = MAX_DISPLAY_NAME - len(ext) - 1
        name = (
            f"{stem[:keep]}.{ext}"
            if dot and 0 < len(ext) <= 10 and keep > 0
            else name[:MAX_DISPLAY_NAME]
        )
    return name or "upload"


def _extension(display_name: str) -> str:
    return Path(display_name).suffix.lower()


def _size_mb(size: int) -> float:
    return round(size / BYTES_PER_MB, 1)


def check_size(size_bytes: int, settings: Settings) -> None:
    """Reject files over the profile limit before reading content (FR-ING-01 AC1)."""
    limit = settings.limits.max_file_mb
    if size_bytes > limit * BYTES_PER_MB:
        local_limit = settings.profiles[ExecutionProfileName.LOCAL].max_file_mb
        raise TabPilotError(ErrorCode.FILE_TOO_LARGE, size=_size_mb(size_bytes), limit=limit,
                            local_limit=local_limit)  # fmt: skip
    if size_bytes == 0:
        raise TabPilotError(ErrorCode.FILE_EMPTY)


def check_content(head: bytes, tail: bytes, fmt: FileFormat) -> None:
    """Verify the content matches the extension using magic bytes (FR-ING-01, §24.2)."""
    if head.startswith(MAGIC_ZIP) or head.startswith(MAGIC_GZIP):
        raise TabPilotError(ErrorCode.ARCHIVE_UNSUPPORTED)
    if fmt == "parquet":
        if not (head.startswith(MAGIC_PARQUET) and tail.endswith(MAGIC_PARQUET)):
            raise TabPilotError(ErrorCode.FILE_CORRUPT, format="Parquet")
        return
    label = "CSV" if fmt == "csv" else "delimited text"
    if head.startswith(MAGIC_OLE) or head.startswith(MAGIC_PARQUET):
        raise TabPilotError(ErrorCode.FILE_CORRUPT, format=label)
    if b"\x00" in head and not head.startswith(UTF16_BOMS):
        raise TabPilotError(ErrorCode.FILE_CORRUPT, format=label)


def _seekable(source: BinaryIO, settings: Settings) -> tuple[BinaryIO, int]:
    """Return a seekable stream and its size, never reading more than limit + 1 bytes."""
    if source.seekable():
        start = source.tell()
        size = source.seek(0, os.SEEK_END) - start
        source.seek(start)
        return source, size
    cap = settings.limits.max_file_mb * BYTES_PER_MB + 1
    data = source.read(cap)
    return io.BytesIO(data), len(data)


def accept_upload(
    source: BinaryIO, filename: str, settings: Settings, run_id: str | None = None
) -> StoredUpload:
    """Validate and store an upload as `runs/{run_id}/input/raw.{ext}` (FR-ING-01).

    Order: extension → size (no content read) → magic bytes → store. Nothing is parsed here.
    """
    display_name = sanitize_display_name(filename)
    ext = _extension(display_name)
    if ext in ARCHIVE_EXTENSIONS:
        raise TabPilotError(ErrorCode.ARCHIVE_UNSUPPORTED)
    if ext not in EXTENSIONS:
        raise TabPilotError(ErrorCode.FILE_TYPE, ext=ext or "(none)")
    fmt = EXTENSIONS[ext]
    source, size = _seekable(source, settings)
    check_size(size, settings)
    start = source.tell()
    head = source.read(SNIFF_BYTES)
    source.seek(start + max(size - len(MAGIC_PARQUET), 0))
    tail = source.read(len(MAGIC_PARQUET))
    check_content(head, tail, fmt)

    run_id = run_id or str(ULID())
    run_dir = settings.data_dir / "runs" / run_id
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    path = input_dir / f"raw{ext}"
    source.seek(start)
    with path.open("wb") as out:
        while chunk := source.read(1024 * 1024):
            out.write(chunk)
    return StoredUpload(run_id=run_id, path=path, display_name=display_name, format=fmt,
                        size_bytes=size)  # fmt: skip
