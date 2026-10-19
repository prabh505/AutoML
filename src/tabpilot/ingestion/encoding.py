"""Text encoding detection (PRD §8.2; PI-43)."""

from __future__ import annotations

import codecs
from dataclasses import dataclass

from charset_normalizer import from_bytes

from tabpilot.schemas import ErrorCode, TabPilotError, WarningCode

SAMPLE_BYTES = 1024 * 1024
MIN_CONFIDENCE = 0.8
UTF8_BOM = b"\xef\xbb\xbf"
UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


@dataclass(frozen=True)
class EncodingResult:
    """Detected encoding and whether the latin-1 fallback was used."""

    encoding: str
    fallback: bool = False
    confidence: float | None = None


def _allowed(name: str) -> bool:
    """§8.2 allowlist: cp1252, latin-1, utf-16, iso-8859-*."""
    canonical = codecs.lookup(name).name
    return canonical in {"cp1252", "utf-16", "utf-16-le", "utf-16-be"} or canonical.startswith(
        "iso8859-"
    )


def _is_utf8(sample: bytes) -> bool:
    """Strict UTF-8 check that tolerates a multi-byte character cut at the sample end."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    try:
        decoder.decode(sample, final=len(sample) < SAMPLE_BYTES)
    except UnicodeDecodeError:
        return False
    return True


def detect_encoding(data: bytes) -> EncodingResult:
    """Detect the encoding of `data` from its first 1 MB (§8.2 steps 1–5).

    Confidence for charset-normalizer is `1 - chaos` (PI-43).
    """
    sample = data[:SAMPLE_BYTES]
    if sample.startswith(UTF8_BOM):
        return EncodingResult("utf-8-sig")
    if sample.startswith(UTF16_BOMS):
        return EncodingResult("utf-16")
    if _is_utf8(sample):
        return EncodingResult("utf-8")
    best = from_bytes(sample).best()
    if best is not None:
        confidence = 1.0 - best.chaos
        if confidence >= MIN_CONFIDENCE and _allowed(best.encoding):
            return EncodingResult(codecs.lookup(best.encoding).name, confidence=confidence)
    return EncodingResult("latin-1", fallback=True)


def decode(data: bytes, result: EncodingResult) -> str:
    """Decode the whole file; failures after detection are E-ENCODING (§8.2 step 6)."""
    try:
        return data.decode(result.encoding)
    except UnicodeDecodeError as exc:
        raise TabPilotError(ErrorCode.ENCODING) from exc


def fallback_warning(text: str) -> str:
    """W-ENCODING-FALLBACK naming up to 3 lines with non-ASCII characters (no line content).

    The UI can show those lines to the user. Line content is dataset content, so it is never
    stored in metadata (DEC-026).
    """
    lines = [i for i, line in enumerate(text.splitlines(), start=1) if not line.isascii()][:3]
    return f"{WarningCode.ENCODING_FALLBACK.value}:lines={','.join(map(str, lines))}"
