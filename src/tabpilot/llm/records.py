"""`llm_calls.jsonl` writer (PRD §30.1, §22.2, §23.8).

One line per attempt or cache hit: metadata plus the SHA-256 of the request payload and of the
response content. Prompt and response bodies are never written (I-11).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FILE_NAME = "llm_calls.jsonl"


class LLMCallRecorder:
    """Appends call metadata to `<records_dir>/llm_calls.jsonl`."""

    def __init__(
        self, records_dir: Path, now: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        records_dir.mkdir(parents=True, exist_ok=True)
        self.path = records_dir / FILE_NAME
        self.path.touch()
        self._now = now

    def write(self, **fields: Any) -> None:
        """Write one record. Allowed fields are metadata only (see `RECORD_FIELDS`)."""
        unknown = set(fields) - RECORD_FIELDS
        if unknown:
            raise ValueError(f"not an llm_calls field: {sorted(unknown)}")
        record = {"ts": self._now().isoformat(), **fields}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


RECORD_FIELDS = frozenset({
    "purpose", "provider", "model_id", "attempt", "outcome", "error_class", "latency_s",
    "input_tokens", "output_tokens", "tokens_estimated", "cached", "request_sha256",
    "response_sha256",
})  # fmt: skip
