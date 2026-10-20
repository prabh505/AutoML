"""LLM response cache (PRD §6.4, §23.8).

Key: SHA-256 of the canonical JSON of (provider, model_id, prompt template identity, request data,
temperature). The template identity is the SHA-256 of the system prompt, so a template change is
a new key. Hosted: in memory for the run only (nothing is written). Local: on disk under
`~/.cache/tabpilot/llm` with a 7-day TTL. `enabled=False` turns it off (benchmark).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tabpilot.schemas import LLMRequest, LLMResponse

TTL_DAYS = 7
DEFAULT_DIR = Path.home() / ".cache" / "tabpilot" / "llm"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_hash(req: LLMRequest) -> str:
    """Identity of the request payload alone (T-LLM-06: identical across providers)."""
    return sha256(canonical({"purpose": req.purpose, "system": req.system,
                             "user": req.user_json, "max_output_tokens": req.max_output_tokens,
                             "temperature": req.temperature}))  # fmt: skip


def cache_key(provider: str, model_id: str, req: LLMRequest) -> str:
    """§6.4 key."""
    return sha256(canonical({"provider": provider, "model_id": model_id,
                             "template": sha256(req.system), "data": req.user_json,
                             "temperature": req.temperature}))  # fmt: skip


class LLMCache:
    """In-memory (hosted) or on-disk with TTL (local) response cache."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        directory: Path | None = None,
        ttl_s: float = TTL_DAYS * 86_400,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.enabled = enabled
        self.directory = directory
        self.ttl_s = ttl_s
        self._clock = clock
        self._memory: dict[str, LLMResponse] = {}

    def get(self, key: str) -> LLMResponse | None:
        if not self.enabled:
            return None
        if key in self._memory:
            return self._memory[key]
        if self.directory is None:
            return None
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        if self._clock() - path.stat().st_mtime > self.ttl_s:
            path.unlink(missing_ok=True)
            return None
        return LLMResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, response: LLMResponse) -> None:
        if not self.enabled:
            return
        self._memory[key] = response
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / f"{key}.json").write_text(response.model_dump_json(),
                                                        encoding="utf-8")  # fmt: skip
