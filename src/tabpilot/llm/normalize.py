"""Response normalization and token estimation (PRD §23.6)."""

from __future__ import annotations

import json
from typing import Any

from tabpilot.schemas import LLMResponse

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """chars / 4, rounded up (§23.6). Used when the provider reports no usage."""
    return -(-len(text) // CHARS_PER_TOKEN)


def extract_json(content: str) -> dict[str, Any] | None:
    """Strip code fences and any text before the first `{`, then parse; None on failure."""
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def normalize(
    *,
    provider: str,
    model_id: str,
    content: str,
    input_tokens: int | None,
    output_tokens: int | None,
    prompt_text: str,
    latency_s: float,
) -> LLMResponse:
    """Build an `LLMResponse`; missing token counts are estimated and flagged."""
    estimated = input_tokens is None or output_tokens is None
    return LLMResponse(
        provider=provider,
        model_id=model_id,
        content=content,
        parsed=extract_json(content),
        input_tokens=input_tokens if input_tokens is not None else estimate_tokens(prompt_text),
        output_tokens=output_tokens if output_tokens is not None else estimate_tokens(content),
        tokens_estimated=estimated,
        latency_s=max(0.0, latency_s),
    )
