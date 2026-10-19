"""LLM router contracts: request, response, and per-run budget (PRD §23.1, §23.7; A-004)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, JsonValue
from pydantic.json_schema import SkipJsonSchema

from tabpilot.schemas._base import Contract, NonNegFloat, NonNegInt

LLMPurpose = Literal[
    "plan", "plan_repair", "reflect", "reflect_repair", "narrate", "narrate_repair"
]


class LLMRequest(Contract):
    """One LLM request (`llm_request/1.0`, PRD §23.1).

    `response_schema` is a Python type used for parsing. It is excluded from serialization and
    from the exported JSON Schema (PI-25).
    """

    schema_version: Literal["llm_request/1.0"] = "llm_request/1.0"
    purpose: LLMPurpose
    system: str
    user_json: dict[str, JsonValue]
    response_schema: SkipJsonSchema[type[BaseModel] | None] = Field(default=None, exclude=True)
    max_output_tokens: int = Field(ge=1, le=8000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(default=90.0, ge=20.0, le=240.0)


class LLMResponse(Contract):
    """One normalized LLM response (`llm_response/1.0`, PRD §23.1, §23.6)."""

    schema_version: Literal["llm_response/1.0"] = "llm_response/1.0"
    provider: str
    model_id: str
    content: str
    parsed: dict[str, JsonValue] | None = None
    input_tokens: NonNegInt | None = None
    output_tokens: NonNegInt | None = None
    tokens_estimated: bool = False
    latency_s: NonNegFloat
    cached: bool = False
    attempts: int = Field(default=1, ge=1)


class LLMBudget(Contract):
    """Hard per-run LLM ceilings (`llm_budget/1.0`, PRD §23.7; A-004; DEC-017)."""

    schema_version: Literal["llm_budget/1.0"] = "llm_budget/1.0"
    max_calls: int = Field(default=12, ge=0, le=25)
    max_input_tokens: NonNegInt = 40_000
    max_output_tokens: NonNegInt = 8_000
    max_wall_s: NonNegFloat
    per_call_timeout_s: float = Field(default=90.0, ge=20.0, le=240.0)
    max_input_tokens_per_call: NonNegInt = 6_000
