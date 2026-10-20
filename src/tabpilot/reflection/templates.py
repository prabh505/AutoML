"""Versioned reflector prompt templates (PRD §19; P12). Same loader and rules as the planner's."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from tabpilot.planning.templates import Template, compact_model_schema, fill_template
from tabpilot.schemas import ReflectionProposal

PROMPTS = Path(__file__).parent / "prompts"
REFLECTOR_SYSTEM = "reflector_system"
REFLECTOR_REPAIR = "reflector_repair"
VERSION = "1.0"


@cache
def load(name: str, version: str = VERSION) -> Template:
    """Load a reflector template; `{{SCHEMA}}` is the compact `ReflectionProposal` schema."""
    schema = json.dumps(compact_model_schema(ReflectionProposal), separators=(",", ":"),
                        sort_keys=True)  # fmt: skip
    return fill_template(PROMPTS, name, version, {"{{SCHEMA}}": schema})


def versions() -> dict[str, str]:
    """Template versions for `ExperimentConfig.prompt_template_versions`."""
    return {REFLECTOR_SYSTEM: VERSION, REFLECTOR_REPAIR: VERSION}
