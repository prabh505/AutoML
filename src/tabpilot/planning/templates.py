"""Versioned planner prompt templates (PRD §12.4; P11 deliverable 2).

Templates are files named `<name>@<version>.md` in `prompts/`. Placeholders are filled with fixed,
code-generated text only (the compact JSON Schema and a synthetic worked example), never with
profile data: column names reach the LLM only inside the JSON data payload (§24.8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from tabpilot.features.dsl.catalog import CATALOG
from tabpilot.schemas import PlannerOutput
from tabpilot.schemas.features import FEATURE_NAME_PATTERN

PROMPTS = Path(__file__).parent / "prompts"
PLANNER_SYSTEM = "planner_system"
PLANNER_REPAIR = "planner_repair"
VERSION = "1.0"
DROP_KEYS = frozenset({"title", "description", "examples"})

# Synthetic example (an invented loan dataset), shown for format only. Not user data.
EXAMPLE_INPUT: dict[str, Any] = {
    "task": "binary", "primary_metric": "roc_auc",
    "profile": {"columns": [
        {"name": "loan_amount", "type": "numeric", "missing": 0.0, "skew": 2.4},
        {"name": "annual_income", "type": "numeric", "missing": 0.04, "skew": 1.9},
        {"name": "term", "type": "categorical", "missing": 0.0, "distinct": 2},
        {"name": "num_late_payments", "type": "numeric", "missing": 0.0, "distinct": 9},
    ]},
    "available_models": ["linear", "random_forest", "lightgbm"],
}  # fmt: skip
EXAMPLE_OUTPUT: dict[str, Any] = {
    "schema_version": "plan/1.0",
    "column_decisions": [{"column": "num_late_payments", "action": "treat_as_categorical",
                          "reason": "Few distinct small integer counts."}],
    "imputation": {"numeric": "median", "categorical": "most_frequent",
                   "add_missing_indicators": True},
    "encoding": {"low_cardinality": "onehot", "high_cardinality": "frequency"},
    "features": [{"id": "f01", "name": "loan_to_income", "op": "ratio",
                  "inputs": ["loan_amount", "annual_income"], "params": {"eps": 1e-6},
                  "rationale": "Repayment burden relative to income."},
                 {"id": "f02", "name": "log_income", "op": "log1p", "inputs": ["annual_income"],
                  "rationale": "Income is right-skewed."}],
    "models": [{"name": "lightgbm", "priority": 1, "class_weight": "none"},
               {"name": "linear", "priority": 2, "class_weight": "none"}],
    "tuning_allocation": {"lightgbm": 0.7, "linear": 0.3},
    "target_transform": "none", "cv_folds": 5, "metric_recommendation": "roc_auc",
    "rationale": "Loan size relative to income is the main risk signal; boosting first.",
}  # fmt: skip


@dataclass(frozen=True)
class Template:
    """A loaded template: `name@version` and its final text."""

    name: str
    version: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


def _strip(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k not in DROP_KEYS}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def compact_schema() -> dict[str, Any]:
    """`PlannerOutput` JSON Schema with one generic feature object (per-op arity and parameter
    ranges are in the `dsl_catalog` of the data payload) and no titles or descriptions.
    """
    schema = PlannerOutput.model_json_schema()
    feature = {"type": "object", "required": ["id", "name", "op", "inputs"],
               "additionalProperties": False,
               "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,15}$"},
                              "name": {"type": "string", "pattern": FEATURE_NAME_PATTERN},
                              "op": {"enum": sorted(CATALOG)},
                              "inputs": {"type": "array", "items": {"type": "string"}},
                              "params": {"type": "object"},
                              "rationale": {"type": "string", "maxLength": 300}}}  # fmt: skip
    schema["properties"]["features"]["items"] = feature
    used = json.dumps({k: v for k, v in schema.items() if k != "$defs"})
    schema["$defs"] = {k: v for k, v in schema["$defs"].items() if f"#/$defs/{k}" in used}
    return dict(_strip(schema))


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


@cache
def load(name: str, version: str = VERSION) -> Template:
    """Load `prompts/<name>@<version>.md` and fill its placeholders."""
    text = (PROMPTS / f"{name}@{version}.md").read_text(encoding="utf-8")
    fills = {"{{SCHEMA}}": _compact(compact_schema()), "{{EXAMPLE_INPUT}}": _compact(EXAMPLE_INPUT),
             "{{EXAMPLE_OUTPUT}}": _compact(EXAMPLE_OUTPUT)}  # fmt: skip
    for placeholder, value in fills.items():
        text = text.replace(placeholder, value)
    if "{{" in text:
        raise ValueError(f"{name}@{version}: unfilled placeholder")
    return Template(name, version, text.strip())


def versions() -> dict[str, str]:
    """Template versions for `ExperimentConfig.prompt_template_versions`."""
    return {PLANNER_SYSTEM: VERSION, PLANNER_REPAIR: VERSION}
