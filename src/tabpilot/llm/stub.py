"""Offline stub provider for `llm_mode="stub"` (PRD §28.2 `RunConfig.llm_mode`).

A deterministic, network-free provider that answers the three TabPilot purposes from the
request data alone, so the full LLM path (router, budgets, records, validation, fact-check)
runs in CI and demos without keys:

- plan: every available model (up to the plan limit) with equal tuning shares, and one `square`
  feature on the first numeric column;
- reflect: toggle the leading model's class weight (classification), else no change;
- narrate: four short paragraphs that cite the facts table.

`mode` selects a failure behaviour for reliability tests (T-E2E-05): `timeout`, `server_error`,
`auth`, `invalid_json`, or `adversarial` (prohibited plan fields, an always-proposing reflector,
an invented number in the narrative). Its answers carry no data beyond the request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tabpilot.llm.consent import ConsentStore
from tabpilot.llm.errors import AuthError, LLMTimeout, ServerError
from tabpilot.schemas import LLMRequest, LLMResponse

STUB_NAME = "stub"
STUB_MODEL = "stub-1"
MODES = ("ok", "timeout", "server_error", "auth", "invalid_json", "adversarial")


class StubConsent(ConsentStore):
    """Consent for the offline stub only (it sends nothing anywhere)."""

    def allows(self, provider: str) -> bool:
        return provider == STUB_NAME


def _inner(data: dict[str, Any], key: str) -> dict[str, Any]:
    """The payload of a first call, or of a repair call (which wraps it under `key`)."""
    inner = data.get(key, data)
    return dict(inner) if isinstance(inner, dict) else data


def _plan(data: dict[str, Any], adversarial: bool) -> dict[str, Any]:
    models = list(data.get("available_models", []))[
        : int(data.get("budget", {}).get("max_models", 5))
    ]
    share = round(1.0 / len(models), 6) if models else 0.0
    plan: dict[str, Any] = {
        "models": [{"name": m, "priority": min(i + 1, 5)} for i, m in enumerate(models)],
        "tuning_allocation": dict.fromkeys(models, share),
        "features": [],
        "rationale": "Stub plan: every available model with equal tuning shares.",
    }
    numeric = [c["name"] for c in data.get("profile", {}).get("columns", [])
               if c.get("type") == "numeric"]  # fmt: skip
    if numeric:
        plan["features"] = [{"id": "f1", "name": "sq_stub", "op": "square", "inputs": [numeric[0]]}]
    if adversarial:
        plan["primary_metric"] = "accuracy"  # not a plan field: schema-invalid, then fallback
        plan["models"].append({"name": "not_a_model", "priority": 1})
    return plan


def _reflect(data: dict[str, Any], adversarial: bool) -> dict[str, Any]:
    top = data.get("leaderboard_top") or [{}]
    model = top[0].get("model")
    classification = data.get("task") != "regression"
    if adversarial and model:
        return {
            "changes": [{"type": "toggle_class_weight", "model": model, "value": "balanced"}]
            if classification
            else [{"type": "change_target_transform", "value": "log1p"}]
        }
    if classification and model:
        return {
            "changes": [{"type": "toggle_class_weight", "model": model, "value": "balanced"}],
            "expected_effect": "Balanced class weights may help the minority class.",
        }
    return {"changes": []}


def _fact(facts: dict[str, Any], key: str) -> tuple[str, str] | None:
    """(displayed value, fact id) of the fact with `key`."""
    for fid, fact in facts.items():
        if fact.get("key") == key:
            return str(fact["value"]), fid
    return None


def _cite(found: tuple[str, str] | None, text: str, fallback: str) -> str:
    return text.format(value=found[0], id=found[1]) if found else fallback


def _narrate(data: dict[str, Any], adversarial: bool) -> dict[str, Any]:
    facts = data.get("facts", {})
    paragraphs = [
        _cite(_fact(facts, "final_model"), "The selected model is {value} [{id}].",
              "A model was selected."),
        _cite(_fact(facts, "cv_mean"), "Its mean cross-validated score is {value} [{id}].",
              "Cross-validated scores are shown in the leaderboard."),
        _cite(_fact(facts, "holdout_score"),
              "On the rows set aside for a final check it scored {value} [{id}].",
              "Only cross-validated scores are available for this run."),
        _cite(_fact(facts, "top_feature_1"),
              "The input this model relied on most was `{value}` [{id}].",
              "Importance describes how this model uses its inputs."),
    ]  # fmt: skip
    if adversarial:
        paragraphs[1] += " It reached 0.9999 on every fold."
    return {"paragraphs": paragraphs}


@dataclass
class StubProvider:
    """`LLMProvider` (§23.1) that answers from the request data, without any network."""

    mode: str = "ok"
    name: str = STUB_NAME
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown stub mode {self.mode!r}")

    def available(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    def privacy_notice(self) -> str:
        return "Offline stub: no data leaves this computer."

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req.purpose)
        if self.mode == "timeout":
            raise LLMTimeout("stub timeout")
        if self.mode == "server_error":
            raise ServerError("stub 503")
        if self.mode == "auth":
            raise AuthError("stub 401")
        adversarial = self.mode == "adversarial"
        data: dict[str, Any] = dict(req.user_json)
        if req.purpose.startswith("plan"):
            body = _plan(_inner(data, "planner_input"), adversarial)
        elif req.purpose.startswith("reflect"):
            body = _reflect(_inner(data, "reflection_input"), adversarial)
        else:
            body = _narrate(data, adversarial)
        content = "not json {" if self.mode == "invalid_json" else json.dumps(body)
        return LLMResponse(provider=self.name, model_id=STUB_MODEL, content=content,
                           parsed=None if self.mode == "invalid_json" else body,
                           input_tokens=len(json.dumps(data)) // 4,
                           output_tokens=len(content) // 4, latency_s=0.0)  # fmt: skip
