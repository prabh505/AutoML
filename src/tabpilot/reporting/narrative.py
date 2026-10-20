"""Plain-English narrative: LLM with fact-check, or the deterministic template (PRD §21.7).

```
narrator call (narrator_system@1.0, facts table as data) → schema + fact-check
  failed → one repair call with the violations (≤ 1,500 chars) → schema + fact-check
  still failing, or no provider, or any router failure → template narrative
```

The template narrative is built from the facts table only and always passes the fact-check
(tested). The report labels which narrative was used (I-8, I-9).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tabpilot.llm import BudgetTracker, LLMRouter, LLMUnavailable
from tabpilot.planning.templates import Template, compact_model_schema, fill_template
from tabpilot.reporting.factcheck import MAX_WORDS, PARAGRAPHS, fact_check
from tabpilot.reporting.facts import Fact, FactsTable
from tabpilot.schemas import LLMRequest, RunEvent
from tabpilot.schemas.llm import LLMPurpose
from tabpilot.schemas.report import NarrativeSection

PROMPTS = Path(__file__).parent / "prompts"
NARRATOR_SYSTEM = "narrator_system"
NARRATOR_REPAIR = "narrator_repair"
VERSION = "1.0"
NARRATE_MAX_OUTPUT_TOKENS = 1200
REPAIR_CHARS = 1500
PREVIOUS_CHARS = 4000


class NarratorOutput(BaseModel):
    """The narrator's JSON answer: exactly 4 paragraphs."""

    model_config = ConfigDict(extra="forbid")

    paragraphs: list[str] = Field(min_length=PARAGRAPHS, max_length=PARAGRAPHS)


@dataclass
class Narrative:
    """The shipped narrative and how it was produced."""

    text: str
    source: Literal["llm", "template"]
    fact_check: Literal["passed", "failed", "not_run"]
    violations: list[str] = field(default_factory=list)
    fallback_reason: str | None = None

    def section(self) -> NarrativeSection:
        return NarrativeSection(source=self.source, fact_check=self.fact_check, text=self.text)


@cache
def load(name: str, version: str = VERSION) -> Template:
    """Load a narrator template; `{{SCHEMA}}` is the compact `NarratorOutput` schema."""
    schema = json.dumps(compact_model_schema(NarratorOutput), separators=(",", ":"),
                        sort_keys=True)  # fmt: skip
    fills = {"{{SCHEMA}}": schema, "{{MAX_WORDS}}": str(MAX_WORDS),
             "{{PARAGRAPHS}}": str(PARAGRAPHS)}  # fmt: skip
    return fill_template(PROMPTS, name, version, fills)


def versions() -> dict[str, str]:
    """Template versions for `ExperimentConfig.prompt_template_versions`."""
    return {NARRATOR_SYSTEM: VERSION, NARRATOR_REPAIR: VERSION}


# ---------------------------------------------------------------- template narrative
def _ref(fact: Fact) -> str:
    shown = (
        f"`{fact.value}`"
        if fact.key in ("target",) or fact.key.startswith("top_feature")
        else fact.display()
    )
    return f"{shown} [{fact.id}]"


def _f(facts: FactsTable, key: str) -> Fact:
    found = facts.by_key(key)
    if found is None:
        raise KeyError(key)
    return found


def _opt(facts: FactsTable, key: str) -> Fact | None:
    return facts.by_key(key)


def template_narrative(facts: FactsTable) -> Narrative:
    """Deterministic 4-paragraph narrative from the facts table (§21.7 step 4)."""

    def f(key: str) -> str:
        return _ref(_f(facts, key))

    p1 = (f"This run predicts {f('target')} as a {f('task')} task. TabPilot compared "
          f"{f('n_models_compared')} model types with cross-validation over {f('n_folds')} folds "
          f"on the development rows and selected {f('final_model')} by its {f('primary_metric')} "
          "score.")  # fmt: skip
    p2 = (f"The selected model's mean cross-validated score is {f('cv_mean')}, with a "
          f"fold-to-fold standard deviation of {f('cv_std')}.")  # fmt: skip
    if _opt(facts, "runner_up_model") is not None:
        p2 += (
            f" The runner-up, {f('runner_up_model')}, averaged {f('runner_up_cv_mean')}; the "
            f"selected model scored higher on {f('folds_won')} of {f('n_folds')} folds. With "
            "this few folds the comparison is descriptive, not a significance test."
        )
    p2 += (
        " Cross-validated scores of the selected model are optimistic, because the same folds "
        "were used to choose it."
    )
    if _opt(facts, "holdout_score") is not None:
        p3 = (f"On the independent test rows, used once after selection, the selected model "
              f"scored {f('holdout_score')}.")  # fmt: skip
        if _opt(facts, "holdout_baseline_score") is not None:
            p3 += (f" The reference baseline, {f('baseline_model')}, scored "
                   f"{f('holdout_baseline_score')}.")  # fmt: skip
        if _opt(facts, "holdout_ci_low") is not None:
            p3 += (f" The {_f(facts, 'ci_level').display()}% [{_f(facts, 'ci_level').id}] "
                   f"bootstrap interval for the selected model runs from {f('holdout_ci_low')} "
                   f"to {f('holdout_ci_high')}.")  # fmt: skip
    else:
        p3 = (
            "No rows were set aside for a final check, so only cross-validated scores are "
            "available for this run."
        )
    tops = [_ref(fact) for i in range(1, 6) if (fact := _opt(facts, f"top_feature_{i}"))]
    if tops:
        listed = ", ".join(tops[:-1]) + (" and " if len(tops) > 1 else "") + tops[-1]
        p4 = f"The inputs this model relied on most were {listed}."
    else:
        p4 = "Feature importance was not available for this run."
    p4 += (
        " Importance describes how this model uses its inputs, not what would happen in the "
        "world if a value changed; correlated inputs can share or swap importance."
    )
    if _opt(facts, "ensemble_members") is not None:
        p4 += (f" The final model blends {f('ensemble_members')} models, kept because the blend "
               "scored higher on most folds.")  # fmt: skip
    text = "\n\n".join([p1, p2, p3, p4])
    return Narrative(text=text, source="template", fact_check="not_run")


def checked_template(facts: FactsTable, events: Sequence[RunEvent],
                     reason: str | None) -> Narrative:  # fmt: skip
    """The template narrative with its own fact-check result recorded."""
    narrative = template_narrative(facts)
    result = fact_check(narrative.text, facts, events)
    narrative.fact_check = "passed" if result.passed else "failed"
    narrative.violations = result.violations
    narrative.fallback_reason = reason
    return narrative


# ---------------------------------------------------------------- LLM narrative
def _call(router: LLMRouter, budget: BudgetTracker, purpose: LLMPurpose, template: str,
          user_json: dict[str, Any]) -> Any:  # fmt: skip
    tpl = load(template)
    request = LLMRequest(purpose=purpose, system=tpl.text, user_json=user_json,
                         response_schema=NarratorOutput,
                         max_output_tokens=NARRATE_MAX_OUTPUT_TOKENS, temperature=0.0,
                         timeout_s=budget.budget.per_call_timeout_s)  # fmt: skip
    return router.complete(request, budget, template=tpl.id)


def _assess(response: Any, facts: FactsTable,
            events: Sequence[RunEvent]) -> tuple[str | None, list[str]]:  # fmt: skip
    raw = response.parsed if response.parsed is not None else response.content
    try:
        output = NarratorOutput.model_validate(raw if isinstance(raw, dict) else json.loads(raw))
    except (ValidationError, ValueError, TypeError) as exc:
        name = type(exc).__name__
        return None, [f"response is not a JSON object with {PARAGRAPHS} paragraphs ({name})"]
    text = "\n\n".join(p.strip() for p in output.paragraphs)
    return text, fact_check(text, facts, events).violations


def _violation_text(violations: list[str]) -> str:
    text = "\n".join(violations)
    return text if len(text) <= REPAIR_CHARS else text[: REPAIR_CHARS - 1] + "…"


def narrate(
    facts: FactsTable,
    router: LLMRouter | None,
    budget: BudgetTracker | None,
    events: Sequence[RunEvent] = (),
) -> Narrative:
    """§21.7 steps 2–4: LLM narrative if it passes the fact-check (after ≤ 1 repair), else the
    template narrative. Never raises for LLM problems (I-8).
    """
    if router is None or budget is None or not router.usable():
        return checked_template(facts, events, "no LLM provider available")
    payload: dict[str, Any] = {"facts": facts.payload()}
    try:
        first = _call(router, budget, "narrate", NARRATOR_SYSTEM, payload)
        text, violations = _assess(first, facts, events)
        if violations:
            repair = {"facts": payload["facts"],
                      "previous_response": str(first.content)[:PREVIOUS_CHARS],
                      "violations": _violation_text(violations)}  # fmt: skip
            second = _call(router, budget, "narrate_repair", NARRATOR_REPAIR, repair)
            text, violations = _assess(second, facts, events)
    except LLMUnavailable as exc:
        return checked_template(facts, events, exc.reason)
    if violations or text is None:
        narrative = checked_template(facts, events, "LLM narrative failed the fact-check")
        narrative.violations = violations
        return narrative
    return Narrative(text=text, source="llm", fact_check="passed")
