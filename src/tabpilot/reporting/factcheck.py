"""Strict fact-check of a narrative (PRD §21.7 step 3; FR-RPT-01, I-9).

Rules, applied per paragraph:

1. Shape: exactly 4 paragraphs (blank-line separated), at most 300 words.
2. Citations `[F12]` must name existing facts.
3. Every number (integer, decimal, or percentage) must equal a fact's value under that fact's
   declared rounding (percentages only match `%` facts), and the paragraph must cite one of the
   matching facts.
4. Every model family mentioned must be a model named in the facts table; every identifier in
   backticks and every snake_case identifier must be a name in the facts table.
5. Banned claims: causal language and guarantees.
6. Actions must be in the event log: "tuned" needs completed tuning trials, "ensemble" needs a
   kept ensemble, "reflection" an accepted round, "test set" an evaluated holdout, "LLM" a
   successful LLM call.

Names from the facts table are masked before rules 3–6, so a column called `cause_code` or
`Noise 1` cannot trip a rule. A failed check never ships the text (NarrativeSection).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from tabpilot.reporting.facts import FactsTable
from tabpilot.schemas import EventType, ModelName, RunEvent

MAX_WORDS = 300
PARAGRAPHS = 4
MIN_TRIALS_FOR_TUNING = 2
CITATION = re.compile(r"\[F(\d+)\]")
NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w])")
BACKTICK = re.compile(r"`([^`]*)`")
SNAKE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
MASK = " • "
MODEL_PATTERNS: dict[str, re.Pattern[str]] = {
    ModelName.LINEAR.value: re.compile(r"\blinear\b|\blogistic regression\b|\bridge\b", re.I),
    ModelName.RANDOM_FOREST.value: re.compile(r"\brandom[ _-]?forests?\b", re.I),
    ModelName.LIGHTGBM.value: re.compile(r"\blight ?gbm\b", re.I),
    ModelName.XGBOOST.value: re.compile(r"\bxgboost\b", re.I),
    ModelName.CATBOOST.value: re.compile(r"\bcatboost\b", re.I),
    ModelName.HIST_GB.value: re.compile(r"\bhist[ _]?gb\b|\bhistogram gradient boosting\b", re.I),
}
BANNED: dict[str, re.Pattern[str]] = {
    "causal language": re.compile(
        r"\bcaus(?:e|es|ed|ing|al|ally|ation)\b|\bleads? to\b|\bled to\b|\bbecause of\b"
        r"|\bdue to\b|\bdrives?\b|\bdriven by\b|\bresults? in\b|\bimpacts?\b|\beffect of\b",
        re.I),
    "guarantee": re.compile(
        r"\bguarantee(?:s|d)?\b|\bwill (?:achieve|perform|deliver|reach|generalize|work)\b"
        r"|\bcertain(?:ly)?\b|\balways\b|\bprove[sn]?\b", re.I),
}  # fmt: skip


def _tuned(e: RunEvent) -> bool:
    if e.type == EventType.TRIAL_COMPLETED:
        return True
    trials = e.data.get("trials")
    return (e.type == EventType.CANDIDATE_COMPLETED and isinstance(trials, int)
            and trials >= MIN_TRIALS_FOR_TUNING)  # fmt: skip


ACTIONS: dict[str, tuple[re.Pattern[str], Callable[[RunEvent], bool]]] = {
    "tuning": (re.compile(r"\btun(?:ed|ing|es)\b|\bhyper-?parameters?\b", re.I), _tuned),
    "ensemble": (re.compile(r"\bensembl\w*|\bblend\w*", re.I),
                 lambda e: e.type == EventType.ENSEMBLE_DECIDED and e.data.get("kept") is True),
    "reflection": (re.compile(r"\breflect\w*|\brefine\w*", re.I),
                   lambda e: e.type == EventType.REFLECTION_DECIDED
                   and e.data.get("accepted") is True),
    "holdout": (re.compile(r"\bhold-?out\b|\bindependent test\b|\btest (?:set|rows|data)\b"
                           r"|\bunseen data\b", re.I),
                lambda e: e.type == EventType.HOLDOUT_EVALUATED and e.data.get("present") is True),
    "llm": (re.compile(r"\bLLMs?\b|\blanguage models?\b|\bAI\b", re.I),
            lambda e: e.type == EventType.LLM_CALL_SUCCEEDED),
}  # fmt: skip


@dataclass
class FactCheckResult:
    """`passed` iff there are no violations."""

    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _mask_names(text: str, names: Sequence[str]) -> str:
    for name in sorted(names, key=len, reverse=True):
        if len(name) >= 2:
            text = re.sub(re.escape(name), MASK, text, flags=re.I)
    return text


def _number(token: str) -> tuple[Decimal | None, bool]:
    percent = token.endswith("%")
    try:
        return Decimal(token.rstrip("%").replace(",", "")), percent
    except InvalidOperation:
        return None, percent


def _check_numbers(paragraph: str, cited: set[str], facts: FactsTable, n: int) -> list[str]:
    out = []
    for token in NUMBER.findall(paragraph):
        value, percent = _number(token)
        matches = {
            f.id
            for f in facts.facts.values()
            if f.is_number and (f.unit == "%") == percent and f.decimal() == value
        }
        if not matches:
            out.append(f"paragraph {n}: number {token} is not in the facts table")
        elif not matches & cited:
            out.append(f"paragraph {n}: number {token} does not cite " + "/".join(sorted(matches)))
    return out


def _check_names(paragraph: str, facts: FactsTable, n: int) -> list[str]:
    out = []
    names = facts.names()
    models = facts.model_names()
    unmasked = _mask_names(paragraph, sorted(names - models))  # e.g. a column `linear_trend`
    for family, pattern in MODEL_PATTERNS.items():
        if pattern.search(unmasked) and family not in models:
            out.append(f"paragraph {n}: model {family} is not in the facts table")
    for token in BACKTICK.findall(paragraph):
        if token not in names:
            out.append(f"paragraph {n}: name `{token}` is not in the facts table")
    return out


def fact_check(text: str, facts: FactsTable, event_log: Sequence[RunEvent]) -> FactCheckResult:
    """§21.7 step 3. Returns every violation found (empty = passed)."""
    result = FactCheckResult()
    paras = paragraphs(text)
    if len(paras) != PARAGRAPHS:
        result.violations.append(f"expected {PARAGRAPHS} paragraphs, found {len(paras)}")
    if len(text.split()) > MAX_WORDS:
        result.violations.append(f"more than {MAX_WORDS} words")
    done = {name for name, (_, happened) in ACTIONS.items() if any(map(happened, event_log))}
    for n, paragraph in enumerate(paras, start=1):
        cited = {f"F{i}" for i in CITATION.findall(paragraph)}
        result.violations.extend(f"paragraph {n}: unknown fact {c}"
                                 for c in sorted(cited - set(facts.facts)))  # fmt: skip
        result.violations.extend(_check_names(paragraph, facts, n))
        plain = _mask_names(CITATION.sub(MASK, paragraph), sorted(facts.names()))
        plain = BACKTICK.sub(MASK, plain)
        result.violations.extend(f"paragraph {n}: unknown name {token}"
                                 for token in SNAKE.findall(plain))  # fmt: skip
        result.violations.extend(_check_numbers(plain, cited, facts, n))
        for label, pattern in BANNED.items():
            if (hit := pattern.search(plain)) is not None:
                result.violations.append(f"paragraph {n}: {label} ({hit.group(0)!r})")
        for action, (pattern, _) in ACTIONS.items():
            if action not in done and (hit := pattern.search(plain)) is not None:
                result.violations.append(f"paragraph {n}: mentions {action} ({hit.group(0)!r}) "
                                         "but the event log has none")  # fmt: skip
    return result
