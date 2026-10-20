"""Reflection round budget and stopping rules (PRD §19.4 as amended by A-004, §19.6).

A-004: reflection may use ≤ 15% of the run budget across all rounds. One round gets
min(25% of the total, the reflection share still unused, the time left before the reserved slice
at 70%). Reflection is skipped when the remaining non-reserved time is < 15% of the total. Extra
trials per round: 30% of the initial round's trial count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

REFLECTION_SHARE = 0.15
ROUND_SHARE = 0.25
STOP_FRACTION = 0.70
MIN_REMAINING_SHARE = 0.15
TRIAL_SHARE = 0.30


@dataclass(frozen=True)
class RoundBudget:
    """What one reflection round may spend, or why it is skipped."""

    seconds: float
    trials: int
    skip_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skip_reason is not None


def reflection_budget(
    total_s: float, elapsed_s: float, reflection_used_s: float, initial_trials: int
) -> RoundBudget:
    """The budget of the next round (§19.4, A-004)."""
    before_reserved = STOP_FRACTION * total_s - elapsed_s
    if before_reserved < MIN_REMAINING_SHARE * total_s:
        return RoundBudget(0.0, 0, "budget")
    seconds = min(ROUND_SHARE * total_s, REFLECTION_SHARE * total_s - reflection_used_s,
                  before_reserved)  # fmt: skip
    if seconds <= 0:
        return RoundBudget(0.0, 0, "budget")
    return RoundBudget(seconds, max(1, math.floor(TRIAL_SHARE * initial_trials)))


@dataclass(frozen=True)
class RoundOutcome:
    """Facts about the round that just ended, for `should_stop`."""

    rounds_used: int
    max_rounds: int
    accepted: bool = False
    budget_skipped: bool = False
    llm_unavailable: bool = False
    proposal_invalid: bool = False
    proposal_empty: bool = False


def should_stop(outcome: RoundOutcome) -> tuple[bool, str | None]:
    """§19.6: the first rule that holds gives the reason."""
    rules = (
        (outcome.rounds_used >= outcome.max_rounds, "max rounds reached"),
        (outcome.budget_skipped, "budget below minimum"),
        (outcome.llm_unavailable, "LLM unavailable or LLM budget exhausted"),
        (outcome.proposal_invalid, "proposal invalid after one repair"),
        (outcome.proposal_empty, "proposal empty"),
        (not outcome.accepted, "no change accepted this round"),
    )
    for holds, reason in rules:
        if holds:
            return True, reason
    return False, None
