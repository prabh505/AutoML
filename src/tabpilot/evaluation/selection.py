"""Deterministic model selection (PRD §18.2–§18.3; FR-EVL-01; I-5).

Selection uses computed CV scores only. Once `SelectionFreeze.freeze()` has been called (just
before the holdout opens), any further `select()` call raises, so holdout scores can never feed
back into a decision (FR-EVL-02).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tabpilot.evaluation.metrics import selection_score
from tabpilot.schemas import MetricName
from tabpilot.schemas.experiment import LeaderboardEntry, RunnerUpPaired, SelectionRecord

NEAR_TIE_SEM_FRACTION = 0.25
FAMILY_RANK = {"linear": 0, "random_forest": 1, "lightgbm": 2, "xgboost": 2, "catboost": 2,
               "hist_gb": 2, "ensemble": 3}  # fmt: skip
RULE = "max mean CV score; near-ties (0.25 SEM): simpler, fewer features, faster, id"
TIE_KEYS = ("simpler model family", "fewer output features", "lower predict seconds",
            "lexicographic entry id")  # fmt: skip


class NoValidModelError(RuntimeError):
    """No entry is `ok` and comparable (§18.2)."""


class SelectionFrozenError(RuntimeError):
    """`select()` was called after selection was frozen."""


@dataclass
class SelectionFreeze:
    """The frozen flag the `HoldoutGuard` checks; set once, never cleared."""

    frozen: bool = False

    def freeze(self) -> None:
        self.frozen = True

    def is_frozen(self) -> bool:
        return self.frozen


def _sel(entry: LeaderboardEntry) -> float:
    assert entry.mean is not None  # noqa: S101 - eligible entries have fold scores
    return selection_score(MetricName(entry.metric.name), entry.mean)


def _tie_key(entry: LeaderboardEntry) -> tuple[int, int, float, str]:
    return (FAMILY_RANK.get(str(entry.model), 3), entry.n_features_out, entry.predict_seconds,
            entry.entry_id)  # fmt: skip


def _deciding_key(near: list[LeaderboardEntry]) -> str | None:
    keys = sorted(_tie_key(e) for e in near)
    for i, name in enumerate(TIE_KEYS):
        if keys[0][i] != keys[1][i]:
            return name
    return None


def paired_differences(winner: LeaderboardEntry, runner: LeaderboardEntry) -> RunnerUpPaired:
    """Winner − runner-up per fold in selection-score units; descriptive only (§18.3)."""
    metric = MetricName(winner.metric.name)
    diffs = [selection_score(metric, w) - selection_score(metric, r)
             for w, r in zip(winner.fold_scores, runner.fold_scores, strict=True)]  # fmt: skip
    return RunnerUpPaired(runner_up=runner.entry_id, mean_diff=sum(diffs) / len(diffs),
                          min_diff=min(diffs), max_diff=max(diffs),
                          folds_won=sum(d > 0 for d in diffs), n_folds=len(diffs))  # fmt: skip


def select(
    entries: Sequence[LeaderboardEntry], freeze: SelectionFreeze | None = None
) -> SelectionRecord:
    """The §18.2 rule: best mean selection score; among near-ties (within 0.25 × the best entry's
    SEM) prefer the simpler family, then fewer output features, lower predict time, entry id.
    """
    if freeze is not None and freeze.frozen:
        raise SelectionFrozenError("selection is frozen; the holdout may already be open")
    eligible = [e for e in entries if e.status == "ok" and e.comparable and e.mean is not None]
    if not eligible:
        raise NoValidModelError("no successful, comparable candidate")
    best = max(eligible, key=lambda e: (_sel(e), -_tie_key(e)[0]))
    margin = NEAR_TIE_SEM_FRACTION * (best.sem or 0.0)
    near = [e for e in eligible if _sel(best) - _sel(e) <= margin]
    near.sort(key=_tie_key)
    winner = near[0]
    others = sorted((e for e in eligible if e.entry_id != winner.entry_id),
                    key=lambda e: (-_sel(e), _tie_key(e)))  # fmt: skip
    paired = None
    if others and len(others[0].fold_scores) == len(winner.fold_scores):
        paired = paired_differences(winner, others[0])
    return SelectionRecord(
        winner=winner.entry_id,
        rule=RULE,
        near_ties=[e.entry_id for e in near if e.entry_id != winner.entry_id],
        tie_break=_deciding_key(near) if len(near) > 1 else None,
        runner_up_paired=paired,
    )
