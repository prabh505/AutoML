"""Paired-fold acceptance rule (PRD §19.5, §20.4).

The single source of truth for both reflection (child vs parent) and ensembling (ensemble vs best
single model). Deltas are in selection-score units, so "better" is always positive whatever the
metric direction. The relative gain is computed on the parent's metric magnitude.

```
d_k = sel(C', k) - sel(C, k);  mean_gain = mean(d_k)
rel_gain = mean_gain / max(|mean_metric(C)|, 1e-12)
accept iff rel_gain ≥ 0.002 and #(d_k > 0) ≥ ceil(0.6 · K) and C' ok and comparable
```
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from tabpilot.evaluation.metrics import selection_score
from tabpilot.schemas import AcceptanceDecision, MetricName
from tabpilot.schemas.experiment import LeaderboardEntry

MIN_REL_GAIN = 0.002  # 0.2% [SRC]
FOLD_MAJORITY = 0.6
EPS = 1e-12


def paired_decision(
    *,
    parent_id: str,
    child_id: str,
    parent_scores: Sequence[float],
    child_scores: Sequence[float],
    metric: MetricName,
    child_usable: bool = True,
    min_rel_gain: float = MIN_REL_GAIN,
    majority: float = FOLD_MAJORITY,
) -> AcceptanceDecision:
    """§19.5 on raw fold scores of the same folds (same order)."""
    if len(parent_scores) != len(child_scores) or not parent_scores:
        raise ValueError("paired comparison needs the same non-empty folds")
    deltas = [selection_score(metric, c) - selection_score(metric, p)
              for p, c in zip(parent_scores, child_scores, strict=True)]  # fmt: skip
    mean_gain = math.fsum(deltas) / len(deltas)
    parent_mean = math.fsum(parent_scores) / len(parent_scores)
    rel_gain = mean_gain / max(abs(parent_mean), EPS)
    better = sum(d > 0 for d in deltas)
    required = math.ceil(majority * len(deltas))
    if not child_usable:
        accepted, reason = False, "challenger is not ok or not comparable"
    elif rel_gain < min_rel_gain:
        accepted, reason = False, f"relative gain {rel_gain:.4%} below {min_rel_gain:.1%}"
    elif better < required:
        accepted, reason = False, f"better in {better} of {len(deltas)} folds; needs {required}"
    else:
        accepted = True
        reason = f"improved by {rel_gain:.2%} and better in {better} of {len(deltas)} folds"
    return AcceptanceDecision(parent_entry_id=parent_id, child_entry_id=child_id,
                              fold_deltas=deltas, mean_gain=mean_gain, rel_gain=rel_gain,
                              folds_better=better, folds_required=required, accepted=accepted,
                              reason=reason)  # fmt: skip


def accept(parent: LeaderboardEntry, child: LeaderboardEntry) -> AcceptanceDecision:
    """§19.5 for two leaderboard entries. A child that failed, is not comparable, or used other
    folds is never accepted.
    """
    metric = MetricName(parent.metric.name)
    usable = (child.status == "ok" and child.comparable and child.split_id == parent.split_id
              and child.metric.name == parent.metric.name
              and len(child.fold_scores) == len(parent.fold_scores))  # fmt: skip
    # an unusable challenger is compared as equal to the parent: all deltas 0, always rejected
    child_scores = child.fold_scores if usable else parent.fold_scores
    return paired_decision(parent_id=parent.entry_id, child_id=child.entry_id,
                           parent_scores=parent.fold_scores, child_scores=child_scores,
                           metric=metric, child_usable=usable)  # fmt: skip
