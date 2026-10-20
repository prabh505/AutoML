"""Cross-fitted greedy ensemble selection (PRD §20; FR-ENS-01).

Caruana-style hill climbing with replacement over OOF predictions. For the reported CV score,
weights for fold k are learned on the other folds only, so the ensemble's fold scores are
comparable to a single model's. Final weights (for export) use all OOF rows. The ensemble is kept
only if it passes the same paired-fold rule as reflection (§19.5) against the best single model,
evaluated on the same OOF fold partition, and only within the cost limits of §20.5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tabpilot.evaluation.metrics import Problem, score, selection_score
from tabpilot.reflection.accept import paired_decision
from tabpilot.schemas import EnsembleResult, MetricName, SplitSpec, Task
from tabpilot.schemas.experiment import LeaderboardEntry
from tabpilot.schemas.reflection import EnsembleMember
from tabpilot.splitting import iter_folds

MAX_CANDIDATES = 8
MAX_PER_FAMILY = 2
MAX_MEMBERS = 5
MAX_STEPS = 20
MIN_IMPROVEMENT = 1e-6
MAX_INFERENCE_RATIO = 5.0
ENSEMBLE_ID = "ensemble"


@dataclass(frozen=True)
class Candidate:
    """A leaderboard entry with its OOF predictions (rows in `split.dev_indices` order)."""

    entry: LeaderboardEntry
    oof: NDArray[np.float32] | None


def _sel(entry: LeaderboardEntry, metric: MetricName) -> float:
    return selection_score(metric, entry.mean if entry.mean is not None else 0.0)


def eligible(cands: Sequence[Candidate], metric: MetricName) -> list[Candidate]:
    """§20.2: ok, comparable, finite OOF; best variant per family-iteration; ≤ 2 per family;
    top 8 by mean selection score. Order: score descending, then entry id.
    """
    usable = [c for c in cands if c.entry.status == "ok" and c.entry.comparable
              and c.entry.mean is not None and c.entry.model != ENSEMBLE_ID
              and c.oof is not None and np.all(np.isfinite(c.oof))]  # fmt: skip
    usable.sort(key=lambda c: (-_sel(c.entry, metric), c.entry.entry_id))
    seen_variant: set[tuple[str, int]] = set()
    per_family: dict[str, int] = {}
    out: list[Candidate] = []
    for c in usable:
        family = str(c.entry.model)
        if (family, c.entry.iteration) in seen_variant or per_family.get(
            family, 0
        ) >= MAX_PER_FAMILY:
            continue
        seen_variant.add((family, c.entry.iteration))
        per_family[family] = per_family.get(family, 0) + 1
        out.append(c)
    return out[:MAX_CANDIDATES]


def blend(
    preds: Sequence[NDArray[Any]], weights: NDArray[np.float64], problem: Problem
) -> NDArray[np.float64]:
    """Weighted average; classification rows are renormalized to sum to 1 (§20.3)."""
    out = np.zeros_like(np.asarray(preds[0], dtype=np.float64))
    for w, p in zip(weights, preds, strict=True):
        if w:
            out += w * np.asarray(p, dtype=np.float64)
    if problem.task != Task.REGRESSION:
        totals = out.sum(axis=1, keepdims=True)
        out = np.divide(out, totals, out=np.full_like(out, 1.0 / out.shape[1]), where=totals > 0)
    return out


def _score(
    pred: NDArray[np.float64], y: NDArray[Any], problem: Problem, metric: MetricName
) -> float:
    shaped = pred[:, 0] if problem.task == Task.REGRESSION else pred
    return score(metric, y, shaped, problem)


def hill_climb(
    preds: Sequence[NDArray[Any]],
    y: NDArray[Any],
    problem: Problem,
    metric: MetricName,
    *,
    max_steps: int = MAX_STEPS,
    max_members: int = MAX_MEMBERS,
) -> NDArray[np.float64]:
    """Greedy selection with replacement; ties go to the lower index (entry-id order)."""
    n = len(preds)

    def sel(counts: NDArray[np.float64]) -> float:
        return selection_score(metric, _score(blend(preds, counts / counts.sum(), problem), y,
                                              problem, metric))  # fmt: skip

    singles = [sel(np.eye(n)[i]) for i in range(n)]
    best = int(np.argmax(singles))  # argmax returns the first maximum: lowest index on ties
    counts = np.zeros(n)
    counts[best] = 1
    current = singles[best]
    for _ in range(max_steps):
        if int(np.count_nonzero(counts)) >= max_members:
            break
        trials = []
        for i in range(n):
            trial = counts.copy()
            trial[i] += 1
            trials.append(sel(trial))
        pick = int(np.argmax(trials))
        if trials[pick] <= current + MIN_IMPROVEMENT:
            break
        counts[pick] += 1
        current = trials[pick]
    return np.asarray(counts / counts.sum(), dtype=np.float64)


def fold_partition(split: SplitSpec) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """(train positions, valid positions) into the OOF arrays for every repeat × fold."""
    position = {row: i for i, row in enumerate(split.dev_indices)}
    out = []
    for _, _, train, valid in iter_folds(split):
        out.append(
            (np.asarray([position[r] for r in train]), np.asarray([position[r] for r in valid]))
        )
    return out


def oof_fold_scores(
    pred: NDArray[Any], y: NDArray[Any], split: SplitSpec, problem: Problem, metric: MetricName
) -> list[float]:
    """Fold scores of one OOF array on the run's fold partition."""
    return [_score(np.asarray(pred, dtype=np.float64)[va], y[va], problem, metric)
            for _, va in fold_partition(split)]  # fmt: skip


def cross_fitted_ensemble(
    preds: Sequence[NDArray[Any]],
    y: NDArray[Any],
    split: SplitSpec,
    problem: Problem,
    metric: MetricName,
    *,
    max_members: int = MAX_MEMBERS,
    max_steps: int = MAX_STEPS,
) -> tuple[list[float], NDArray[np.float64]]:
    """§20.3: fold k scored with weights learned on the other folds; final weights on all rows."""
    fold_scores = []
    for tr, va in fold_partition(split):
        w = hill_climb([np.asarray(p)[tr] for p in preds], y[tr], problem, metric,
                       max_steps=max_steps, max_members=max_members)  # fmt: skip
        fold_scores.append(_score(blend([np.asarray(p)[va] for p in preds], w, problem), y[va],
                                  problem, metric))  # fmt: skip
    final = hill_climb(preds, y, problem, metric, max_steps=max_steps, max_members=max_members)
    return fold_scores, final


def build_ensemble(
    cands: Sequence[Candidate],
    y_dev: NDArray[Any],
    split: SplitSpec,
    problem: Problem,
    metric: MetricName,
    *,
    artifact_limit_bytes: int,
) -> EnsembleResult:
    """§20.2–§20.5. `y_dev` is aligned to `split.dev_indices`."""
    pool = eligible(cands, metric)
    if len(pool) < 2:
        return EnsembleResult(kept=False, reason="fewer than 2 eligible candidates")
    preds = [c.oof for c in pool if c.oof is not None]
    fold_scores, weights = cross_fitted_ensemble(preds, y_dev, split, problem, metric)
    members = [EnsembleMember(entry_id=c.entry.entry_id, weight=float(w))
               for c, w in zip(pool, weights, strict=True) if w > 0]  # fmt: skip
    best = pool[0]
    if len(members) < 2:
        return EnsembleResult(kept=False, reason="hill climbing selected a single model",
                              members=members, fold_scores=fold_scores)  # fmt: skip
    chosen = [c.entry for c, w in zip(pool, weights, strict=True) if w > 0]
    ratio = sum(e.predict_seconds for e in chosen) / max(best.entry.predict_seconds, 1e-9)
    size = sum(e.model_size_bytes for e in chosen)
    single = oof_fold_scores(preds[0], y_dev, split, problem, metric)  # same partition
    decision = paired_decision(parent_id=best.entry.entry_id, child_id=ENSEMBLE_ID,
                               parent_scores=single, child_scores=fold_scores,
                               metric=metric)  # fmt: skip
    if ratio > MAX_INFERENCE_RATIO:
        reason, kept = f"predicted inference {ratio:.1f}× the best single model (limit 5×)", False
    elif size > artifact_limit_bytes:
        reason, kept = "members exceed the artifact size limit", False
    else:
        reason, kept = decision.reason, decision.accepted
    return EnsembleResult(kept=kept, reason=reason[:300], members=members, fold_scores=fold_scores,
                          decision=decision, predicted_inference_ratio=ratio)  # fmt: skip
