"""Leaderboard entries from CV results (PRD §18.1, §18.4; FR-EVL-01).

`mean`, `std` (ddof=1) and `sem` are computed only from the stored fold scores, so they can be
recomputed exactly (T-EVL-01). An entry is comparable only when it used the run's split, primary
metric, and DEV row set (§10.4).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tabpilot.evaluation.metrics import SECONDARY, Problem, score
from tabpilot.evaluation.metrics import spec as metric_spec
from tabpilot.schemas import MetricName, ModelName, SplitSpec, Task
from tabpilot.schemas.experiment import LeaderboardEntry, TrainingResult, fold_statistics
from tabpilot.splitting import iter_folds


def secondary_metrics(
    oof: NDArray[np.float32] | None,
    y: NDArray[Any],
    split: SplitSpec,
    problem: Problem,
    primary: MetricName,
) -> dict[MetricName, float]:
    """Mean over folds of each secondary metric, computed from the stored OOF rows (§18.1).

    Repeated CV uses the repeat-averaged OOF on the first repeat's folds. A metric that is
    undefined on some fold (e.g. RMSLE with negative values) is left out.
    """
    if oof is None:
        return {}
    position = {row: i for i, row in enumerate(split.dev_indices)}
    first_repeat = [f for f in iter_folds(split) if f[0] == 0]
    out: dict[MetricName, float] = {}
    for name in SECONDARY[problem.task]:
        if name == primary:
            continue
        values = []
        try:
            for _, _, _, valid in first_repeat:
                pred = oof[[position[r] for r in valid]].astype(np.float64)
                pred_in = pred[:, 0] if problem.task == Task.REGRESSION else pred
                values.append(score(name, y[valid], pred_in, problem))
        except ValueError:
            continue
        if all(math.isfinite(v) for v in values):
            out[name] = float(np.mean(values))
    return out


def make_entry(
    result: TrainingResult,
    *,
    run_id: str,
    model: ModelName,
    iteration: int,
    feature_set: str,
    metric: MetricName,
    split: SplitSpec,
    run_split_id: str,
    run_metric: MetricName,
    secondary: dict[MetricName, float] | None = None,
    model_size_bytes: int = 0,
    n_features_out: int = 0,
    warnings: Sequence[str] = (),
) -> LeaderboardEntry:
    """One leaderboard row (§18.1). Failed/skipped rows keep their status and have no stats."""
    comparable = split.split_id == run_split_id and metric == run_metric
    scores = list(result.fold_scores) if result.status == "ok" else []
    stats: dict[str, float | None] = {"mean": None, "std": None, "sem": None}
    if scores:
        mean, std, sem = fold_statistics(scores)
        stats = {"mean": mean, "std": std, "sem": sem}
    notes = list(warnings) + ([result.error[:200]] if result.error else [])
    return LeaderboardEntry(
        entry_id=result.entry_id, experiment_id=run_id, iteration=iteration, model=model,
        feature_set=feature_set, params=dict(result.best_params), metric=metric_spec(metric),
        fold_scores=scores, secondary=secondary or {}, fit_seconds=result.fit_seconds_mean,
        predict_seconds=result.predict_seconds_mean, model_size_bytes=model_size_bytes,
        n_features_out=n_features_out, status=result.status, comparable=comparable,
        warnings=notes, split_id=split.split_id, **stats,
    )  # fmt: skip


def not_comparable(entries: Sequence[LeaderboardEntry]) -> list[LeaderboardEntry]:
    """Entries for the separate "Not comparable" table (§18.4)."""
    return [e for e in entries if not e.comparable]
