"""Report figures as PNG (PRD §21.2).

Matplotlib's object-oriented API (`Figure`) renders with the Agg canvas and touches no global
pyplot state, so it is safe in worker threads and never needs a display. User-derived text
(column names, model names) has `$` escaped so it is never parsed as mathtext.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from tabpilot.evaluation.metrics import Problem
from tabpilot.explain import ErrorAnalysis, ExplanationResult, Importance, ShapOutput
from tabpilot.schemas import Task
from tabpilot.schemas.experiment import LeaderboardEntry

TOP_BARS = 20
TOP_ENTRIES = 15
RELIABILITY_BINS = 10
DPI = 100


def label(text: object) -> str:
    """Plain text for matplotlib: `$` would start mathtext."""
    return str(text).replace("$", r"\$")


def _figure(width: float = 6.4, height: float = 4.0) -> Figure:
    return Figure(figsize=(width, height), dpi=DPI, layout="constrained")


def _ranked(entries: Sequence[LeaderboardEntry]) -> list[LeaderboardEntry]:
    ok = [e for e in entries if e.status == "ok" and e.comparable and e.mean is not None]
    sign = 1.0 if ok and ok[0].metric.direction == "maximize" else -1.0
    return sorted(ok, key=lambda e: (-sign * (e.mean or 0.0), e.entry_id))[:TOP_ENTRIES]


def leaderboard_chart(entries: Sequence[LeaderboardEntry]) -> Figure | None:
    """Mean CV score per entry with ±std error bars (CV on DEV only, T-EVL-06)."""
    ranked = _ranked(entries)
    if not ranked:
        return None
    fig = _figure(6.4, 1.0 + 0.35 * len(ranked))
    ax = fig.add_subplot()
    y = np.arange(len(ranked))[::-1]
    ax.barh(y, [e.mean or 0.0 for e in ranked], xerr=[e.std or 0.0 for e in ranked],
            color="#4C72B0", capsize=3)  # fmt: skip
    ax.set_yticks(y, [label(f"{e.model} ({e.entry_id[-8:]})") for e in ranked])
    ax.set_xlabel(label(f"{ranked[0].metric.name.value} (CV mean ± std)"))
    lows = [(e.mean or 0.0) - (e.std or 0.0) for e in ranked]
    highs = [(e.mean or 0.0) + (e.std or 0.0) for e in ranked]
    pad = 0.1 * max(max(highs) - min(lows), 1e-6)
    ax.set_xlim(min(lows) - pad, max(highs) + pad)
    return fig


def fold_strip(entries: Sequence[LeaderboardEntry]) -> Figure | None:
    """Fold scores per entry (strip plot), fold-paired by colour order."""
    ranked = _ranked(entries)
    if not ranked:
        return None
    fig = _figure(6.4, 1.0 + 0.35 * len(ranked))
    ax = fig.add_subplot()
    rng = np.random.default_rng(0)
    for row, entry in enumerate(ranked[::-1]):
        scores = np.asarray(entry.fold_scores, dtype=float)
        ax.scatter(scores, row + rng.uniform(-0.15, 0.15, len(scores)), s=14, color="#4C72B0")
    ax.set_yticks(range(len(ranked)), [label(e.model) for e in ranked[::-1]])
    ax.set_xlabel(label(f"{ranked[0].metric.name.value} per fold"))
    return fig


def importance_bar(items: Sequence[Importance], xlabel: str) -> Figure | None:
    """Horizontal bars for up to 20 source columns (std whiskers when present)."""
    shown = list(items)[:TOP_BARS]
    if not shown:
        return None
    fig = _figure(6.4, 1.0 + 0.3 * len(shown))
    ax = fig.add_subplot()
    y = np.arange(len(shown))[::-1]
    errs = [i.std for i in shown]
    ax.barh(y, [i.value for i in shown], xerr=errs if any(errs) else None, color="#55A868",
            capsize=2)  # fmt: skip
    ax.set_yticks(y, [label(i.name) for i in shown])
    ax.set_xlabel(label(xlabel))
    return fig


def beeswarm(output: ShapOutput) -> Figure | None:
    """SHAP value per row for the 20 preprocessed columns with the largest mean |SHAP|;
    colour is the column's value (min–max scaled per column; grey when not numeric).
    """
    values = output.values
    if values.size == 0:
        return None
    order = np.argsort(-np.abs(values).mean(axis=0), kind="stable")[:TOP_BARS]
    fig = _figure(6.4, 1.0 + 0.3 * len(order))
    ax = fig.add_subplot()
    rng = np.random.default_rng(0)
    for row, j in enumerate(order[::-1]):
        col = output.data[:, j]
        finite = np.isfinite(col)
        span = np.ptp(col[finite]) if finite.any() else 0.0
        scaled = (
            np.where(finite, (col - np.nanmin(col)) / span if span else 0.5, np.nan)
            if finite.any()
            else np.full(len(col), np.nan)
        )
        jitter = row + rng.uniform(-0.25, 0.25, len(col))
        ax.scatter(values[finite, j], jitter[finite], c=scaled[finite], cmap="coolwarm", s=6,
                   vmin=0.0, vmax=1.0)  # fmt: skip
        ax.scatter(values[~finite, j], jitter[~finite], color="#999999", s=6)
    ax.set_yticks(range(len(order)), [label(output.columns[j]) for j in order[::-1]])
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    title = "SHAP value (contribution to the model output)"
    ax.set_xlabel(label(f"{title}, class {output.label}" if output.label else title))
    return fig


def confusion_chart(errors: ErrorAnalysis) -> Figure | None:
    """Normalized OOF confusion matrix (rows = true class)."""
    matrix = errors.confusion_normalized
    if matrix is None:
        return None
    k = len(errors.labels)
    fig = _figure(1.5 + 0.6 * k, 1.2 + 0.6 * k)
    ax = fig.add_subplot()
    ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ticks = [label(x) for x in errors.labels]
    ax.set_xticks(range(k), ticks, rotation=45, ha="right")
    ax.set_yticks(range(k), ticks)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(k):
        for j in range(k):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[i, j] > 0.5 else "black",
            )
    return fig


def _positive(oof: NDArray[Any], y: NDArray[Any], problem: Problem) -> tuple[Any, Any]:
    p = np.asarray(oof, dtype=float)
    p = p[:, problem.positive_index] if p.ndim == 2 else p
    return (np.asarray(y) == problem.positive_index).astype(int), p


def roc_pr(oof: NDArray[Any], y: NDArray[Any], problem: Problem) -> tuple[Figure, Figure]:
    """ROC and precision–recall curves of the binary OOF predictions."""
    from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: PLC0415

    truth, p = _positive(oof, y, problem)
    fpr, tpr, _ = roc_curve(truth, p)
    roc = _figure(4.8, 4.0)
    ax = roc.add_subplot()
    ax.plot(fpr, tpr, color="#4C72B0")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    precision, recall, _ = precision_recall_curve(truth, p)
    pr = _figure(4.8, 4.0)
    ax2 = pr.add_subplot()
    ax2.plot(recall, precision, color="#4C72B0")
    ax2.axhline(float(truth.mean()), linestyle="--", color="#999999")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    return roc, pr


def reliability(oof: NDArray[Any], y: NDArray[Any], problem: Problem) -> Figure:
    """10-bin reliability curve: positive class (binary) or top-label confidence (multiclass)."""
    p = np.asarray(oof, dtype=float)
    if problem.task == Task.BINARY:
        hit, conf = _positive(oof, y, problem)
    else:
        conf = p.max(axis=1)
        hit = (np.argmax(p, axis=1) == np.asarray(y)).astype(int)
    edges = np.linspace(0.0, 1.0, RELIABILITY_BINS + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, RELIABILITY_BINS - 1)
    xs, ys = [], []
    for b in range(RELIABILITY_BINS):
        mask = bins == b
        if mask.any():
            xs.append(float(conf[mask].mean()))
            ys.append(float(hit[mask].mean()))
    fig = _figure(4.8, 4.0)
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    ax.plot(xs, ys, marker="o", color="#4C72B0")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    return fig


def regression_charts(errors: ErrorAnalysis) -> dict[str, Figure]:
    """Residuals vs predicted, residual histogram, actual vs predicted (OOF)."""
    pred, actual, resid = errors.predicted, errors.actual, errors.residuals
    if pred is None or actual is None or resid is None:
        return {}
    out = {}
    fig = _figure(4.8, 4.0)
    ax = fig.add_subplot()
    ax.scatter(pred, resid, s=6, alpha=0.5, color="#4C72B0")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual − predicted)")
    out["residuals_vs_predicted"] = fig
    fig = _figure(4.8, 4.0)
    ax = fig.add_subplot()
    ax.hist(resid, bins=30, color="#4C72B0")
    ax.set_xlabel("Residual")
    out["residual_histogram"] = fig
    fig = _figure(4.8, 4.0)
    ax = fig.add_subplot()
    ax.scatter(actual, pred, s=6, alpha=0.5, color="#4C72B0")
    lo, hi = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#999999")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    out["actual_vs_predicted"] = fig
    return out


def render_figures(
    *,
    entries: Sequence[LeaderboardEntry],
    explanation: ExplanationResult | None,
    errors: ErrorAnalysis | None,
    oof: NDArray[Any] | None,
    y: NDArray[Any] | None,
    problem: Problem,
    out_dir: Path,
) -> dict[str, str]:
    """Write every applicable §21.2 figure; returns {name: path relative to the report dir}."""
    figures: dict[str, Figure | None] = {
        "leaderboard": leaderboard_chart(entries),
        "fold_scores": fold_strip(entries),
    }
    if explanation is not None:
        figures["shap_importance"] = importance_bar(explanation.global_importance,
                                                    "mean |SHAP| (source columns)")  # fmt: skip
        for k, output in enumerate(explanation.shap):
            figures[f"shap_beeswarm_{k + 1}"] = beeswarm(output)
        figures["permutation_importance"] = importance_bar(explanation.permutation,
                                                           "score drop when permuted")  # fmt: skip
        figures["gain_importance"] = importance_bar(explanation.gain, "native importance")
    if errors is not None:
        figures["confusion_matrix"] = confusion_chart(errors)
        figures.update(regression_charts(errors))
    if oof is not None and y is not None and problem.task != Task.REGRESSION:
        if problem.task == Task.BINARY:
            figures["roc_curve"], figures["pr_curve"] = roc_pr(oof, y, problem)
        figures["reliability"] = reliability(oof, y, problem)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, fig in figures.items():
        if fig is not None:
            path = out_dir / f"{name.replace('_', '-')}.png"  # no Markdown specials in links
            fig.savefig(path, format="png")
            written[name] = f"{out_dir.name}/{path.name}"
    return written
