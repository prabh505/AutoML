"""Leaderboard with fold details and the score chart (PRD §27.2, US-08, US-09).

Sortable table of valid models; fold scores in expanders; failed or skipped models in a separate
collapsed list; the selected model is marked "★ Selected" in text, not by color alone (§27.5),
and the selection rule is shown next to it. The chart shows mean ± std with the metric name and
direction in the axis label. Its caption gives the key numbers as a text alternative.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

from tabpilot.runtime.catalog import metric_direction
from tabpilot.schemas import LeaderboardEntry
from tabpilot.ui.text import md_escape

SELECTED = "★ Selected"


def _fmt(value: float | None) -> float | None:
    return None if value is None else round(value, 5)


def rows(entries: Sequence[LeaderboardEntry], selected: str | None) -> list[dict[str, Any]]:
    """One table row per valid entry (failed/skipped entries are listed separately)."""
    return [{
        "": SELECTED if e.entry_id == selected else "",
        "Model": e.model if isinstance(e.model, str) else e.model.value,
        "Features": e.feature_set, "Iteration": e.iteration,
        f"Mean {e.metric.name.value}": _fmt(e.mean), "Std": _fmt(e.std),
        "Folds": len(e.fold_scores), "Fit s": round(e.fit_seconds, 1),
        "Comparable": "yes" if e.comparable else "no", "Entry": e.entry_id,
    } for e in entries if e.status == "ok"]  # fmt: skip


def chart_spec(entries: Sequence[LeaderboardEntry]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Vega-Lite bar of the mean selection metric with ± std error bars."""
    valid = [e for e in entries if e.status == "ok" and e.mean is not None]
    metric = valid[0].metric.name if valid else None
    data = [{"entry": e.entry_id, "mean": e.mean, "low": e.mean - (e.std or 0.0),
             "high": e.mean + (e.std or 0.0)} for e in valid if e.mean is not None]  # fmt: skip
    title = f"{metric.value} ({metric_direction(metric)})" if metric else "score"
    spec = {
        "layer": [
            {"mark": "bar", "encoding": {
                "y": {"field": "entry", "type": "nominal", "sort": "-x", "title": None},
                "x": {"field": "mean", "type": "quantitative", "title": title}}},
            {"mark": {"type": "errorbar", "ticks": True}, "encoding": {
                "y": {"field": "entry", "type": "nominal", "sort": "-x"},
                "x": {"field": "low", "type": "quantitative", "title": title},
                "x2": {"field": "high"}}},
        ],
    }  # fmt: skip
    return data, spec


def caption(entries: Sequence[LeaderboardEntry], selected: str | None) -> str:
    """Text alternative for the chart (§27.5)."""
    valid = [e for e in entries if e.status == "ok" and e.mean is not None]
    if not valid:
        return "No model produced a valid score."
    parts = [f"{e.entry_id}: {e.mean:.4f} ± {(e.std or 0.0):.4f}"
             + (" (selected)" if e.entry_id == selected else "") for e in valid]  # fmt: skip
    return "Mean ± std over folds — " + "; ".join(parts)


def render(entries: Sequence[LeaderboardEntry], selected: str | None,
           selection: dict[str, Any]) -> None:  # fmt: skip
    table = rows(entries, selected)
    if not table:
        st.info("No model produced a valid score.")
    else:
        st.dataframe(table, hide_index=True)
    rule = selection.get("rule")
    if rule:
        st.caption(f"{SELECTED}: chosen by the rule “{md_escape(rule)}”. Selection uses computed "
                   "scores only, never the LLM.")  # fmt: skip
    data, spec = chart_spec(entries)
    if data:
        st.vega_lite_chart(data, spec, alt=caption(entries, selected))
        st.caption(md_escape(caption(entries, selected)))
    for e in entries:
        if e.status == "ok" and e.fold_scores:
            with st.expander(f"Fold scores — {e.entry_id}"):
                st.dataframe([{"Fold": i + 1, "Score": round(s, 5)}
                              for i, s in enumerate(e.fold_scores)], hide_index=True)  # fmt: skip
    failed = [e for e in entries if e.status != "ok"]
    if failed:
        with st.expander(f"Failed or skipped models ({len(failed)})", expanded=False):
            st.dataframe([{"Entry": e.entry_id, "Status": e.status,
                           "Reason": "; ".join(e.warnings)[:300]} for e in failed],
                         hide_index=True)  # fmt: skip
