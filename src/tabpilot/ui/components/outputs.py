"""Results-page sections: summary, data profile, plan, explainability, report, downloads
(PRD §27.1 Results tabs, §27.2, §24.7 "Delete now", US-10, US-11, FR-EXP-01, US-F8).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from tabpilot.runtime import RunArtifacts
from tabpilot.schemas import EventType, LeaderboardEntry, RunWarning
from tabpilot.ui.components.columns import warning_text
from tabpilot.ui.text import md_escape, quote, size

JOBLIB_WARNING = ("model.joblib is a Python pickle. Loading it runs code, so only load files "
                  "you trust, with the library versions in requirements.txt.")  # fmt: skip
NO_HOLDOUT = "No independent test set (dataset < 2,000 rows)."
METHOD_TEXT = {
    "shap_tree": "SHAP (tree explainer)", "shap_linear": "SHAP (linear explainer)",
    "shap_member_plus_permutation": "SHAP on the ensemble's members plus permutation importance",
    "permutation": "permutation importance", "gain": "the model's built-in importance (gain)",
    "none": "no explanation",
}  # fmt: skip


def selected_entry(art: RunArtifacts) -> LeaderboardEntry | None:
    chosen = art.selected_entry_id
    return next((e for e in art.leaderboard if e.entry_id == chosen), None)


def _score(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float | int) else str(value)


def holdout_lines(holdout: dict[str, Any]) -> list[str]:
    """US-10: holdout results in their own section (or the reason there is none)."""
    if not holdout.get("present"):
        return [
            NO_HOLDOUT
            if "2,000" in str(holdout.get("reason", "")) or not holdout.get("reason")
            else f"No independent test set: {holdout['reason']}"
        ]
    out = []
    for label, key in (("Selected model", "final"), ("Baseline", "baseline")):
        scores = holdout.get(key) or {}
        items = [f"{k} {_score(v)}" for k, v in scores.items() if k != "ci95"]
        if "ci95" in scores:
            low, high = scores["ci95"]
            items.append(f"95% CI {low:.4f}–{high:.4f}")
        if items:
            out.append(f"{label}: " + ", ".join(items))
    return out


def summary(art: RunArtifacts, warnings: Sequence[RunWarning]) -> None:
    entry = selected_entry(art)
    if entry is not None:
        model = entry.model if isinstance(entry.model, str) else entry.model.value
        st.metric(f"Cross-validated {entry.metric.name.value} (selected: {model}, "
                  f"{entry.feature_set})", f"{_score(entry.mean)} ± {_score(entry.std or 0.0)}",
                  border=True)  # fmt: skip
    st.markdown("**Cross-validation (development data)**")
    st.write(f"{sum(e.status == 'ok' for e in art.leaderboard)} valid configurations compared "
             "on the same folds.")  # fmt: skip
    st.markdown("**Independent test set (holdout, scored once)**")
    for text in holdout_lines(art.holdout):
        st.write(text)
    all_warnings = [f"{w.code.value}: {w.message}" for w in warnings]
    if art.report is not None:  # raw `W-CODE:column` tokens are explained (KI-063)
        all_warnings += [warning_text(w) if w.startswith("W-") else w
                         for w in art.report.sections.warnings]  # fmt: skip
    if all_warnings:
        st.markdown("**Warnings**")
        st.markdown("\n".join(f"- {md_escape(w)}" for w in all_warnings))


def data_profile(art: RunArtifacts, columns: list[dict[str, Any]] | None) -> None:
    sections = art.report.sections if art.report else None
    if sections is not None:
        st.json(sections.summary, expanded=True)
        if sections.data_decisions:
            st.markdown("**Data decisions**")
            st.dataframe([{k: json.dumps(v) if isinstance(v, dict | list) else v
                           for k, v in d.items()} for d in sections.data_decisions],
                         hide_index=True)  # fmt: skip
    if columns:
        st.markdown("**Columns**")
        st.dataframe(columns, hide_index=True)


LLM_UNAVAILABLE = "LLM unavailable — rule-based plan used"  # US-F2 wording


def plan_source_text(art: RunArtifacts, source: object) -> str:
    """LLM plan, rule-based by choice, or the US-F2 fallback with its reason (KI-061)."""
    if source == "llm":
        return "LLM planner (validated)"
    llm = art.experiment.get("config", {}).get("llm", {})
    if not llm.get("enabled"):
        return "rule-based plan (No LLM was selected)"
    reasons = [str(e.data.get("reason", "")) for e in art.events
               if e.type == EventType.PLANNER_FALLBACK and e.data.get("reason")]  # fmt: skip
    return LLM_UNAVAILABLE + (f" ({reasons[0]})" if reasons else "")


def figure_caption(art: RunArtifacts, caption: str) -> str:
    """Figure caption with its key numbers as a text alternative (§27.5, KI-063)."""
    expl = art.report.sections.explanations if art.report else None
    lowered = caption.lower()
    if expl is not None and expl.top_features and any(
            k in lowered for k in ("shap", "importance", "permutation")):  # fmt: skip
        return f"{caption} — top features: {', '.join(expl.top_features[:5])}"
    entry = selected_entry(art)
    if entry is not None and entry.mean is not None:
        return (f"{caption} — selected model, cross-validated {entry.metric.name.value} "
                f"{entry.mean:.4f} ± {(entry.std or 0.0):.4f}")  # fmt: skip
    return caption


def plan(art: RunArtifacts) -> None:
    source = art.experiment.get("plan_source") or (
        art.report.sections.plan.get("source") if art.report else None
    )
    st.write("Plan source: " + plan_source_text(art, source))
    for event in art.events:
        if event.type == EventType.LLM_RATIONALE:
            source = md_escape(event.data.get("source", ""))
            text = quote(event.data.get("text", ""))
            st.info(f"**LLM-generated rationale ({source})**\n\n{text}",
                    icon=":material/smart_toy:")  # fmt: skip
    if art.report is not None and art.report.sections.plan:
        st.json(art.report.sections.plan, expanded=False)
    validation = art.experiment.get("plan_validation")
    if validation:
        with st.expander("Plan validation"):
            st.json(validation, expanded=False)


def explainability(art: RunArtifacts) -> None:
    expl = art.report.sections.explanations if art.report else None
    if expl is not None:
        st.write(f"Method: {METHOD_TEXT.get(expl.method, expl.method)}.")
        if expl.method not in ("shap_tree", "shap_linear"):
            st.warning("SHAP was not available for this model; a fallback method is shown."
                       + (f" Reason: {md_escape(expl.unavailable_reason)}."
                          if expl.unavailable_reason else ""))  # fmt: skip
        if expl.top_features:
            st.caption("Top features: " + md_escape(", ".join(expl.top_features)))
    if not art.figures:
        st.info("No charts were produced for this run.")
    for fig in art.figures:
        text = figure_caption(art, fig.caption)
        st.image(str(fig.path), caption=text, alt=text)


def report(art: RunArtifacts) -> None:
    if art.report is None:
        st.info("The report is not available.")
        return
    narrative = art.report.sections.narrative
    label = ("Written by the LLM and fact-checked against the computed results"
             if narrative.source == "llm" else "Written from a template (no LLM)")  # fmt: skip
    st.caption(label)
    for paragraph in narrative.text.split("\n\n"):
        if paragraph.strip():
            st.markdown(md_escape(paragraph))  # names in the text cannot form links/markup
    if art.report.sections.limitations:
        st.markdown("**Limitations**")
        st.markdown("\n".join(f"- {md_escape(x)}" for x in art.report.sections.limitations))
    for path, mime in ((art.report_html, "text/html"), (art.report_md, "text/markdown")):
        if path is not None:
            st.download_button(f"Download {path.name}", data=_reader(path), file_name=path.name,
                               mime=mime, key=f"dl_{path.name}")  # fmt: skip


def _reader(path: Path) -> Callable[[], bytes]:
    return path.read_bytes


def downloads(art: RunArtifacts, on_delete: Callable[[], None] | None) -> None:
    """Bundle with size and SHA-256, report download, joblib warning, "Delete my data now"."""
    if art.bundle is None:
        st.info("No bundle is available for this result." if art.precomputed is None else
                "Precomputed sample results have no download. Run the sample yourself to get "
                "the bundle.")  # fmt: skip
    else:
        st.download_button(f"Download bundle ({size(art.bundle.size_bytes)})",
                           data=_reader(art.bundle.path), file_name=art.bundle.path.name,
                           mime="application/zip", type="primary", key="dl_bundle")  # fmt: skip
        st.caption("SHA-256 of the ZIP:")
        st.code(art.bundle.sha256, language=None)
        st.warning(JOBLIB_WARNING, icon=":material/warning:")
    if art.report_html is not None:
        st.download_button("Download the report (HTML)", data=_reader(art.report_html),
                           file_name="report.html", mime="text/html", key="dl_report")  # fmt: skip
    if on_delete is not None:
        st.button("Delete my data now", on_click=on_delete, key="delete_run",
                  help="Deletes the run directory: upload, model, report and bundle.")  # fmt: skip
